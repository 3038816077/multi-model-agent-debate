"""研讨增益对照实验：单模型 / 并行 / 研讨，三种模式同题对比。

回答一个从没被量化过的问题：多模型研讨到底比单模型好多少，值不值它的成本。

用法（先起服务）：

    python run.py                                # 另开一个终端
    python scripts/benchmark_debate.py --base http://127.0.0.1:8000

先小规模验证接线：

    python scripts/benchmark_debate.py --limit 2 --dry-run

它做四件事：
  1. 同一批题分别用「单模型 / 并行 / 研讨」各跑一遍
  2. 每次请求前后对比 data/usage.jsonl 的新增行，算出这一轮的调用数、token、真实成本
  3. 把三份答案打乱顺序，交给裁判模型匿名打分（裁判不知道哪份来自哪种模式）
  4. 输出一张「质量 / 成本 / 质量成本比」的表

注意：
  - 裁判模型不是绝对真理。跑完抽 3-5 条看 reason，确认和你自己的判断一致。
  - 单模型基线默认用服务端的 DEFAULT_MODEL_ID。要对比强模型，就用 --single-model 指定；
    想让对比更公平，可以让 baseline 和 chief 用同一个模型。
  - 成本按 data/usage.jsonl 的增量统计，所以跑的时候别同时有别人在调这个服务。
  - 裁判的调用成本不计入表里（它不走 /chat 接口），实际开销会略高于表内数字。
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

USAGE_LOG = os.path.join(REPO_ROOT, "data", "usage.jsonl")
JUDGE_PROMPT = """你在评审对同一个问题的 THREE 份回答，标为 甲 / 乙 / 丙，顺序已打乱。
其中有些回答可能是多份并列答案拼在一起的（多个模型各自作答），这本身不代表它更好或更差。

评分依据（参考要点，不是唯一正确答案，允许合理补充）：

{ref}

评分维度（1-10 分）：
- 准确性 50%：有没有与事实或参考要点冲突的错误。有硬错误直接低分。
- 覆盖度 30%：命中了参考要点里的几项。
- 可执行性 20%：能不能直接照着做。

明确规则：
- 参考要点之外的额外内容，如果不相关，不加分也不算错。
- **篇幅长短不是评分因素**，不要因为更长或更短就给更高或更低的分。
- 多份答案拼接造成的重复，最多扣 1 分。

只输出 JSON，不要任何解释文字：
{"甲": {"score": 8, "hit": "命中的要点", "error": "错误或无", "reason": "一句话"}, "乙": {...}, "丙": {...}}
"""


# ---------------- 基础工具 ----------------

def load_api_key():
    key = os.getenv("API_KEY")
    if key:
        return key
    try:
        with open(os.path.join(REPO_ROOT, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ""


def usage_len():
    if not os.path.exists(USAGE_LOG):
        return 0
    with open(USAGE_LOG, "rb") as f:
        return sum(1 for _ in f)


def usage_index():
    """把 usage.jsonl 读成 {trace_id: 这条请求的调用数/成本}。

    按 trace_id 归因，而不是按文件行数增量——这样并发跑题也不会串账。
    """
    idx = {}
    if not os.path.exists(USAGE_LOG):
        return idx
    with open(USAGE_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = rec.get("trace_id") or ""
            if not tid:
                continue
            cell = idx.setdefault(tid, dict(calls=0, tokens_in=0, tokens_out=0, cost=0.0))
            cell["calls"] += 1
            cell["tokens_in"] += int(rec.get("prompt_tokens") or 0)
            cell["tokens_out"] += int(rec.get("completion_tokens") or 0)
            cell["cost"] += float(rec.get("cost_usd") or 0)
    return idx


def post_json(url, payload, api_key, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key or "",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_sse(url, payload, api_key, timeout):
    """读 SSE 流，返回解析后的事件列表。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key or "",
        "Accept": "text/event-stream",
    })
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body:
                continue
            try:
                events.append(json.loads(body))
            except Exception:
                continue
    return events


# ---------------- 三种模式 ----------------

def run_single(base, question, args, api_key):
    payload = {"message": question, "auto_route": False}
    if args.single_model:
        payload["user_model_name"] = args.single_model
    t0 = time.time()
    resp = post_json(base + "/chat", payload, api_key, args.timeout)
    elapsed = time.time() - t0
    return resp.get("answer") or resp.get("error") or "", elapsed, resp.get("trace_id")


def run_group(base, question, mode, args, api_key):
    payload = {
        "message": question,
        "chief_model": args.chief,
        "staff_models": args.staff,
        "mode": mode,
    }
    t0 = time.time()
    events = post_sse(base + "/chat/group_stream", payload, api_key, args.timeout)
    elapsed = time.time() - t0

    speaks, summary, divergence, errors = [], "", "", []
    trace_id = None
    for ev in events:
        kind = ev.get("event")
        if kind == "speak":
            speaks.append((ev.get("model"), ev.get("content") or ""))
        elif kind == "summary":
            summary = ev.get("content") or summary
        elif kind == "divergence":
            divergence = ev.get("content") or divergence
        elif kind == "done":
            trace_id = ev.get("trace_id") or trace_id
        elif kind == "error":
            errors.append(str(ev.get("data")))

    first_round = "\n\n".join("[%s]\n%s" % (m, c) for m, c in speaks)

    if mode == "debate":
        # 用户最终读到的是首席总结；没有总结时退回第一轮
        answer = summary or first_round
    else:
        # 并行模式没有总结，用户读到的是四份独立答案 + 共识分歧
        answer = first_round
        if divergence:
            answer += "\n\n[共识与分歧]\n" + divergence

    if not answer and errors:
        answer = "[请求失败] " + "; ".join(errors)
    return answer, elapsed, len(speaks), trace_id


# ---------------- 裁判 ----------------

def judge_three(question, answers, judge_client, judge_model, timeout, ref=""):
    """answers: {mode: text}。打乱顺序后匿名打分，返回 {mode: (score, reason)}。"""
    modes = list(answers.keys())
    labels = ["甲", "乙", "丙"][:len(modes)]
    shuffled = modes[:]
    random.shuffle(shuffled)
    label_of = {labels[i]: shuffled[i] for i in range(len(modes))}

    blocks = []
    for lab in labels:
        blocks.append("【" + lab + "】\n" + (answers[label_of[lab]] or "(空)")[:3000])
    user_content = "问题：\n" + question + "\n\n" + "\n\n".join(blocks)

    resp = judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT.replace("{ref}", ref or "（本题未提供参考要点，请按你的知识判断）")},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = text[text.find("{"):text.rfind("}") + 1] if "{" in text else "{}"
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}

    result = {}
    for lab in labels:
        item = parsed.get(lab) or {}
        score = item.get("score")
        result[label_of[lab]] = (float(score) if isinstance(score, (int, float)) else None,
                                 item.get("reason") or "(解析失败)")
    return result


# ---------------- 主流程 ----------------

def run_one(item, args, api_key, base, modes, judge_client, judge_model):
    """跑一道题的三种模式并打分，返回一行结果。"""
    qid, question, ref = item.get("id"), item.get("q"), item.get("ref", "")
    answers, metas = {}, {}

    for mode in modes:
        trace_id = None
        try:
            if args.dry_run:
                answers[mode] = "(dry-run) 假的%s答案" % mode
                elapsed, speakers = 0.0, 0
            elif mode == "single":
                answers[mode], elapsed, trace_id = run_single(base, question, args, api_key)
                speakers = 1
            else:
                answers[mode], elapsed, speakers, trace_id = run_group(base, question, mode, args, api_key)
        except Exception as e:
            answers[mode] = "[调用失败] %s" % e
            elapsed, speakers = 0.0, 0
        metas[mode] = dict(elapsed=elapsed, speakers=speakers, trace_id=trace_id)

    if args.dry_run:
        scores = {m: (random.uniform(6, 9), "dry-run") for m in modes}
    else:
        try:
            scores = judge_three(question, answers, judge_client, judge_model, args.timeout, ref=ref)
        except Exception as e:
            scores = {m: (None, "judge failed: %s" % str(e)[:80]) for m in modes}

    row = dict(id=qid, cat=item.get("cat"), q=question, ref=ref, modes={})
    for m in modes:
        sc, reason = scores.get(m, (None, ""))
        row["modes"][m] = dict(answer=answers[m], score=sc, reason=reason, **metas[m])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default=os.path.join(REPO_ROOT, "eval", "debate_eval_set.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    ap.add_argument("--chief", default=os.getenv("BENCH_CHIEF", "qwen-plus"))
    ap.add_argument("--staff", nargs="+", default=[x for x in os.getenv("BENCH_STAFF", "").split(",") if x],
                    help="副模型列表，空格分隔")
    ap.add_argument("--single-model", default=os.getenv("BENCH_SINGLE", ""),
                    help="单模型基线用哪个模型（留空则用服务端默认）")
    ap.add_argument("--judge-model", default=os.getenv("BENCH_JUDGE", ""),
                    help="裁判模型（建议用比被测模型更强的）")
    ap.add_argument("--modes", default="single,parallel,debate")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="同时跑几道题（服务端记得 RATE_LIMIT_ENABLED=0）")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "debate_bench.json"))
    ap.add_argument("--dry-run", action="store_true", help="不真的调用，用假答案跑通流程")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    api_key = load_api_key()
    base = args.base.rstrip("/")

    questions = []
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit:
        questions = questions[:args.limit]
    if args.dry_run and args.out.endswith(".json"):
        args.out = args.out[:-5] + ".dryrun.json"

    if not args.dry_run and not args.staff:
        raise SystemExit("缺少 --staff（副模型列表），例如 --staff qwen-plus deepseek-chat")

    judge_client, judge_model = None, args.judge_model
    if not args.dry_run:
        from openai import OpenAI
        from app.core.config import get_model_registry
        registry = get_model_registry()
        entry = registry.get(judge_model) or {}
        if not entry:
            raise SystemExit("裁判模型 %s 不在 MODEL_REGISTRY 里，用 --judge-model 指定一个" % judge_model)
        judge_client = OpenAI(api_key=entry.get("api_key"), base_url=entry.get("base_url"),
                              timeout=args.timeout)
        judge_model = entry.get("model") or judge_model

    t_start = time.time()
    rows, done = [], 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(run_one, item, args, api_key, base, modes,
                               judge_client, judge_model): item for item in questions}
        for fut in as_completed(futures):
            item = futures[fut]
            done += 1
            try:
                row = fut.result()
            except Exception as e:
                print("[%d/%d] %s 失败: %s" % (done, len(questions), item.get("id"), e))
                continue
            rows.append(row)
            brief = " ".join("%s=%s" % (m, ("%.1f" % row["modes"][m]["score"])
                                        if row["modes"][m]["score"] is not None else "?")
                             for m in modes)
            print("[%d/%d] %s %-4s %s  (%.0fs)" % (
                done, len(questions), row["id"], row.get("cat") or "", brief, time.time() - t_start))

    # 成本按 trace_id 归因
    idx = {} if args.dry_run else usage_index()
    for r in rows:
        for m, v in r["modes"].items():
            cell = idx.get(v.get("trace_id") or "", dict(calls=0, tokens_in=0, tokens_out=0, cost=0.0))
            v["cost"] = cell["cost"]
            v["calls"] = cell["calls"]
            v["tokens_in"] = cell["tokens_in"]
            v["tokens_out"] = cell["tokens_out"]

    rows.sort(key=lambda r: str(r["id"]))
    print("")
    print("=" * 84)
    print("%-10s %8s %12s %10s %10s %8s" % ("模式", "平均分", "平均成本($)", "平均耗时(s)", "平均调用", "样本"))
    print("-" * 84)
    for m in modes:
        vals = [r["modes"][m] for r in rows if m in r["modes"]]
        if not vals:
            continue
        sc = [v["score"] for v in vals if v.get("score") is not None]
        avg_score = sum(sc) / len(sc) if sc else float("nan")
        avg_cost = sum(v["cost"] for v in vals) / len(vals)
        avg_time = sum(v["elapsed"] for v in vals) / len(vals)
        avg_calls = sum(v["calls"] for v in vals) / len(vals)
        ratio = ("%.0f" % (avg_score / avg_cost)) if avg_cost > 0 else "-"
        print("%-10s %8.2f %12.6f %10.1f %10.1f %8d   (分/美元 %s)"
              % (m, avg_score, avg_cost, avg_time, avg_calls, len(sc), ratio))
    print("=" * 84)
    print("总耗时 %.1f 分钟 | 并发 %d" % ((time.time() - t_start) / 60, args.concurrency))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), rows=rows), f, ensure_ascii=False, indent=2)
    print("明细已写入: %s" % args.out)
    print("提示：裁判分不是真理。看 analyze_bench.py 的分类表，再抽 3-5 条核对 reason。")


if __name__ == "__main__":
    main()
