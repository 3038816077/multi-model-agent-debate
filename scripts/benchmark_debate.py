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
JUDGE_PROMPT = """你是一名严格的技术答案评审。下面是对同一个问题 THREE 份回答，分别标为 甲 / 乙 / 丙，顺序已打乱。

评分标准（1-10 分）：
- 正确性 40%：有没有事实错误、逻辑错误
- 完整性 30%：有没有遗漏关键点
- 可执行性 20%：给出的内容能不能直接照着做
- 简洁性 10%：有没有注水

重要：不要因为回答更长就给更高分。有错误信息必须扣分。

只输出 JSON，不要任何解释文字，格式：
{"甲": {"score": 8, "reason": "一句话"}, "乙": {...}, "丙": {...}}
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


def usage_delta(start_line):
    """读 usage.jsonl 从 start_line 之后的新增记录，汇总调用数 / token / 成本。"""
    calls, tokens_in, tokens_out, cost = 0, 0, 0, 0.0
    if not os.path.exists(USAGE_LOG):
        return dict(calls=0, tokens_in=0, tokens_out=0, cost=0.0)
    with open(USAGE_LOG, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            calls += 1
            tokens_in += int(rec.get("prompt_tokens") or 0)
            tokens_out += int(rec.get("completion_tokens") or 0)
            cost += float(rec.get("cost_usd") or 0)
    return dict(calls=calls, tokens_in=tokens_in, tokens_out=tokens_out, cost=cost)


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
    return resp.get("answer") or resp.get("error") or "", elapsed


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
    for ev in events:
        kind = ev.get("event")
        if kind == "speak":
            speaks.append((ev.get("model"), ev.get("content") or ""))
        elif kind == "summary":
            summary = ev.get("content") or summary
        elif kind == "divergence":
            divergence = ev.get("content") or divergence
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
    return answer, elapsed, len(speaks)


# ---------------- 裁判 ----------------

def judge_three(question, answers, judge_client, judge_model, timeout):
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
            {"role": "system", "content": JUDGE_PROMPT},
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default=os.path.join(REPO_ROOT, "eval", "debate_eval_set.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    ap.add_argument("--chief", default=os.getenv("BENCH_CHIEF", "qwen-plus"))
    ap.add_argument("--staff", nargs="+", default=[s for s in os.getenv("BENCH_STAFF", "").split(",") if s],
                    help="副模型列表，空格分隔")
    ap.add_argument("--single-model", default=os.getenv("BENCH_SINGLE", ""),
                    help="单模型基线用哪个模型（留空则用服务端默认）")
    ap.add_argument("--judge-model", default=os.getenv("BENCH_JUDGE", ""),
                    help="裁判模型（建议用比被测模型更强的）")
    ap.add_argument("--modes", default="single,parallel,debate")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "debate_bench.json"))
    ap.add_argument("--settle", type=float, default=3.0,
                    help="每次请求后等待几秒再统计用量（后台记忆抽取是异步的）")
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
        judge_client = OpenAI(
            api_key=entry.get("api_key"), base_url=entry.get("base_url"), timeout=args.timeout)
        judge_model = entry.get("model") or judge_model

    rows = []
    for idx, item in enumerate(questions, 1):
        qid, question = item.get("id"), item.get("q")
        print("\n[%d/%d] %s  %s" % (idx, len(questions), qid, question[:40]))
        answers, stats = {}, {}

        for mode in modes:
            start = usage_len()
            try:
                if args.dry_run:
                    answers[mode] = "(dry-run) 这是一段假的%s答案，用于验证流程。" % mode
                    elapsed, speakers = 0.0, 0
                elif mode == "single":
                    answers[mode], elapsed = run_single(base, question, args, api_key)
                    speakers = 1
                else:
                    answers[mode], elapsed, speakers = run_group(base, question, mode, args, api_key)
            except Exception as e:
                answers[mode] = "[调用失败] %s" % e
                elapsed, speakers = 0.0, 0
            if not args.dry_run and args.settle > 0:
                time.sleep(args.settle)
            usage = usage_delta(start)
            stats[mode] = dict(elapsed=elapsed, speakers=speakers, **usage)
            print("   %-9s %6.1fs  调用 %d  成本 %.6f  %d 字"
                  % (mode, elapsed, usage["calls"], usage["cost"], len(answers[mode])))

        if args.dry_run:
            scores = {m: (random.uniform(6, 9), "dry-run") for m in modes}
        else:
            print("   裁判打分中 ...")
            try:
                scores = judge_three(question, answers, judge_client, judge_model, args.timeout)
            except Exception as e:
                print("   裁判失败: %s" % e)
                scores = {m: (None, "judge failed: %s" % str(e)[:80]) for m in modes}

        row = dict(id=qid, cat=item.get("cat"), q=question, modes={})
        for m in modes:
            row["modes"][m] = dict(
                answer=answers[m], stats=stats[m],
                score=scores.get(m, (None, ""))[0], reason=scores.get(m, (None, ""))[1])
        rows.append(row)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("%-10s %8s %12s %10s %10s %8s" % ("模式", "平均分", "平均成本($)", "平均耗时(s)", "平均调用", "样本"))
    print("-" * 78)
    for m in modes:
        vals = [r["modes"][m] for r in rows]
        sc = [v["score"] for v in vals if v["score"] is not None]
        avg_score = sum(sc) / len(sc) if sc else float("nan")
        avg_cost = sum(v["stats"]["cost"] for v in vals) / len(vals)
        avg_time = sum(v["stats"]["elapsed"] for v in vals) / len(vals)
        avg_calls = sum(v["stats"]["calls"] for v in vals) / len(vals)
        ratio = ("%.0f" % (avg_score / avg_cost)) if avg_cost > 0 else "-"
        print("%-10s %8.2f %12.6f %10.1f %10.1f %8d   (分/美元 %s)"
              % (m, avg_score, avg_cost, avg_time, avg_calls, len(sc), ratio))
    print("=" * 78)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), rows=rows), f, ensure_ascii=False, indent=2)
    print("\n明细已写入: %s" % args.out)
    print("提示：裁判分不是真理。抽 3-5 条看 reason 字段，确认和你自己的判断一致。")


if __name__ == "__main__":
    main()
