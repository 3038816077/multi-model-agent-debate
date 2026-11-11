"""模型分层路由的对照实验脚本。

用法（先起服务）：
    SERVER_PORT=8010 AUTO_ROUTE=true python run.py
    python scripts/benchmark_route.py --base http://127.0.0.1:8010

它做两件事：
  1. 把同一批任务分别用「关路由」和「开路由」各跑一遍
  2. 从 data/usage.jsonl 里按 trace_id 取出每次调用的 token 与成本，汇总对比

输出一张表：请求数 / 总 token / 总成本 / 平均延迟 / 每档命中数。
答案质量需要人工打分（建议 1-10），脚本不自动评判。
"""

import argparse
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from collections import Counter

TASKS = [
    # (任务说明, 提问, 期望档位)
    ("翻译", "把下面这句话翻译成英文：这个项目支持多模型研讨。", "cheap"),
    ("翻译", "把下面这句话翻译成日文：今天天气不错，适合出门。", "cheap"),
    ("格式化", "请把这段 JSON 格式化一下：{\"a\":1,\"b\":[2,3]}", "cheap"),
    ("归类", "把这些问题分成三类：登录失败、支付异常、页面卡顿。", "cheap"),
    ("闲聊", "你好", "cheap"),
    ("改写", "帮我把这句话改写得更正式：这玩意儿不好使。", "cheap"),
    ("摘要", "帮我把下面这段话总结成一句话：人工智能正在改变软件开发的流程。", "cheap"),
    ("提取", "从这句话里提取人名：张三和李四昨天去了北京。", "cheap"),
    ("代码", "这段 Python 为什么报 KeyError？给出排查步骤。", "strong"),
    ("代码", "用 LangGraph 实现一个人工审核节点，写出关键代码。", "strong"),
    ("架构", "设计一个支持多租户的权限模型，说明表结构和扩展方式。", "strong"),
    ("分析", "分析一下多模型研讨相比单模型的实际收益和成本。", "strong"),
    ("对比", "对比 BM25 和向量检索在专业文档场景下的优劣。", "strong"),
    ("推理", "一个班有 40 人，60% 及格，及格的人里一半是女生，问女生及格多少人？", "strong"),
    ("方案", "给一个日活 10 万的客服系统做降本方案。", "strong"),
    ("长文", "请从架构、成本、可维护性、团队能力四个维度评估是否应该自研 Agent 框架，并给出结论。" * 3, "strong"),
]


def load_key():
    """优先环境变量，其次从 .env 里读 API_KEY。"""
    key = os.getenv("API_KEY")
    if key:
        return key
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ""


def ask(base, key, message, auto_route, force_model=None, timeout=180):
    body = {
        "message": message,
        "session_id": str(uuid.uuid4()),
        "auto_route": auto_route,
    }
    if force_model:
        # A 组固定用同一个强模型，两组才有可比性
        body["user_model_name"] = force_model
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat",
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": key},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}", "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:80], "ms": int((time.time() - t0) * 1000)}
    return {
        "ok": data.get("code") == 200,
        "error": data.get("error"),
        "ms": int((time.time() - t0) * 1000),
        "trace_id": data.get("trace_id"),
        "routed_model": data.get("routed_model"),
        "answer": data.get("answer", ""),
    }


def read_usage(path):
    """把 usage.jsonl 读成 {trace_id: record}，文件不存在就返回空。"""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.setdefault(rec.get("trace_id"), []).append(rec)
    except FileNotFoundError:
        pass
    return out


def run_mode(base, key, tasks, auto_route, usage_path, force_model=None):
    rows = []
    for label, message, expect in tasks:
        before = read_usage(usage_path)
        r = ask(base, key, message, auto_route, force_model=force_model)
        time.sleep(0.2)
        after = read_usage(usage_path)
        trace = r.get("trace_id")
        new = [rec for rec in after.get(trace, []) if rec not in before.get(trace, [])]
        rows.append({
            "label": label,
            "expect": expect,
            "ok": r.get("ok"),
            "error": r.get("error"),
            "wall_ms": r.get("ms"),
            "model": r.get("routed_model") or (new[0]["model"] if new else None),
            "tier": new[0].get("tier") if new else None,
            "prompt_tokens": sum(x.get("prompt_tokens", 0) for x in new),
            "completion_tokens": sum(x.get("completion_tokens", 0) for x in new),
            "cost_usd": sum(x.get("cost_usd", 0) for x in new),
            "answer": r.get("answer", ""),
        })
    return rows


def summarize(name, rows):
    ok = [r for r in rows if r["ok"]]
    total_cost = sum(r["cost_usd"] for r in rows)
    avg_ms = int(sum(r["wall_ms"] for r in rows) / max(len(rows), 1))
    tiers = Counter(r["tier"] for r in rows if r["tier"])
    print(f"\n===== {name} =====")
    print(f"成功 {len(ok)}/{len(rows)}   总成本 ${total_cost:.6f}   平均延迟 {avg_ms}ms")
    print(f"档位分布: {dict(tiers)}")
    print(f"{'任务':8} {'档位':7} {'模型':16} {'in':>6} {'out':>5} {'成本$':>10} {'耗时':>7}")
    for r in rows:
        flag = "" if r["ok"] else "  ❌" + str(r["error"])[:40]
        print(f"{r['label']:8} {str(r['tier'] or '-'):7} {str(r['model'] or '-'):16} "
              f"{r['prompt_tokens']:>6} {r['completion_tokens']:>5} {r['cost_usd']:>10.6f} "
              f"{r['wall_ms']:>6}ms{flag}")
    return {"cost": total_cost, "avg_ms": avg_ms, "ok": len(ok), "total": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--usage", default="data/usage.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务，快速验证用")
    ap.add_argument("--baseline", default="gpt-6.1-sol",
                    help="A 组的固定模型（对照组），默认 gpt-6.1-sol")
    ap.add_argument("--save", default="data/bench_answers.json",
                    help="把两组答案存成文件，方便人工打分对比")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("⚠️  没找到 API_KEY，若服务端未开启鉴权可以忽略")

    tasks = TASKS[: args.limit] if args.limit else TASKS
    print(f"共 {len(tasks)} 个任务，base={args.base}")

    # 预检查：自动路由没生效就直接退出，避免跑完 8 分钟才发现数据作废
    probe = ask(args.base, key, "把这句话翻译成英文：路由预检查。", auto_route=True)
    if not probe.get("ok"):
        print(f"❌ 服务没起来或鉴权失败：{probe.get('error')}")
        return
    if not probe.get("routed_model"):
        print("❌ 服务端的自动路由没有生效，本次结果无效。")
        print("   检查：启动服务前是否设置了环境变量 AUTO_ROUTE=true")
        print("   Windows CMD 正确写法（注意不要写成一行加 &&，会带尾随空格）：")
        print("       set AUTO_ROUTE=true")
        print("       python run.py")
        return
    print(f"✅ 自动路由已生效，预检查命中 {probe['routed_model']}")

    off = run_mode(args.base, key, tasks, auto_route=False, usage_path=args.usage,
                   force_model=args.baseline)
    s_off = summarize(f"A 组：关闭路由（全部用 {args.baseline}）", off)
    on = run_mode(args.base, key, tasks, auto_route=True, usage_path=args.usage)
    s_on = summarize("B 组：开启路由（按档位选模型）", on)

    print("\n===== 结论 =====")
    if s_off["cost"] > 0:
        drop = (s_off["cost"] - s_on["cost"]) / s_off["cost"] * 100
        print(f"成本：${s_off['cost']:.6f} -> ${s_on['cost']:.6f}   降幅 {drop:.1f}%")
    print(f"平均延迟：{s_off['avg_ms']}ms -> {s_on['avg_ms']}ms")
    print("质量：需要人工打分或另接 judge 模型，脚本不自动评判")

    # 把两组答案存下来，方便逐条对比打分
    try:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({
                "baseline_model": args.baseline,
                "A_no_route": [{"task": r["label"], "answer": r["answer"]} for r in off],
                "B_routed": [{"task": r["label"], "answer": r["answer"]} for r in on],
            }, f, ensure_ascii=False, indent=2)
        print(f"两组答案已保存到 {args.save}")
    except Exception as e:
        print(f"保存答案失败：{e}")


if __name__ == "__main__":
    main()
