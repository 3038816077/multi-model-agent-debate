"""评测「分歧分析」本身：真实的矛盾有没有被点出来，没有矛盾时会不会编。

题集 eval/divergence_set.jsonl 每条都有明确的期望：
  conflict=true  两份答案结论相反  → 应该在【分歧】里指出，并命中预期关键词
  conflict=false 两份答案结论一致  → 不应该出现【分歧】

用法：
    python scripts/benchmark_divergence.py --models qwen-plus glm-4-flash
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI  # noqa: E402

from app.api.chat import (  # noqa: E402
    DIVERGENCE_ANSWER_MAX_CHARS,
    GROUP_DIVERGENCE_PROMPT,
    render_divergence,
)
from app.core.config import get_model_extra_body, get_model_registry, resolve_model_id  # noqa: E402


def analyze(item, model_key, registry, timeout=180):
    """用指定模型跑一次分歧分析，返回 (渲染文本, 耗时, tokens)。"""
    entry = registry[model_key]
    client = OpenAI(api_key=entry["api_key"], base_url=entry["base_url"], timeout=timeout)
    body = "\n\n".join(
        "【回答%d】\n%s" % (i + 1, ans[:DIVERGENCE_ANSWER_MAX_CHARS])
        for i, ans in enumerate(item["answers"])
    )
    prompt = GROUP_DIVERGENCE_PROMPT.format(
        n=len(item["answers"]), query=item["query"], answers=body
    )
    extra = get_model_extra_body(model_key)
    kwargs = {"extra_body": extra} if extra else {}
    started = time.time()
    resp = client.chat.completions.create(
        model=resolve_model_id(model_key),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=entry.get("max_tokens") or 800,
        **kwargs,
    )
    elapsed = time.time() - started
    raw = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0) if usage else 0
    return render_divergence(raw), elapsed, tokens


def grade(item, rendered):
    """返回 (是否正确, 说明)。"""
    has_conflict_section = "**分歧**" in rendered
    if item["conflict"]:
        if not has_conflict_section:
            return False, "漏报：该指出分歧却给了无冲突"
        hit = [k for k in item.get("keywords", []) if k.lower() in rendered.lower()]
        if hit:
            return True, "检出（命中 %s）" % "/".join(hit[:3])
        return False, "方向对但没命中预期要点"
    if has_conflict_section:
        return False, "误报：本无冲突却编出了分歧"
    return True, "正确判为无冲突"


def run_model(model_key, items, registry, workers):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze, it, model_key, registry): it for it in items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                rendered, elapsed, tokens = fut.result()
                ok, why = grade(item, rendered)
            except Exception as e:
                rendered, elapsed, tokens = "", 0.0, 0
                ok, why = False, "调用失败：%s" % str(e)[:60]
            rows.append({
                "id": item["id"], "conflict": item["conflict"], "ok": ok, "why": why,
                "elapsed": round(elapsed, 1), "tokens": tokens, "output": rendered,
            })
    return rows


def summarize(model_key, rows):
    conflict_rows = [r for r in rows if r["conflict"]]
    clean_rows = [r for r in rows if not r["conflict"]]
    tp = sum(1 for r in conflict_rows if r["ok"])
    tn = sum(1 for r in clean_rows if r["ok"])
    total = len(rows)
    return {
        "model": model_key,
        "total": total,
        "recall": round(tp / len(conflict_rows), 3) if conflict_rows else 0,
        "false_alarm": round(1 - tn / len(clean_rows), 3) if clean_rows else 0,
        "accuracy": round((tp + tn) / total, 3) if total else 0,
        "avg_seconds": round(sum(r["elapsed"] for r in rows) / total, 1) if total else 0,
        "avg_tokens": int(sum(r["tokens"] for r in rows) / total) if total else 0,
        "missed": [r["id"] for r in conflict_rows if not r["ok"]],
        "false_alarms": [r["id"] for r in clean_rows if not r["ok"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval/divergence_set.jsonl")
    ap.add_argument("--models", nargs="+", default=["qwen-plus", "glm-4-flash"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/divergence_bench.json")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.set, encoding="utf-8") if l.strip()]
    registry = get_model_registry()
    missing = [m for m in args.models if m not in registry]
    if missing:
        raise SystemExit("注册表里没有：%s" % missing)

    print("题集 %d 条（有冲突 %d / 无冲突 %d）| 参评模型 %s" % (
        len(items), sum(1 for i in items if i["conflict"]),
        sum(1 for i in items if not i["conflict"]), args.models))

    results = {}
    for model_key in args.models:
        t0 = time.time()
        rows = run_model(model_key, items, registry, args.workers)
        stats = summarize(model_key, rows)
        results[model_key] = {"stats": stats, "rows": rows}
        print("")
        print("== %s ==" % model_key)
        print("  准确率 %.1f%% | 检出率 %.1f%% | 误报率 %.1f%% | 平均 %.1fs | %d tokens" % (
            stats["accuracy"] * 100, stats["recall"] * 100,
            stats["false_alarm"] * 100, stats["avg_seconds"], stats["avg_tokens"]))
        if stats["missed"]:
            print("  漏报: %s" % ", ".join(stats["missed"]))
        if stats["false_alarms"]:
            print("  误报: %s" % ", ".join(stats["false_alarms"]))
        print("  耗时 %.1f 分钟" % ((time.time() - t0) / 60))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"items": len(items), "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("")
    print("已写入 %s" % args.out)


if __name__ == "__main__":
    main()
