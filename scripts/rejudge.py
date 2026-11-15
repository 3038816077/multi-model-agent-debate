"""给已有结果重新打分（不重新生成答案）。

用途：某次评测答案都跑完了，但裁判模型中途额度耗尽，分数不全。
重跑生成很贵，重新打分只要几分钱。

用法：
    python scripts/rejudge.py --in data/debate_bench_family.json \
        --judge-model qwen-plus --out data/debate_bench_family_scored.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_debate import judge_three  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", default="")
    ap.add_argument("--judge-model", default="qwen-plus")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    data = json.load(open(args.src, encoding="utf-8"))
    rows = data["rows"]
    modes = [m for m in ("single", "parallel", "debate") if m in rows[0]["modes"]]

    from openai import OpenAI
    from app.core.config import get_model_registry
    reg = get_model_registry()
    entry = reg.get(args.judge_model) or {}
    if not entry:
        raise SystemExit("裁判模型 %s 不在注册表里" % args.judge_model)
    client = OpenAI(api_key=entry.get("api_key"), base_url=entry.get("base_url"),
                    timeout=args.timeout)
    model = entry.get("model") or args.judge_model

    print("裁判: %s (%s) | 待打分 %d 题" % (args.judge_model, model, len(rows)))

    def score_one(row):
        answers = {m: row["modes"][m]["answer"] for m in modes}
        t0 = time.time()
        try:
            res = judge_three(row["q"], answers, client, model, args.timeout,
                              ref=row.get("ref", ""))
        except Exception as e:
            return row, None, "judge failed: %s" % str(e)[:80], time.time() - t0
        return row, res, "", time.time() - t0

    t_start = time.time()
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(score_one, r) for r in rows]
        for fut in as_completed(futures):
            row, res, err, dt = fut.result()
            done += 1
            if err or not res:
                failed += 1
                print("[%d/%d] %s 失败: %s" % (done, len(rows), row["id"], err))
                continue
            brief = " ".join("%s=%s" % (m, res[m][0]) for m in modes if m in res)
            print("[%d/%d] %s %s (%.0fs)" % (done, len(rows), row["id"], brief, dt))
            for m in modes:
                if m in res:
                    row["modes"][m]["score"], row["modes"][m]["reason"] = res[m]

    out = args.dst or args.src.replace(".json", "_scored.json")
    data.setdefault("judge", {})["model"] = args.judge_model
    data["judge"]["rejudged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("")
    print("=" * 70)
    for m in modes:
        sc = [r["modes"][m]["score"] for r in rows if r["modes"][m].get("score") is not None]
        print("%-10s 平均 %.2f  (有效 %d/%d)" % (m, sum(sc) / len(sc) if sc else 0, len(sc), len(rows)))
    print("失败 %d 题 | 耗时 %.1f 分钟" % (failed, (time.time() - t_start) / 60))
    print("已写入: %s" % out)


if __name__ == "__main__":
    main()
