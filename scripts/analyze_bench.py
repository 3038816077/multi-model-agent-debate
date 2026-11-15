"""把 debate_bench.json 变成能看懂的结论。

用法：
    python scripts/analyze_bench.py --in data/debate_bench.json

输出三块：
  1. 总表：每模式的质量 / 成本 / 质量成本比
  2. 分类表：按题型拆开看研讨到底在哪类题上赢
  3. 诊断：长度与分数的相关性、分歧最大的题
"""

import argparse
import json
import os

MODES = ["single", "parallel", "debate"]


def cell(v, key):
    """兼容两种结构：新脚本把统计放在顶层，旧版本放在 stats 里。"""
    if "stats" in v:
        return v["stats"].get(key, 0)
    return v.get(key, 0)


def pearson(pairs):
    n = len(pairs)
    if n < 3:
        return 0.0
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    sx = (sum((x - mx) ** 2 for x, _ in pairs)) ** 0.5
    sy = (sum((y - my) ** 2 for _, y in pairs)) ** 0.5
    return cov / (sx * sy + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/debate_bench.json")
    ap.add_argument("--out", dest="dst", default="", help="可选：输出 markdown 文件")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit("找不到 %s，先跑 benchmark_debate.py" % args.src)
    data = json.load(open(args.src, encoding="utf-8"))
    rows = data["rows"]
    modes = [m for m in MODES if any(m in r["modes"] for r in rows)]

    lines = []
    add = lines.append

    add("## 总表（%d 题）" % len(rows))
    add("")
    add("| 模式 | 平均分 | 平均成本($) | 平均耗时(s) | 平均调用 | 分/美元 |")
    add("|---|---|---|---|---|---|")
    for m in modes:
        vals = [r["modes"][m] for r in rows if m in r["modes"]]
        sc = [v["score"] for v in vals if v.get("score") is not None]
        avg_score = sum(sc) / len(sc) if sc else 0.0
        avg_cost = sum(cell(v, "cost") for v in vals) / len(vals)
        avg_time = sum(cell(v, "elapsed") for v in vals) / len(vals)
        avg_calls = sum(cell(v, "calls") for v in vals) / len(vals)
        ratio = ("%.0f" % (avg_score / avg_cost)) if avg_cost > 0 else "-"
        add("| %s | %.2f | %.6f | %.1f | %.1f | %s |"
            % (m, avg_score, avg_cost, avg_time, avg_calls, ratio))

    cats = {}
    for r in rows:
        cats.setdefault(r.get("cat") or "未分类", []).append(r)
    add("")
    add("## 分类表（研讨相对单模型的增益）")
    add("")
    add("| 题型 | 题数 | 单模型 | 并行 | 研讨 | 研讨-单模型 |")
    add("|---|---|---|---|---|---|")
    for cat, rs in cats.items():
        cells = []
        for m in modes:
            sc = [r["modes"][m]["score"] for r in rs
                  if m in r["modes"] and r["modes"][m].get("score") is not None]
            cells.append(sum(sc) / len(sc) if sc else 0.0)
        delta = ""
        if "single" in modes and "debate" in modes:
            delta = "%+.2f" % (cells[modes.index("debate")] - cells[modes.index("single")])
        add("| %s | %d | %s | %s |" % (cat, len(rs),
            " | ".join("%.2f" % c for c in cells), delta))

    pairs = []
    for r in rows:
        for m in modes:
            v = r["modes"].get(m)
            if v and v.get("score") is not None:
                pairs.append((len(v["answer"]), v["score"]))
    add("")
    add("## 诊断")
    add("")
    r_val = pearson(pairs)
    flag = "（裁判可能在惩罚长度，需要改提示词）" if r_val < -0.3 else "（正常）"
    add("- 长度 vs 分数 相关系数 r = %.2f %s" % (r_val, flag))

    if "single" in modes and "debate" in modes:
        diffs = []
        for r in rows:
            a = r["modes"]["single"].get("score")
            b = r["modes"]["debate"].get("score")
            if a is not None and b is not None:
                diffs.append((abs(b - a), r))
        diffs.sort(key=lambda x: -x[0])
        add("")
        add("- 分歧最大的题：")
        for d, r in diffs[:3]:
            add("  - %s %s：单模型 %.0f / 研讨 %.0f"
                % (r["id"], (r.get("q") or "")[:30], r["modes"]["single"]["score"],
                   r["modes"]["debate"]["score"]))

    text = "\n".join(lines)
    print(text)
    if args.dst:
        open(args.dst, "w", encoding="utf-8").write(text + "\n")
        print("")
        print("已写入 %s" % args.dst)


if __name__ == "__main__":
    main()
