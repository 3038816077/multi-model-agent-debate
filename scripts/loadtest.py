"""极简压测工具：只用标准库 + requests，不引入 locust。

为什么自己写：
  1. 这个环境装不上第三方包（PyPI 常超时）
  2. 我们要压的是「服务自身开销」，场景简单（固定并发打同一个接口）

用法：
    python scripts/loadtest.py --path /models --concurrency 20 --duration 20
    python scripts/loadtest.py --path /sessions --concurrency 50 --duration 20
    python scripts/loadtest.py --path /sessions/archive --method POST \
        --body '{"session_ids":["..."],"archived":true}' --concurrency 10

说明：
  - 先用 /session/token 换令牌（浏览器走 Cookie，脚本走 Bearer）
  - 延迟记录每次请求的耗时，最后给 p50/p95/p99
  - 需要服务端关闭限流（RATE_LIMIT_ENABLED=0），否则测的是限流器
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

import requests

RESULTS = []
LOCK = threading.Lock()


def get_token(base: str) -> str:
    try:
        resp = requests.post(base + "/session/token", timeout=10)
        return resp.json()["data"]["token"]
    except Exception:
        return ""


def worker(base, path, method, body, token, stop_at, results):
    headers = {}
    if token:
        headers["Authorization"] = "Bearer " + token
    if body:
        headers["Content-Type"] = "application/json"
    session = requests.Session()
    while time.time() < stop_at:
        started = time.time()
        try:
            if method == "POST":
                resp = session.post(base + path, data=body or "", headers=headers, timeout=30)
            else:
                resp = session.get(base + path, headers=headers, timeout=30)
            ok = resp.status_code < 400
        except Exception:
            ok = False
        results.append((time.time() - started, ok))


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[idx]


def run(base, path, method, body, concurrency, duration, label=""):
    token = get_token(base)
    stop_at = time.time() + duration
    results = []
    threads = [
        threading.Thread(target=worker, args=(base, path, method, body, token, stop_at, results))
        for _ in range(concurrency)
    ]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started

    latencies = [r[0] for r in results]
    ok_count = sum(1 for r in results if r[1])
    total = len(results)
    summary = {
        "label": label or (method + " " + path),
        "concurrency": concurrency,
        "duration_s": round(elapsed, 1),
        "requests": total,
        "rps": round(total / elapsed, 1) if elapsed else 0,
        "error_rate": round((total - ok_count) / total * 100, 2) if total else 0,
        "p50_ms": round(percentile(latencies, 50) * 1000),
        "p95_ms": round(percentile(latencies, 95) * 1000),
        "p99_ms": round(percentile(latencies, 99) * 1000),
        "max_ms": round(max(latencies) * 1000) if latencies else 0,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--path", default="/health")
    ap.add_argument("--method", default="GET", choices=["GET", "POST"])
    ap.add_argument("--body", default="")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    summary = run(args.base.rstrip("/"), args.path, args.method, args.body,
                  args.concurrency, args.duration)
    print(json.dumps(summary, ensure_ascii=False))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
