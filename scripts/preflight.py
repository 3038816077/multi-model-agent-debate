"""部署前自检：把「换台机器跑不起来」的常见原因提前查出来。

用法：
    python scripts/preflight.py              # 基础检查（不联网、不花钱）
    python scripts/preflight.py --docker     # 额外校验 docker / compose
    python scripts/preflight.py --check-models  # 逐个模型发最小请求（会花钱）

退出码：0 = 可以启动；1 = 有致命问题
"""
import argparse
import importlib.util
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OK, WARN, FAIL = "[OK]  ", "[WARN]", "[FAIL]"
problems = []


def ok(msg):
    print(OK, msg)


def warn(msg):
    print(WARN, msg)
    problems.append(("warn", msg))


def fail(msg):
    print(FAIL, msg)
    problems.append(("fail", msg))


def check_python():
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        ok("Python %d.%d" % (major, minor))
    else:
        fail("Python %d.%d 太旧，需要 3.10+" % (major, minor))


def check_files():
    required = [
        "app/main.py",
        "requirements.txt",
        "env.txt",
        "static/index.html",
        "docker-compose.yml",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        fail("缺少文件：%s" % ", ".join(missing))
    else:
        ok("关键文件齐全")


def _parse_env(path: str) -> dict:
    """手写解析 .env：部署机上往往还没装 python-dotenv。"""
    values = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                values[key.strip()] = value
    except OSError:
        pass
    return values


def check_dotenv():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        if os.path.exists(os.path.join(ROOT, "env.txt")):
            fail(".env 不存在。先执行：cp env.txt .env 然后填 key")
        else:
            fail(".env 和 env.txt 都不存在")
        return

    values = _parse_env(env_path)
    empty = [k for k, v in values.items() if k and not (v or "").strip()]
    if empty:
        warn("这些变量值为空：%s" % ", ".join(empty[:6]))

    api_key = (values.get("API_KEY") or "").strip()
    if not api_key:
        warn("API_KEY 为空：接口没有鉴权，公网部署前务必设置")
    elif api_key == "my-secret-key-123":
        warn("API_KEY 还是默认值 my-secret-key-123，公网部署前必须改")
    else:
        ok("API_KEY 已设置")

    if values.get("MODEL_REGISTRY"):
        ok(".env 存在且读到 MODEL_REGISTRY")
    else:
        fail("MODEL_REGISTRY 为空：一个模型都没配，聊天会全部失败")


def check_registry():
    try:
        from app.core.config import get_model_registry
        from app.core.router import MODEL_PRICE_PER_M, MODEL_TIERS
        from app.core.selfcheck import check_registry, check_tiers, check_prices, has_fatal
    except Exception as e:
        warn("本机没装 Python 依赖，跳过注册表检查（用 Docker 部署不需要）：%s" % str(e)[:60])
        return

    registry = get_model_registry()
    found = check_registry(registry) + check_tiers(registry, MODEL_TIERS) + check_prices(
        MODEL_PRICE_PER_M, registry
    )
    if has_fatal(found):
        for item in found:
            if item["level"] == "fatal":
                fail(item["message"])
        return
    ok("注册表可用：%d 个模型（%s）" % (len(registry), ", ".join(registry)))
    for item in found:
        if item["level"] == "warn":
            warn(item["message"])


def check_dirs():
    from app.core.selfcheck import check_writable

    issues = check_writable([
        os.path.join(ROOT, "data"),
        os.path.join(ROOT, "data", "uploads"),
    ])
    if issues:
        for item in issues:
            fail(item["message"])
    else:
        ok("data/ 与 data/uploads/ 可写")


def check_port():
    # 不 import app.core.config：部署机上往往还没有依赖
    values = _parse_env(os.path.join(ROOT, ".env"))
    raw = values.get("SERVER_PORT") or os.getenv("SERVER_PORT") or "8000"
    try:
        port = int(raw)
    except ValueError:
        warn("SERVER_PORT 不是数字：%s，按 8000 处理" % raw)
        port = 8000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        ok("端口 %d 空闲" % port)
    except OSError:
        warn("端口 %d 已被占用：可能已有实例在跑（单实例锁会阻止再起一个）" % port)
    finally:
        sock.close()


def check_docker():
    if importlib.util.find_spec("docker") is None:
        pass
    import shutil
    import subprocess

    if not shutil.which("docker"):
        warn("未安装 docker，跳过容器检查（用 systemd 部署也一样）")
        return
    try:
        out = subprocess.run(["docker", "compose", "config", "--quiet"], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            ok("docker compose 配置合法")
        else:
            fail("docker compose 配置有误：%s" % (out.stderr or out.stdout)[:200])
    except Exception as e:
        warn("docker 检查跳过：%s" % e)


def check_models():
    """逐个模型发一次最小请求——会花钱，默认不跑。"""
    from openai import OpenAI

    from app.core.config import get_model_registry, resolve_model_id
    from app.core.router import has_price

    registry = get_model_registry()
    for key, cfg in registry.items():
        try:
            client = OpenAI(api_key=cfg.get("api_key"), base_url=cfg.get("base_url"), timeout=30)
            client.chat.completions.create(
                model=resolve_model_id(key),
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            ok("模型 %s 连通%s" % (key, "" if has_price(key) else "（未配价格）"))
        except Exception as e:
            fail("模型 %s 不可用：%s" % (key, str(e)[:100]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docker", action="store_true", help="额外校验 docker compose")
    ap.add_argument("--check-models", action="store_true", help="逐个模型发请求（会花钱）")
    args = ap.parse_args()

    print("=" * 60)
    print("部署前自检  ", ROOT)
    print("=" * 60)

    check_python()
    check_files()
    check_dotenv()
    check_registry()
    check_dirs()
    check_port()
    if args.docker:
        check_docker()
    if args.check_models:
        check_models()

    fails = [m for level, m in problems if level == "fail"]
    warns = [m for level, m in problems if level == "warn"]
    print("-" * 60)
    if fails:
        print("有 %d 个致命问题，先修完再启动：" % len(fails))
        for m in fails:
            print("  ✗", m)
        return 1
    if warns:
        print("可以启动，但有 %d 个提醒：" % len(warns))
        for m in warns:
            print("  !", m)
    else:
        print("全部通过，可以启动了。")
    print("\n下一步：")
    print("  python run.py                       # 本地直接跑")
    print("  docker compose up -d --build        # 或用容器")
    print("  curl localhost:%s/health" % os.getenv("SERVER_PORT", "8000"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
