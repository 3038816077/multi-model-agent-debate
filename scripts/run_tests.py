"""测试运行器：没有 pytest 也能跑全部用例。

为什么需要它：这台机器装 pytest 很慢，但"没有测试运行器"不是理由。
约定：tests/test_*.py 里的 test_* 函数，参数名是 tmp_path 的传临时目录，
其余函数直接调用。有 pytest 时用 pytest 跑效果一样（fixture 由 pytest 提供）。

用法：
    python scripts/run_tests.py          # 跑全部
    python scripts/run_tests.py divergence   # 只跑文件名包含 divergence 的
"""
import importlib.util
import io
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")


def load_module(path):
    name = "run_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(fn, tmp):
    """按参数名决定传什么：只认 tmp_path，其余一律不带参。"""
    import inspect

    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []
    if params and params[0] == "tmp_path":
        return fn(tmp)
    return fn()


def main(argv):
    keyword = argv[1] if len(argv) > 1 else ""
    files = sorted(
        f for f in os.listdir(TESTS)
        if f.startswith("test_") and f.endswith(".py") and keyword in f
    )
    if not files:
        print("没有匹配的测试文件")
        return 1

    total = passed = 0
    failures = []
    # 很多模块会往 stdout 打日志，统一重定向，保证输出干净
    real_stdout = sys.stdout
    for filename in files:
        path = os.path.join(TESTS, filename)
        buf = io.StringIO()
        sys.stdout = buf
        try:
            module = load_module(path)
        except Exception:
            sys.stdout = real_stdout
            print("[加载失败] " + filename)
            print(buf.getvalue()[-800:])
            print(traceback.format_exc(limit=3))
            failures.append((filename, "<import>", "导入失败"))
            total += 1
            continue

        names = [k for k in dir(module) if k.startswith("test_") and callable(getattr(module, k))]
        for name in sorted(names):
            total += 1
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    call(getattr(module, name), tmp)
                    passed += 1
                except Exception as e:
                    failures.append((filename, name, "%s: %s" % (type(e).__name__, e)))
        sys.stdout = real_stdout
        print("  %-28s %d 项" % (filename, len(names)))

    sys.stdout = real_stdout
    print("")
    if failures:
        print("失败 %d 项：" % len(failures))
        for f, n, msg in failures:
            print("  %s :: %s\n      %s" % (f, n, msg))
        print("\n通过 %d/%d" % (passed, total))
        return 1
    print("全部通过：%d 项" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
