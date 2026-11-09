# 👇 放在文件绝对的第一行（在 uvicorn 之前）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys

# 用 print(flush=True) 而不是 logger：此时 logging 还没配好，
# 而且重依赖（torch / sentence-transformers / faiss）导入要好几秒，
# 以前这段时间屏幕上什么都没有，很容易被误判成"卡死"。
print("[1/3] 正在导入依赖（torch / transformers / faiss），首次启动可能需要 10-60 秒，请勿关闭窗口 ...", flush=True)

import uvicorn
from app.core.config import SERVER_PORT

print("[2/3] 依赖导入完成，正在加载应用与向量模型 ...", flush=True)

from app.core.logger import logger
from app.main import app

def _port_in_use(port: int) -> bool:
    """端口已被占用时，uvicorn 会丢一大段 traceback，看起来像"卡死"；先自己查一下。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    if _port_in_use(SERVER_PORT):
        print(f"[!] 端口 {SERVER_PORT} 已被占用，多半是上一次的 python 没退干净。", flush=True)
        print(f"    Windows:  netstat -ano | findstr :{SERVER_PORT}", flush=True)
        print( "              然后 taskkill /PID <上面查到的PID> /F", flush=True)
        print(f"    或者改 .env 里的 SERVER_PORT 换一个端口。", flush=True)
        sys.exit(1)
    print(f"[3/3] 启动完成，正在监听 0.0.0.0:{SERVER_PORT}", flush=True)
    logger.info(f"服务启动成功，接口文档：http://localhost:{SERVER_PORT}/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=SERVER_PORT, reload=False)