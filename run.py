import uvicorn
from app.core.config import SERVER_PORT
from app.core.logger import logger
from app.main import app

if __name__ == "__main__":
    logger.info(f"服务启动成功，接口文档：http://localhost:{SERVER_PORT}/docs")
    # 👇 加上 reload=False，容器内绝对不要开热重载
    uvicorn.run("app.main:app", host="0.0.0.0", port=SERVER_PORT, reload=False)