import os
from dotenv import load_dotenv
from openai import OpenAI
from app.core.logger import logger

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 20))

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
MODEL_NAME = "qwen-plus"