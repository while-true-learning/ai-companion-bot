import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)

DB_PATH = os.path.join(BASE_DIR, "memory.db")
USER_ID = "local_user"
MODEL_REPLY = "gpt-5.4-mini"
MODEL_DECIDER = "gpt-5.4-nano"
MODEL_SUMMARIZE = "gpt-5.4-nano"
CONTEXT_LIMIT = 50

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")