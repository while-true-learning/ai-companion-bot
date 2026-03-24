import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "memory.db"
USER_ID = "local_user"
MODEL_MAIN = "gpt-5.4-mini"
MODEL_DECISION = "gpt-5.4-nano"
CONTEXT_LIMIT = 14
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")