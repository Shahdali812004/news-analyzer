import os
import dotenv
from dotenv import load_dotenv
#load environment variables from .env file
load_dotenv(override=True) 

#get environment variables
OPEN_AI_KEY = os.getenv("open-ai-key")
APP_NAME = os.getenv("APP_NAME")
APP_VERSION = os.getenv("APP_VERSION")
VLLM_URL = os.getenv("VLLM_URL")

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_ID = "Shahdddddd/news-analyzer"
