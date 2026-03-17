"""
Configuration file for API keys and other settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = "xxxx"
OPENAI_BASE_URL = "xxxxx"

# API Rate Limiting Configuration
CALLS_PER_MINUTE = 20
PERIOD = 60
MAX_RETRIES = 3
RETRY_DELAY = 120

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"  # please specify your preferred LLM model
DEFAULT_MAX_TOKENS = 250

# Embedding Configuration
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # please specify your preferred embedding model
# EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MODEL = "BAAI/bge-m3"

# --- New Advanced Retrieval Configuration ---
# 稠密检索模型 (Dense)
DENSE_MODEL_NAME = "BAAI/bge-m3"

# 重排序模型 (Rerank)
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"

# 检索超参数
RETRIEVAL_TOP_K_CANDIDATES = 100  # 双路召回各自获取的候选数量
RERANK_TOP_K = 10  # 最终重排序后返回给 LLM 的数量

# Embedding Configuration
EMBEDDING_BATCH_SIZE = 16  # BGE-M3 较大，显存不足可调小至 8 或 4

# Cache Configuration
CACHE_DIR = "cache"
RESULT_DIR = "result"
