import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COSMOS_CONN = os.getenv("COSMOS_MONGO_CONN", "")
DB_NAME = os.getenv("COSMOS_DB", "ragdb")
COLL_NAME = os.getenv("COSMOS_COLLECTION", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

if not COSMOS_CONN:
    raise RuntimeError("COSMOS_MONGO_CONN is not set")

mongo = MongoClient(COSMOS_CONN)
db = mongo[DB_NAME]
collection = db[COLL_NAME]

oa = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)