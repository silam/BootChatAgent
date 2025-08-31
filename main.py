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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
####from openai_tools import create_openai_tools_chain


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


collection.create_index(
    [("embedding", "cosmosSearch")],
    cosmosSearchOptions={
        "kind": "vector-ivf",
        "numLists": 800,
        "similarity": "COS",
        "dimensions": 1536
    }
)


oa = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

app = FastAPI(title="RAG with CosmosDB + OpenAI + LangChain")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestItem(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    items: List[IngestItem]

class ChatRequest(BaseModel):
    question: str
    k: Optional[int] = None

def get_embedding(text: str) -> List[float]:
    resp = oa.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def cosmos_vector_search(query_embedding: List[float], k: int):
    pipeline = [
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": k
                }
            }
        },
        {"$limit": k},
        {"$project": {"_id": 0, "id": 1, "text": 1, "metadata": 1, "score": {"$meta": "searchScore"}}}
    ]
    return list(collection.aggregate(pipeline))

def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ingest")
def ingest(payload: IngestRequest):
    docs = []
    for i, item in enumerate(payload.items):
        if not item.text.strip():
            continue
        # Optional simple chunking
        chunks = split_text(item.text)
        for j, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            docs.append({
                "id": item.id or f"{i}-{j}",
                "text": chunk,
                "embedding": emb,
                "metadata": item.metadata or {}
            })
    if docs:
        collection.insert_many(docs)
    return {"ingested": len(docs)}

tools_description = """
- `fetch_price(product_id)` → returns the latest price of a product.
- `fetch_inventory(product_id)` → returns stock availability.
- `fetch_details(product_id)` → returns product description.
"""


SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the answer isn't in the context, say you don't know. Be concise.

You cannot guess prices. Always use the available tools to answer questions about products.
If the user asks for the price of a product, 
call the `fetch_price` tool with the given product_id. 

You have access to the following tools:

- `fetch_price(product_id)` → returns the latest price of a product.
- `fetch_inventory(product_id)` → returns stock availability.
- `fetch_details(product_id)` → returns product description.

General rules:
1. Always use the appropriate tool if a user asks about information the tool can provide.  
2. Never guess or invent answers — rely on tools for facts.  
3. If a tool returns data, include it in your natural-language response.  
4. If no tool is relevant, politely explain that you cannot answer.  
5. Do not expose raw JSON or tool calls directly — summarize results for the user.  
6. If multiple tools are needed, call them in sequence until you can answer.  

After the tool returns the result, 
explain it in natural language back to the user.
Return a "Sources" section listing the IDs of the most relevant chunks you used.
"""

USER_PROMPT = """Question:
{question}

Context chunks:
{context}

Answer:
"""

@app.post("/chat")
def chat(payload: ChatRequest):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if len(question) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"Question too long (> {MAX_INPUT_CHARS} chars).")

    k = payload.k or TOP_K
    q_emb = get_embedding(question)
    hits = cosmos_vector_search(q_emb, k)

    # Build context block
    context_parts = []
    used_ids = []
    for h in hits:
        chunk_id = str(h.get("id", ""))
        context_parts.append(f"[{chunk_id}] {h.get('text','')}")
        used_ids.append(chunk_id)
    context_str = "\n\n".join(context_parts) if context_parts else "N/A"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format_messages(question=question, context=context_str)

    #lm_with_tools = llm.bind_tools([fetch_price])

    #chain = prompt | llm_with_tools
    #####esponse = chain.invoke()
    #response = llm_with_tools.invoke(prompt)
    #print(response)
    # Call LLM via LangChain ChatOpenAI
    response = llm.invoke(prompt)

    # Bind tools + chain so LLM incorporates tool outputs
    ####llm_with_tools = llm.bind_tools([fetch_price])
    ####chain = create_openai_tools_chain(llm_with_tools, [fetch_price])

    # Prompt
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful shopping assistant."),
    #     ("user", "What is the price of product 001?")
    # ])

    # Run chain with invoke (final answer includes tool result)
    ####response = chain.invoke(prompt.format())
    #print(result["output"])

    return {
        "answer": response.content,
        "sources": used_ids,
        "matches": hits
    }

    



############################################################
## GetPRice tool
###########################################################
@tool
def fetch_price(product_id: str) -> str:
    """Fetch the price of a product by product_id from MCP server."""
    #resp = requests.get("http://localhost:8000/price", params={"product_id": product_id})
    #data = resp.json()
    print('Fetch price called')
    return f"The price of product is $300.99."
