from fastapi import FastAPI, Request
import openai
import pinecone
import os
from dotenv import load_dotenv

openai.api_key = os.getenv('OPENAI_API_KEY')
pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Connect to index
index = pc.Index("tradetron-vector-search")

load_dotenv()

# FastAPI app
app = FastAPI()

@app.post("/query")
async def query_endpoint(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
    except Exception as e:
        return {"error": "Invalid JSON", "details": str(e)}
    # 1. Get embedding for query
    embed_response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small",
        dimensions=1024
    )
    query_embedding = embed_response.data[0].embedding
    # 2. Query Pinecone
    search_result = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    # 3. Extract context
    context_chunks = [match["metadata"]["text"] for match in search_result["matches"]]
    return context_chunks
