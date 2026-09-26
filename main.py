import os
import json
import hashlib
import random
from typing import List, Union, Optional
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Bedrock & Azure OpenAI Mock Server")

# Helper function
def generate_vector(text: str, dim: int) -> List[float]:
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return [round(rng.uniform(-0.05, 0.05), 6) for _ in range(dim)]

# --- Service 1: Bedrock Titan Embeddings Mock ---
@app.post("/model/{model_id:path}/invoke")
async def invoke_bedrock_model(model_id: str, request: Request):
    try:
        raw_body = await request.body()
        data = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    input_text = data.get("inputText")
    if not input_text:
        raise HTTPException(status_code=422, detail="'inputText' is required")

    dim = int(data.get("dimensions", 1024))
    vector = generate_vector(input_text, dim)
    token_count = max(1, len(input_text) // 4)

    return JSONResponse(
        content={
            "embedding": vector,
            "inputTextTokenCount": token_count
        }
    )

# --- Service 2: Azure OpenAI Embeddings Mock ---
class AzureEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    dimensions: Optional[int] = 3072

@app.post("/openai/deployments/{deployment_id}/embeddings")
async def create_azure_embeddings(
    deployment_id: str,
    payload: AzureEmbeddingRequest,
    api_version: Optional[str] = Query(None, alias="api-version")
):
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    dim = payload.dimensions or 3072

    data_elements = []
    total_tokens = 0
    for idx, text in enumerate(texts):
        vector = generate_vector(text, dim)
        tokens = max(1, len(text) // 4)
        total_tokens += tokens
        data_elements.append({
            "object": "embedding",
            "index": idx,
            "embedding": vector
        })

    return {
        "object": "list",
        "data": data_elements,
        "model": deployment_id,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
