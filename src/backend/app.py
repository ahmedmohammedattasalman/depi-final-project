from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.models.inference import RecSysInference

app = FastAPI(title='Amazon RecSys API')
model = RecSysInference(model_dir="data")

class RecRequest(BaseModel):
    user_id: str
    top_k: int = 10

class RecResponse(BaseModel):
    recommendations: List[Dict]

@app.post('/recommend', response_model=RecResponse)
def recommend(req: RecRequest):
    try:
        recs = model.recommend(req.user_id, req.top_k)
        return {'recommendations': recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/')
def root():
    return {'message': 'Amazon RecSys API is running'}
