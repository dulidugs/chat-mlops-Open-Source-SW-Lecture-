from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# 전역 변수로 파이프라인(텍스트 생성 모델) 저장
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI 서버 시작 시 딱 한 번만 모델 로드
    global model_pipeline
    print("Loading text generation model...")
    # 하드웨어 자원이 제한적인 환경이므로 가벼운 모델인 distilgpt2 사용 (또는 microsoft/DialoGPT-small)
    model_pipeline = pipeline("text-generation", model="distilgpt2")
    print("Model loaded successfully!")
    yield
    # 서버 종료 시 필요한 리소스 정리
    model_pipeline = None
    print("Model resources released.")

app = FastAPI(title="MLOps Chat Prediction API", lifespan=lifespan)

class ChatRequest(BaseModel):
    history: list[str]

@app.post("/predict-next")
async def predict_next(request: ChatRequest):
    global model_pipeline
    
    if not model_pipeline or not request.history:
        return {"next_message": "모델이 준비되지 않았거나 히스토리가 없습니다."}

    # 입력된 대화 내역을 하나의 문맥으로 병합
    user_context = " ".join(request.history)
    
    # 모델 추론 수행 (텍스트 이어서 생성)
    # max_new_tokens로 너무 긴 문장 생성 방지, truncation으로 입력 제한 통과
    result = model_pipeline(user_context, max_new_tokens=20, num_return_sequences=1, truncation=True)
    generated_text = result[0]['generated_text']
    
    # 입력된 문맥 이후의 문자열(즉 새로 생성된 다음 문장)만 추출
    next_message = generated_text[len(user_context):].strip()
    
    if not next_message:
        next_message = "더 이상 할 말이 생각나지 않네요."
        
    return {"next_message": next_message}
