from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Simple Chat Prediction API")

class ChatRequest(BaseModel):
    history: list[str]

@app.post("/predict-next")
async def predict_next(request: ChatRequest):
    # 채팅 내역에 '과제'라는 단어가 있는지 검사
    for msg in request.history:
        if "과제" in msg:
            return {"next_message": "파이팅!"}
            
    return {"next_message": "계속 이야기해 주세요."}
