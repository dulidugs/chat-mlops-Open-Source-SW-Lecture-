from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import pipeline

# 1. 글로벌 변수로 텍스트 생성 파이프라인 생성 (초기값은 None)
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    try:
        # 2. 앱 시작 시 한 번만 모델 로드
        print("Loading Hugging Face model (distilgpt2)...")
        # CPU 환경에서 동작하도록 가볍고 작은 영문 모델(distilgpt2)를 사용합니다.
        model_pipeline = pipeline("text-generation", model="distilgpt2", device=-1)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    # 자원 정리
    print("Shutting down and releasing resources.")
    model_pipeline = None

# FastAPI 인스턴스 생성
app = FastAPI(title="1:1 Chat Helper API (MLOps Version)", lifespan=lifespan)

# Pydantic BaseModel 정의
class ChatRequest(BaseModel):
    history: list[str]

# POST /predict-next 엔드포인트
@app.post("/predict-next")
async def predict_next(request: ChatRequest):
    global model_pipeline
    
    if not request.history:
        return {"next_message": "먼저 대화 내용을 입력해 주세요."}
        
    try:
        # a) history 리스트를 "\\n"으로 join 해서 긴 대화 context 문자열을 만든다.
        context = "\\n".join(request.history)
        
        # b) context 끝에 "나:" 프롬프트를 붙여서 다음 할 말을 유도한다.
        if not context.endswith("\\n"):
            context += "\\n"
        context += "나: "
        
        # 만약 모델 로드가 실패했다면 fallback 텍스트 반환
        if model_pipeline is None:
            return {"next_message": "현재 AI 모델을 사용할 수 없습니다. (Fallback: 그렇구나!)"}
            
        # c) Hugging Face pipeline을 이용해 생성 (max_new_tokens=30)
        # 패딩 경고 방지를 위해 pad_token_id 설정, 입력 제한을 피하기 위해 truncation 추가
        result = model_pipeline(
            context,
            max_new_tokens=30,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=model_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        
        # d) 입력해 준 context 부분을 잘라내고 새로 생성된 텍스트만 정리한다.
        new_text = generated_text[len(context):]
        
        # 여러 줄이 생성되었을 경우 첫 번째 줄(한 줄)만 가져오고 양쪽 공백 정리
        next_message_candidate = new_text.split("\\n")[0].strip()
        
        if not next_message_candidate:
            next_message_candidate = "음... 무슨 말을 할지 고민되네."
            
        return {"next_message": next_message_candidate}
        
    except Exception as e:
        # e) 에러 발생 시를 대비한 fallback 문장
        print(f"Prediction Error: {e}")
        return {"next_message": "요청을 모델로 추론하는 중 오류가 발생했습니다. (Fallback: 무슨 일 있었어?)"}

# 이전에 만든 웹 UI (루트 경로 유지)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>1:1 Chat Helper</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .card {
            background: #ffffff;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 500px;
            box-sizing: border-box;
        }
        h1 {
            font-size: 1.6rem;
            text-align: center;
            color: #1c1e21;
            margin-top: 0;
            margin-bottom: 24px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4b4f56;
            font-size: 0.95rem;
        }
        textarea {
            width: 100%;
            height: 180px;
            padding: 14px;
            border: 1px solid #ccd0d5;
            border-radius: 10px;
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            box-sizing: border-box;
            line-height: 1.5;
            transition: border-color 0.2s;
        }
        textarea:focus {
            outline: none;
            border-color: #1877f2;
        }
        button {
            width: 100%;
            padding: 14px;
            background-color: #1877f2;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.05rem;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 8px;
        }
        button:hover {
            background-color: #166fe5;
        }
        button:disabled {
            background-color: #9cb4d8;
            cursor: not-allowed;
        }
        .result-container {
            margin-top: 24px;
            display: none;
        }
        .bubble {
            background-color: #e4e6eb;
            color: #050505;
            padding: 16px;
            border-radius: 18px;
            border-bottom-right-radius: 4px;
            font-size: 1rem;
            line-height: 1.4;
            word-break: break-word;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>

<div class="card">
    <h1>1:1 Chat Helper</h1>
    
    <div class="form-group">
        <label for="chat-history">나와 상대의 대화 내역</label>
        <textarea id="chat-history" placeholder="상대: 오늘 점심 뭐 먹을래?\\n나: 글쎄, 너는 땡기는 거 있어?\\n상대: 난 아무거나 괜찮아!"></textarea>
    </div>
    
    <button id="predict-btn" onclick="predictNext()">내 다음 말 추천받기</button>
    
    <div class="result-container" id="result-container">
        <div id="result-bubble" class="bubble"></div>
    </div>
</div>

<script>
async function predictNext() {
    const textContent = document.getElementById('chat-history').value;
    const btn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const resultBubble = document.getElementById('result-bubble');
    
    // textarea 내용 줄바꿈 기준 분리 & 공백 줄 제거
    const rawLines = textContent.split('\\n');
    const history = rawLines.filter(line => line.trim() !== '');
    
    // 로딩 상태 처리
    btn.disabled = true;
    btn.innerText = "추천 중...";
    resultContainer.style.display = 'none';
    
    try {
        // POST 요청으로 history 배열 전송 (상대경로 활용)
        const response = await fetch("/predict-next", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ history: history })
        });
        
        if (!response.ok) {
            throw new Error(`서버 에러가 발생했습니다. HTTP 상태: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 응답 값 말풍선에 렌더링 (나: 추천된 문장 형태)
        resultBubble.innerText = `나: ${data.next_message}`;
        resultContainer.style.display = 'block';
    } catch (error) {
        alert("추천 중 오류가 발생했습니다:\\n" + error.message);
    } finally {
        btn.disabled = false;
        btn.innerText = "내 다음 말 추천받기";
    }
}
</script>

</body>
</html>"""
    return HTMLResponse(content=html_content)
