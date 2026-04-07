from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Chat MLOps Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .card {
            background: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 480px;
            box-sizing: border-box;
        }
        h1 {
            font-size: 1.5rem;
            text-align: center;
            color: #212529;
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
            color: #495057;
            font-size: 0.9rem;
        }
        textarea {
            width: 100%;
            height: 160px;
            padding: 12px;
            border: 1px solid #ced4da;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            box-sizing: border-box;
            line-height: 1.5;
            transition: border-color 0.15s ease-in-out;
        }
        textarea:focus {
            outline: none;
            border-color: #86b7fe;
        }
        .speaker-select {
            display: flex;
            gap: 20px;
        }
        .speaker-select label {
            display: flex;
            align-items: center;
            font-weight: 400;
            cursor: pointer;
            color: #212529;
            font-size: 0.95rem;
        }
        .speaker-select input {
            margin-right: 6px;
        }
        button {
            width: 100%;
            padding: 14px;
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #0b5ed7;
        }
        button:disabled {
            background-color: #6ea8fe;
            cursor: not-allowed;
        }
        .result-container {
            margin-top: 24px;
            display: none;
        }
        .bubble {
            background-color: #f1f3f5;
            color: #212529;
            padding: 16px;
            border-radius: 16px;
            border-bottom-left-radius: 4px;
            font-size: 0.95rem;
            line-height: 1.5;
            word-break: break-word;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>

<div class="card">
    <h1>Chat MLOps Demo</h1>
    
    <div class="form-group">
        <label for="chat-history">여러 화자의 채팅 내역</label>
        <textarea id="chat-history" placeholder="유저: 안녕&#10;봇: 리눅스 과제 다 했어?&#10;유저: 아직..."></textarea>
    </div>
    
    <div class="form-group">
        <label>다음 화자 선택</label>
        <div class="speaker-select">
            <label><input type="radio" name="next_speaker" value="유저" checked> 유저</label>
            <label><input type="radio" name="next_speaker" value="봇"> 봇</label>
        </div>
    </div>
    
    <button id="predict-btn" onclick="predictNext()">다음 말 예측하기</button>
    
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
    
    // 1) textarea 내용 줄바꿈 기준 분리 & 공백 줄 제거
    const rawLines = textContent.split('\\n');
    const history = rawLines.filter(line => line.trim() !== '');
    
    // 2) 선택된 다음 화자 프롬프트 배열 끝에 추가
    const nextSpeakerEls = document.getElementsByName('next_speaker');
    let nextSpeaker = "유저";
    for (const el of nextSpeakerEls) {
        if (el.checked) {
            nextSpeaker = el.value;
            break;
        }
    }
    history.push(`${nextSpeaker}: `);
    
    // 로딩 상태 처리
    btn.disabled = true;
    btn.innerText = "예측 중...";
    resultContainer.style.display = 'none';
    
    try {
        // 3) POST 요청으로 history 배열 전송
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
        
        // 4) 응답 값 말풍선에 렌더링
        resultBubble.innerText = `${nextSpeaker}: ${data.next_message}`;
        resultContainer.style.display = 'block';
    } catch (error) {
        alert("예측 중 오류가 발생했습니다:\\n" + error.message);
    } finally {
        btn.disabled = false;
        btn.innerText = "다음 말 예측하기";
    }
}
</script>

</body>
</html>"""
    return HTMLResponse(content=html_content)
