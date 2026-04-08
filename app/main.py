from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 전역 변수로 토크나이저와 모델 선언
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    try:
        print("Loading Hugging Face model (skt/kogpt2-base-v2)...")
        tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
        model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
        
        # 모델 재생성시 pad_token_id가 없으면 발생하는 경고를 피하기 위해 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # GPU 사용 가능 시 cuda로 이동
        if torch.cuda.is_available():
            model.to("cuda")
            
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("Shutting down and releasing resources.")
    tokenizer = None
    model = None

app = FastAPI(title="1:1 Chat Helper API (KoGPT2)", lifespan=lifespan)

class ChatRequest(BaseModel):
    history: list[str]

@app.post("/predict-next")
async def predict_next(request: ChatRequest):
    global tokenizer, model
    
    if not request.history:
        return {"next_message": "먼저 대화 내용을 입력해 주세요."}
        
    try:
        context = "\\n".join(request.history)
        if not context.endswith("\\n"):
            context += "\\n"
        context += "나:"
        
        if model is None or tokenizer is None:
            return {"next_message": "현재 AI 모델을 사용할 수 없습니다. (Fallback: 그렇구나!)"}
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(context, return_tensors="pt").to(device)
        
        # 한국어 텍스트 생성 옵션 적용 (창의적이면서도 반복 피하기)
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력된 프롬프트(context)를 제외한 새로 생성된 부분만 가져오기
        new_text = generated_text[len(context):]
        
        # 첫 번째 줄바꿈 전까지만 남기기
        next_message_candidate = new_text.split("\\n")[0].strip()
        
        # 간혹 "나:" 또는 유사한 접두사가 다시 생성된 경우 제거
        if next_message_candidate.startswith("나:"):
            next_message_candidate = next_message_candidate[2:].strip()
            
        # [changed / fixed] 
        # 버그 원인: Python에서는 빈 문자열("")이 모든 문자열의 부분문자열로 평가되므로,
        # '"" in string'은 항상 True를 반환합니다. 이 때문에 매번 무조건 fallback이 적용되었습니다.
        # 수정: .strip()을 사용해 공백뿐인 결과를 걸러내고, 유니코드 깨진 문자() 검사를 추가했습니다.
        if not next_message_candidate.strip() or "" in next_message_candidate:
            next_message_candidate = "음... 무슨 말을 해야 할지 잘 모르겠네."
            
        return {"next_message": next_message_candidate}
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"next_message": "추론 중 오류가 발생했습니다. (Fallback: 무슨 일 있었어?)"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>1:1 Chat Helper (KoGPT2)</title>
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
            margin-bottom: 8px;
        }
        p.subtitle {
            text-align: center;
            color: #606770;
            margin-top: 0;
            margin-bottom: 24px;
            font-size: 0.95rem;
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
    <h1>1:1 한국어 채팅 예측용 Helper</h1>
    <p class="subtitle">KoGPT2 기반 AI 다음 문장 생성 데모</p>
    
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
        
        // 응답 값 말풍선에 렌더링
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
