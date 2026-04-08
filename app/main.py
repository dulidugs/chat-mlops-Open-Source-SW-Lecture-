from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# [모델 및 토크나이저 상수 명시]
# 모델 이름과 프롬프트를 이렇게 지정한 이유:
# base 모델(skt/kogpt2-base-v2)은 사람 간의 대화보다는 위키/문서 형태의 텍스트 생성에 맞춰져 있어 단순 채팅맥락을 이어나가기엔 한계가 있습니다.
# 이를 대신해, base 모델을 바탕으로 대화/명령(instruction) 데이터셋을 학습시킨 'KoAlpaca' 계열 같은 Chat 전용 모델을 사용하면 
# "사용자: ~ 봇: ~" 형태의 핑퐁 대화 구조를 명확히 이해하고, 답변을 훨씬 자연스럽고 알맞게 생성해낼 수 있습니다.
MODEL_NAME = "beomi/KoAlpaca-Polyglot-1.3B"  # 한국어 대화형 모델 (CPU/Local에서 돌릴 수 있는 1.3B 파라미터 경량 챗봇 모델)

# 전역 변수로 토크나이저와 모델 선언
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    try:
        print(f"Loading Hugging Face Chat model ({MODEL_NAME})...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if torch.cuda.is_available():
            model.to("cuda")
            
        # 추론용 평가 모드로 전환
        model.eval()
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("Shutting down and releasing resources.")
    tokenizer = None
    model = None

app = FastAPI(title="1:1 Chat Helper API (KoAlpaca Chat)", lifespan=lifespan)

class ChatRequest(BaseModel):
    history: list[str]

@app.post("/predict-next")
async def predict_next(request: ChatRequest):
    global tokenizer, model
    
    if not request.history:
        return {"next_message": "먼저 대화 내용을 입력해 주세요."}
        
    try:
        # [프롬프트 변환 로직]
        # "나: ..." -> "사용자: ..."
        # "상대: ..." -> "봇: ..."
        converted_history = []
        for line in request.history:
            if line.startswith("나:"):
                converted_history.append("사용자: " + line[len("나:"):].strip())
            elif line.startswith("상대:"):
                converted_history.append("봇: " + line[len("상대:"):].strip())
            else:
                converted_history.append(line)
                
        # 대화형 모델에 맞는 프롬프트 템플릿 구성
        prompt = "사용자와 챗봇의 대화가 아래와 같다.\\n\\n"
        prompt += "\\n".join(converted_history)
        if not prompt.endswith("\\n"):
            prompt += "\\n"
        # 최종적으로 "내가 다음에 할 말"을 모델이 유추하게 만드므로, "사용자:"를 끝에 배치하여 생성을 유도함.
        prompt += "사용자:"
        
        if model is None or tokenizer is None:
            return {"next_message": "현재 AI 모델을 사용할 수 없습니다."}
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids_length = inputs["input_ids"].shape[1]
        
        # evaluation 모드에서 no_grad()로 그래디언트 연산 해제 후 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,          # 과도하게 긴 말뭉치 방지
                do_sample=True,
                temperature=0.7,            # 너무 높으면 이상한 말 가능성 상승
                top_p=0.85, 
                repetition_penalty=1.2,     # 같은 단어 반복 방지
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 모델 출력물에서 "새로 만들어진 토큰"만 디코드 수행
        new_token_ids = outputs[0][input_ids_length:]
        new_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # 첫 번째 줄바꿈 전까지만 남기기
        next_message_candidate = new_text.split("\\n")[0].strip()
        
        # 접두사 중복 생성 제거 (사용자:, 봇:, 나:)
        for prefix in ["사용자:", "봇:", "나:"]:
            if next_message_candidate.startswith(prefix):
                next_message_candidate = next_message_candidate[len(prefix):].strip()
                
        print("\\n=== Debug Info ===")
        print(f"prompt (converted context):\\n{prompt}")
        print(f"decoded new_text: {repr(new_text)}")
        print(f"final next_message_candidate: {repr(next_message_candidate)}")
        print("==================\\n")

        # 결과 텍스트가 완전히 비어있거나, 단순 공백만 있으면 최소한의 점(...)으로 반환
        if not next_message_candidate.strip():
            next_message_candidate = "..."
            
        return {"next_message": next_message_candidate}
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return {"next_message": f"추론 중 예외 발생: {e}"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>1:1 Chat Helper (KoAlpaca)</title>
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
    <p class="subtitle">대화형 파인튜닝 모델(KoAlpaca) 데모</p>
    
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
