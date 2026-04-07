from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="1:1 Chat Helper API")

class ChatRequest(BaseModel):
    history: list[str]

@app.post("/predict-next")
async def predict_next(request: ChatRequest):
    if not request.history:
        return {"next_message": "먼저 대화 내용을 입력해 주세요."}

    # 전체 문맥을 하나의 문자열로 합쳐서 키워드 단어 검사 수행
    user_context = " ".join(request.history)
    
    if "피곤" in user_context or "졸려" in user_context:
        next_message = "오늘은 조금 일찍 자는 게 좋겠다."
    elif "밥" in user_context or "배고파" in user_context or "점심" in user_context:
        next_message = "간단하게 먹을 수 있는 걸로 먼저 챙겨 먹자."
    elif "약속" in user_context or "시간" in user_context or "언제" in user_context:
        next_message = "언제 시간이 괜찮은지 먼저 물어봐야겠다."
    else:
        next_message = "그렇구나, 조금 더 자세히 얘기해 줄 수 있어?"
        
    return {"next_message": next_message}

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
        <textarea id="chat-history" placeholder="상대: 오늘 점심 뭐 먹을래?&#10;나: 글쎄, 너는 땡기는 거 있어?&#10;상대: 난 아무거나 괜찮아!"></textarea>
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
        // POST 요청으로 history 배열 전송
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
