# server.py (하이브리드 버전)
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse # 추가
from pydantic import BaseModel
from typing import List, Dict, Any
from langserve import add_routes # 👈 LangServe 다시 등판!
from graph_agent import get_graph_app

# 1. FastAPI 앱 생성
app = FastAPI(
    title="BabySquad Hybrid Server",
    description="Custom API + LangServe 모두 지원하는 강력한 서버"
)
graph_app = get_graph_app()

# ----------------------------------------------------------------
# [A] LangServe 영역 (개발자/관리자용) 🛠️
# - 주소: /baby-agent
# - 용도: Playground에서 테스트하거나, 문서 볼 때 사용
# ----------------------------------------------------------------
add_routes(
    app,
    graph_app,
    path="/baby-agent",
)

# 루트 접속 시 Playground로 리다이렉트 (편의성)
@app.get("/")
async def redirect_root_to_playground():
    return RedirectResponse("/baby-agent/playground")


# ----------------------------------------------------------------
# [B] 커스텀 API 영역 (실제 앱용) 📱
# - 주소: /chat
# - 용도: Streamlit이나 모바일 앱에서 thread_id를 콕 집어 보낼 때 사용
# ----------------------------------------------------------------
class ChatRequest(BaseModel):
    thread_id: str
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # LangGraph가 원하는 Config 수동 주입 (확실한 제어)
        config = {"configurable": {"thread_id": req.thread_id}}
        input_data = {"messages": [{"role": "user", "content": req.message}]}
        
        result = graph_app.invoke(input_data, config=config)
        
        last_message = result["messages"][-1]
        return {"response": last_message.content}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Server Running...")
    print(" - App Endpoint: http://localhost:8000/chat")
    print(" - Dev Playground: http://localhost:8000/baby-agent/playground")
    uvicorn.run(app, host="0.0.0.0", port=8000)