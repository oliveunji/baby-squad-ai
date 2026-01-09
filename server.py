import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from graph_agent import get_graph_app

# 🤖 심판을 위한 LLM 준비
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="BabySquad Smart HITL Server")
graph_app = get_graph_app()

# 심판관 모델 (빠르고 저렴한 gpt-4o-mini 추천, 없으면 gpt-4o)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

class ChatRequest(BaseModel):
    thread_id: str
    message: Optional[str] = None

# ---------------------------------------------------------
# 🛡️ 안전 심판관 함수 (Risk Analyzer)
# ---------------------------------------------------------
def analyze_risk(text: str) -> bool:
    """
    답변 내용을 분석해서 '위험(RISK)' 여부를 판단합니다.
    True: 위험함 (사람 검토 필요)
    False: 안전함 (자동 승인)
    """
    system_prompt = (
        "당신은 AI 답변 안전 관리자입니다. 아래 [답변]을 분석하여 위험 여부를 판단하세요.\n\n"
        "[위험 기준 (RISK)]\n"
        "1. 구체적인 의약품 '투약 용량'이나 '복용법'을 지시하는 경우 (예: 5ml 먹이세요, 교차 복용하세요)\n"
        "2. 응급 상황에 대한 직접적인 대처법을 조언하는 경우 (예: 바늘로 따세요, 억지로 토하게 하세요)\n"
        "3. 의사의 진단 없이 병명을 확정 짓는 경우\n\n"
        "[안전 기준 (SAFE)]\n"
        "1. 단순한 의학 용어의 정의 설명 (예: 해열제란 열을 내리는 약입니다)\n"
        "2. 병원에 가보라는 일반적인 권유\n"
        "3. 영양제 추천이나 일반적인 육아 상식\n\n"
        "결과는 오직 'RISK' 또는 'SAFE' 단어 하나만 출력하세요."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{text}"),
    ])
    
    chain = prompt | judge_llm
    result = chain.invoke({"text": text}).content.strip()
    
    print(f"  👮 [심판 판정] {result} (내용: {text[:30]}...)")
    return result == "RISK"

# ---------------------------------------------------------
# 채팅 엔드포인트
# ---------------------------------------------------------
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # 1. 실행 (전문가 노드 후 멈춤)
    input_data = {"messages": [{"role": "user", "content": req.message}]}
    graph_app.invoke(input_data, config=config)
    
    # 2. 상태 확인
    state = graph_app.get_state(config)
    
    # Human_Review 단계에 걸려 있다면?
    if state.next and "Human_Review" in state.next:
        last_message = state.values["messages"][-1]
        draft_response = last_message.content
        
        # 🔍 [스마트 로직] LLM에게 위험도 판단 요청
        is_risky = analyze_risk(draft_response)
        
        if is_risky:
            # 🚨 진짜 위험한 조언임 -> 검토 요청
            return {
                "status": "review_needed",
                "draft_response": draft_response
            }
        else:
            # ✅ 키워드는 있어도 내용은 안전함 -> 자동 승인
            print("✅ 안전한 내용으로 판단되어 자동 승인합니다.")
            result = graph_app.invoke(None, config=config)
            return {
                "status": "completed",
                "response": result["messages"][-1].content
            }
    
    return {
        "status": "completed",
        "response": state.values["messages"][-1].content
    }

@app.post("/approve")
def approve_endpoint(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph_app.invoke(None, config=config)
    return {
        "status": "completed",
        "response": result["messages"][-1].content
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)