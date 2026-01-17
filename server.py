import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from graph_agent_scalable import get_graph_app

# 🤖 심판을 위한 LLM 준비
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="BabySquad Smart HITL Server")

# CORS 설정 (Streamlit과 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraph 앱 초기화
graph_app = get_graph_app()

# 심판관 모델 (빠르고 저렴한 gpt-4o-mini 추천)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ================================================================================
# 요청/응답 모델
# ================================================================================
class ChatRequest(BaseModel):
    thread_id: str
    message: Optional[str] = None

class ChatResponse(BaseModel):
    status: str  # "completed", "review_needed"
    response: Optional[str] = None
    draft_response: Optional[str] = None
    # 메타데이터
    selected_experts: Optional[list] = []
    complexity: Optional[str] = "simple"
    expert_count: Optional[int] = 0
    stage: Optional[str] = "completed"
    synthesis_performed: Optional[bool] = False

# ================================================================================
# 🛡️ 안전 심판관 함수 (Risk Analyzer)
# ================================================================================
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

# ================================================================================
# 메타데이터 추출 함수
# ================================================================================
def extract_metadata(state) -> dict:
    """
    LangGraph state에서 메타데이터 추출
    """
    metadata = state.values.get("metadata", {})
    
    return {
        "selected_experts": metadata.get("selected_experts", []),
        "complexity": metadata.get("complexity", "simple"),
        "expert_count": metadata.get("expert_count", 0),
        "stage": metadata.get("stage", "completed"),
        "synthesis_performed": metadata.get("synthesis_performed", False)
    }

# ================================================================================
# 채팅 엔드포인트
# ================================================================================
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    채팅 메시지 처리
    - 자동으로 리스크 분석 후 안전하면 자동 승인
    - 위험하면 사람 검토 요청
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    
    try:
        # 1. 첫 번째 실행 (interrupt까지)
        input_data = {"messages": [{"role": "user", "content": req.message}]}
        graph_app.invoke(input_data, config=config)
        
        # 2. 상태 확인
        state = graph_app.get_state(config)
        
        # 메타데이터 추출
        metadata = extract_metadata(state)
        
        # 3. Human_Review 단계에 걸려 있는지 확인
        if state.next and "Human_Review" in state.next:
            last_message = state.values["messages"][-1]
            draft_response = last_message.content
            
            # 🔍 [스마트 로직] LLM에게 위험도 판단 요청
            is_risky = analyze_risk(draft_response)
            
            if is_risky:
                # 🚨 진짜 위험한 조언임 -> 검토 요청
                print("⚠️ 위험한 내용으로 판단되어 사람 검토를 요청합니다.")
                return ChatResponse(
                    status="review_needed",
                    draft_response=draft_response,
                    **metadata
                )
            else:
                # ✅ 키워드는 있어도 내용은 안전함 -> 자동 승인
                print("✅ 안전한 내용으로 판단되어 자동 승인합니다.")
                result = graph_app.invoke(None, config=config)
                
                # 최종 상태에서 메타데이터 다시 추출
                final_state = graph_app.get_state(config)
                metadata = extract_metadata(final_state)
                
                return ChatResponse(
                    status="completed",
                    response=result["messages"][-1].content,
                    **metadata
                )
        
        # 4. interrupt 없이 바로 완료된 경우
        return ChatResponse(
            status="completed",
            response=state.values["messages"][-1].content,
            **metadata
        )
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return ChatResponse(
            status="error",
            response=f"오류가 발생했습니다: {str(e)}"
        )

# ================================================================================
# 승인 엔드포인트
# ================================================================================
@app.post("/approve")
def approve_endpoint(req: ChatRequest):
    """
    사람이 검토 후 승인한 경우 처리
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    
    try:
        # interrupt 이후 계속 실행
        result = graph_app.invoke(None, config=config)
        
        # 메타데이터 추출
        final_state = graph_app.get_state(config)
        metadata = extract_metadata(final_state)
        
        return {
            "status": "completed",
            "response": result["messages"][-1].content,
            **metadata
        }
        
    except Exception as e:
        return {
            "status": "error",
            "response": f"승인 처리 중 오류: {str(e)}"
        }

# ================================================================================
# 헬스 체크
# ================================================================================
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "service": "BabySquad AI Backend",
        "features": [
            "Multi-Agent System (영양/수면/놀이)",
            "Smart Risk Analysis",
            "Human-in-the-Loop",
            "Real-time Expert Tracking"
        ]
    }

# ================================================================================
# 전문가 목록 조회 (선택 사항)
# ================================================================================
@app.get("/experts")
async def get_experts():
    """등록된 전문가 목록 반환"""
    from graph_agent_scalable import registry
    
    experts = []
    for name, config in registry.experts.items():
        experts.append({
            "name": name,
            "display_name": config.display_name,
            "keywords": config.keywords[:5],  # 처음 5개만
        })
    
    return {
        "experts": experts,
        "total_count": len(experts)
    }

# ================================================================================
# 실행
# ================================================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"""
    
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          👶 BabySquad AI Backend Server                ║
    ║                                                          ║
    ║  🚀 Features:                                           ║
    ║     • Multi-Agent System (영양/수면/놀이)              ║
    ║     • Smart Risk Analysis                               ║
    ║     • Real-time Expert Tracking                         ║
    ║     • Human-in-the-Loop Safety                          ║
    ║                                                          ║
    ║  📡 Running on: http://localhost:{port}                ║
    ║  📚 API Docs: http://localhost:{port}/docs             ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    """)
    uvicorn.run(app, host="0.0.0.0", port=port)