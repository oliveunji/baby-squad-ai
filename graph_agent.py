# graph_agent.py
import os
import operator
from typing import Annotated, List, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm_expert = ChatOpenAI(model="gpt-4o", temperature=0) 
llm_supervisor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_store = PineconeVectorStore(
    index_name="baby-index",
    embedding=embeddings
)

def retrieve_knowledge(query: str, category: str) -> str:
    if not vector_store: return "정보 없음"
    print(f"  🔍 [{category}] 문서 검색 중: '{query}'")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )
    try:
        results = retriever.invoke(query)
    except Exception as e:
        print(f"  ⚠️ 검색 경고: 기준을 넘는 문서가 없습니다. ({e})")
        return "관련된 가이드라인을 찾지 못했습니다."

    if not results:
        return "관련된 가이드라인을 찾지 못했습니다."
    
    context = "\n".join([f"- {doc.page_content}" for doc in results])
    return context

# [상태 정의] - 전문가별 답변 저장 추가
class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add] 
    next: str
    nutrition_answer: str  # 영양 전문가 답변
    sleep_answer: str      # 수면 전문가 답변
    needs_nutrition: bool  # 영양 전문가 필요 여부
    needs_sleep: bool      # 수면 전문가 필요 여부

# [전문가 노드들 - 답변을 state에 저장하도록 수정]
def nutrition_expert_node(state: AgentState):
    messages = convert_to_messages(state["messages"])
    last_message = messages[-1].content
    context = retrieve_knowledge(last_message, "Nutrition")
    
    system_msg = f"""당신은 소아 영양 전문가입니다. 부모들에게 과학적 근거에 기반한 실용적인 영양 조언을 제공합니다.

# 역할 및 전문성
- 영유아 및 아동의 성장 단계별 영양 요구사항 전문가
- 이유식, 유아식, 어린이 식단 계획 전문가
- 식품 알레르기, 편식, 영양 불균형 문제 해결 전문가

# 응답 스타일 - 매우 중요!
**기본 원칙: 간결하게, 핵심만**
- 첫 응답은 2-4문장으로 핵심 조언만 제공
- 불필요한 설명, 배경지식, 서론 생략
- 불렛포인트는 최대 3개까지만 사용

**응답 구조 (간결 버전)**
1. 핵심 답변 1-2문장
2. 실행 방법 2-3줄 (필요시)
3. 중요 주의사항 1줄 (필수인 경우만)

# 주요 고려사항
- 아이의 연령/개월 수 (필수 확인)
- 알레르기 유무 및 기저질환
- 현재 식습관

# 안전 관련은 반드시 언급
- 의학적 조언 필요 시 → "소아청소년과 상담 권장"
- 알레르기 위험 → 반드시 주의사항 포함
- 질식 위험 식품 → 경고 필수

# 참고 지식베이스
{context}

**기억하세요: 짧고 명확하게!**"""

    response = llm_expert.invoke([SystemMessage(content=system_msg)] + messages)
    print(f"  💊 [영양 전문가] 답변 생성 완료")
    
    return {
        "nutrition_answer": response.content
    }

def sleep_expert_node(state: AgentState):
    messages = convert_to_messages(state["messages"])
    last_message = messages[-1].content
    context = retrieve_knowledge(last_message, "Sleep")
    
    system_msg = f"""당신은 소아 수면 전문가입니다. 부모들에게 과학적 근거에 기반한 실용적인 수면 조언을 제공합니다.

# 역할 및 전문성
- 영유아 및 아동의 발달 단계별 수면 패턴 전문가
- 수면 교육, 수면 환경 설계, 수면 문제 해결 전문가
- SIDS 예방 등 안전 가이드라인 숙지

# 응답 스타일 - 매우 중요!
**기본 원칙: 간결하게, 핵심만**
- 첫 응답은 2-4문장으로 핵심 조언만 제공
- 불필요한 설명, 이론, 연구 결과 언급 생략
- 불렛포인트는 최대 3개까지만 사용

**응답 구조 (간결 버전)**
1. 핵심 답변 1-2문장
2. 실행 방법 2-3줄 (바로 적용 가능한 것만)
3. 안전 주의사항 1줄 (필수인 경우만)

# 주요 고려사항
- 아이의 연령/개월 수
- 현재 수면 패턴 (낮잠, 밤잠)
- 가장 힘든 문제 (입면, 야간 깨기 등)

# 안전 관련은 반드시 언급 (간결하게)
- 1세 미만: "천장 보고, 단단한 매트리스, 이불 없이"
- 안전 위험 있으면: 한 문장으로 경고
- 심각한 문제 의심: "수면 클리닉 상담 권장"

# 참고 지식베이스
{context}

**기억하세요: 짧고 실용적으로!**"""

    response = llm_expert.invoke([SystemMessage(content=system_msg)] + messages)
    print(f"  😴 [수면 전문가] 답변 생성 완료")
    
    return {
        "sleep_answer": response.content
    }

# [핵심 개선] Supervisor가 필요한 전문가를 복수 선택
def supervisor_node(state: AgentState):
    messages = convert_to_messages(state["messages"])
    
    system_prompt = """당신은 육아 상담 관리자입니다. 
부모의 질문을 분석하여 어떤 전문가의 도움이 필요한지 판단하세요.

**선택 가능한 전문가:**
- Nutrition_Expert: 이유식, 식단, 영양, 음식, 알레르기, 편식, 먹이기, 분유, 모유 등
- Sleep_Expert: 수면, 잠, 낮잠, 밤잠, 수면교육, 야간수유, 깨다, 재우다 등

**매우 중요**: 
- 질문에 "밤에 배고파서", "저녁 먹고 자는", "이유식이랑 수면" 같은 표현이 있으면 **반드시 둘 다** 선택!
- 애매하면 둘 다 선택하는 것이 안전합니다.

**출력 형식**:
- 영양만: "Nutrition_Expert"
- 수면만: "Sleep_Expert"  
- **둘 다**: "Nutrition_Expert, Sleep_Expert"

**예시 (매우 중요):**
Q: "7개월 아기가 밤에 자주 깨는데 이유식이랑 관련 있을까?"
A: "Nutrition_Expert, Sleep_Expert" ← 반드시 둘 다!

Q: "통잠 자려면 저녁 이유식 양 늘려야 해?"
A: "Nutrition_Expert, Sleep_Expert" ← 반드시 둘 다!

Q: "돌잔치 준비중인데 수면 루틴 망가지지 않으면서 영양 챙기는 방법"
A: "Nutrition_Expert, Sleep_Expert" ← 반드시 둘 다!
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm_supervisor
    result = chain.invoke(messages)
    decision = result.content.strip()
    
    # 강제 로깅
    print(f"\n{'='*60}")
    print(f"📋 [Supervisor] 원본 질문: {messages[-1].content}")
    print(f"🎯 [Supervisor] AI 판단: {decision}")
    
    needs_nutrition = "Nutrition" in decision
    needs_sleep = "Sleep" in decision
    
    print(f"✅ [Supervisor] 영양 필요: {needs_nutrition}")
    print(f"✅ [Supervisor] 수면 필요: {needs_sleep}")
    print(f"{'='*60}\n")
    
    return {
        "needs_nutrition": needs_nutrition,
        "needs_sleep": needs_sleep,
        "next": "route_to_experts"
    }

# [2] Synthesizer 강화 - 통합 품질 개선
def synthesizer_node(state: AgentState):
    """전문가 답변들을 하나로 통합"""
    
    nutrition_ans = state.get("nutrition_answer", "")
    sleep_ans = state.get("sleep_answer", "")
    
    print(f"\n{'='*60}")
    print(f"🔄 [Synthesizer] 통합 시작")
    print(f"   - 영양 답변 존재: {bool(nutrition_ans)} ({len(nutrition_ans)} chars)")
    print(f"   - 수면 답변 존재: {bool(sleep_ans)} ({len(sleep_ans)} chars)")
    
    # 둘 다 있으면 통합
    if nutrition_ans and sleep_ans:
        print(f"   → 두 전문가 답변 통합 모드")
        
        messages = convert_to_messages(state["messages"])
        original_question = messages[-1].content
        
        synthesis_prompt = f"""당신은 육아 전문가 팀 리더입니다.
영양 전문가와 수면 전문가의 답변을 받았습니다. 이를 **하나의 자연스러운 답변**으로 통합하세요.

**원래 질문**: {original_question}

**영양 전문가 답변**:
{nutrition_ans}

**수면 전문가 답변**:
{sleep_ans}

**통합 원칙** (필수):
1. "영양 전문가는...", "수면 전문가는..." 같은 표현 절대 금지
2. 두 답변의 핵심을 자연스럽게 융합
3. 인과관계 명확히 (예: 이유식을 충분히 먹으면 → 수면이 개선됨)
4. 실행 순서 제시 (무엇을 먼저 해야 하는지)
5. 전체 길이: 4-7문장
6. 한 영역의 조언이 다른 영역에 미치는 영향 설명

**통합 예시 (좋은 답변)**:
질문: "7개월 아기가 밤에 자주 깨는데 이유식이랑 관련 있을까?"

"네, 관련이 있을 수 있습니다. 7개월이면 하루 2-3회 이유식을 먹는 시기인데, 특히 저녁 이유식을 충분히 먹이는 것이 중요합니다. 
배가 고프면 밤에 자주 깰 수 있으니, 저녁 6-7시에 단백질과 탄수화물이 포함된 이유식을 주세요. 
동시에 일정한 취침 루틴(이유식→목욕→책→잠)을 만들고 매일 같은 시간에 재우면 수면 패턴이 안정됩니다. 
방은 어둡고 시원하게 유지하세요."

**나쁜 예시 (절대 금지)**:
"영양 전문가에 따르면 이유식을... 그리고 수면 전문가는 수면 루틴을..."

이제 통합된 답변을 작성하세요:
"""
        
        synthesis_msg = llm_expert.invoke([
            SystemMessage(content=synthesis_prompt)
        ])
        
        final_answer = synthesis_msg.content
        print(f"   ✅ 통합 완료 ({len(final_answer)} chars)")
        
    elif nutrition_ans:
        print(f"   → 영양 전문가 답변만 사용")
        final_answer = nutrition_ans
    elif sleep_ans:
        print(f"   → 수면 전문가 답변만 사용")
        final_answer = sleep_ans
    else:
        print(f"   ⚠️ 답변 없음!")
        final_answer = "죄송합니다. 적절한 답변을 생성하지 못했습니다."
    
    print(f"{'='*60}\n")
    
    return {
        "messages": [AIMessage(content=final_answer)]
    }

# [라우팅 로직]
def route_to_experts(state: AgentState):
    """어떤 전문가를 호출할지 결정"""
    needs_nutrition = state.get("needs_nutrition", False)
    needs_sleep = state.get("needs_sleep", False)
    
    if needs_nutrition and needs_sleep:
        return "both"
    elif needs_nutrition:
        return "nutrition_only"
    elif needs_sleep:
        return "sleep_only"
    else:
        return "nutrition_only"  # 기본값

# [그래프 구성]
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Nutrition_Expert", nutrition_expert_node)
workflow.add_node("Sleep_Expert", sleep_expert_node)
workflow.add_node("Synthesizer", synthesizer_node)

# 시작점
workflow.set_entry_point("Supervisor")

# Supervisor에서 라우팅
workflow.add_conditional_edges(
    "Supervisor",
    route_to_experts,
    {
        "both": "Nutrition_Expert",          # 둘 다 필요하면 영양부터
        "nutrition_only": "Nutrition_Expert",
        "sleep_only": "Sleep_Expert"
    }
)

# 영양 전문가 후 처리
def after_nutrition(state: AgentState):
    if state.get("needs_sleep", False):
        return "sleep"  # 수면 전문가도 필요
    else:
        return "synthesize"  # 바로 통합

workflow.add_conditional_edges(
    "Nutrition_Expert",
    after_nutrition,
    {
        "sleep": "Sleep_Expert",
        "synthesize": "Synthesizer"
    }
)

# 수면 전문가는 항상 통합으로
workflow.add_edge("Sleep_Expert", "Synthesizer")

# 통합 후 종료
workflow.add_edge("Synthesizer", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def get_graph_app():
    return app