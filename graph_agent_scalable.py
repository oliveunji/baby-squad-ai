# graph_agent_scalable.py
import os
import operator
from typing import Annotated, List, Any, Dict, Callable
from typing_extensions import TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ================================================================================
# 설정
# ================================================================================
llm_expert = ChatOpenAI(model="gpt-4o", temperature=0) 
llm_supervisor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_store = PineconeVectorStore(
    index_name="baby-index",
    embedding=embeddings
)

def retrieve_knowledge(query: str, category: str) -> str:
    """지식베이스 검색"""
    if not vector_store: 
        return "정보 없음"
    
    print(f"  🔍 [{category}] 검색: '{query[:30]}...'")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )
    try:
        results = retriever.invoke(query)
        if not results:
            return "관련 가이드라인을 찾지 못했습니다."
        context = "\n".join([f"- {doc.page_content}" for doc in results])
        return context
    except Exception as e:
        return "관련 가이드라인을 찾지 못했습니다."

# ================================================================================
# 상태 정의
# ================================================================================
class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    question: str  # 원본 질문
    complexity: str  # "simple" or "complex"
    selected_experts: List[str]  # 선택된 전문가 리스트
    expert_answers: Dict[str, str]  # 전문가별 답변 {expert_name: answer}
    final_answer: str
    metadata: Dict[str, Any]  # 메타데이터 (프론트엔드 전달용)

# ================================================================================
# 전문가 정의 클래스
# ================================================================================
@dataclass
class ExpertConfig:
    """전문가 설정"""
    name: str
    display_name: str
    keywords: List[str]
    category: str  # 검색용 카테고리
    system_prompt: str

# ================================================================================
# 전문가 레지스트리
# ================================================================================
class ExpertRegistry:
    """전문가 등록 및 관리"""
    
    def __init__(self):
        self.experts: Dict[str, ExpertConfig] = {}
    
    def register(self, config: ExpertConfig):
        """전문가 등록"""
        self.experts[config.name] = config
        print(f"  ✅ 전문가 등록: {config.display_name}")
    
    def get_expert_list(self) -> List[str]:
        """등록된 전문가 목록"""
        return list(self.experts.keys())
    
    def get_expert_descriptions(self) -> str:
        """전문가 설명 (Orchestrator용)"""
        descriptions = []
        for name, config in self.experts.items():
            keywords = ", ".join(config.keywords[:5])
            descriptions.append(
                f"- {name}: {config.display_name} (키워드: {keywords})"
            )
        return "\n".join(descriptions)
    
    def select_experts_by_keywords(self, question: str) -> List[str]:
        """키워드 기반 빠른 전문가 선택"""
        q_lower = question.lower()
        selected = []
        for name, config in self.experts.items():
            if any(kw in q_lower for kw in config.keywords):
                selected.append(name)
        return selected

# 전역 레지스트리 생성
registry = ExpertRegistry()

# ================================================================================
# 전문가 등록
# ================================================================================

# 영양 전문가
registry.register(ExpertConfig(
    name="nutrition",
    display_name="영양 전문가",
    keywords=["이유식", "먹", "음식", "영양", "분유", "우유", "알레르기", "편식", "식단", "단백질", "철분"],
    category="Nutrition",
    system_prompt="""당신은 소아 영양 전문가입니다.

# 응답 원칙
- 2-4문장으로 핵심만 전달
- 구체적 수치와 방법 포함
- 안전 관련은 반드시 언급

# 참고 자료
{context}

간결하고 실용적으로 답변하세요."""
))

# 수면 전문가
registry.register(ExpertConfig(
    name="sleep",
    display_name="수면 전문가",
    keywords=["자", "잠", "수면", "깨", "낮잠", "밤잠", "재우", "통잠", "수면교육", "야간수유"],
    category="Sleep",
    system_prompt="""당신은 소아 수면 전문가입니다.

# 응답 원칙
- 2-4문장으로 핵심만 전달
- 안전 수칙 (SIDS 예방 등) 필수
- 실행 가능한 루틴 제시

# 참고 자료
{context}

간결하고 실용적으로 답변하세요."""
))

# 🆕 놀이 전문가 (추가 예시)
registry.register(ExpertConfig(
    name="play",
    display_name="놀이 전문가",
    keywords=["놀이", "장난감", "활동", "발달놀이", "오감", "그림책", "놀아주"],
    category="Play",  # 별도 카테고리 있다면
    system_prompt="""당신은 아동 놀이 전문가입니다.

# 응답 원칙
- 2-4문장으로 핵심만 전달
- 연령별 적합한 놀이 추천
- 집에서 쉽게 할 수 있는 방법

# 참고 자료
{context}

간결하고 실용적으로 답변하세요."""
))

# ================================================================================
# 노드 1: 복잡도 판단
# ================================================================================
def complexity_router_node(state: AgentState):
    """질문의 복잡도 판단"""
    messages = convert_to_messages(state["messages"])
    question = messages[-1].content
    
    # 1단계: 키워드 기반 빠른 전문가 선택
    selected = registry.select_experts_by_keywords(question)
    
    print(f"\n{'='*60}")
    print(f"📋 질문: {question}")
    print(f"🔍 키워드 매칭: {selected}")
    
    # 2단계: 복잡도 판단
    # - 2명 이상 전문가 필요 = 복합
    # - 질문 길이 긴 경우 = 복합
    # - 단순 패턴 = 단순
    
    is_complex = False
    
    if len(selected) >= 2:
        is_complex = True
        print(f"  → 복수 도메인 감지 (복합)")
    elif len(question) > 40 and not any(p in question for p in ["?", "언제", "얼마", "몇"]):
        is_complex = True
        print(f"  → 긴 질문 (복합)")
    else:
        # LLM으로 최종 판단 (애매한 경우)
        complexity_prompt = f"""이 질문이 복합적인지 단순한지 판단하세요.

질문: {question}

기준:
- 복합(COMPLEX): 여러 영역 관련, 깊은 분석/조언 필요, 인과관계 질문
- 단순(SIMPLE): 단일 팩트 질문, 짧고 명확한 답 가능

답변: COMPLEX 또는 SIMPLE"""
        
        result = llm_supervisor.invoke([
            SystemMessage(content=complexity_prompt)
        ]).content.strip()
        
        is_complex = "COMPLEX" in result
        print(f"  → LLM 판단: {result}")
    
    complexity = "complex" if is_complex else "simple"
    print(f"✅ 최종 복잡도: {complexity.upper()}")
    print(f"{'='*60}\n")
    
    return {
        "question": question,
        "complexity": complexity,
        "selected_experts": selected,
        "metadata": {
            "stage": "routing",
            "complexity": complexity,
            "initial_experts": selected
        }
    }

# ================================================================================
# 노드 2: 단순 질문 직접 답변
# ================================================================================
def direct_answer_node(state: AgentState):
    """간단한 질문은 바로 답변 (Single Agent처럼)"""
    question = state["question"]
    selected = state.get("selected_experts", [])
    
    # 선택된 전문가 중 첫 번째 사용 (또는 기본값)
    if selected:
        expert_name = selected[0]
        expert_config = registry.experts[expert_name]
    else:
        # 기본값: 영양 전문가
        expert_name = "nutrition"
        expert_config = registry.experts["nutrition"]
    
    print(f"  ⚡ [직접 답변] {expert_config.display_name} 호출")
    
    # 컨텍스트 검색
    context = retrieve_knowledge(question, expert_config.category)
    
    # 답변 생성
    system_msg = expert_config.system_prompt.format(context=context)
    messages = convert_to_messages(state["messages"])
    
    response = llm_expert.invoke([
        SystemMessage(content=system_msg),
        *messages
    ])
    
    return {
        "final_answer": response.content,
        "messages": [response],
        "selected_experts": [expert_name],  # 실제 사용된 전문가
        "metadata": {
            "stage": "completed",
            "complexity": "simple",
            "selected_experts": [expert_name],
            "expert_count": 1
        }
    }

# ================================================================================
# 노드 3: Orchestrator (전문가 선택)
# ================================================================================
def orchestrator_node(state: AgentState):
    """복합 질문에서 필요한 전문가들을 최종 선택"""
    question = state["question"]
    initial_selected = state.get("selected_experts", [])
    
    # LLM에게 정확한 전문가 선택 요청
    expert_descriptions = registry.get_expert_descriptions()
    
    orchestrator_prompt = f"""당신은 육아 상담 관리자입니다.
질문을 분석하여 필요한 전문가를 선택하세요.

**등록된 전문가:**
{expert_descriptions}

**질문:** {question}

**규칙:**
- 질문과 관련된 전문가를 모두 선택
- 애매하면 관련 가능성 있는 전문가 포함
- 전문가 이름만 쉼표로 구분하여 출력

**예시:**
질문: "밤에 배고파서 깨는 것 같아요"
답변: nutrition, sleep

질문: "7개월 아기 놀이 추천해주세요"
답변: play

이제 위 질문에 대해 필요한 전문가를 선택하세요 (이름만 출력):"""

    result = llm_supervisor.invoke([
        SystemMessage(content=orchestrator_prompt)
    ]).content.strip()
    
    # 파싱
    selected_experts = [
        name.strip() 
        for name in result.split(",")
        if name.strip() in registry.experts
    ]
    
    # 빈 경우 초기 선택 사용
    if not selected_experts:
        selected_experts = initial_selected if initial_selected else ["nutrition"]
    
    print(f"  🎯 [Orchestrator] 선택된 전문가: {selected_experts}")
    
    return {
        "selected_experts": selected_experts,
        "metadata": {
            "stage": "expert_selection",
            "selected_experts": selected_experts,
            "expert_count": len(selected_experts)
        }
    }

# ================================================================================
# 노드 4: 전문가 실행 (동적)
# ================================================================================
def expert_execution_node(state: AgentState):
    """선택된 전문가들을 병렬로 실행"""
    question = state["question"]
    selected_experts = state["selected_experts"]
    
    expert_answers = {}
    
    print(f"\n{'='*60}")
    print(f"🔧 전문가 실행 시작 ({len(selected_experts)}명)")
    
    for expert_name in selected_experts:
        if expert_name not in registry.experts:
            print(f"  ⚠️ 알 수 없는 전문가: {expert_name}")
            continue
        
        expert_config = registry.experts[expert_name]
        print(f"  💬 [{expert_config.display_name}] 실행 중...")
        
        # 컨텍스트 검색
        context = retrieve_knowledge(question, expert_config.category)
        
        # 답변 생성
        system_msg = expert_config.system_prompt.format(context=context)
        messages = convert_to_messages(state["messages"])
        
        response = llm_expert.invoke([
            SystemMessage(content=system_msg),
            *messages
        ])
        
        expert_answers[expert_name] = response.content
        print(f"     ✅ 답변 생성 완료 ({len(response.content)} chars)")
    
    print(f"{'='*60}\n")
    
    return {
        "expert_answers": expert_answers,
        "metadata": {
            "stage": "consulting",
            "selected_experts": selected_experts,
            "expert_count": len(selected_experts)
        }
    }

# ================================================================================
# 노드 5: 답변 통합
# ================================================================================
def synthesizer_node(state: AgentState):
    """여러 전문가 답변을 하나로 통합"""
    question = state["question"]
    expert_answers = state.get("expert_answers", {})
    selected_experts = state.get("selected_experts", [])
    
    print(f"  🔄 [통합] {len(expert_answers)}개 답변 통합 중...")
    
    # 전문가가 1명이면 그대로 사용
    if len(expert_answers) == 1:
        final_answer = list(expert_answers.values())[0]
        print(f"     → 단일 답변 사용")
    
    # 여러 명이면 통합
    else:
        # 전문가별 답변 포맷팅
        answers_text = "\n\n".join([
            f"**{registry.experts[name].display_name} 답변:**\n{answer}"
            for name, answer in expert_answers.items()
        ])
        
        synthesis_prompt = f"""당신은 육아 전문가 팀 리더입니다.
여러 전문가의 답변을 하나의 자연스러운 답변으로 통합하세요.

**원래 질문:** {question}

{answers_text}

**통합 원칙:**
1. "~전문가는..." 같은 표현 금지
2. 핵심만 자연스럽게 융합
3. 인과관계와 우선순위 명확히
4. 전체 4-7문장 이내
5. 중복 제거, 모순 조정

통합된 답변을 작성하세요:"""
        
        synthesis_msg = llm_expert.invoke([
            SystemMessage(content=synthesis_prompt)
        ])
        
        final_answer = synthesis_msg.content
        print(f"     ✅ 통합 완료 ({len(final_answer)} chars)")
    
    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "metadata": {
            "stage": "completed",
            "complexity": "complex",
            "selected_experts": selected_experts,
            "expert_count": len(expert_answers),
            "synthesis_performed": len(expert_answers) > 1
        }
    }

# ================================================================================
# 라우팅 함수
# ================================================================================
def route_by_complexity(state: AgentState) -> str:
    """복잡도에 따라 라우팅"""
    complexity = state.get("complexity", "simple")
    return complexity

# ================================================================================
# 그래프 구성
# ================================================================================
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("ComplexityRouter", complexity_router_node)
workflow.add_node("DirectAnswer", direct_answer_node)
workflow.add_node("Orchestrator", orchestrator_node)
workflow.add_node("ExpertExecution", expert_execution_node)
workflow.add_node("Synthesizer", synthesizer_node)

# 시작: 복잡도 판단
workflow.set_entry_point("ComplexityRouter")

# 복잡도에 따라 분기
workflow.add_conditional_edges(
    "ComplexityRouter",
    route_by_complexity,
    {
        "simple": "DirectAnswer",      # 단순 → 직접 답변
        "complex": "Orchestrator"       # 복합 → 전문가 팀
    }
)

# 직접 답변 → 종료
workflow.add_edge("DirectAnswer", END)

# 복합 질문 플로우
workflow.add_edge("Orchestrator", "ExpertExecution")
workflow.add_edge("ExpertExecution", "Synthesizer")
workflow.add_edge("Synthesizer", END)

# 컴파일
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def get_graph_app():
    return app

# ================================================================================
# 사용 예시
# ================================================================================
if __name__ == "__main__":
    app = get_graph_app()
    
    test_questions = [
        "이유식 언제 시작해?",  # 단순 → 직접 답변
        "7개월 아기가 밤에 자주 깨는데 이유식이랑 관련 있을까?",  # 복합 → 영양+수면
        "돌 아기 놀이 추천해줘",  # 단순 → 놀이 전문가
    ]
    
    for i, q in enumerate(test_questions, 1):
        print(f"\n{'#'*80}")
        print(f"테스트 {i}: {q}")
        print(f"{'#'*80}")
        
        response = app.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"configurable": {"thread_id": f"test_{i}"}}
        )
        
        # 메타데이터 확인
        metadata = response.get("metadata", {})
        
        print(f"\n📊 메타데이터:")
        print(f"   - 복잡도: {metadata.get('complexity', 'N/A')}")
        print(f"   - 선택된 전문가: {metadata.get('selected_experts', [])}")
        print(f"   - 전문가 수: {metadata.get('expert_count', 0)}")
        print(f"   - 단계: {metadata.get('stage', 'N/A')}")
        
        print(f"\n📝 최종 답변:")
        print(response["messages"][-1].content)
        print(f"\n{'='*80}\n")

# ================================================================================
# API 응답 헬퍼 함수 (백엔드 API용)
# ================================================================================
def get_response_with_metadata(state: AgentState) -> dict:
    """
    프론트엔드에 전달할 응답 데이터 생성
    
    Returns:
        {
            "response": "최종 답변",
            "selected_experts": ["nutrition", "sleep"],
            "complexity": "complex",
            "expert_count": 2,
            "stage": "completed"
        }
    """
    metadata = state.get("metadata", {})
    
    return {
        "response": state.get("final_answer", ""),
        "selected_experts": metadata.get("selected_experts", []),
        "complexity": metadata.get("complexity", "simple"),
        "expert_count": metadata.get("expert_count", 0),
        "stage": metadata.get("stage", "completed"),
        "synthesis_performed": metadata.get("synthesis_performed", False)
    }