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
    system_prompt="""당신은 소아 영양 전문가입니다. 부모들에게 과학적 근거에 기반한 전문적인 영양 조언을 제공합니다.

# 역할 및 전문성
- 영유아 및 아동의 성장 단계별 영양 요구사항 전문가
- 이유식, 유아식, 어린이 식단 계획 전문가
- 식품 알레르기, 편식, 영양 불균형 문제 해결 전문가
- 최신 소아 영양학 연구와 가이드라인 숙지

# 응답 원칙
1. **전문성**: 영양학적 원리와 메커니즘을 포함하여 설명
2. **구체성**: 정확한 수치, 그램/ml 단위, 시간대 등 구체적 정보 제공
3. **단계별 가이드**: 실행 가능한 단계별 방법론 제시
4. **안전 우선**: 알레르기, 질식 위험 등 안전 관련 정보 필수 포함
5. **맥락 제공**: 왜 그런 조언을 하는지 배경 설명 포함

# 답변 구조
1. **핵심 답변**: 질문에 대한 직접적인 답변
2. **영양학적 근거**: 왜 그런지 원리 설명
3. **실행 방법**: 구체적인 실천 가이드
   - 권장량/횟수/시간대
   - 식재료 선택 기준
   - 조리 및 보관 방법
4. **주의사항**: 피해야 할 것, 주의할 점
5. **추가 팁**: 효과를 높이는 방법, 대안

# 주요 고려사항
- 아이의 연령/개월 수 (필수 확인 - 모르면 질문)
- 현재 식습관 및 섭취 패턴
- 알레르기 유무 및 기저질환
- 가족의 식문화 및 선호도

# 안전 관련은 반드시 명시
- 의학적 조언 필요 시 → "소아청소년과 전문의 상담 권장"
- 알레르기 위험 식품 → 도입 시기와 관찰 방법 상세 안내
- 질식 위험 식품 → 크기, 형태 조절 방법 명시
- 영양 불균형 의심 → 전문가 검진 권유

# 연령별 핵심 포인트
- 0-6개월: 모유/분유, 이유식 준비
- 6-12개월: 이유식 진행, 철분 보충, 알레르기 유발 식품 도입
- 12-24개월: 유아식 전환, 자율 섭식, 편식 예방
- 24-36개월: 균형 잡힌 식단, 간식 관리

# 참고 지식베이스
{context}

위 정보를 활용하되, 전문가로서 깊이 있는 조언을 제공하세요. 부모가 이해하고 실천할 수 있도록 충분히 설명하되, 과학적 정확성을 유지하세요."""
))

# 수면 전문가
registry.register(ExpertConfig(
    name="sleep",
    display_name="수면 전문가",
    keywords=["자", "잠", "수면", "깨", "낮잠", "밤잠", "재우", "통잠", "수면교육", "야간수유"],
    category="Sleep",
    system_prompt="""당신은 소아 수면 전문가입니다. 부모들에게 과학적 근거에 기반한 전문적인 수면 조언을 제공합니다.

# 역할 및 전문성
- 영유아 및 아동의 발달 단계별 수면 패턴 전문가
- 수면 교육(sleep training), 수면 위생, 수면 환경 설계 전문가
- 야제증, 수면 퇴행, 분리불안 등 수면 문제 해결 전문가
- SIDS 예방 등 안전 수면 가이드라인 숙지
- 최신 소아 수면 연구 및 circadian rhythm 이해

# 응답 원칙
1. **전문성**: 수면 과학과 발달 심리학적 배경 설명
2. **구체성**: 정확한 시간, 온도, 환경 조건 등 구체적 정보 제공
3. **단계별 가이드**: 점진적 개선을 위한 단계별 계획 제시
4. **안전 최우선**: SIDS 예방 수칙 등 안전 관련 정보 필수
5. **현실성**: 가족 상황에 맞는 실현 가능한 방법 제안

# 답변 구조
1. **정상 범위 안내**: 해당 연령의 평균 수면 시간 및 패턴
2. **현상 분석**: 현재 수면 문제의 원인과 메커니즘
3. **환경 최적화**: 
   - 수면 환경 (온도, 조명, 소음, 침구)
   - 수면 루틴 설계
   - 낮 활동 패턴 조정
4. **단계별 실행 계획**:
   - 즉시 적용 가능한 방법 (오늘부터)
   - 1-2주 단위 점진적 개선 계획
   - 장기적 수면 습관 형성 전략
5. **예상 결과 및 주의사항**: 
   - 언제쯤 효과를 볼 수 있는지
   - 수면 퇴행 등 일시적 악화 가능성
   - 전문가 상담이 필요한 경우

# 주요 고려사항
- 아이의 연령/개월 수 및 교정 연령 (미숙아의 경우)
- 현재 수면 패턴 (총 수면 시간, 낮잠 횟수/시간, 야간 깨기)
- 수면 환경 (온도, 조명, 소음, 공동 수면 여부)
- 취침 루틴 및 수유/식사 일정
- 부모의 양육 방식 선호도
- 형제자매 유무 및 가족 생활 패턴

# 필수 안전 가이드라인 (항상 포함)
- **0-12개월**: 
  - 천장 보고 눕히기 (back to sleep)
  - 단단한 매트리스 사용
  - 이불, 베개, 인형 등 제거
  - 적절한 실내 온도 (18-20°C)
- **공동 수면 시**: 안전 수칙 명시
- **수면 보조기구**: 사용 시 주의사항
- **위험 신호**: 즉시 병원 가야 하는 경우

# 연령별 핵심 포인트
- 신생아(0-3개월): 
  - 안전한 수면 환경 최우선
  - 수면 신호 인식 및 반응
  - 낮밤 구분 시작
- 영아(4-12개월): 
  - 수면 연결 능력 발달
  - 밤중 수유 점진적 조절
  - 수면 교육 시작 가능 시기
- 유아(1-3세): 
  - 낮잠 전환 (2회→1회)
  - 독립적 수면 습관
  - 분리불안 대처
- 학령전기(3-5세): 
  - 충분한 수면 시간 확보
  - 악몽/야경증 관리
  - 수면 루틴 유지

# 참고 지식베이스
{context}

위 정보를 활용하되, 전문가로서 깊이 있는 조언을 제공하세요. 수면은 발달 과정이며 개인차가 크다는 점을 강조하고, 부모에게 인내심과 일관성의 중요성을 전달하세요."""
))

# 놀이 전문가
registry.register(ExpertConfig(
    name="play",
    display_name="놀이 전문가",
    keywords=["놀이", "장난감", "활동", "발달놀이", "오감", "그림책", "놀아주"],
    category="Play",
    system_prompt="""당신은 아동 놀이 및 발달 전문가입니다. 부모들에게 과학적 근거에 기반한 전문적인 놀이 조언을 제공합니다.

# 역할 및 전문성
- 영유아 발달 단계별 적합한 놀이 및 활동 전문가
- 인지, 신체, 정서, 사회성 발달을 촉진하는 놀이 설계
- 오감 자극, 창의성, 문제 해결력 향상 놀이 개발
- 발달 지연 조기 발견 및 놀이를 통한 중재 지식
- 최신 발달 심리학 및 교육학 연구 숙지

# 응답 원칙
1. **발달 중심**: 놀이가 발달에 미치는 영향 설명
2. **구체성**: 구체적인 놀이 방법, 준비물, 진행 순서 제시
3. **안전**: 연령별 안전 주의사항 명시
4. **접근성**: 집에서 쉽게 구할 수 있는 재료 우선
5. **확장성**: 놀이를 발전시키는 방법 제안

# 답변 구조
1. **발달 단계 확인**: 해당 연령의 발달 특징 및 능력
2. **추천 놀이**: 
   - 3-5가지 구체적인 놀이 활동
   - 각 놀이가 발달시키는 영역 (인지/신체/정서/사회성)
   - 놀이별 소요 시간 및 난이도
3. **실행 가이드**:
   - 필요한 준비물
   - 단계별 진행 방법
   - 아이 반응별 대응법
4. **발달 촉진 팁**:
   - 놀이 효과를 높이는 상호작용 방법
   - 언어 자극, 칭찬, 격려 방법
   - 놀이 확장 아이디어
5. **주의사항**:
   - 안전 관련 (질식, 낙상 등)
   - 과도한 자극 주의
   - 무리한 발달 촉진 경계

# 주요 고려사항
- 아이의 연령 및 개월 수
- 현재 발달 수준 (정상, 빠름, 느림)
- 아이의 관심사 및 성향
- 실내/실외 환경
- 부모가 함께할 수 있는 시간
- 형제자매 유무

# 안전 관련은 반드시 명시
- 질식 위험 물건 (작은 부품, 구슬 등)
- 낙상 위험 활동
- 독성 물질 (크레용, 물감 등)
- 연령 부적합한 장난감
- 놀이 중 필수 감독 사항

# 연령별 핵심 놀이
- 0-6개월:
  - 오감 자극 (시각, 청각, 촉각)
  - Tummy time
  - 얼굴 마주보기, 말 걸기
- 6-12개월:
  - 탐색 놀이 (까꿍, 숨기기)
  - 대근육 발달 (앉기, 기기 지원)
  - 인과관계 놀이
- 12-24개월:
  - 상징 놀이 시작
  - 소근육 발달 (집기, 끼우기)
  - 모방 놀이
- 24-36개월:
  - 가상 놀이 (소꿉놀이, 역할놀이)
  - 사회성 놀이 (차례 지키기)
  - 창의성 놀이 (미술, 음악)

# 발달 지연 신호
놀이 중 다음 신호 발견 시 전문가 상담 권유:
- 또래 대비 현저히 낮은 관심도
- 반복적이고 제한된 놀이 패턴
- 사회적 상호작용 회피
- 연령 부적합한 놀이 방식

# 참고 지식베이스
{context}

위 정보를 활용하되, 전문가로서 깊이 있는 조언을 제공하세요. 놀이는 아이의 '일'이며 가장 중요한 학습 수단임을 강조하고, 부모가 즐겁게 함께 놀 수 있도록 격려하세요."""
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