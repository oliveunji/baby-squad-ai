import os
import operator
from typing import Annotated, List, TypedDict, Union
from dotenv import load_dotenv

# 라이브러리 임포트
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings # DB 검색용
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # 🧠 기억력 모듈

# 1. 환경 설정
load_dotenv()

# [중요] 모델 설정
# - 추론/대화: OpenAI (GPT-4o)
# - 임베딩/검색: Google (기존 DB와 호환성 유지)
if not os.getenv("OPENAI_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: API Key가 부족합니다. .env를 확인하세요.")
    exit()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. 벡터 DB 로드 (기존에 만든 chroma_db 폴더가 있어야 함!)
if os.path.exists("./chroma_db"):
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="baby_knowledge"
    )
    print("✅ RAG 데이터베이스 연결 성공")
else:
    vector_store = None
    print("⚠️ 경고: chroma_db 폴더가 없습니다. ingest.py를 먼저 실행하세요.")

# ---------------------------------------------------------
# [Step 1] 상태(State) 정의
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# ---------------------------------------------------------
# [Step 2] 도구 함수 (RAG 검색)
# ---------------------------------------------------------
def retrieve_knowledge(query: str, category: str) -> str:
    """DB에서 관련 정보를 검색해서 반환"""
    if not vector_store:
        return "죄송합니다. 지식 DB가 없습니다."
    
    print(f"  🔍 [{category}] 문서 검색 중: '{query}'")
    results = vector_store.similarity_search(query, k=3)
    
    context = "\n".join([f"- {doc.page_content} (출처: {doc.metadata.get('source', 'unknown')})" for doc in results])
    return context

# ---------------------------------------------------------
# [Step 3] 전문가 노드 (Workers) - RAG 적용됨!
# ---------------------------------------------------------
def nutrition_expert_node(state: AgentState):
    print("  🥦 [영양 전문가] 가 작업을 시작합니다.")
    last_message = state["messages"][-1].content
    
    # 1. 문서 검색
    context = retrieve_knowledge(last_message, "Nutrition")
    
    # 2. 답변 생성 (Context 주입)
    system_msg = (
        "당신은 영양 전문가입니다. 아래의 [검색된 가이드라인]을 바탕으로 답변하세요.\n"
        "출처를 반드시 명시하세요.\n\n"
        f"[검색된 가이드라인]\n{context}"
    )
    
    response = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}

def sleep_expert_node(state: AgentState):
    print("  💤 [수면 전문가] 가 작업을 시작합니다.")
    last_message = state["messages"][-1].content
    
    # 1. 문서 검색
    context = retrieve_knowledge(last_message, "Sleep")
    
    # 2. 답변 생성
    system_msg = (
        "당신은 수면 전문가입니다. 아래의 [검색된 가이드라인]을 바탕으로 답변하세요.\n"
        "출처를 반드시 명시하세요.\n\n"
        f"[검색된 가이드라인]\n{context}"
    )
    
    response = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}

# ---------------------------------------------------------
# [Step 4] 관리자 노드 (Supervisor)
# ---------------------------------------------------------
def supervisor_node(state: AgentState):
    print("\n👮 [관리자(GPT-4o)] 가 질문을 분석 중입니다...")
    
    options = ["Nutrition_Expert", "Sleep_Expert"]
    
    system_prompt = (
        "당신은 BabySquad 팀의 관리자입니다."
        "대화 내용을 보고 다음 중 누구에게 업무를 배정할지 결정하세요: {options}."
        "답변은 반드시 전문가의 이름만 말하세요."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(options=str(options))
    
    supervisor_chain = prompt | llm
    
    result = supervisor_chain.invoke(state["messages"])
    decision = result.content.strip()
    
    # 안전장치
    if "Nutrition" in decision: decision = "Nutrition_Expert"
    elif "Sleep" in decision: decision = "Sleep_Expert"
    else: decision = "Nutrition_Expert"
        
    print(f"👉 판단 결과: '{decision}' 에게 배정합니다.")
    return {"next": decision}

# ---------------------------------------------------------
# [Step 5] 그래프 연결 (Wiring)
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Nutrition_Expert", nutrition_expert_node)
workflow.add_node("Sleep_Expert", sleep_expert_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        "Nutrition_Expert": "Nutrition_Expert",
        "Sleep_Expert": "Sleep_Expert"
    }
)

workflow.add_edge("Nutrition_Expert", END)
workflow.add_edge("Sleep_Expert", END)

# [핵심] 기억력 장착! 🧠
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def get_graph_app():
    return app
# ---------------------------------------------------------
# [Step 6] 대화형 실행 (Chat Loop)
# ---------------------------------------------------------
# if __name__ == "__main__":
#     from langchain_community.callbacks import get_openai_callback
    
#     # thread_id: 대화 세션을 구분하는 ID (이게 같으면 기억을 공유함)
#     config = {"configurable": {"thread_id": "session_1"}}
    
#     print("========== BabySquad AI (Type 'quit' to exit) ==========")
    
#     while True:
#         user_input = input("\n🧑 사용자: ")
#         if user_input.lower() in ["quit", "exit"]:
#             break
            
#         with get_openai_callback() as cb:
#             # invoke 대신 stream을 쓰면 한 글자씩 나오지만, 지금은 간단히 invoke
#             result = app.invoke(
#                 {"messages": [HumanMessage(content=user_input)]},
#                 config=config # 설정(thread_id) 전달
#             )
#             print(f"🤖 AI: {result['messages'][-1].content}")
#             print(f"   (Cost: ${cb.total_cost:.5f})")