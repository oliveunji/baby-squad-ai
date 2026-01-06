# graph_agent.py
import os
import operator
from typing import Annotated, List, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Expert는 똑똑해야 하니까 gpt-4o 유지
llm_expert = ChatOpenAI(model="gpt-4o", temperature=0) 

# Supervisor는 빠르고 싼 gpt-4o-mini 사용
llm_supervisor = ChatOpenAI(model="gpt-4o-mini", temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# [DB 로드]
if os.path.exists("./chroma_db"):
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="baby_knowledge"
    )
else:
    vector_store = None

def retrieve_knowledge(query: str, category: str) -> str:
    if not vector_store: return "정보 없음"
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

# [상태 정의]
class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add] 
    next: str

# [노드 정의]
def nutrition_expert_node(state: AgentState):
    messages = convert_to_messages(state["messages"])
    last_message = messages[-1].content
    context = retrieve_knowledge(last_message, "Nutrition")
    system_msg = f"당신은 영양 전문가입니다. 정보: {context}"
    response = llm_supervisor.invoke([SystemMessage(content=system_msg)] + messages)
    return {"messages": [response]}

def sleep_expert_node(state: AgentState):
    messages = convert_to_messages(state["messages"])
    last_message = messages[-1].content
    context = retrieve_knowledge(last_message, "Sleep")
    system_msg = f"당신은 수면 전문가입니다. 정보: {context}"
    response = llm_expert.invoke([SystemMessage(content=system_msg)] + messages)
    return {"messages": [response]}

def supervisor_node(state: AgentState):
    messages = convert_to_messages(state["messages"])
    options = ["Nutrition_Expert", "Sleep_Expert"]
    system_prompt = (
        "당신은 관리자입니다. 질문을 보고 담당자를 정하세요: {options}. "
        "답변은 담당자 이름만 하세요."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(options=str(options))
    chain = prompt | llm_supervisor
    result = chain.invoke(messages)
    decision = result.content.strip()
    
    if "Nutrition" in decision: decision = "Nutrition_Expert"
    elif "Sleep" in decision: decision = "Sleep_Expert"
    else: decision = "Nutrition_Expert"
    return {"next": decision}

# ---------------------------------------------------------
# [New Node] 사람 검토 노드 (검문소) 👮
# ---------------------------------------------------------
def human_review_node(state: AgentState):
    # 이 노드는 실제로 아무것도 안 하고 통과만 시킵니다.
    # 하지만 이 노드 '앞'에서 멈추는 게 목적입니다.
    print("  ✅ [사람 검토] 승인 완료. 프로세스를 종료합니다.")
    return {}

# [그래프 연결]
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Nutrition_Expert", nutrition_expert_node)
workflow.add_node("Sleep_Expert", sleep_expert_node)
workflow.add_node("Human_Review", human_review_node)

workflow.set_entry_point("Supervisor")
workflow.add_conditional_edges("Supervisor", lambda x: x["next"], 
                               {"Nutrition_Expert": "Nutrition_Expert", "Sleep_Expert": "Sleep_Expert"})
workflow.add_edge("Nutrition_Expert", "Human_Review")
workflow.add_edge("Sleep_Expert", END)

# 검토가 끝나야 진짜 끝(END)
workflow.add_edge("Human_Review", END)

memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["Nutrition_Expert", "Sleep_Expert"]
)

# [핵심] 그냥 app만 리턴합니다. (LangServe 설정 다 뺌)
def get_graph_app():
    return app