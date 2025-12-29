import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

# 우리가 만든 그래프 엔진 가져오기
from graph_agent import get_graph_app

# 1. 페이지 설정
st.set_page_config(page_title="BabySquad AI 2.0", page_icon="👶")
st.title("👶 BabySquad: AI 육아 전문가 팀 (Agent Ver.)")

# 2. 세션 상태 초기화 (대화기록, Thread ID)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 세션 ID 생성 (새로고침 해도 유지되도록 하려면 별도 처리 필요하지만 지금은 간단히)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 3. 사이드바 (상태 모니터링)
with st.sidebar:
    st.header("🔧 엔진 상태")
    st.write(f"Session ID: `{st.session_state.thread_id}`")
    st.info("LangGraph 기반 멀티 에이전트가 동작 중입니다.")
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4()) # ID 바꿔서 기억 초기화 효과
        st.rerun()

# 4. 이전 대화 내용 화면에 표시
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 5. 사용자 입력 처리
if prompt := st.chat_input("육아 고민을 물어보세요 (예: 이유식 언제 시작해?)"):
    # (1) 사용자 메시지 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # (2) LangGraph 엔진 실행
    app = get_graph_app()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🕵️‍♀️")
        
        # 비용 추적과 함께 실행
        with get_openai_callback() as cb:
            # 스트리밍 없이 일단 결과만 한번에 받기 (invoke)
            # 입력: 현재까지의 모든 메시지가 아니라, '이번 턴의 새 메시지'만 줘도 
            # LangGraph의 MemorySaver가 나머지를 기억함.
            result = app.invoke(
                {"messages": [HumanMessage(content=prompt)]}, 
                config=config
            )
            
            # 최종 답변 추출
            final_response = result["messages"][-1].content
            
            # 화면 표시
            display_text = f"{final_response}\n\n---\n*💰 Cost: ${cb.total_cost:.5f} | Tokens: {cb.total_tokens}*"
            message_placeholder.markdown(display_text)
            
            # (3) AI 메시지 세션에 저장
            st.session_state.messages.append(AIMessage(content=final_response))