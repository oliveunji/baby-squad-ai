# app.py
import streamlit as st
import uuid
import requests

st.set_page_config(page_title="BabySquad Client", page_icon="👶")
st.title("👶 BabySquad: 육아 전문가 (API Ver.)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("🔌 연결 상태")
    st.info(f"Session ID: {st.session_state.thread_id}")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 대화 내용 표시
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking... 📡")
        
        try:
            # 🚀 우리가 만든 커스텀 API(/chat) 호출
            response = requests.post(
                "http://localhost:8000/chat",
                json={
                    "thread_id": st.session_state.thread_id,
                    "message": prompt
                }
            )
            
            if response.status_code == 200:
                result_text = response.json()["response"]
                placeholder.markdown(result_text)
                st.session_state.messages.append({"role": "assistant", "content": result_text})
            else:
                placeholder.error(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            placeholder.error(f"연결 실패: {e}")