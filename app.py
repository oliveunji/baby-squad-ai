import streamlit as st
import uuid
import requests

# 1. 페이지 설정
st.set_page_config(page_title="BabySquad AI", page_icon="👶")
st.title("👶 BabySquad: 안전한 AI 육아 상담소")

# 2. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# [핵심] 승인 대기 상태를 저장할 변수 추가
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None  # None이면 대기 없음, 값이 있으면 대기 중인 답변 텍스트

# 3. 사이드바
with st.sidebar:
    st.title("🔧 제어판")
    st.info(f"Session ID: {st.session_state.thread_id}")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 4. 대화 내용 표시 (이전 대화들)
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# =========================================================
# 5. UI 분기 처리 (승인 대기 중 vs 일반 대화)
# =========================================================

# (A) 승인 대기 중일 때 (질문 입력창 숨기고 승인 버튼 보여주기)
if st.session_state.pending_approval:
    draft = st.session_state.pending_approval
    
    with st.chat_message("assistant"):
        # 경고 박스 표시
        st.warning(f"🛡️ [안전 모드] 답변 승인 대기 중\n\n---\n{draft}")
        
        col1, col2 = st.columns(2)
        
        # ✅ 승인 버튼 로직 (이제 chat_input 밖이라서 잘 동작함!)
        with col1:
            if st.button("✅ 승인 (Approve)", use_container_width=True):
                try:
                    # 1. /approve API 호출
                    res = requests.post(
                        "http://localhost:8000/approve",
                        json={"thread_id": st.session_state.thread_id}
                    )
                    final_res = res.json()["response"]
                    
                    # 2. 결과 저장 및 상태 해제
                    st.session_state.messages.append({"role": "assistant", "content": final_res})
                    st.session_state.pending_approval = None # 대기 상태 해제
                    st.rerun() # 화면 갱신
                    
                except Exception as e:
                    st.error(f"승인 오류: {e}")

        # ❌ 반려 버튼 로직
        with col2:
            if st.button("❌ 반려 (Reject)", use_container_width=True):
                st.error("답변이 반려되었습니다.")
                st.session_state.pending_approval = None # 대기 상태 해제
                st.rerun()

# (B) 일반 대화 상태일 때 (평소처럼 질문 입력창 표시)
else:
    if prompt := st.chat_input("육아 고민을 물어보세요..."):
        # 1. 사용자 질문 표시
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. API 호출
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking... 📡")
            
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"thread_id": st.session_state.thread_id, "message": prompt}
                )
                data = response.json()
                
                # Case 1: 검토 필요 (Review Needed)
                if data.get("status") == "review_needed":
                    # [핵심] 화면에 바로 띄우지 않고, session_state에 저장 후 리런!
                    st.session_state.pending_approval = data["draft_response"]
                    st.rerun() # 다시 실행해서 (A) 흐름으로 보냄
                
                # Case 2: 자동 승인/완료 (Completed)
                else:
                    final_res = data.get("response", "응답 없음")
                    placeholder.markdown(final_res)
                    st.session_state.messages.append({"role": "assistant", "content": final_res})
                    
            except Exception as e:
                placeholder.error(f"연결 실패: {e}")