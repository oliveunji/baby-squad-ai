import os
import streamlit as st
import uuid
import requests
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(page_title="BabySquad AI", page_icon="👶")
st.title("👶 BabySquad: 안전한 AI 육아 상담소")

# =========================================================
# 2. 세션 상태 초기화
# =========================================================
def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None
    
    if "example_clicked" not in st.session_state:
        st.session_state.example_clicked = None
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

initialize_session_state()

# =========================================================
# 3. 사이드바
# =========================================================
def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.title("🔧 제어판")
        st.info(f"Session ID: {st.session_state.thread_id}")
        
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.session_state.pending_approval = None
            st.session_state.example_clicked = None
            st.session_state.processing = False
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
        
        st.divider()
        st.markdown("### 👥 전문가 팀")
        st.markdown("""
        - 🍎 **영양 전문가**: 이유식, 식단, 알레르기
        - 😴 **수면 전문가**: 수면교육, 수면패턴
        - 🎨 **놀이 전문가**: 발달놀이, 활동
        """)

render_sidebar()

# =========================================================
# 4. 예시 질문 영역
# =========================================================
def render_example_questions():
    """예시 질문 버튼 렌더링"""
    st.markdown("##### 💡 이런 질문을 해보세요")
    
    example_questions = [
        "6개월 아기 이유식은 하루에 몇 번 먹일까요?",
        "이유식 먹고 변이 딱딱해졌는데 괜찮은가요?",
        "아기가 낮잠을 너무 오래 자는데 깨워야 하나요?",
        "아기 열나는데 타이레놀 10ml 먹여도 되?",
    ]
    
    # CSS 스타일링
    st.markdown("""
    <style>
    div[data-testid="column"] > div > div > div > div.stButton {
        height: 80px;
        display: flex;
        align-items: stretch;
    }
    
    div.stButton > button {
        text-align: left !important;
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
        transition: all 0.2s;
        width: 100%;
        height: 100%;
        white-space: normal;
        line-height: 1.4;
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: flex-start !important;
    }
    
    div.stButton > button:hover {
        border-color: #4a90e2;
        background-color: #e8f4ff;
        box-shadow: 0 2px 8px rgba(74,144,226,0.15);
        transform: translateY(-1px);
    }
    
    div.stButton > button > div {
        width: 100%;
        text-align: left !important;
        display: flex;
        justify-content: flex-start !important;
    }
    
    div.stButton > button p {
        text-align: left !important;
        margin: 0;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 버튼 렌더링
    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.example_clicked = q
                st.rerun()

render_example_questions()

# =========================================================
# 5. 대화 내용 표시
# =========================================================
def render_chat_history():
    """이전 대화 내용 표시"""
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

render_chat_history()

# =========================================================
# 6. 유틸리티 함수
# =========================================================
def get_expert_icon(expert_name: str) -> str:
    """전문가별 아이콘 반환"""
    icons = {
        "nutrition": "🍎",
        "sleep": "😴",
        "play": "🎨",
        "health": "🏥",
        "development": "📈"
    }
    return icons.get(expert_name, "👤")

# =========================================================
# 7. API 호출 함수
# =========================================================
def process_question(prompt: str):
    """질문 처리 및 답변 생성"""
    st.session_state.processing = True
    
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        try:
            # 단계별 상태 표시
            status_placeholder.info("🔍 **질문 분석 중...**")
            time.sleep(0.3)
            
            status_placeholder.info("🎯 **전문가 배정 중...**")
            
            # API 호출
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"thread_id": st.session_state.thread_id, "message": prompt}
            )
            data = response.json()
            
            # 전문가 상담 중 표시
            selected_experts = data.get("selected_experts", [])
            if selected_experts:
                expert_str = " ".join([f"{get_expert_icon(e)} {e}" for e in selected_experts])
                status_placeholder.success(f"💬 **전문가 상담 중:** {expert_str}")
                time.sleep(0.5)
            else:
                status_placeholder.success("💬 **전문가 상담 중...**")
                time.sleep(0.5)
            
            # 답변 통합 중 표시
            if len(selected_experts) > 1:
                status_placeholder.info("🔄 **답변 통합 중...**")
                time.sleep(0.3)
            
            # 결과 처리
            if data.get("status") == "review_needed":
                # 검토 필요
                status_placeholder.empty()
                st.session_state.pending_approval = data["draft_response"]
                st.session_state.processing = False
                st.rerun()
            else:
                # 자동 승인/완료
                final_res = data.get("response", "응답 없음")
                
                status_placeholder.success("✅ **답변 생성 완료!**")
                time.sleep(0.3)
                
                status_placeholder.empty()
                st.markdown(final_res)
                
                st.session_state.messages.append({"role": "assistant", "content": final_res})
                st.session_state.processing = False
                
        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ 연결 실패: {e}")
            st.session_state.processing = False

# =========================================================
# 8. 승인 대기 UI
# =========================================================
def render_approval_ui():
    """승인 대기 중 UI 렌더링"""
    draft = st.session_state.pending_approval
    
    with st.chat_message("assistant"):
        st.warning(f"🛡️ [안전 모드] 답변 승인 대기 중\n\n---\n{draft}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ 승인 (Approve)", use_container_width=True):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/approve",
                        json={"thread_id": st.session_state.thread_id}
                    )
                    final_res = res.json()["response"]
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_res})
                    st.session_state.pending_approval = None
                    st.session_state.processing = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"승인 오류: {e}")
        
        with col2:
            if st.button("❌ 반려 (Reject)", use_container_width=True):
                st.error("답변이 반려되었습니다.")
                st.session_state.pending_approval = None
                st.session_state.processing = False
                st.rerun()

# =========================================================
# 9. 사용자 입력 처리
# =========================================================
def handle_user_input(prompt: str):
    """사용자 입력 처리 (예시 질문 + 채팅 입력 공통)"""
    # 사용자 질문 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # API 호출
    process_question(prompt)
    st.rerun()

# =========================================================
# 10. 메인 로직
# =========================================================

# 승인 대기 중이면 승인 UI 표시
if st.session_state.pending_approval:
    render_approval_ui()

# 일반 대화 상태
else:
    # 예시 질문 클릭 처리
    if st.session_state.example_clicked and not st.session_state.processing:
        prompt = st.session_state.example_clicked
        st.session_state.example_clicked = None
        handle_user_input(prompt)
    
    # 일반 채팅 입력 처리
    if not st.session_state.processing:
        if prompt := st.chat_input("육아 고민을 물어보세요..."):
            handle_user_input(prompt)