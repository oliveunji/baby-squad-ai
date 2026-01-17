import os
import streamlit as st
import uuid
import requests
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# 1. 페이지 설정
st.set_page_config(page_title="BabySquad AI", page_icon="👶")
st.title("👶 BabySquad: 안전한 AI 육아 상담소")

# 2. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

# 3. 사이드바
with st.sidebar:
    st.title("🔧 제어판")
    st.info(f"Session ID: {st.session_state.thread_id}")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.markdown("### 👥 전문가 팀")
    st.markdown("""
    - 🍎 **영양 전문가**: 이유식, 식단, 알레르기
    - 😴 **수면 전문가**: 수면교육, 수면패턴
    - 🎨 **놀이 전문가**: 발달놀이, 활동
    """)

# 4. 대화 내용 표시 (이전 대화들)
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# =========================================================
# 전문가 아이콘 및 상태 표시 함수
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

def show_processing_status(status_container, stage: str, experts: list = None):
    """처리 상태를 시각적으로 표시"""
    
    if stage == "analyzing":
        status_container.info("🔍 **질문 분석 중...**")
    
    elif stage == "routing":
        status_container.info("🎯 **전문가 배정 중...**")
    
    elif stage == "consulting":
        if experts:
            expert_str = " ".join([f"{get_expert_icon(e)} {e}" for e in experts])
            status_container.success(f"💬 **전문가 상담 중:** {expert_str}")
        else:
            status_container.success("💬 **전문가 상담 중...**")
    
    elif stage == "synthesizing":
        status_container.info("🔄 **답변 통합 중...**")
    
    elif stage == "complete":
        status_container.success("✅ **답변 생성 완료!**")

# =========================================================
# 5. UI 분기 처리 (승인 대기 중 vs 일반 대화)
# =========================================================

# (A) 승인 대기 중일 때
if st.session_state.pending_approval:
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
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"승인 오류: {e}")

        with col2:
            if st.button("❌ 반려 (Reject)", use_container_width=True):
                st.error("답변이 반려되었습니다.")
                st.session_state.pending_approval = None
                st.rerun()

# (B) 일반 대화 상태일 때
else:
    if prompt := st.chat_input("육아 고민을 물어보세요..."):
        # 1. 사용자 질문 표시
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. API 호출 with 상태 표시
        with st.chat_message("assistant"):
            # 상태 표시용 컨테이너
            status_container = st.empty()
            answer_container = st.empty()
            
            try:
                # 단계별 상태 표시 (시뮬레이션)
                show_processing_status(status_container, "analyzing")
                time.sleep(0.3)  # 시각적 효과
                
                show_processing_status(status_container, "routing")
                
                # 실제 API 호출
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"thread_id": st.session_state.thread_id, "message": prompt}
                )
                data = response.json()
                
                # 백엔드에서 전문가 정보를 받아왔다면 표시
                # (백엔드 수정 필요: selected_experts 정보 포함)
                selected_experts = data.get("selected_experts", [])
                
                if selected_experts:
                    show_processing_status(status_container, "consulting", selected_experts)
                    time.sleep(0.5)
                else:
                    show_processing_status(status_container, "consulting")
                    time.sleep(0.5)
                
                # 복합 질문이면 통합 단계 표시
                if len(selected_experts) > 1:
                    show_processing_status(status_container, "synthesizing")
                    time.sleep(0.3)
                
                # Case 1: 검토 필요
                if data.get("status") == "review_needed":
                    status_container.empty()  # 상태 메시지 지우기
                    st.session_state.pending_approval = data["draft_response"]
                    st.rerun()
                
                # Case 2: 자동 승인/완료
                else:
                    final_res = data.get("response", "응답 없음")
                    
                    # 완료 상태 표시
                    show_processing_status(status_container, "complete")
                    time.sleep(0.3)
                    status_container.empty()  # 상태 메시지 지우기
                    
                    # 최종 답변 표시
                    answer_container.markdown(final_res)
                    st.session_state.messages.append({"role": "assistant", "content": final_res})
                    
            except Exception as e:
                status_container.empty()
                answer_container.error(f"❌ 연결 실패: {e}")