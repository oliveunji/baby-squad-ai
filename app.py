import streamlit as st
import uuid
import requests
import json

# [중요] 우리가 만든 baseline.py에서 '단일 에이전트 함수'를 가져옵니다.
# (파일이 같은 폴더에 있어야 합니다)
try:
    from baseline import simple_rag_answer
except ImportError:
    st.error("❌ 'baseline.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")
    simple_rag_answer = None

# 1. 페이지 설정
st.set_page_config(page_title="BabySquad AI", page_icon="👶", layout="wide")

# 2. 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 3. 사이드바 (모드 선택)
with st.sidebar:
    st.title("🔧 제어판")
    
    # 모드 선택 스위치
    mode = st.radio("모드 선택", ["💬 일반 대화 모드", "🆚 비교 모드 (A/B Test)"])
    
    st.divider()
    st.info(f"Session ID:\n{st.session_state.thread_id}")
    
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# =========================================================
# [Mode 1] 일반 대화 모드 (기존과 동일)
# =========================================================
if mode == "💬 일반 대화 모드":
    st.header("👶 BabySquad: AI 육아 전문가 팀")
    st.caption("멀티 에이전트 시스템과 대화해보세요.")

    # 대화 기록 표시
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # 사용자 입력
    if prompt := st.chat_input("육아 고민을 물어보세요..."):
        # 내 메시지 표시
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # API 호출
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking... 📡")
            
            try:
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
                    placeholder.error(f"서버 에러 ({response.status_code}): {response.text}")
                    
            except Exception as e:
                placeholder.error(f"서버 연결 실패! server.py가 켜져 있나요?\n에러: {e}")

# =========================================================
# [Mode 2] 비교 모드 (A/B Test) - 면접 시연용 🔥
# =========================================================
else:
    st.header("🆚 성능 비교 (A/B Test)")
    st.markdown("""
    **Single Agent(기본 RAG)**와 **Multi-Agent(제안 모델)**의 답변 품질을 실시간으로 비교합니다.
    """)

    # 비교 전용 입력창
    if prompt := st.chat_input("비교할 질문을 입력하세요 (예: 5개월 아기 이유식 스케줄)"):
        
        # 질문 표시
        st.write(f"### ❓ 질문: {prompt}")
        st.divider()

        # 화면을 좌우로 나눔
        col1, col2 = st.columns(2)

        # [왼쪽] 청코너: Single Agent (Baseline)
        with col1:
            st.subheader("🔵 Single Agent (Baseline)")
            status1 = st.empty()
            status1.info("답변 생성 중...")
            
            try:
                # baseline.py 함수 직접 실행
                if simple_rag_answer:
                    result_a = simple_rag_answer(prompt)
                    status1.empty()
                    st.success("완료")
                    st.markdown(result_a)
                else:
                    st.error("baseline.py 로드 실패")
            except Exception as e:
                status1.error(f"에러 발생: {e}")

        # [오른쪽] 홍코너: Multi-Agent (Proposed)
        with col2:
            st.subheader("🔴 Multi-Agent (Proposed)")
            status2 = st.empty()
            status2.info("API 서버 호출 중...")
            
            try:
                # server.py API 호출
                # (비교 모드에서는 매번 새로운 스레드로 가정하거나, 기존 스레드 유지 선택 가능)
                # 여기서는 공정한 비교를 위해 기존 스레드를 사용해 문맥을 유지하도록 함
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "thread_id": st.session_state.thread_id, # 문맥 유지
                        "message": prompt
                    }
                )
                
                if response.status_code == 200:
                    result_b = response.json()["response"]
                    status2.empty()
                    st.success("완료")
                    st.markdown(result_b)
                else:
                    status2.error(f"API 에러: {response.text}")
                    
            except Exception as e:
                status2.error(f"연결 실패: {e}")