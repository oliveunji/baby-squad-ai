import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# ADK 및 Agent 관련 임포트
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# 1. 기본 설정 및 API 키 로드
load_dotenv()

st.set_page_config(page_title="BabySquad", page_icon="👶")
st.title("👶 BabySquad: AI 육아 전문가 팀")
st.caption("🚀 4개월 아기 부모를 위한 수면 & 영양 맞춤 솔루션")

with st.sidebar:
    st.header("About BabySquad")
    st.markdown("""
    - 👩‍💼 **Head Nanny:** 팀장
    - 💤 **Sleep Expert:** 수면 전문가
    - 🥦 **Nutritionist:** 영양 전문가
    """)
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# 2. 에이전트 팀 설정 (캐싱 사용)
# ---------------------------------------------------------
@st.cache_resource
def setup_agent_system():
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("API Key가 없습니다. .env 파일을 확인해주세요.")
        return None, None, None

    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    MODEL_NAME = "openai/gpt-4.1-mini"

    # (1) 도구 정의
    def get_sleep_guide(month: str) -> dict:
        return {"status": "success", "guide": f"{month}개월 아기는 낮잠 변환기입니다. 깨시를 1시간 30분~2시간으로 잡으세요."}

    def get_feeding_guide(month: str) -> dict:
        return {"status": "success", "guide": f"{month}개월은 수유량이 줄어들 수 있습니다. 하루 총량 800ml 이상이면 괜찮습니다."}

    # (2) 에이전트 생성
    sleep_expert = Agent(
        name="sleep_expert",
        model=LiteLlm(model=MODEL_NAME),
        description="수면 전문",
        instruction="수면 문제(잠투정, 깨시, 통잠)에 대해 다정하게 조언하세요. 'get_sleep_guide' 도구를 사용하세요.",
        tools=[get_sleep_guide]
    )
    
    nutrition_expert = Agent(
        name="nutrition_expert",
        model=LiteLlm(model=MODEL_NAME),
        description="영양 전문",
        instruction="수유량과 이유식 문제에 대해 전문적으로 조언하세요. 'get_feeding_guide' 도구를 사용하세요.",
        tools=[get_feeding_guide]
    )

    head_nanny = Agent(
        name="head_nanny",
        model=LiteLlm(model=MODEL_NAME),
        sub_agents=[sleep_expert, nutrition_expert],
        description="메인 상담사",
        instruction="BabySquad 팀장입니다. 수면은 sleep_expert, 영양은 nutrition_expert에게 맡기세요. 인사는 직접 하세요."
    )
    
    # (3) 서비스 및 러너 생성
    session_service = InMemorySessionService()
    runner = Runner(agent=head_nanny, app_name="baby_squad_web", session_service=session_service)
    session_id = "web_session_001"
        
    return runner, session_service, session_id

runner, session_service, session_id = setup_agent_system()

# ---------------------------------------------------------
# 3. 채팅 UI 구현
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 육아 고민이 있으신가요? 수면이나 이유식, 무엇이든 물어봐주세요. 😊"}
    ]

for msg in st.session_state.messages:
    avatar = "🧑‍🍼" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🍼"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        with st.spinner("전문가들이 회의 중입니다..."):
            
            # [수정된 부분] try-except로 "이미 존재함" 에러 무시하기
            async def run_agent():
                APP_NAME = "baby_squad_web"
                USER_ID = "web_user"
                
                # 1. 세션 생성 시도 (이미 있으면 에러가 나므로 try-except로 감쌈)
                try:
                    # 세션 만들기를 시도합니다.
                    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                except Exception:
                    # 에러가 나면 "아, 이미 세션이 있구나" 하고 그냥 넘어갑니다.
                    pass 

                # 2. 실행
                content = types.Content(role='user', parts=[types.Part(text=prompt)])
                result_text = "답변을 생성하지 못했습니다."
                
                async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            result_text = event.content.parts[0].text
                        break
                return result_text

            # Streamlit 비동기 실행
            try:
                response_text = asyncio.run(run_agent())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response_text = loop.run_until_complete(run_agent())
                loop.close()

        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})