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

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
# [2] 벡터 DB 로드 (Global State에 저장)
# ---------------------------------------------------------
if "vector_store" not in st.session_state:
    if os.getenv("GOOGLE_API_KEY"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        if os.path.exists("./chroma_db"):
            st.session_state.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name="baby_knowledge"
            )
            print("✅ Vector DB Loaded into Session State")
        else:
            st.error("⚠️ 'chroma_db' 폴더가 없습니다. ingest.py를 먼저 실행하세요.")
            st.session_state.vector_store = None
    else:
        st.error("API Key가 없습니다.")
        st.session_state.vector_store = None

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

    # ★★★ [핵심] RAG 검색 도구 정의 ★★★
    def search_knowledge_base(query: str) -> str:
        """
        육아 가이드 문서(수면, 영양 등)에서 질문과 관련된 내용을 검색합니다.
        질문에 대한 구체적인 해결책이나 수치를 찾을 때 사용하세요.
        """
        # 전역 세션 상태에서 가져옴 (스코프 에러 방지)
        db = st.session_state.get("vector_store")
        
        if db is None:
            return "죄송합니다. 지식 데이터베이스가 연결되지 않았습니다."
            
        print(f"🔍 검색 수행: {query}")
        try:
            # 유사도 검색 (상위 3개 문서 추출)
            results = db.similarity_search(query, k=3)
            
            # 검색된 내용을 하나의 텍스트로 합침
            context_text = "\n\n".join([f"- {doc.page_content}" for doc in results])
            return f"[검색된 가이드라인]\n{context_text}"
        except Exception as e:
            return f"검색 중 오류 발생: {str(e)}"

    # (1) 수면 전문가 (이제 검색 도구를 사용함)
    sleep_expert = Agent(
        name="sleep_expert",
        model=LiteLlm(model=MODEL_NAME),
        description="수면 전문",
        instruction="""
        당신은 수면 컨설턴트입니다.
        사용자의 질문이 들어오면 반드시 'search_knowledge_base' 도구를 사용해 가이드라인을 검색하세요.
        검색된 내용(깨시, 수면의식 등)을 기반으로 따뜻하게 조언해주세요.
        """,
        tools=[search_knowledge_base] # 검색 도구 장착!
    )
    
    # (2) 영양 전문가
    nutrition_expert = Agent(
        name="nutrition_expert",
        model=LiteLlm(model=MODEL_NAME),
        description="영양 전문",
        instruction="""
        당신은 영양사입니다.
        이유식이나 수유량 질문이 오면 'search_knowledge_base' 도구를 사용해 정확한 수치를 찾으세요.
        검색된 근거(철분, 용량 등)를 인용하여 전문적으로 답변하세요.
        """,
        tools=[search_knowledge_base]
    )

    # (3) 헤드 내니
    head_nanny = Agent(
        name="head_nanny",
        model=LiteLlm(model=MODEL_NAME),
        sub_agents=[sleep_expert, nutrition_expert],
        description="메인 상담사",
        instruction="BabySquad 팀장입니다. 수면/영양 전문가를 적절히 호출하고, 인사는 직접 하세요."
    )
    
    # (3) 서비스 및 러너 생성
    session_service = InMemorySessionService()
    runner = Runner(agent=head_nanny, app_name="baby_squad_web", session_service=session_service)
    session_id = "rag_session_001"
        
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
        
        # with st.spinner("전문가들이 회의 중입니다..."):
        with st.spinner("가이드라인 검색 및 분석 중..."):
            # [수정된 부분] try-except로 "이미 존재함" 에러 무시하기
            async def run_agent():
                APP_NAME = "baby_squad_web"
                USER_ID = "rag_user"
                
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