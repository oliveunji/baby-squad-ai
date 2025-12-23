import os
from dotenv import load_dotenv
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ 에러: .env 파일을 찾을 수 없거나 GOOGLE_API_KEY가 없습니다.")
elif not os.getenv("OPENAI_API_KEY"):
    print("❌ 에러: .env 파일을 찾을 수 없거나 OPENAI_API_KE가 없습니다.")
else:
    print("✅ 환경 변수 로드 완료")

MODEL_GEMINI_2_0_FLASH = "gemini/gemini-2.0-flash"
MODEL_GPT_4O = "openai/gpt-4.1-mini"

# ---------------------------------------------------------
# [Step 1] 육아 전용 도구(Tools) 정의 (Mock Data)
# ---------------------------------------------------------

def get_sleep_guide(month: str) -> dict:
    """월령별 수면 가이드(적정 깨시, 낮잠 횟수)를 제공합니다."""
    print(f"--- 💤 Tool Call: get_sleep_guide(month={month}) ---")
    
    # 더미 데이터 (나중에는 RAG나 DB로 교체)
    guide_db = {
        "4": "4개월 아기의 적정 '깨어있는 시간(Wake Window)'은 1시간 30분 ~ 2시간입니다. 낮잠은 3~4회 변환기입니다.",
        "5": "5개월 아기는 활동량이 늘어납니다. 깨시 2시간 ~ 2시간 30분을 목표로 하세요.",
    }
    
    # 간단한 키워드 매칭
    for key, value in guide_db.items():
        if key in month:
            return {"status": "success", "guide": value}
            
    return {"status": "general", "guide": "해당 월령의 구체적인 데이터가 없습니다. 일반적인 수면 패턴을 안내합니다."}

def get_feeding_guide(month: str) -> dict:
    """월령별 수유 및 이유식 가이드를 제공합니다."""
    print(f"--- 🍼 Tool Call: get_feeding_guide(month={month}) ---")
    
    if "4" in month:
        return {"status": "success", "guide": "4개월은 수유량 800~1000ml 유지 시기입니다. 이유식은 아직 이르거나, 쌀미음 정도만 시도해볼 수 있습니다."}
    elif "6" in month:
        return {"status": "success", "guide": "6개월은 이유식(소고기 포함)을 시작해야 하는 필수 시기입니다. 철분 섭취가 중요합니다."}
        
    return {"status": "general", "guide": "아기의 몸무게와 발달 상황에 따라 수유량을 조절하세요."}

# ---------------------------------------------------------
# [Step 2] 서브 에이전트(Sub-Agents) 정의
# ---------------------------------------------------------

# 1. 수면 전문가 (Sleep Expert)
sleep_expert = Agent(
    name="sleep_expert",
    model=LiteLlm(model=MODEL_GEMINI_2_0_FLASH), # 모델명 주의!
    description="수면 문제(잠투정, 수면교육, 깨시)를 전담하는 전문가입니다.",
    instruction="""
    당신은 따뜻하고 전문적인 '수면 컨설턴트'입니다.
    사용자가 수면 문제(안 자요, 자주 깨요, 낮잠 등)를 물어보면 'get_sleep_guide' 도구를 사용해 조언해주세요.
    말투는 육아에 지친 부모를 위로하는 부드러운 말투(~해요, ~랍니다)를 사용하세요.
    """,
    tools=[get_sleep_guide],
)

# 2. 영양 전문가 (Nutrition Expert)
nutrition_expert = Agent(
    name="nutrition_expert",
    model=LiteLlm(model=MODEL_GEMINI_2_0_FLASH),
    description="수유, 분유량, 이유식 관련 질문을 전담하는 전문가입니다.",
    instruction="""
    당신은 꼼꼼한 '영양사'입니다.
    사용자가 먹는 문제(분유량, 이유식 시기, 식단)를 물어보면 'get_feeding_guide' 도구를 사용해 조언해주세요.
    과학적인 근거(WHO, AAP 가이드라인)를 기반으로 설명하는 듯한 전문적인 톤을 유지하세요.
    """,
    tools=[get_feeding_guide],
)

print("✅ 전문가 에이전트 생성 완료!")

# ---------------------------------------------------------
# [Step 3] 루트 에이전트 (Head Nanny) 정의
# ---------------------------------------------------------

head_nanny = Agent(
    name="head_nanny",
    model=LiteLlm(model=MODEL_GPT_4O),
    # 서브 에이전트 등록
    sub_agents=[sleep_expert, nutrition_expert], 
    description="사용자의 질문을 분석하여 적절한 육아 전문가(수면/영양)를 호출하는 메인 관리자입니다.",
    instruction="""
    당신은 'BabySquad' 팀의 리더인 '헤드 내니(Head Nanny)'입니다.
    당신의 임무는 부모님의 고민을 듣고, 우리 팀의 전문가에게 연결해주는 것입니다.

    [판단 기준]
    1. '잠', '낮잠', '새벽', '통잠', '깨시' 관련 질문 -> 'sleep_expert'에게 위임하세요.
    2. '분유', '모유', '이유식', '먹다', '수유' 관련 질문 -> 'nutrition_expert'에게 위임하세요.
    3. 인사는 직접 받아주고, 그 외 복합적인 질문은 전문가들의 의견을 종합해서 답변하세요.
    
    사용자에게는 항상 든든하고 친절한 파트너처럼 대화하세요.
    """
)

print(f"✅ 팀장 에이전트({head_nanny.name}) 생성 완료!")

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")

async def main():
    root_agent_var_name = 'head_nanny'
    
    print("\n--- Testing Agent Team Delegation ---")
    session_service = InMemorySessionService()
    APP_NAME = "baby_squad_agent_team"
    USER_ID = "user_1_agent_team"
    SESSION_ID = "session_001_agent_team"
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    actual_root_agent = globals()[root_agent_var_name]
    runner_agent_team = Runner( # Or use InMemoryRunner
        agent=actual_root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    print(f"Runner created for agent '{actual_root_agent.name}'.")

    # 1. 인사
    await call_agent_async(query="안녕? 너네 팀은 뭘 해줄 수 있어?", 
                        runner=runner_agent_team,
                        user_id=USER_ID,
                        session_id=SESSION_ID)
    
    # 2. 수면 질문 (Sleep Expert 호출 확인)
    await call_agent_async(query="우리 아기가 4개월인데 낮잠을 안 자려고 해. 깨시가 얼마나 돼야 해?",
                        runner=runner_agent_team,
                        user_id=USER_ID,
                        session_id=SESSION_ID)
    
    # 3. 영양 질문 (Nutrition Expert 호출 확인)
    await call_agent_async(query="지금 4개월인데 이유식 시작해도 될까? 분유는 800 먹어.", 
                        runner=runner_agent_team,
                        user_id=USER_ID,
                        session_id=SESSION_ID)        

if __name__ == "__main__":
    asyncio.run(main())