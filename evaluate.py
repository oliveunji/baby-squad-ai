# evaluate.py
import pandas as pd
from baseline import simple_rag_answer # 청코너
from graph_agent import get_graph_app   # 홍코너
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 심판 모델 (GPT-4)
judge_llm = ChatOpenAI(model="gpt-4", temperature=0)

# 2. 테스트 데이터셋 (검증하고 싶은 질문들)
test_questions = [
    "5개월 아기 이유식 스케줄 짜줘",           # 복합 질문
    "아기가 밤에 계속 깨는데 수면 교육 어떻게 해?", # 수면 전문 지식 필요
    "철분 부족하면 뭘 먹여야 해?",             # 영양 전문 지식 필요
    "돌 지난 아기 우유 얼마나 마셔?",           # 구체적 수치 필요
]

# 3. 채점 프롬프트 (가장 중요!)
judge_prompt = ChatPromptTemplate.from_template("""
너는 AI 답변 평가자다. 사용자의 질문에 대해 두 가지 답변(A, B)이 있다.
더 '전문적'이고, '구체적'이며, '도움이 되는' 답변을 선택하고 이유를 설명하라.

[질문]: {question}

[답변 A (Single Agent)]:
{answer_a}

[답변 B (Multi-Agent)]:
{answer_b}

평가 기준:
1. 정확성: 검색된 정보에 기반했는가?
2. 전문성: 전문가다운 어조와 깊이가 있는가?
3. 구조: 읽기 편하게 정리되었는가?

결과 형식:
- 승자: (A 또는 B)
- 이유: (한 줄 요약)
""")

# 4. 평가 실행 함수
def run_evaluation():
    results = []
    app = get_graph_app() # Multi-Agent
    
    config = {"configurable": {"thread_id": "eval_user"}}
    eval_chain = judge_prompt | judge_llm
    print("🚀 평가 시작...\n")
    
    for q in test_questions:
        print(f"Testing: {q}")
        
        # A: Baseline 실행
        try:
            ans_a = simple_rag_answer(q)
        except Exception as e:
            ans_a = f"Error: {e}"
        
        # B: Multi-Agent 실행
        response = app.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
        ans_b = response["messages"][-1].content
        
        # 심판 채점
        eval_result_msg = eval_chain.invoke({
            "question": q,
            "answer_a": ans_a,
            "answer_b": ans_b
        })

        eval_content = eval_result_msg.content # 결과 텍스트 추출
        print(f"   -> 심판 판정: {eval_content[:50]}...") # 로그 살짝 출력
        
        results.append({
            "Question": q,
            "Winner": "Multi-Agent" if "승자: B" in eval_content or "승자:B" in eval_content else "Single Agent",
            "Evaluation": eval_content
        })

    # 5. 결과 출력
    df = pd.DataFrame(results)
    print("\n📊 최종 평가 결과:")
    print(df[["Question", "Winner"]])
    
    # 엑셀로 저장 (증거 자료)
    df.to_excel("evaluation_results.xlsx")
    print("\n✅ 결과가 'evaluation_results.xlsx'로 저장되었습니다.")

if __name__ == "__main__":
    run_evaluation()