# evaluate.py
import pandas as pd
from baseline import simple_rag_answer
from graph_agent import get_graph_app
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ✨ 개선된 테스트 데이터셋
test_questions = [
    # 복합 질문 (Multi-Agent 강점)
    {
        "q": "7개월 아기가 밤에 자주 깨는데 이유식이랑 관련 있을까? 둘 다 조언해줘",
        "type": "복합",
        "expected_winner": "Multi-Agent"
    },
    {
        "q": "돌잔치 준비중인데 수면 루틴 망가지지 않으면서 영양 챙기는 방법 알려줘",
        "type": "복합",
        "expected_winner": "Multi-Agent"
    },
    
    # 전문 지식 필요 (전문가별 깊이 테스트)
    {
        "q": "철분 보충제 먹이면서 수면 패턴 바뀔 수 있어?",
        "type": "전문",
        "expected_winner": "Multi-Agent"
    },
    {
        "q": "통잠 자려면 저녁 이유식 양을 늘려야 해?",
        "type": "전문",
        "expected_winner": "Multi-Agent"
    },
    
    # 단순 질문 (공정성 테스트)
    {
        "q": "5개월 아기 이유식 언제 시작해?",
        "type": "단순",
        "expected_winner": "Either"
    },
    {
        "q": "신생아 평균 수면 시간은?",
        "type": "단순",
        "expected_winner": "Either"
    },
]

# ✨ 개선된 평가 프롬프트
judge_prompt = ChatPromptTemplate.from_template("""
당신은 육아 AI 시스템 평가 전문가입니다.

**평가 맥락**: 
- 답변 A는 일반 AI (단일 에이전트)
- 답변 B는 전문가 팀 AI (영양/수면 전문가로 구성된 멀티 에이전트)

**사용자 질문**: {question}

**답변 A (Single Agent)**:
{answer_a}

**답변 B (Multi-Agent)**:
{answer_b}

**평가 기준** (각 5점 만점):

1. **정확성** (5점): 사실 기반 정확한 정보 제공
   - 잘못된 정보나 과장 없음
   - 연령별 적절한 조언

2. **전문성 깊이** (5점): 전문가 수준의 인사이트
   - 일반론이 아닌 구체적 메커니즘 설명
   - 전문 용어의 적절한 사용
   - "왜 그런지" 원리 설명

3. **실행 가능성** (5점): 부모가 바로 적용 가능한가
   - 구체적 수치/시간/방법 제시
   - 단계별 가이드 제공

4. **복합 질문 처리** (5점): (해당되는 경우만)
   - 영양+수면 등 여러 측면 통합 답변
   - 도메인 간 연결 설명
   - 우선순위 제시

5. **간결성** (5점): 불필요한 장황함 없이 핵심 전달
   - 2-4문장으로 핵심 답변
   - 부가 설명은 필요시만

**출력 형식** (반드시 준수):
```
승자: A 또는 B
점수: A=00/25, B=00/25
이유: [1-2문장으로 핵심 차이점]
세부:
- 정확성: A=0, B=0
- 전문성: A=0, B=0  
- 실행성: A=0, B=0
- 복합처리: A=0, B=0
- 간결성: A=0, B=0
```

**중요**: 길다고 좋은 것이 아닙니다. 짧아도 핵심을 정확히 전달하면 높은 점수입니다.
""")

def run_evaluation():
    results = []
    app = get_graph_app()
    config = {"configurable": {"thread_id": "eval_user"}}
    eval_chain = judge_prompt | judge_llm
    
    print("🚀 평가 시작...\n")
    print("=" * 80)
    
    for item in test_questions:
        q = item["q"]
        q_type = item["type"]
        
        print(f"\n{'#'*80}")
        print(f"📝 질문 유형: [{q_type}]")
        print(f"질문: {q}")
        print(f"{'#'*80}\n")
        
        # A: Single Agent
        print("🔵 Single Agent 실행 중...")
        try:
            ans_a = simple_rag_answer(q)
            print(f"   답변 길이: {len(ans_a)} chars")
            print(f"   미리보기: {ans_a[:100]}...")
        except Exception as e:
            ans_a = f"Error: {e}"
            print(f"   ⚠️ 오류: {e}")
        
        # B: Multi-Agent
        print(f"\n🔴 Multi-Agent 실행 중...")
        try:
            response = app.invoke(
                {"messages": [{"role": "user", "content": q}]}, 
                config=config
            )
            ans_b = response["messages"][-1].content
            print(f"   답변 길이: {len(ans_b)} chars")
            print(f"   미리보기: {ans_b[:100]}...")
        except Exception as e:
            ans_b = f"Error: {e}"
            print(f"   ⚠️ 오류: {e}")
        
        # 심판 평가
        print(f"\n⚖️ 심판 평가 중...")
        eval_result_msg = eval_chain.invoke({
            "question": q,
            "answer_a": ans_a,
            "answer_b": ans_b
        })
        
        eval_content = eval_result_msg.content
        
        # 승자 파싱
        if "승자: B" in eval_content or "승자:B" in eval_content:
            winner = "Multi-Agent"
        elif "승자: A" in eval_content or "승자:A" in eval_content:
            winner = "Single Agent"
        else:
            winner = "Unknown"
        
        print(f"\n{'='*80}")
        print(f"🏆 판정: {winner}")
        print(f"{'='*80}")
        print(eval_content)
        print(f"{'='*80}\n")
        
        results.append({
            "Question": q,
            "Type": q_type,
            "Expected": item["expected_winner"],
            "Winner": winner,
            "Answer_A_Length": len(ans_a),
            "Answer_B_Length": len(ans_b),
            "Evaluation": eval_content,
            "Answer_A": ans_a,
            "Answer_B": ans_b
        })
        
        # 각 질문 후 구분선
        print("\n" + "█"*80 + "\n")
    
    # 결과 분석
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("📊 최종 평가 결과:")
    print("=" * 80)
    print(df[["Type", "Winner", "Expected"]])
    
    # 승률 계산
    multi_wins = len(df[df["Winner"] == "Multi-Agent"])
    single_wins = len(df[df["Winner"] == "Single Agent"])
    
    print(f"\n🎯 승률:")
    print(f"   Multi-Agent: {multi_wins}/{len(df)} ({multi_wins/len(df)*100:.1f}%)")
    print(f"   Single Agent: {single_wins}/{len(df)} ({single_wins/len(df)*100:.1f}%)")
    
    # 질문 유형별 승률
    print(f"\n📈 유형별 결과:")
    for qtype in df["Type"].unique():
        subset = df[df["Type"] == qtype]
        multi_type_wins = len(subset[subset["Winner"] == "Multi-Agent"])
        print(f"   {qtype}: Multi-Agent {multi_type_wins}/{len(subset)}승")
    
    # 엑셀 저장
    df.to_excel("evaluation_results_v2.xlsx", index=False)
    print("\n✅ 상세 결과가 'evaluation_results_v2.xlsx'로 저장되었습니다.")
    
    return df

if __name__ == "__main__":
    df = run_evaluation()