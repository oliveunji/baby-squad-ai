# batch_evaluate_v2.py
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from baseline import simple_rag_answer
from graph_agent_scalable import get_graph_app  # 새 확장 가능한 구조
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================================================================================
# 설정
# ================================================================================
class Config:
    # 평가 설정
    MAX_WORKERS = 3
    BATCH_SIZE = 10
    TIMEOUT = 60
    
    # 모델 설정
    JUDGE_MODEL = "gpt-4o"
    JUDGE_TEMP = 0
    
    # 로깅 설정
    LOG_LEVEL = logging.WARNING
    SAVE_DIR = Path("evaluation_results")
    
    # 재시도 설정
    MAX_RETRIES = 2
    RETRY_DELAY = 3

# ================================================================================
# 로깅 설정
# ================================================================================
Config.SAVE_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================================================================
# 테스트 데이터셋 생성 (3개 전문가 포함)
# ================================================================================
def generate_test_dataset(size: int = 100) -> list:
    """
    영양, 수면, 놀이 3개 전문가를 고려한 테스트 질문 생성
    """
    
    templates = {
        # 복합 질문 (2개 이상 도메인)
        "복합_영양수면": [
            "{month}개월 아기가 밤에 자주 깨는데 이유식이랑 관련 있을까?",
            "통잠 자려면 저녁 이유식 양을 늘려야 해?",
            "철분 보충제 먹이면서 수면 패턴 바뀔 수 있어?",
            "낮잠 줄이면 밤에 더 잘 먹을까?",
            "밤중 수유 끊으면 영양 부족할까?",
            "저녁 이유식 시간을 늦추면 아침에 일찍 안 깨?",
        ],
        "복합_영양놀이": [
            "이유식 거부하는 아기 놀이로 관심 돌릴 수 있어?",
            "식사 시간에 놀이 활동 같이 하면 좋을까?",
            "편식 심한데 놀이로 음식 관심 유도하는 방법은?",
            "음식 탐색 놀이 어떻게 해?",
        ],
        "복합_수면놀이": [
            "낮에 활동량 늘리면 밤에 더 잘 자?",
            "자기 전 놀이 시간은 언제가 적당해?",
            "낮잠 안 자려고 하는데 놀이로 에너지 소진시키면 돼?",
            "흥분시키는 놀이는 자기 전 몇 시간 전에 끝내야 해?",
        ],
        "복합_3개도메인": [
            "돌잔치 준비중인데 수면 루틴 망가지지 않으면서 영양 챙기고 놀이도 하는 방법",
            "{month}개월 아기 하루 일과표 짜줘 (식사, 놀이, 수면 포함)",
            "어린이집 가기 전 준비 어떻게 해? (식습관, 놀이 적응, 수면 패턴)",
        ],
        
        # 단일 도메인
        "영양": [
            "{month}개월 아기 이유식 스케줄 짜줘",
            "철분 부족하면 뭘 먹여야 해?",
            "돌 지난 아기 우유 얼마나 마셔?",
            "계란 알레르기 있는데 단백질 어떻게 보충해?",
            "이유식 거부하는데 어떻게 해?",
            "편식 심한 아기 영양 보충 방법은?",
            "분유 먹는 양이 줄었는데 괜찮아?",
        ],
        "수면": [
            "아기가 밤에 계속 깨는데 수면 교육 어떻게 해?",
            "낮잠 언제 줄여야 해?",
            "신생아 평균 수면 시간은?",
            "{month}개월 아기 적정 수면 시간은?",
            "밤중 수유 언제 끊어야 해?",
            "수면 퇴행 어떻게 대처해?",
            "통잠은 언제부터 자?",
        ],
        "놀이": [
            "{month}개월 아기 발달 놀이 추천해줘",
            "집에서 할 수 있는 오감 놀이는?",
            "그림책 언제부터 읽어주면 좋아?",
            "장난감 없이 놀아주는 방법 알려줘",
            "혼자 노는 시간 어떻게 늘려?",
            "돌 아기 놀이 추천",
            "바깥 활동 하루에 얼마나 해야 해?",
        ],
        
        # 단순 질문
        "단순": [
            "이유식 언제 시작해?",
            "통잠은 언제부터 자?",
            "분유 얼마나 먹여야 해?",
            "몇 개월부터 걸어?",
            "치아는 언제 나와?",
        ]
    }
    
    questions = []
    months = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 24, 30, 36]
    
    # 비율 설정
    # 복합 50% (영양수면 20%, 영양놀이 10%, 수면놀이 10%, 3개도메인 10%)
    # 단일 40% (영양 15%, 수면 15%, 놀이 10%)
    # 단순 10%
    
    target_counts = {
        "복합_영양수면": int(size * 0.20),
        "복합_영양놀이": int(size * 0.10),
        "복합_수면놀이": int(size * 0.10),
        "복합_3개도메인": int(size * 0.10),
        "영양": int(size * 0.15),
        "수면": int(size * 0.15),
        "놀이": int(size * 0.10),
        "단순": int(size * 0.10),
    }
    
    for q_type, count in target_counts.items():
        for i in range(count):
            template = templates[q_type][i % len(templates[q_type])]
            month = months[i % len(months)]
            
            question = template.format(month=month)
            
            # 질문 유형 분류
            if "복합" in q_type:
                category = "복합"
                if "3개도메인" in q_type:
                    sub_category = "영양+수면+놀이"
                elif "영양수면" in q_type:
                    sub_category = "영양+수면"
                elif "영양놀이" in q_type:
                    sub_category = "영양+놀이"
                elif "수면놀이" in q_type:
                    sub_category = "수면+놀이"
            else:
                category = q_type
                sub_category = q_type
            
            questions.append({
                "id": len(questions) + 1,
                "q": question,
                "type": category,
                "sub_type": sub_category,
                "expected_winner": "Multi-Agent" if category == "복합" else "Either"
            })
    
    return questions

# ================================================================================
# 평가 함수
# ================================================================================
judge_llm = ChatOpenAI(model=Config.JUDGE_MODEL, temperature=Config.JUDGE_TEMP)

judge_prompt = ChatPromptTemplate.from_template("""
당신은 육아 AI 시스템 평가 전문가입니다.

**평가 맥락**: 
- 답변 A는 일반 AI (단일 에이전트)
- 답변 B는 전문가 팀 AI (영양/수면/놀이 전문가로 구성된 멀티 에이전트)

**사용자 질문**: {question}

**답변 A (Single Agent)**:
{answer_a}

**답변 B (Multi-Agent)**:
{answer_b}

**평가 기준** (각 5점 만점):

1. **정확성** (5점): 사실 기반 정확한 정보 제공
2. **전문성 깊이** (5점): 전문가 수준의 인사이트
3. **실행 가능성** (5점): 부모가 바로 적용 가능한가
4. **복합 질문 처리** (5점): 여러 측면 통합 답변 (해당 시에만)
5. **간결성** (5점): 핵심만 전달 (길다고 좋은 것 아님)

**특별 고려사항**:
- 질문이 여러 영역(영양+수면, 영양+놀이, 수면+놀이 등)에 걸쳐있다면, 복합 질문 처리 능력을 중요하게 평가
- 단일 영역 질문이면 복합처리 점수는 N/A로 표시하고 다른 4개 항목만 평가

**출력 형식** (반드시 준수):
```
승자: A 또는 B
점수: A=00/25, B=00/25
이유: [1-2문장]
세부:
- 정확성: A=0, B=0
- 전문성: A=0, B=0  
- 실행성: A=0, B=0
- 복합처리: A=0, B=0 (또는 N/A)
- 간결성: A=0, B=0
```
""")

def evaluate_single_question(item: dict, app, config: dict) -> dict:
    """단일 질문 평가"""
    question = item["q"]
    q_id = item["id"]
    
    result = {
        "id": q_id,
        "question": question,
        "type": item["type"],
        "sub_type": item["sub_type"],
        "expected": item["expected_winner"],
        "status": "success",
        "error": None,
        "answer_a": None,
        "answer_b": None,
        "winner": None,
        "evaluation": None,
        "time_taken": 0,
    }
    
    start_time = time.time()
    
    try:
        # A: Single Agent (재시도 로직)
        for attempt in range(Config.MAX_RETRIES + 1):
            try:
                result["answer_a"] = simple_rag_answer(question)
                break
            except Exception as e:
                if attempt == Config.MAX_RETRIES:
                    raise
                logger.warning(f"Q{q_id} Single Agent 재시도 {attempt+1}/{Config.MAX_RETRIES}")
                time.sleep(Config.RETRY_DELAY)
        
        # B: Multi-Agent (재시도 로직)
        for attempt in range(Config.MAX_RETRIES + 1):
            try:
                response = app.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=config
                )
                result["answer_b"] = response["messages"][-1].content
                break
            except Exception as e:
                if attempt == Config.MAX_RETRIES:
                    raise
                logger.warning(f"Q{q_id} Multi-Agent 재시도 {attempt+1}/{Config.MAX_RETRIES}")
                time.sleep(Config.RETRY_DELAY)
        
        # 심판 평가
        eval_chain = judge_prompt | judge_llm
        eval_result = eval_chain.invoke({
            "question": question,
            "answer_a": result["answer_a"],
            "answer_b": result["answer_b"]
        })
        
        eval_content = eval_result.content
        result["evaluation"] = eval_content
        
        # 승자 파싱
        if "승자: B" in eval_content or "승자:B" in eval_content:
            result["winner"] = "Multi-Agent"
        elif "승자: A" in eval_content or "승자:A" in eval_content:
            result["winner"] = "Single Agent"
        else:
            result["winner"] = "Unknown"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["winner"] = "Error"
        logger.error(f"Q{q_id} 평가 실패: {e}")
    
    result["time_taken"] = time.time() - start_time
    return result

# ================================================================================
# 배치 평가 메인 함수
# ================================================================================
def run_batch_evaluation(num_questions: int = 100):
    """배치 평가 실행"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Config.SAVE_DIR / f"batch_results_{timestamp}.xlsx"
    checkpoint_file = Config.SAVE_DIR / f"checkpoint_{timestamp}.json"
    
    print("="*80)
    print(f"🚀 배치 평가 시작 (3개 전문가: 영양, 수면, 놀이)")
    print(f"   - 평가 질문 수: {num_questions}")
    print(f"   - 동시 실행: {Config.MAX_WORKERS}")
    print(f"   - 결과 저장: {results_file}")
    print("="*80 + "\n")
    
    # 테스트 데이터 생성
    print("📝 테스트 데이터 생성 중...")
    test_questions = generate_test_dataset(num_questions)
    print(f"   ✅ {len(test_questions)}개 질문 생성 완료\n")
    
    # 질문 유형 분포 출력
    type_counts = pd.Series([q["type"] for q in test_questions]).value_counts()
    sub_type_counts = pd.Series([q["sub_type"] for q in test_questions]).value_counts()
    
    print("📊 질문 유형 분포:")
    for q_type, count in type_counts.items():
        print(f"   - {q_type}: {count}개 ({count/len(test_questions)*100:.1f}%)")
    
    print("\n📊 세부 유형 분포:")
    for q_type, count in sub_type_counts.items():
        print(f"   - {q_type}: {count}개 ({count/len(test_questions)*100:.1f}%)")
    print()
    
    # Multi-Agent 앱 초기화
    app = get_graph_app()
    
    # 병렬 평가 실행
    results = []
    failed_count = 0
    
    print("⚙️ 평가 진행 중...\n")
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                evaluate_single_question,
                item,
                app,
                {"configurable": {"thread_id": f"eval_{item['id']}"}}
            ): item for item in test_questions
        }
        
        with tqdm(total=len(test_questions), desc="평가 진행") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result["status"] == "error":
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "성공": len(results) - failed_count,
                    "실패": failed_count
                })
                
                # 중간 저장
                if len(results) % Config.BATCH_SIZE == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 평가 완료!\n")
    
    # 결과 분석
    df = pd.DataFrame(results)
    df = df.sort_values('id').reset_index(drop=True)
    
    # 통계 계산
    print("="*80)
    print("📊 최종 평가 결과")
    print("="*80)
    
    success_df = df[df["status"] == "success"]
    if len(success_df) > 0:
        multi_wins = len(success_df[success_df["winner"] == "Multi-Agent"])
        single_wins = len(success_df[success_df["winner"] == "Single Agent"])
        unknown = len(success_df[success_df["winner"] == "Unknown"])
        
        print(f"\n🎯 전체 승률 (성공한 {len(success_df)}건 기준):")
        print(f"   - Multi-Agent:  {multi_wins:3d}승 ({multi_wins/len(success_df)*100:5.1f}%)")
        print(f"   - Single Agent: {single_wins:3d}승 ({single_wins/len(success_df)*100:5.1f}%)")
        print(f"   - Unknown:      {unknown:3d}건 ({unknown/len(success_df)*100:5.1f}%)")
        
        # 대분류별 승률
        print(f"\n📈 질문 유형별 결과:")
        for q_type in success_df["type"].unique():
            subset = success_df[success_df["type"] == q_type]
            multi_type_wins = len(subset[subset["winner"] == "Multi-Agent"])
            total = len(subset)
            print(f"   - {q_type:6s}: Multi-Agent {multi_type_wins:2d}/{total:2d}승 ({multi_type_wins/total*100:5.1f}%)")
        
        # 세부 유형별 승률 (복합 질문만)
        complex_df = success_df[success_df["type"] == "복합"]
        if len(complex_df) > 0:
            print(f"\n🔍 복합 질문 세부 분석:")
            for sub_type in complex_df["sub_type"].unique():
                subset = complex_df[complex_df["sub_type"] == sub_type]
                multi_sub_wins = len(subset[subset["winner"] == "Multi-Agent"])
                total = len(subset)
                print(f"   - {sub_type:15s}: Multi-Agent {multi_sub_wins:2d}/{total:2d}승 ({multi_sub_wins/total*100:5.1f}%)")
        
        # 평균 소요 시간
        avg_time = success_df["time_taken"].mean()
        print(f"\n⏱️ 평균 응답 시간: {avg_time:.2f}초")
        
    if failed_count > 0:
        print(f"\n⚠️ 실패한 질문: {failed_count}건 ({failed_count/len(df)*100:.1f}%)")
    
    print("="*80 + "\n")
    
    # 엑셀 저장 (여러 시트)
    with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='전체결과', index=False)
        
        if len(success_df) > 0:
            success_df.to_excel(writer, sheet_name='성공한평가', index=False)
            
            # 복합 질문만
            complex_results = success_df[success_df["type"] == "복합"]
            if len(complex_results) > 0:
                complex_results.to_excel(writer, sheet_name='복합질문', index=False)
            
            # 단일 도메인만
            single_domain = success_df[success_df["type"].isin(["영양", "수면", "놀이"])]
            if len(single_domain) > 0:
                single_domain.to_excel(writer, sheet_name='단일도메인', index=False)
        
        failed_df = df[df["status"] == "error"]
        if len(failed_df) > 0:
            failed_df.to_excel(writer, sheet_name='실패한평가', index=False)
        
        # 통계 요약
        summary_data = {
            "항목": [
                "전체 질문 수", "성공", "실패", 
                "Multi-Agent 승", "Single Agent 승", "Unknown",
                "복합 질문 수", "복합 질문 Multi 승률(%)",
                "평균 소요시간(초)"
            ],
            "값": [
                len(df),
                len(success_df),
                failed_count,
                multi_wins if len(success_df) > 0 else 0,
                single_wins if len(success_df) > 0 else 0,
                unknown if len(success_df) > 0 else 0,
                len(complex_df) if len(success_df) > 0 else 0,
                f"{len(complex_df[complex_df['winner']=='Multi-Agent'])/len(complex_df)*100:.1f}" if len(complex_df) > 0 else "N/A",
                f"{avg_time:.2f}" if len(success_df) > 0 else "N/A"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='요약', index=False)
    
    print(f"💾 결과 저장 완료: {results_file}")
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return df

# ================================================================================
# 메인 실행
# ================================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='배치 평가 실행 (3개 전문가)')
    parser.add_argument('-n', '--num', type=int, default=100, 
                        help='평가할 질문 수 (기본값: 100)')
    parser.add_argument('-w', '--workers', type=int, default=3,
                        help='동시 실행 개수 (기본값: 3)')
    
    args = parser.parse_args()
    
    Config.MAX_WORKERS = args.workers
    
    df = run_batch_evaluation(num_questions=args.num)
    
    print("\n✅ 모든 작업 완료!")