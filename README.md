
# 👶 BabySquad: AI Multi-Agent Parenting Assistant

> **"It takes a village to raise a child. BabySquad is your AI village."**

BabySquad는 0~12개월 영아 부모를 위해 설계된 **계층형 멀티 에이전트(Hierarchical Multi-Agent) 육아 상담 시스템**입니다.
단일 LLM의 일반적인 답변이 아닌, 각 분야(수면, 영양)에 특화된 전문가 에이전트들이 협업하여 상충될 수 있는 육아 고민에 대해 최적의 개인화 솔루션을 제공합니다.

## 🎯 Key Features

- **Multi-Agent Orchestration**: 사용자의 질문 의도를 분석하여 적절한 전문가(Sub-agent)에게 위임(Routing)하는 중앙 관리자(`Head Nanny`) 구조.
- **Specialized Experts**:
  - 💤 **Sleep Consultant**: 수면 퇴행, 적정 깨어있는 시간(Wake Window), 수면 교육 가이드 제공.
  - 🥦 **Nutritionist**: 월령별 수유량, 이유식 시작 시기 및 식단 가이드 제공.
- **Conflict Resolution**: 수면과 영양 문제가 복합적으로 얽힌 상황(예: 밤중 수유 vs 통잠)에서 종합적인 판단 수행.

## 🛠️ Architecture

이 프로젝트는 **Google Gemini 2.0 Flash**와 **GPT 4o-mini** 모델을 기반으로 하며, 에이전트 간의 통신 및 도구(Tools) 호출을 위해 Python 기반의 Agent Framework를 사용했습니다.

```mermaid
graph TD
    User((사용자)) -->|질문 입력| Streamlit[Streamlit Frontend]
    Streamlit -->|API 요청 (/chat)| FastAPI[FastAPI Server]
    
    subgraph "BabySquad Brain (LangGraph)"
        FastAPI --> Supervisor{관리자 에이전트}
        Supervisor -->|라우팅| Experts[전문가 에이전트<br>(영양/수면)]
        Experts -->|답변 초안 작성| Draft[초안 생성]
    end
    
    Draft -->|멈춤 (Interrupt)| Guardrail{🛡️ AI 안전 심판관}
    
    Guardrail -->|SAFE (안전)| AutoApprove[✅ 자동 승인]
    Guardrail -->|RISK (위험)| HumanReview[🚨 사람 검토 요청]
    
    HumanReview -->|응답: review_needed| Streamlit
    Streamlit -->|사용자/관리자| Button{승인 버튼 클릭}
    Button -->|API 요청 (/approve)| FastAPI
    
    AutoApprove -->|최종 답변| Final[🚀 답변 전송]
    FastAPI -->|재개 (Resume)| Final

```

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Google Gemini API Key
* OPEN AI API Key

### Installation

1. Repository 클론

```bash
git clone [https://github.com/oliveunji/baby-squad.git](https://github.com/oliveunji/baby-squad.git)
cd baby_squad_agent

```

2. 의존성 패키지 설치

```bash
pip install -r requirements.txt

```

3. 환경 변수 설정 (.env 파일 생성)

```text
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=False
```

4. 실행

```bash
python agent_team.py
```

## 📝 Usage Example

```text
User: "4개월 아기인데 낮잠을 너무 안 자요. 분유량이 부족해서 그런 걸까요?"

[System Log]
> Head Nanny analyzing intent...
> Detected topics: Sleep (낮잠), Nutrition (분유량)
> Routing to: Sleep Expert & Nutritionist

Head Nanny: "아기의 수면과 식사 문제로 고민이 많으시군요. 전문가들과 분석해본 결과..."

```

## 👨‍💻 Tech Stack

* **Language**: Python
* **LLM**: Google Gemini 2.0 Flash
* **Framework**: Google Gen AI SDK (ADK) / LiteLLM

---

*Developed by oliveunji as a personal AI upskilling project.*