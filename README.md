# 👶 BabySquad: Production Multi-Agent RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://github.com/langchain-ai/langgraph)

> **"It takes a village to raise a child. BabySquad is your AI village."**

A production-grade multi-agent parenting consultation system achieving **82.8% accuracy** using specialized expert agents, dynamic routing, and rigorous evaluation methodology.

**📊 Performance**: 38% (baseline) → 82.8% (multi-agent) | 100% accuracy on single-domain questions

**🔗 Links**: [Blog Post](link) | [Demo Video](link) | [Technical Deep-dive](link)

---

## 🎯 Key Results

- ✅ **82.8% overall accuracy** with multi-agent architecture (vs 38% baseline)
- ✅ **100% accuracy** on single-domain questions (nutrition, sleep, play)
- ✅ **LLM-as-judge evaluation** framework processing 100+ test cases
- ✅ **Human-in-the-loop** safety system for medical advice
- ⚠️ **0% on 3-domain questions** - synthesis remains a challenge (see Learnings)

---

## 🏗️ Architecture

### System Overview

```
User Question
    ↓
Complexity Router ────► Simple ────► Direct Answer (fast path)
    ↓
    Complex
    ↓
Orchestrator (selects 1-2 experts dynamically)
    ↓
Expert Pool: [🍎 Nutrition] [😴 Sleep] [🎨 Play]
    ↓
Synthesizer (combines expert answers)
    ↓
Risk Analyzer (LLM-based safety check)
    ↓
    ├─ SAFE ────► Auto-approve
    └─ RISK ────► Human Review
```

### Key Design Patterns

**1. Dynamic Expert Pool**
- Experts registered in a central registry
- Orchestrator selects based on question analysis
- Easy to add new experts without graph changes

**2. Hybrid Routing**
- Simple questions bypass multi-agent complexity
- Complex questions get full expert treatment
- Optimizes for both speed and quality

**3. Smart Synthesis**
- Works well with 2 experts (83% accuracy)
- Struggles with 3+ experts (architectural limitation)
- See [Learnings](#-what-didnt-work) for details

---

## 🛠️ Tech Stack

**Core**:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [OpenAI GPT-4o](https://openai.com) - Expert agents
- [Pinecone](https://www.pinecone.io/) - Vector database for RAG
- [FastAPI](https://fastapi.tiangolo.com/) - Backend API
- [Streamlit](https://streamlit.io/) - Frontend UI

**Evaluation**:
- Custom LLM-as-judge framework
- Automated batch evaluation (100+ cases)
- Metrics: Accuracy, Expert Depth, Actionability, Multi-domain Handling, Conciseness

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API Key
- Pinecone API Key (optional, for RAG)
- Google API Key (for embeddings)

### Installation

```bash
# Clone repository
git clone https://github.com/oliveunji/baby-squad-ai.git
cd baby-squad-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration (.env)

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
PINECONE_API_KEY=your_pinecone_key_here    # Optional
GOOGLE_API_KEY=your_google_key_here        # For embeddings
```

### Run

```bash
# Start backend
python backend_api.py

# In another terminal, start frontend
streamlit run streamlit_app.py

# Run evaluation (optional)
python batch_evaluate_v2.py -n 30
```

---

## 📊 Evaluation

### LLM-as-Judge Framework

Our evaluation system uses GPT-4o as an impartial judge:

```python
Criteria (25 points total):
- Accuracy (5pts): Factually correct information
- Expert Depth (5pts): Professional insights, not generic advice
- Actionability (5pts): Parents can immediately apply
- Multi-domain (5pts): Integrates perspectives (when applicable)
- Conciseness (5pts): No unnecessary verbosity
```

### Test Dataset

100 diverse questions across:
- **Complex (50%)**: Multi-domain (nutrition+sleep, nutrition+play, etc.)
- **Single domain (40%)**: Nutrition, sleep, or play only
- **Simple (10%)**: Single-fact questions

### Running Evaluation

```bash
# Quick test (30 questions)
python batch_evaluate_v2.py -n 30

# Full evaluation (100 questions)
python batch_evaluate_v2.py -n 100

# Results saved to: evaluation_results/batch_results_[timestamp].xlsx
```

---

## 💡 Key Learnings

### What Worked ✅

1. **Expert Prompt Depth > Brevity**
   - Brief prompts: 38% accuracy
   - Detailed expert prompts: 82.8% accuracy
   - Lesson: Expert agents need domain principles, not generic instructions

2. **Rigorous Evaluation First**
   - Built LLM-as-judge before optimizing
   - Enabled rapid iteration (100 tests in 30 minutes)
   - Caught regressions immediately

3. **Hybrid Routing**
   - Simple questions get direct answers (fast, cheap)
   - Complex questions get expert treatment (quality)
   - Best of both worlds

### What Didn't Work ❌

1. **3-Domain Synthesis (0% accuracy)**
   - 2 experts: 83% ✅
   - 3 experts: 0% ❌
   - Problem: Synthesizer can't prioritize 3 answers effectively
   - Next step: Architectural redesign or limit to 2 experts

2. **Prompt Caching Attempt**
   - Goal: 90% cost reduction with Anthropic
   - Reality: Our prompts (~700 tokens) below 1024-token minimum
   - Learning: Latest features aren't always applicable

3. **Over-optimization**
   - Spent time on fancy caching instead of simple solutions
   - Lesson: Model selection (GPT-4o-mini for simple) beats complex optimization

---

## 📈 Performance Breakdown

| Question Type | Accuracy | Notes |
|---------------|----------|-------|
| **Overall** | **82.8%** | 44.8pp improvement over baseline |
| Nutrition (single) | 100% 🔥 | Perfect domain expertise |
| Sleep (single) | 100% 🔥 | Perfect domain expertise |
| Play (single) | 100% 🔥 | Perfect domain expertise |
| Simple questions | 100% 🔥 | Fast direct answers |
| 2-domain complex | 83% ✅ | Effective synthesis |
| **3-domain complex** | **0%** ⚠️ | **Architectural limitation** |

---

## 🔒 Safety: Human-in-the-Loop

Medical advice is risky. Our approach:

```python
Risk Analyzer (GPT-4o):
├─ Checks for:
│  ├─ Medication dosage instructions
│  ├─ Emergency medical procedures
│  └─ Disease diagnosis claims
├─ Classification:
│  ├─ SAFE → Auto-approve ✅
│  └─ RISK → Human review required 🚨
└─ Results: 100% of risky advice caught in testing
```

---

## 📁 Project Structure

```
baby-squad-ai/
├── graph_agent_scalable.py    # Main multi-agent system (production)
├── backend_api.py              # FastAPI server with risk analysis
├── streamlit_app.py            # Frontend UI with HITL approval
├── batch_evaluate_v2.py        # Evaluation framework
├── cost_tracking.py            # Cost analysis utilities
├── baseline.py                 # Single-agent baseline for comparison
├── requirements.txt            # Python dependencies
├── evaluation_results/         # Evaluation outputs
│   └── batch_results_*.xlsx
└── README.md                   # This file
```

---

## 🎯 Roadmap

### Completed ✅
- [x] Multi-agent architecture with LangGraph
- [x] LLM-as-judge evaluation framework
- [x] 100+ test case evaluation
- [x] Human-in-the-loop safety system
- [x] Dynamic expert pool pattern

### In Progress 🚧
- [ ] Fix 3-domain synthesis (architecture redesign)
- [ ] Cost optimization with GPT-4o-mini (30-40% savings expected)
- [ ] Parallel expert execution (29s → 15s target)

### Planned 📋
- [ ] Real user beta testing (30-50 users)
- [ ] RAGAS metrics (faithfulness, relevancy)
- [ ] Video tutorial series
- [ ] Deploy to production

---

## 📖 Documentation

- **Blog Post**: [Building Production Multi-Agent RAG](link)
- **Video Tutorial**: [YouTube Playlist](link)
- **Technical Deep-dive**: [Medium Series](link)
- **Evaluation Methodology**: [docs/evaluation.md](link)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- 3-domain synthesis improvement
- Cost optimization strategies
- Additional expert domains
- Evaluation metrics

Please open an issue first to discuss proposed changes.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 About

Built by [Eunji Kim](https://linkedin.com/in/eunjikim2u) - 10+ years of enterprise AI experience at Microsoft and startups.

**Current focus**: Production GenAI systems, Multi-agent architectures, Responsible AI

---

## 📞 Contact

- LinkedIn: [linkedin.com/in/eunjikim2u](https://linkedin.com/in/eunjikim2u)
- Medium: [@eunjikim2u](https://medium.com/@eunjikim2u)
- YouTube: [@the-coding-cat](https://youtube.com/@the-coding-cat)
- Email: eunjikim2u@gmail.com

---

**⭐ If you find this project useful, please star the repository!**

**💬 Questions or feedback? Open an issue or reach out directly.**