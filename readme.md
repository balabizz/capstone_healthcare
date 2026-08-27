# Agentic Healthcare Assistant for Medical Task Automation - Capstone Project

An intelligent agentic healthcare system that automates medical tasks using LLMs, Retrieval-Augmented Generation (RAG), and autonomous agents with LangChain and LangGraph.

## 🎯 Project Overview

This project implements an **Agentic Healthcare Assistant** that functions as a virtual medical assistant capable of:
- **Booking medical appointments**: Automating slot discovery and scheduling based on patient intent and doctor availability
- **Managing medical records**: Enabling attendants to add or update structured/unstructured patient history
- **Retrieving medical histories**: Summarizing past diagnoses, treatments, and relevant alerts using LLMs
- **Performing medical information searches**: Fetching up-to-date disease information from trusted sources (Medline, WHO)
- **Autonomous task orchestration**: Breaking down complex multi-step patient queries into sequential sub-goals
- **Context retention**: Using memory modules to maintain long-term patient context across interactions

## 📁 Project Structure

```
capstone_healthcare/
├── data/                          # Data directory
│   ├── raw/                       # Raw healthcare documents (PDFs)
│   ├── processed/                 # Processed text documents
│   └── embeddings/                # Vector store data (FAISS, ChromaDB)
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── reference/                 # Reference materials and BRD
│
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│
│   ├── agents/                  # Agentic system components
│   │   ├── planner.py           # Goal decomposition and planning
│   │   ├── executor.py          # Agent execution engine
│   │   └── memory.py            # Patient context memory management
│
│   ├── tools/                   # Tool definitions for agents
│   │   ├── appointment_tool.py  # Doctor appointment booking
│   │   ├── medical_record_tool.py # Patient record management
│   │   ├── search_tool.py       # Medical information search (Web, Medline, WHO)
│   │   └── ehr_tool.py          # EHR database integration
│
│   ├── data_processing/         # Document loading and processing
│   │   ├── pdf_loader.py        # PDF extraction
│   │   └── text_processing.py   # Text chunking and preparation
│   │
│   ├── embeddings/              # Embedding generation
│   │   └── embedding_manager.py # Embedding management
│   │
│   ├── vector_store/            # Vector database interfaces
│   │   ├── chromadb_store.py    # ChromaDB integration
│   │   └── faiss_store.py       # FAISS integration
│   │
│   ├── llm/                     # Language model interfaces
│   │   ├── llm_client.py        # LLM client wrapper
│   │   └── prompt_templates.py  # Specialized prompt engineering
│   │
│   ├── chains/                  # LangChain components
│   │   ├── rag_chain.py         # RAG chain for medical info retrieval
│   │   └── task_chain.py        # Task chaining for multi-step workflows
│   │
│   ├── graphs/                  # LangGraph workflows
│   │   ├── workflow_graph.py    # Healthcare workflow orchestration
│   │   └── state_manager.py     # State management for agent flows
│   │
│   ├── evaluation/              # Model evaluation and monitoring
│   │   ├── qa_eval.py           # QAEvalChain for response evaluation
│   │   ├── metrics.py           # Performance metrics calculation
│   │   └── logger.py            # Agent action logging
│   │
│   └── utils/                   # Utility functions
│       └── helpers.py           # Helper utilities
│
├── app/                         # Streamlit application
│   ├── __init__.py
│   ├── streamlit_app.py         # Main web interface
│   ├── pages/                   # Multi-page Streamlit app
│   │   ├── patient_view.py      # Patient dashboard
│   │   ├── doctor_view.py       # Doctor dashboard
│   │   ├── appointments.py      # Appointment management
│   │   └── monitoring.py        # Agent monitoring and evaluation
│
├── tests/                       # Unit tests
│   └── __init__.py
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup configuration
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.9+
- pip or conda
- OpenAI API key

### Step 1: Clone the Repository
```bash
git clone https://github.com/balabizz/capstone_healthcare.git
cd capstone_healthcare
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your configuration
# Add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

## � Use Case Scenario

### Patient Query Example
> **"My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize the latest treatment methods?"**

### Agent Workflow
1. **Identify patient and context**: Extract patient age, condition, and relationships
2. **Retrieve father's medical history**: Query vector database for relevant patient records
3. **Query doctor calendar**: Access appointment booking API for nephrologist availability
4. **Book appointment**: Autonomously schedule appointment based on availability and patient preferences
5. **Search and summarize**: Retrieve latest treatment methods via RAG pipeline and medical search APIs
6. **Provide comprehensive response**: Return appointment details + personalized treatment summary

### Multi-step Task Decomposition
The agent automatically breaks down the complex query into:
- Patient identification task
- Records retrieval task
- Appointment booking task
- Medical research task
- Response synthesis task

## 🚀 Quick Start

### 1. Setup & Installation
```bash
# Clone repository
git clone https://github.com/balabizz/capstone_healthcare.git
cd capstone_healthcare

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OpenAI, Bing Search, etc.)
```

### 2. Initialize SQLite Patient and Doctor Database
```python
from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore
from src.models.doctor_vo import DoctorVO
from src.models.patient_vo import PatientVO

ehr = SQLiteStore(SQLITE_DB_PATH)
ehr.save_patient(PatientVO(patient_id="patient_123", first_name="John", last_name="Doe"))
ehr.save_doctor(DoctorVO(
    doctor_id="doctor_001",
    first_name="Jane",
    last_name="Smith",
    speciality="Nephrologist",
    license_number="LIC-001",
))
```

SQLite stores structured patient and doctor records. ChromaDB remains separate and stores document chunks and embeddings for RAG retrieval.

The SQLite EHR schema also includes:

- `appointments`: relates one patient to one doctor for a scheduled visit.
- `medical_history`: relates a patient to diagnoses and optionally the recording doctor.
- `prescriptions`: relates a patient and prescribing doctor, optionally to an appointment.
- `billing`: relates a patient and optionally the appointment being billed.

### 3. Initialize Vector Store for Patient Context
```python
from src.data_processing.pdf_loader import PDFLoader
from src.data_processing.text_processing import TextProcessor
from src.vector_store.chromadb_store import ChromaDBStore

# Load medical documents
loader = PDFLoader()
documents = loader.load_multiple_pdfs("data/raw")

# Process and embed
processor = TextProcessor()
chunks = processor.process_documents(documents)

# Create a vector store for document chunks and embeddings used by RAG
store = ChromaDBStore()
store.create_store([c["content"] for c in chunks], 
                   [c["metadata"] for c in chunks])
```

### 4. Initialize Agentic System
```python
from src.agents.planner import AgentPlanner
from src.agents.executor import AgentExecutor
from src.agents.memory import PatientMemory

# Create agent components
planner = AgentPlanner()
memory = PatientMemory(vector_store=store)
executor = AgentExecutor(planner, memory)

# Execute patient query
result = executor.execute(
    query="My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
    patient_id="father_123"
)

# Output: {
#   "appointment": {...},
#   "treatment_summary": "...",
#   "agent_trace": [...]
# }
```

### 5. Run Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- Patient appointment tracking
- Medical information summaries
- Agent planning breakdowns
- Performance evaluation metrics

## 🛠️ Key Components

### Part 1: Agentic System Architecture

#### Agent Planning & Decomposition
- **Planner**: Interprets multi-step patient queries and breaks complex requests into sequential sub-goals
- **Goal Decomposition**: Identifies appropriate tools or APIs to fulfill each task

#### Tool & Memory Setup
- **Appointment Booking Tool**: Integrates Doctor Schedule API for automated slot discovery and booking
- **Medical Records Tool**: Manages structured/unstructured patient history in EHR/Patient database
- **Medical Search Tool**: Fetches up-to-date disease information from Medline, WHO, and web search APIs
- **Memory Modules**: Stores and retrieves patient summaries using FAISS vector database for long-term context

#### Prompt Engineering & Task Chaining
- **Specialized Prompts**: Tailored prompts for each agentic sub-task (planning, summarization, action triggering)
- **Prompt Chains**: Guides LLMs through multi-step workflows with patient context injected via memory lookups

#### Agent Execution Flow
- Multi-step orchestration using LangGraph
- Context-aware decision making based on patient history
- Tool selection and sequencing for optimal task completion

### Part 2: LLMOps (Monitoring & Evaluation)

#### Model Evaluation
- **QAEvalChain**: Assesses accuracy and relevance of generated summaries
- **Metrics Tracking**: Logs success rate of bookings, response precision, and tool effectiveness
- **Performance Analytics**: Per-module performance analysis

#### Data Visualization & UI
- **Streamlit Dashboard**: 
  - Patient and doctor views
  - Real-time appointment tracking
  - Medical information summaries
  - Evaluation metrics display
- **Interactive Testing**: Scenario simulation and tool testing interface
- **Logs & Monitoring**: Agent memory traces, planning breakdowns, tool usage analytics

### Data Processing Components
- **PDFLoader**: Extracts text from healthcare documents
- **TextProcessor**: Chunks documents using RecursiveCharacterTextSplitter

### Vector Storage
- **ChromaDBStore**: Persistent vector storage with metadata
- **FAISSStore**: Fast similarity search using FAISS indices

### LLM Integration
- **LLMClient**: Interface for OpenAI API calls
- **RAGChain**: Retrieval Augmented Generation for medical information
- **PromptTemplates**: Specialized prompts for healthcare tasks

## 📊 Configuration

Edit `.env` to customize (see `.env.example` for full template):

```env
# LLM Settings
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7

# Vector Database Paths
CHROMADB_PATH=./data/embeddings/chromadb
FAISS_INDEX_PATH=./data/embeddings/faiss

# Tool APIs
DOCTOR_SCHEDULE_API_KEY=your_doctor_api_key
DOCTOR_SCHEDULE_API_URL=https://api.hospital.com/schedules

MEDLINE_API_KEY=your_medline_key
BING_SEARCH_API_KEY=your_bing_search_key

EHR_DATABASE_URL=your_ehr_db_connection_string
PATIENT_DATABASE_URL=your_patient_db_connection_string

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENTS=1000

# Agent Settings
AGENT_MAX_ITERATIONS=10
AGENT_TIMEOUT=300

# Evaluation & Monitoring
ENABLE_EVAL_LOGGING=True
EVAL_THRESHOLD=0.75

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

## 💡 Usage Examples

### Basic Agent Execution
```python
from src.agents.executor import AgentExecutor
from src.agents.planner import AgentPlanner
from src.agents.memory import PatientMemory

planner = AgentPlanner()
memory = PatientMemory()
executor = AgentExecutor(planner, memory)

# Single-step task
result = executor.execute(
    query="Book an appointment with Dr. Smith for cardiology check",
    patient_id="patient_123"
)
```

### Multi-step Complex Query
```python
# Complex multi-task query
result = executor.execute(
    query="""
    My father (70 years old, ID: patient_456) has chronic kidney disease.
    1. Book a nephrologist appointment for next week
    2. Summarize latest treatment methods
    3. Alert about potential drug interactions
    """,
    patient_id="patient_456"
)

print(f"Appointment: {result['appointment']}")
print(f"Treatment Summary: {result['treatment_summary']}")
print(f"Alerts: {result['alerts']}")
```

### Medical Information Retrieval
```python
from src.chains.rag_chain import RAGChain
from src.vector_store.chromadb_store import ChromaDBStore

vectorstore = ChromaDBStore().load_store()
rag_chain = RAGChain(vectorstore)

result = rag_chain.query("Latest treatment options for diabetes management")
print(result["answer"])
```

### Appointment Booking
```python
from src.tools.appointment_tool import AppointmentScheduler

scheduler = AppointmentScheduler()
appointment = scheduler.book_appointment(
    doctor_id="dr_smith_001",
    patient_id="patient_123",
    specialty="Cardiology",
    preferred_date="2024-06-15"
)
```

### Patient Record Management
```python
from src.tools.medical_record_tool import MedicalRecordManager

record_manager = MedicalRecordManager()

# Add new patient record
record_manager.add_record(
    patient_id="patient_123",
    record_type="diagnosis",
    data={
        "condition": "Hypertension",
        "severity": "moderate",
        "medications": ["Lisinopril"]
    }
)

# Retrieve patient history
history = record_manager.get_patient_history("patient_123")
```

## 📊 Model Evaluation & Monitoring

### QA Evaluation
```python
from src.evaluation.qa_eval import QAEvaluator

evaluator = QAEvaluator()

# Evaluate generated responses
results = evaluator.evaluate(
    generated_text="Latest treatment includes ACE inhibitors and lifestyle changes",
    reference_text="ACE inhibitors are recommended along with diet and exercise",
    metric="rouge"
)

print(f"ROUGE Score: {results['rouge_score']}")
print(f"Semantic Similarity: {results['semantic_score']}")
```

### Agent Performance Metrics
```python
from src.evaluation.metrics import PerformanceMetrics

metrics = PerformanceMetrics()

# Log agent actions
metrics.log_action(
    agent_id="agent_001",
    task="appointment_booking",
    success=True,
    duration=2.5,
    tool_used="AppointmentAPI"
)

# Get performance report
report = metrics.get_performance_report()
print(report)
```

### Monitoring Agent Traces
```python
from src.evaluation.logger import AgentLogger

logger = AgentLogger()

# View agent decision path
traces = logger.get_agent_traces(agent_id="agent_001", limit=10)
for trace in traces:
    print(f"Step: {trace['step']}")
    print(f"Action: {trace['action']}")
    print(f"Reasoning: {trace['reasoning']}")
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_rag_chain.py -v
```

## 📚 Technologies Used

### Core AI/ML Stack
- **LangChain**: LLM orchestration, chains, and RAG pipelines
- **LangGraph**: Agentic workflow graph orchestration and state management
- **OpenAI API**: GPT-3.5/GPT-4 for language understanding and generation

### Vector & Data Layer
- **FAISS**: Fast similarity search for patient context retrieval
- **ChromaDB**: Persistent vector database with metadata management
- **PyPDF**: PDF document processing and extraction

### Tool Integration
- **Doctor Schedule API**: Appointment booking and calendar management
- **EHR/Patient Database**: Structured patient record management
- **Medline API**: Medical literature and treatment information
- **Bing Search API**: Real-time medical information search
- **Web Search APIs**: Generic medical research retrieval

### Application & Monitoring
- **Streamlit**: Interactive dashboard and monitoring UI
- **LangChain QAEvalChain**: Response quality evaluation
- **Custom Evaluation Metrics**: Agent performance tracking

### Development
- **Python 3.9+**: Core language
- **LangGraph/LangChain**: Agent and workflow frameworks
- **Python-dotenv**: Environment configuration

## 🔐 Security Notes

- Never commit `.env` file with real API keys
- Use environment variables for sensitive data
- Keep OpenAI API key secure
- Validate user inputs before processing

## 📝 Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints for functions
- Add docstrings to all modules and functions

### Adding New Features
1. Create feature branch: `git checkout -b feature/feature-name`
2. Implement changes with tests
3. Push and create pull request

## 🐛 Troubleshooting

### Issue: Agent fails to complete multi-step tasks
- Check `AGENT_MAX_ITERATIONS` in .env file
- Verify all required tools are properly initialized
- Review agent traces in logs for decision failures
- Ensure patient context is loaded in memory module

### Issue: Appointment booking returns no availability
- Verify Doctor Schedule API key and endpoint in .env
- Check doctor calendar has appointments configured
- Ensure date/specialty filters are correct
- Review API response logs for detailed errors

### Issue: Medical information retrieval returns irrelevant results
- Verify vector store is properly populated with medical documents
- Check embedding model is consistent (ada-002)
- Review chunk size and overlap settings
- Use semantic search debugging to test similarity scores

### Issue: "API key not valid" or authentication errors
- Verify all API keys in .env file (OpenAI, Bing Search, Medline, etc.)
- Check API key expiration and quotas
- Confirm API endpoints are current and accessible
- Test API connectivity separately

### Issue: Patient memory not retaining context
- Check FAISS/ChromaDB paths are correct
- Verify vector store persistence is enabled
- Review memory module initialization
- Check patient ID consistency across queries

### Issue: Streamlit dashboard not loading
- Install streamlit: `pip install streamlit`
- Check port 8501 availability
- Verify app/ directory has all required modules
- Run with: `streamlit run app/streamlit_app.py --logger.level=debug`

### Issue: Agent planning breaks down or produces invalid goals
- Review LLM prompt templates for clarity
- Check if patient context is being injected correctly
- Verify LLM model version and temperature settings
- Test planner with simpler queries first

### Issue: Tool execution timeout
- Increase `AGENT_TIMEOUT` in .env file
- Check external API response times
- Verify network connectivity to external services
- Review tool-specific timeout settings

## ✅ BRD Compliance & Implementation Status

### Part 1: Agentic Healthcare Assistant System Design

| Component | Status | Details |
|-----------|--------|---------|
| Agent Planning & Goal Decomposition | 📋 To-Do | Planner module for multi-step query breakdown |
| Tool Setup - Appointment Booking | 📋 To-Do | Doctor Schedule API integration |
| Tool Setup - Medical Records | 📋 To-Do | EHR database integration |
| Tool Setup - Medical Search | 📋 To-Do | Medline/WHO/Web search APIs |
| Memory Management | 📋 To-Do | FAISS vector store for patient context |
| Prompt Engineering | 📋 To-Do | Specialized prompts for each task |
| Task Chaining | 📋 To-Do | Sequential task execution |
| Agent Execution Flow | 📋 To-Do | LangGraph-based orchestration |

### Part 2: LLMOps (Model Evaluation, Monitoring, and Streamlit UI)

| Component | Status | Details |
|-----------|--------|---------|
| Model Evaluation | 📋 To-Do | QAEvalChain for response accuracy |
| Performance Metrics | 📋 To-Do | Booking success rates, response precision |
| Streamlit Dashboard | 📋 To-Do | Patient/Doctor views, appointment tracking |
| Medical Info Display | 📋 To-Do | Summarized medical information UI |
| Evaluation Metrics UI | 📋 To-Do | Performance visualization |
| Agent Traces Interface | 📋 To-Do | Planning breakdown display |
| Memory Logs | 📋 To-Do | Tool usage and success logging |
| Interactive Scenarios | 📋 To-Do | Test different use cases |

### Use Case Implementation
- **Scenario**: 70-year-old father with chronic kidney disease
- **Required Capabilities**: 
  ✓ Multi-step query decomposition
  ✓ Patient history retrieval
  ✓ Nephrologist appointment booking
  ✓ Treatment method summarization
  ✓ Comprehensive response generation

## 👥 Contributors

Healthcare Capstone Team

## 📞 Support

For issues and questions, please refer to the project documentation or create an issue on GitHub.

---

**Last Updated**: May 2026