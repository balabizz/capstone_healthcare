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
│   └── embeddings/                # FAISS vector store data
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

SQLite stores structured patient and doctor records. FAISS stores reference document chunks and embeddings for RAG retrieval.

The SQLite EHR schema also includes:

- `appointments`: relates one patient to one doctor for a scheduled visit.
- `medical_history`: relates a patient to diagnoses and optionally the recording doctor.
- `prescriptions`: relates a patient and prescribing doctor, optionally to an appointment.
- `billing`: relates a patient and optionally the appointment being billed.

### 3. Initialize Vector Store for Patient Context
```python
from src.data_processing.pdf_loader import PDFLoader
from src.data_processing.text_processing import TextProcessor
from src.vector_store.faiss_store import FAISSStore

# Load medical documents
loader = PDFLoader()
documents = loader.load_multiple_pdfs("data/raw")

# Process and embed
processor = TextProcessor()
chunks = processor.process_documents(documents)

# Create a vector store for document chunks and embeddings used by RAG
store = FAISSStore()
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
- **FAISSStore**: Persistent vector storage and fast similarity search using FAISS indices

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

# Database and vector paths
SQLITE_DB_PATH=./data/healthcare.db
FAISS_INDEX_PATH=./data/embeddings/faiss

# Optional private schedule service; leave URL empty for local SQLite scheduling
SCHEDULE_TIMEZONE=Australia/Sydney
SCHEDULE_API_URL=
SCHEDULE_API_TOKEN=

# Optional NCBI contact/key for live PubMed searches
NCBI_EMAIL=
NCBI_API_KEY=
MEDICAL_SEARCH_DAYS=730

# Patient summary embeddings use the OpenAI API
PATIENT_SUMMARY_EMBEDDING_MODEL=text-embedding-3-small

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENTS=1000

# Appointment polling interval, allowed range 2-60 seconds
APPOINTMENT_REFRESH_SECONDS=5

# Evaluation & Monitoring
ENABLE_EVAL_LOGGING=True
EVAL_THRESHOLD=0.75

# Application Settings
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
from src.vector_store.faiss_store import FAISSStore

vectorstore = FAISSStore().load_store()
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
- Check the FAISS path is correct
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
### Family and dependents (Step 12)

The application now stores family links in SQLite `patient_dependents`. Each link
has an owner patient ID, name, relationship (father, mother, spouse, child, sibling,
guardian or other), and an optional linked patient ID. Clinical history and
appointments remain attached to the family member's own patient record. An
unregistered relative can be saved without a linked ID, but must be registered
before booking or accessing records. Multiple children are separate links.

In **Family & dependents**, add a relative and select who the enquiry is for.
A relationship alone grants no access: the linked patient signs in and grants
`book_appointment`, `view_appointments`, or `view_medical` to the enquiring patient's
ID. The same form revokes permission by leaving “Allow access” unchecked.
These checks also apply to parents and children; verified guardian authorization
is a future extension. A mistaken patient-ID link cannot grant access by itself.

For Step 12: “My 70-year-old father has chronic kidney disease. Book a nephrologist
appointment and summarize treatment methods.”

1. Save the father relationship and link his registered patient ID.
2. The father grants booking and medical-enquiry permissions to the caller.
3. Submit the prompt. The planner resolves the father from the caller's links;
   missing, unregistered, ambiguous, or unauthorized tasks show their status.
4. The planner matches the requested specialty, discovers the next available slot,
  and books it automatically through the chat flow.
5. The response shows the booked date, time, doctor, specialty, timezone, and duration.
6. The appointment belongs to the father, not the caller. Viewing his appointments
   additionally requires `view_appointments` permission.

For multiple children, select a specific child first. Mother, spouse, husband,
wife, son and daughter enquiries follow the same flow. Requests naming different
relationships must be split into separate enquiries.

The medical RAG answers from the existing reference collection and does not
guarantee up-to-date treatment information. The multi-step executor can separately
retrieve authorized, saved medical-history entries for final synthesis. Repository and service caller IDs are trusted application inputs and
must come from the authenticated session in any future API. No self-service
permission grant on someone else's record is exposed in the UI.

### OpenAI multi-step planning (BRD §1 / notebook Step 12)

The Streamlit request flow now uses `Planner` → `PlanExecution`, replacing keyword
routing with an OpenAI-generated, strictly structured plan. Configure:

```dotenv
OPENAI_API_KEY=your_key
PLANNER_MODEL=gpt-4o-mini
```

`PLANNER_MODEL` controls planning and final synthesis independently of `LLM_MODEL`
(the existing reference RAG model). It must support Chat Completions strict JSON
schema output. The adapter uses `requests` against OpenAI's API so it does not
require upgrading the existing legacy LangChain/OpenAI SDK integration. No API key,
refusal, invalid output or timeout produces a keyword fallback or executes a plan.

For “My 70-year-old father has CKD. Book a nephrologist and summarize the latest
treatments”, the planning prompt requests:

| Goal | Tool | Dependencies / outcome |
|---|---|---|
| Resolve patient | `patients.resolve` | Uses authenticated caller and saved family selection; never model-supplied IDs |
| Retrieve history | `medical_history.read` | Patient resolution and `view_medical` permission; reads up to 50 latest saved entries |
| Find specialist | `doctors.find_specialist` | Patient resolution; matches the requested specialty against local doctors |
| Book appointment | `appointments.discover` + `appointments.book` | Patient and specialist lookup; books the earliest available matching slot |
| Search recent publications | `medical_search.latest` | Live PubMed/WHO search with dated sources and explicit provider outcomes |
| Summarize outcomes | `responses.summarize` | Uses all actual step results, including empty history, denied access and unavailable tools |

Model-generated plans can vary with wording, but local validation enforces the tool
allowlist, unique step IDs, earlier-only dependencies, patient prerequisites and a
final summary. The executor revalidates before running. No model output can supply
SQL, a tool implementation or an authorization grant. Ambiguous requests return a
clarification instead of executing. One patient is supported per request.

The UI displays the plan, tools, dependencies and outcomes. Appointment discovery and
booking happen only through the chat flow. A missing/denied history step does not block
independent booking work. Each step logs its actual status, and the appointment step is
successful only after the transactional booking completes.
Clinical text is not copied into execution logs; the request and authorized history
may be sent to OpenAI for planning/synthesis. Outcomes are retained only in the
current Streamlit session, not long-term conversational memory.

Limits: long-term memory and evaluation remain separate work.
Medical-record editing is implemented in the attendant workflow described below. History retrieval is a read-only adapter over existing records, not
a complete EHR management workflow. Specialist discovery is a local specialty
match; doctor availability is checked during chat execution. The legacy
`GoalExecution.execute()` remains for direct goal callers; the UI uses
`PlanExecution.run()` for complete plans.

Offline verification: `python -m pytest -q`. Tests inject OpenAI responses and cover
plan validation, refusals/timeouts, family authorization, patient-scoped history,
dependency failures, booking proposals and final synthesis. These tests do not
measure live model planning quality or replace an end-to-end API/UI acceptance run.

### Automated appointment discovery and Doctor Schedule API

Chat appointment requests may follow this generic structure:

`Book/find/show <appointment> with <doctor, specialty, or any doctor> at <location> on/between <date or date range> at/between <time or time range> with <consultation type> for <patient or reason>`

The optional clauses are normalized into doctor, specialty, location, date/time
constraints, consultation type, and reason. Patient identity comes only from the
authenticated session or authorized family selection. When date/time is omitted,
the soonest slot is booked without confirmation, starting four hours after the
request and rolling to the next clinic day when necessary. `today` and `tomorrow`
are converted to clinic-local dates before searching. Location filters stored doctor
addresses and consultation type is persisted with the appointment.

Appointment discovery now extends the existing `GoalExecution.book_appointment()`,
`appointments` table, patient appointment list and doctor schedule view. It does not
create a separate booking database. `ScheduleRepository` supplies calendar checks
and atomic writes beneath that existing booking entry point.

- Doctors configure recurring weekday working windows and whole-day leave through the
  schedule service or administrative tooling. Multiple windows support lunch breaks.
  New doctors have no availability until hours are configured.
- The OpenAI planner extracts specialty, an optional named doctor, date range and
  time window. Relative dates use the configured clinic timezone. “Nephrology”
  matches seeded “Nephrologist” records through a specialty alias map.
- Discovery returns the earliest five matching 30-minute slots, respecting working
  hours, days off, past times, and both doctor and patient appointment conflicts.
  If dates are omitted it searches the next 30 days. It never silently broadens
  explicit date/time/doctor preferences when there are no matches.
- The first available matching slot is booked automatically from chat. If a concurrent
  booking consumes that slot, execution retries the next discovered slot.
- When no date or time is supplied, the search starts four hours after the request.
  It uses remaining same-day slots first, then rolls to the next day when the clinic
  day has ended.
- Booking rechecks permissions, calendars and conflicts inside a SQLite write transaction.
  The same request key returns the original appointment on retry; conflicting reuse is
  rejected. Existing appointments remain when hours change.

For demo data only, after seeding doctors, create example calendars explicitly:

```bash
python -m scripts.seed_calendars
```

This adds alternating doctor schedules without overwriting existing working hours.
For real schedules, use the doctor's calendar screen instead.

A local **Doctor Schedule HTTP API** is included. No external vendor was specified;
this is the capstone's own service, not a claimed integration with a hospital vendor.
By default the application calls the same repository in-process. To exercise HTTP,
configure `.env` with a private random service token of at least 24 characters:

```dotenv
SCHEDULE_TIMEZONE=Australia/Sydney
SCHEDULE_API_URL=http://127.0.0.1:8765
SCHEDULE_API_TOKEN=<your-random-service-token>
```

Run these commands in separate terminals from the project root:

```bash
python -m src.scheduling.api --port 8765
streamlit run app/streamlit_app.py
```

**Both processes must use the same `SQLITE_DB_PATH` and timezone.** Existing naive
appointment timestamps are interpreted in that clinic timezone; changing timezone
after bookings exist requires a data migration. The demo server binds to localhost.
Its bearer token identifies the trusted Streamlit application, which supplies caller
and patient identities from the authenticated session. It is not a public patient
API. An external provider would need its own adapter and identity mapping.

| Endpoint | JSON input | Result |
|---|---|---|
| `POST /specialists` | Optional `specialty`, `doctor_name` | Matching local doctors |
| `POST /slots` | `requester_patient_id`, `patient_id`; optional `specialty`, `doctor_name`, `date_from`, `date_to`, `time_from`, `time_to`, `limit` | Ordered slots with doctor, date, time, timezone and duration |
| `POST /bookings` | `requester_patient_id`, `patient_id`, `doctor_id`, `day`, `slot`, `idempotency_key`; optional `reason` | `booked` with appointment ID, or `slot_unavailable` |

All endpoints require `Authorization: Bearer <service-token>`. The HTTP client never
falls back to local booking on API failure. After an uncertain network outcome,
retry confirmation using the same request key. Working-calendar management and
existing appointment-list screens continue using the shared SQLite database.

Limits: appointments remain 30 minutes, one clinic timezone, and whole-day leave;
location preferences, variable visit durations, external calendar synchronization,
and rescheduling/cancellation are not implemented. Ambiguous or unrepresentable
natural-language constraints should produce a clarification.

Verification includes calendar/leave filtering, concurrent booking, overlapping
legacy appointments, patient conflicts, permissions, request replay and HTTP
contracts. A live OpenAI → local HTTP API → booking/retry smoke test passed using
only a disposable synthetic database. UI compilation passed; interactive UI testing
was blocked in the current Python 3.14 environment because the pinned Streamlit
installation could not build its Pillow dependency (missing JPEG build headers).

### Attendant medical-record management (BRD objective / §2)

The login screen now offers **attendant** alongside patient and doctor. Attendants
open a dedicated **Medical records** screen instead of the patient booking/chat
screen. They can select an assigned patient, add or edit a diagnosis and treatment,
and add or edit a standalone free-text clinical note. Notes are stored with
`record_type=note` and an empty condition name, so they are not presented as a
new diagnosis. Optional diagnosis date and documenting/treating doctor can be
recorded for diagnosis entries; dates cannot be in the future.

Create an attendant from a trusted administrator terminal (use actual existing
patient IDs; repeat `--patient` to assign more than one):

```bash
python -m scripts.manage_attendants create --username attendant.one --first-name Demo --last-name Attendant --patient patient-001
```

The command prompts privately for a password and confirmation. It stores a salted
PBKDF2-SHA256 hash, never the plaintext password. New staff passwords require at
least 12 characters. No shared default attendant credential is created. Then log
in with the chosen username/password and select **attendant**.

Manage assignments and account status with:

```bash
python -m scripts.manage_attendants assign --username attendant.one --patient patient-002
python -m scripts.manage_attendants revoke --username attendant.one --patient patient-002
python -m scripts.manage_attendants disable --username attendant.one
python -m scripts.manage_attendants enable --username attendant.one
```

To target another database, put `--database /path/to/healthcare.db` before the
subcommand. These commands are trusted local administration; attendants cannot
assign themselves patients in the application. Existing patient and doctor login
accounts remain compatible with their current credentials. Their legacy password
storage has not been globally migrated by this feature.

The editor supports:

1. Select the patient by name, ID and date of birth from assigned patients only.
2. Select **Add diagnosis / treatment**, **Add standalone clinical note**, or an
   existing record to edit.
3. Enter documented clinical information and a reason for the entry/correction.
4. Click **Save medical record**. The application validates and saves the entry.
5. Inspect **Record revision history** for prior content, editor, time and reason.

`GoalExecution.medical_records` exposes the underlying application tools:
`list_patients`, `list_records`, `save`, and `revisions`. Every read and write
rechecks the active attendant account and patient assignment; patient/doctor IDs
cannot be used as attendant credentials. The attendant ID must come from the
server's authenticated session in any future API, never from an untrusted client.
The current interface uses explicit form saves, not LLM-generated medical writes.

Data stays in the existing `medical_history` table. The migration adds record type,
version, update time and editor attribution, plus `attendants`, `attendant_patients`
and `medical_record_revisions`. It rebuilds the login-role constraint while keeping
existing accounts. Migration runs automatically and idempotently during SQLite
initialization; older history entries become diagnosis records at version 1. Their
first edit preserves the previous content in revision history.

Saving the current record, its before/after revision and the tool event is one
transaction. Concurrent/stale edits are rejected until the attendant reloads and
reviews the latest version. Repeating the same save request does not create a
duplicate. Patient ownership cannot be changed by editing a record. No record
deletion workflow is exposed. Clinical content and correction reasons are retained
in the protected revision data; general `agent_events` contains only identifiers,
version and status. There is no new encryption-at-rest layer in this feature.

The existing authorized history-retrieval step now includes record type and version,
orders by last modification, and passes saved diagnoses, treatments and notes into
its existing summary flow. This does not add record uploads/OCR, prescription
issuing, clinical decision-making, long-term memory or model evaluation.

Validation: repository/authentication/migration tests cover legacy data, assignment
revocation, disabled accounts, notes, validation, concurrent edits, idempotency,
audit rollback and history integration. An editor callback test covers creation
and stale-edit reload using a Streamlit test double. Full browser testing remains
unverified because of the previously recorded Streamlit/Pillow environment issue.

### Retrieve and summarize patient history (BRD objective / §4)

The assistant now retrieves one authorized patient's **diagnoses, treatments,
standalone clinical notes, prescriptions and recorded alerts**, then summarizes
that evidence with OpenAI. It uses the existing `medical_history` and `prescriptions`
tables plus a new `patient_alerts` table; it does not substitute general medical
RAG results for the patient's actual records.

Try:

- “Summarize my saved diagnoses, treatments, prescriptions and alerts.”
- “What medicines am I prescribed according to my records?”
- “Summarize my father's medical history and recorded allergies.”

For a relative, first create/link the family member and obtain `view_medical`
permission. Select the specific relative when a relationship is ambiguous.
The planner routes these requests through **patient lookup → history retrieval →
final summary**. Existing mixed requests can also include appointment discovery.

The generated summary has separate diagnoses/treatments, notes, prescriptions and
alerts sections, with typed references such as `[prescription:prescription-id]`.
The structured-output schema restricts references to the retrieved records;
application validation rejects invented references, populated sections omitted in
full, omitted retrieved prescriptions and omitted retrieved active alerts.
Reference validation does not by itself prove every generated clinical statement
is faithful; comprehensive model-quality evaluation remains separate work.

The application:

- Preserves uncertainty in notes and instructions to summarize only recorded facts.
- Includes prescription name, dose, frequency, recorded dates and instructions in
  the evidence. Recorded dates are classified explicitly; a prescription order
  does **not** establish that the patient currently takes that medicine. Missing or
  malformed dates are unknown, rather than assumed current.
- Distinguishes active from resolved alerts. Active alerts are retrieved first,
  prioritizing critical/high severity, and shown separately in the UI with their
  recorded description and severity. Alerts are documented data, not automatically
  inferred allergies, drug interactions or new clinical diagnoses.
- Shows the patient ID and retrieval timestamp, expandable **Patient history source
  records**, per-category counts, and warnings about omitted records or shortened text.
- States that missing records do not establish absence of disease, medications or
  allergies. Summaries are limited to stored evidence; they are not treatment advice.

Assigned attendants can maintain documented alerts under **Medical records →
Patient alerts / allergies**. They can add an allergy/clinical/other alert with a
recorded severity and description, then mark it resolved with a documented reason.
The original description, creator, time and resolution are retained; alert creation
and resolution generate identifier-only audit events. This feature reads existing
prescriptions; it does not add a prescription-issuing workflow.

`PatientHistoryRepository.retrieve()` reads a consistent SQLite snapshot, scoped
by the resolved patient ID. Defaults are up to 50 entries per category, up to 4,000
characters per free-text field, and a 24,000-character serialized row budget per
category. Counts explicitly identify omissions and shortened text, including omitted
active alerts. The limits keep the LLM request bounded and do not claim the summary
is the patient's complete lifetime history.

Medical permissions are checked before and after retrieval and before and after
LLM generation. A detected revocation removes the protected results rather than
returning the generated summary. Cached history and history-related chat entries
are rechecked on UI reruns; changing family selection clears displayed context.
A page already delivered to a browser is not remotely erased without a rerun.
Clinical content is sent to the configured OpenAI model for this authorized task,
but is not copied into general execution logs. The patient identifier is omitted
from the summary model payload; it remains in application scope/audit metadata.

Verification: 76 automated tests pass, including isolation, permission revocation,
empty data, prescription-date handling, alert prioritization, truncation, source
validation and existing workflow regression tests. A live OpenAI planning and
summary smoke test passed with a disposable synthetic father/dependent dataset,
including a prescription, clinical note and active allergy. Full browser validation
remains subject to the Streamlit/Pillow environment limitation described above.

### Live current-medical-information search (BRD objective / §2)

`medical_search` now performs live publication retrieval rather than returning an
unavailable placeholder or relying on a static local FAISS index. Try:

> Search recent diabetes treatment publications and summarize what the retrieved sources report.

For the combined BRD scenario, ask for the relative's history, appointment and
recent treatment publications in one request. The existing planner keeps those
steps separate; private history and public literature are summarized from their
respective evidence rather than treating one as proof of the other.

Two provider adapters are enabled by default:

- **PubMed/MEDLINE:** NCBI ESearch finds recent records, sorted by publication date;
  EFetch supplies titles, abstracts, publication dates, journal and publication
  types. PubMed is broader than MEDLINE; returned `index_status` records that
  distinction. Known retracted papers/retraction notices are excluded by query and
  fetched metadata checks.
- **WHO publications:** the WHO publication API searches title/overview topic terms
  and returns dated publication overviews, titles and canonical WHO item links.
  Generic query words are removed; WHO matches topic words after excluding generic
  treatment/search wording. A WHO search can legitimately return no results for a topic.

The implementation follows the [NCBI E-utilities reference](https://www.ncbi.nlm.nih.gov/books/NBK25499/)
and [WHO publication API reference](https://www.who.int/api/hubs/publications/sfhelp).

Configuration (optional `.env` settings):

```dotenv
NCBI_EMAIL=your-contact-email
NCBI_API_KEY=
MEDICAL_SEARCH_DAYS=730
```

Live searches do not require a paid search-service key. Configure a real NCBI
contact email for deployment; an optional **NCBI** key is separate from the OpenAI
key. The existing `OPENAI_API_KEY`/`PLANNER_MODEL` power planning and synthesis.
NCBI requests are spaced at least 0.35 seconds apart across threads in this process;
multi-process deployments must coordinate their combined request rate separately.

The default window is the previous 730 days through the search date (UTC), with up
to five results per provider. Change `MEDICAL_SEARCH_DAYS` to choose another window
(1–3650 days). The planner asks clarification for unsupported explicit historical
ranges. The default is a bounded recent-publication search, not a complete
systematic review or a guarantee of the latest clinical standard of care.

Each request includes:

- Search time, publication window, provider success/failure/no-result status.
- Source titles, provider URLs, publication dates and retrieved abstracts/overviews.
- OpenAI findings constrained to the retrieved source IDs. The application inserts
  links from provider metadata and rejects unknown references/model-generated URLs.
- Separate **Live medical search sources** in the UI. Metadata-only records remain
  visible but cannot support generated medical findings.
- Explicit incomplete-source and provider-failure outcomes. A failed provider does
  not silently fall back to local RAG. If no usable excerpts are available, no
  unsupported medical answer is generated from model memory.

Provider URLs are fixed in code, redirects are not followed, downloads are bounded,
and excerpts are limited to 6,000 characters per source. WHO HTML is converted to
plain text. Retrieved text is treated as untrusted evidence, not tool instructions.
Only a short general topic is sent to search providers, never the patient-history
bundle. The planner is instructed to omit identifiers, and the external boundary
rejects obvious personal references, emails, URLs and long numeric IDs. This guard
is not a complete de-identification system; use general disease/intervention topics.
Neither queries nor clinical excerpts are copied into general execution-event logs.

Limits: abstracts and WHO overviews are used, not full article/PDF downloads. Recent
research is not automatically a guideline, regulatory approval or an appropriate
personal treatment. Provider availability/index coverage and publication metadata
limit the result; clinical claim faithfulness still needs the separate evaluation
workstream. No arbitrary web crawling or additional search provider is implemented.

Validation includes mocked network contracts, query/date filters, known-retraction
exclusion, provider failures, empty results, download limits, citation validation and
mixed history/search isolation. A live OpenAI → PubMed/WHO → cited-summary test
returned five PubMed abstracts and five WHO overviews successfully without using
patient data. Full browser testing remains limited by the existing Streamlit/Pillow
environment issue.

### Patient summary vectors (FAISS)

Patient history summaries are now indexed automatically after the default OpenAI
history-summary workflow completes. Only the validated clinical summary is indexed;
appointment results and public medical-search answers are excluded.

To index existing records or retry a failed index, sign in as a patient, choose
**Myself** or a linked relative under **Family & dependents**, expand **Search saved
patient summary (FAISS)** and click **Rebuild patient summary**. Enter a topic such
as “allergies” or “previous treatments” and click **Search patient summary**.
The selected relative must grant `view_medical`. Results show excerpts and the
underlying source snapshot/coverage, and are not a substitute for full history.

`OPENAI_API_KEY` supplies the summary and embedding calls.
`PATIENT_SUMMARY_EMBEDDING_MODEL` defaults to `text-embedding-3-small`.
Clinical summary text and search queries are sent to OpenAI for embeddings.
FAISS is already a project dependency; the patient pipeline uses its native API
independently of the older LangChain public-document helper.

Each patient has one replaceable FAISS cosine-similarity index. Serialized FAISS
bytes, excerpts, embedding model and source provenance are stored together in the
`patient_summary_vectors` table in the configured SQLite database. This keeps the
index and metadata transactionally consistent and separate from public RAG.
No pickle files are loaded. Rebuilding replaces previous summary vectors.

Search authorizes access before loading that patient's index and checks again after
embedding. A fingerprint includes all diagnosis/note, prescription and alert rows,
including rows omitted by bounded history retrieval. Changes to these records,
the clinic date or embedding model require rebuilding before results are returned.
This prevents stale summary retrieval; it does not erase stale stored bytes until
replacement or patient deletion. Summary coverage remains bounded by the existing
history reader and is shown with results. Index failures preserve the generated
answer, show a retry notice and do not replace an existing index.

Implementation: `src/vector_store/patient_summaries.py`,
`src/repositories/patient_history_repository.py`, `src/agents/plan_execution.py`,
`src/llm/planning_client.py`, and `app/patient_summary_view.py`.

### Long-term conversational memory

BRD Part 1 §2 (page 3) requires long-term patient context, §3 requires memory
lookups in prompts, and Part 2 §8 (page 4) requires visible memory traces. These
are implemented separately from execution logs and the FAISS clinical-summary index.

Patient chat turns persist in SQLite `patient_conversations`, including the original
question, assistant answer, timestamp and turn ID. They survive sign-out and process
restart. Memory is isolated by **signed-in patient account + subject patient**:
caregiver conversations about a father do not mix with self conversations or another
caregiver's conversations. Linked-patient access requires `view_medical` on every
read/write and is checked again after model generation for memory recall.

Before execution the application first plans without memory to resolve an explicit
relationship, then retrieves bounded recent and keyword-related older turns for that
subject and incorporates them into the planner prompt. A second plan cannot switch
those memories to a different patient. Current instructions override older preferences;
remembered dialogue is untrusted historical context, never verified clinical facts or
permission to replay a booking. Current clinical questions still retrieve live DB
history. “What did we discuss previously?” can use the `conversation_recall` tool,
which returns timestamped excerpts with turn references. No saved dialogue returns
an explicit empty-memory response.

To test: select yourself or an authorized dependent, submit a request, sign out and
back in, reselect the same patient, and inspect **Chat History**. Follow up using a
previously stated preference, or ask what was discussed. **Delete saved conversations
for selected patient** deletes that account's saved dialogue for that subject; it does
not delete medical records, clinical summary vectors or execution logs.

The prompt retrieval budget is six turns, with up to 1,500 question and 2,500 answer
characters per turn; longer turns are marked truncated. SQLite retains complete
accepted turns, with a 12,000-character question / 100,000-character answer limit.
Older relevant turns use keyword matching; conversational retrieval does not claim
semantic recall of all past sessions. There is no automatic expiry or extracted
clinical-fact profile. Saved context sent to the planner is sent to OpenAI, as with
other configured LLM requests. Doctor/attendant conversations are not persisted by
this patient memory workflow. Memory save failures are shown separately from answer
generation failures.

### Conversational follow-ups in medical RAG (notebook Step 11)

Saved conversation context now reaches the active medical-question flow directly,
not just the planner. `GoalExecution.answer_patient_question` retrieves bounded
history for the authenticated requester and resolved patient. `RAGChain.query_with_history`
uses the configured OpenAI `PLANNER_MODEL` to resolve the current question into a
standalone reference query before document retrieval. For example, after discussing
diabetes, “What are its complications?” becomes a question about diabetes complications.
The current explicit topic takes precedence over older topics. Ambiguous references
produce a clarification instead of a guessed retrieval query.

Only the resolved question goes to the public-document retriever. Historical dialogue
is used to understand the question, not as verified clinical evidence. The answer is
still grounded in retrieved reference documents, and source documents are retained.
Patient-record questions continue to use the separate history workflow. With no saved
context, the existing plain reference-question path is used.

Access is checked before loading dialogue, after context resolution, after RAG answer
generation and around final summarization. Contextual answers are tagged with their
patient scope so the UI also rechecks permissions on later renders. Switching the
selected patient does not carry another patient's dialogue into RAG. Doctor/general
requests without an authenticated patient context continue without patient memory.

Validation: `tests/test_medical_question_context.py` exercises the actual application
wiring with deterministic model and retrieval doubles: persisted follow-ups after
restart, scoped patient switching, empty history, ambiguity, malformed resolver
responses, bounded history, and revocation before/during generation. These tests do
not measure live model accuracy or replace browser testing.

### Task-specific prompts and chaining — BRD §3; notebook Steps 9 and 14

The active reference RAG chain now passes an application-defined chat prompt through
`RetrievalQA.from_chain_type(..., chain_type_kwargs={"prompt": ...})`. This replaces
the library's default QA instructions. As in notebook Step 9, it explicitly consumes
`context` and `question`; as in Step 14, execution fills them with retrieved documents
and the current (or context-resolved) question.

| Task | Application prompt / contract | Next stage |
| --- | --- | --- |
| Planning | `src/agents/planner.py`: `PLANNING_PROMPT`, strict `PLAN_SCHEMA` | Local validation, then dependency-ordered execution |
| Action extraction | `src/llm/task_prompts.py`: `ACTION_EXTRACTION_PROMPT`, composed into the planner prompt | Validated specialty/date/time/doctor preferences → slot discovery → user confirmation |
| Memory context | `src/llm/conversation_context.py`: `PROMPT`, structured question/clarification output | Standalone question → reference retrieval, or clarification |
| Reference QA | `src/llm/task_prompts.py`: `RAG_SYSTEM_PROMPT`, `RAG_USER_PROMPT` | Evidence-grounded answer with returned source documents |
| Clinical history | `src/llm/history_summary.py`: `SUMMARY_PROMPT`, source-specific schema and renderer | Validated source-attributed summary → patient FAISS indexing |
| Live medical search | `src/llm/medical_search_summary.py`: `PROMPT`, source-specific schema and renderer | Cited PubMed/WHO evidence summary |
| Final operational response | `src/llm/task_prompts.py`: `FINAL_SUMMARY_PROMPT` | Actual tool outcomes, pending confirmations and failures |

The reference QA system prompt separates application instructions from untrusted
question/document text. It requires evidence-only answers, uncertainty and conflict
preservation, and explicit acknowledgement of insufficient evidence. It forbids
invented sources, personal diagnoses, treatment changes, action-completion claims,
and unsupported currentness claims about the local index. If retrieval returns no
source documents, the application replaces any generated answer with an explicit
insufficient-reference response. Prompts guide model behavior; they do not prove
clinical accuracy or eliminate prompt-injection risk.

Action extraction remains part of the structured planner call rather than a second
independent model call. Existing authorization, schema validation, dependency checks,
and booking confirmation enforce the action boundary outside the prompt. Earlier
planning/history/memory implementations already supplied custom prompts; this change
closes the remaining default-reference-QA gap and documents their composition.

`tests/test_task_prompt_chaining.py` uses the actual LangChain retrieval/combine chain
with an offline fake chat model to verify that retrieved context and questions reach
the custom prompt, memory resolution feeds the same prompt, sources survive chaining,
empty retrieval is handled, and action parameters pass local validation. No paid
model call is needed for these integration tests.

### PDF upload and ingestion — notebook Steps 3–8

Sign in as a **doctor** or **attendant** and expand **Reference documents — upload
and index**. Select one or more general medical-reference PDFs, then click **Index
uploaded PDFs**. Each file reports success, duplicate skip, or failure. The panel
shows the indexed-document inventory and a **Preview reference search** field.
Indexed documents are immediately available to the existing medical RAG workflow.

This library is shared across users and all patients. Every authenticated attendant
and doctor can upload general guidelines, educational material and other medical
references; no patient assignment is required for this shared library. Patient notes
and patient-specific documents belong in the medical-record workflow instead.
The upload control is not exposed to patient accounts. The CLI is for trusted local
operators. Neither route accepts uploaded serialized FAISS indexes or pickle files.

From the project root, the same workflow is available without Streamlit:

```bash
# Ingest a PDF or all top-level PDFs from a directory (appends to the shared index)
.venv/bin/python -m scripts.ingest_documents reference/healthcare_project_brd.pdf
.venv/bin/python -m scripts.ingest_documents data/raw/

# Inspect the inventory or run a semantic search
.venv/bin/python -m scripts.ingest_documents --list
.venv/bin/python -m scripts.ingest_documents --search "kidney disease treatment"

# Build a separate index with custom chunking
.venv/bin/python -m scripts.ingest_documents /path/to/guideline.pdf --index-path /tmp/reference-demo --chunk-size 500 --chunk-overlap 50
```

Use actual clinical reference PDFs for medical answers; the BRD command above is only
an ingestion demonstration, not a source of medical guidance. `--index-path` defaults
to `FAISS_INDEX_PATH`. To query an alternate index from the application, configure the
same path there. The command reports JSON results and exits nonzero if any file or
search fails; successful files remain indexed when another file fails.

Pipeline: `PDFLoader.load_pages` → existing `TextProcessor` recursive chunking →
`EmbeddingManager` OpenAI embeddings in batches of 32 → `FAISSStore` persistence.
It uses `OPENAI_API_KEY`, the existing `EMBEDDING_MODEL` (`text-embedding-ada-002`),
`CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MAX_DOCUMENTS`. Extracted text is sent to OpenAI;
raw PDF bytes are not retained by the ingestion service. File content hashes prevent
repeat embedding/indexing, including renamed duplicates. Different file contents
are appended as separate documents even if their filenames match.

Limits: 20 MB, 200 pages, one million extracted characters, and 5,000 chunks per PDF.
Chunk size must be 100–4,000 characters and overlap smaller than the chunk size.
Encrypted, corrupt, empty and textless/scanned PDFs produce actionable errors before
embedding. Pages without text are skipped and counted; OCR is not included. These
limits bound accepted inputs but do not sandbox the PDF parser or guarantee peak
memory usage for unusually compressed PDFs.

Source filename, 1-based page number, chunk index and document hash survive retrieval.
The manifest records chunk settings and ingestion time. Native serialized FAISS bytes,
JSON documents and the manifest are committed together in
`FAISS_INDEX_PATH/references.sqlite`; SQLite serializes ingestion writers, and failure
rolls back the current document. Patient summary vectors remain in their separate
patient database table. Search reloads committed state, so newly indexed documents
are visible without restarting the app.

**Legacy index format:** the new workflow does not deserialize old
`index.faiss`/`index.pkl` pairs. Re-ingest their source PDFs into the new format;
legacy files are left untouched. If the embedding model changes, use a separate
index directory or restore the original model. Identical-file uploads are skipped,
so use a separate index directory to rebuild with different chunk settings.

Implementation: `src/data_processing/ingestion.py`, `scripts/ingest_documents.py`,
`app/document_ingestion_view.py`, and the existing PDF/embedding/FAISS helpers.
Offline tests use real PDF parsing and FAISS storage/search with deterministic
embeddings, plus the CLI and a Streamlit UI double.

### Source evidence in reference answers

`RAGChain` explicitly enables `return_source_documents=True`, including the
history-aware question path. The executor preserves those documents and now appends
an application-generated **Retrieved reference documents** list to the final answer.
Labels such as `[ref:1] guideline.pdf — page 3` match the **Source documents** panel,
which displays the retrieved excerpts as plain text. Missing metadata is labelled
unknown rather than invented. Page numbers from the ingestion workflow are 1-based.

The final operational stage preserves the medical-reference answer instead of asking
another LLM to rewrite it without the retrieved excerpts. These reference labels
identify retrieved evidence, not validated claim-by-claim citations or proof that
all claims are supported. No documents means no fabricated reference list; the RAG
chain already returns its insufficient-evidence response for empty retrieval.
Permission revocation removes contextual answers, sources and reference metadata.

This supports notebook Steps 9/14 (answering from retrieved context) and inspection
for Step 16's faithfulness criteria. It does not implement automated faithfulness
scoring. Clinical-history typed citations and live PubMed/WHO citations remain
separate, unchanged evidence paths. The stored conversational answer retains its
reference labels, but historical conversation storage does not archive full reference
excerpts; the source panel displays the latest request's evidence.

### Model evaluation — BRD §6 and notebook Step 16

Run the synthetic controlled-evidence benchmark from the project root:

```bash
.venv/bin/python -m scripts.evaluate_models
# Optional models, dataset and output file
.venv/bin/python -m scripts.evaluate_models --judge-model gpt-4o-mini --output reports/evaluation/latest.json
```

This command makes paid OpenAI generation and grading requests using `OPENAI_API_KEY`.
The bundled `evaluation/healthcare_cases.json` contains nine author-written synthetic
cases with reference answers and case-specific rubrics: three reference QA, three
clinical-history summaries, and three publication-summary cases. It covers supported
answers, empty evidence, embedded commands, unconfirmed patient reports, expired
prescriptions, active alerts, and inconclusive/conflicting study results. Fictional
medicines/conditions/publications are test data, not clinical guidance. No patient DB
or public production index is read or modified during evaluation.

`src/evaluation/runner.py` provides reference-based LLM judging equivalent in purpose
to QAEvalChain. Generation uses the actual application reference prompts and production
history/search summary methods and renderers. Evidence is fixed by the dataset:
this evaluates controlled-evidence generation, not FAISS retrieval quality, internet
search recall, planner accuracy, appointment success, or the whole application.
Reference generation uses `LLM_MODEL` and `LLM_TEMPERATURE`; clinical/search summaries
use `PLANNER_MODEL`. The default judge is `PLANNER_MODEL`, with temperature zero.
`--answer-model`, `--summary-model`, and `--judge-model` support comparisons.

Metrics:

| Metric | Definition |
| --- | --- |
| Accuracy | Judge score 0/0.5/1 for correctness and coverage against the reference answer/rubric |
| Relevance | Judge score 0/0.5/1 for addressing the actual question |
| Faithfulness | Judge score 0/0.5/1 for grounding factual claims in the supplied evidence |
| Hallucination | Fraction of graded answers containing at least one unsupported/contradictory claim; lower is better |
| Latency | Mean and nearest-rank p95 successful generation time, excluding judge time; individual judge/total times are recorded |
| Completion | Evaluated count / attempted count, with generation and judge failures explicitly distinguished |

Scores average only successfully graded cases; failed cases are never silently scored
as correct or hallucination-free. No graded cases produces null quality scores.
Two separate judge controls use known-supported and deliberately fabricated answers;
they are not included in model-quality averages. Invalid or inconsistent judge output
is a grading failure. The CLI exits nonzero for generation/grading failures or failed
judge controls; exit zero means completion, not a clinical-quality certification.

The atomic JSON report contains model names, generation settings, dataset version/hash,
prompt hash, timestamps, per-case answers/evidence/reference answers, grading rationales,
unsupported claims, latency, aggregate scores and scores by task. Deterministic
empty-evidence abstentions are labelled separately from LLM generations. Doctors and
attendants can view it in **Model evaluation and tool outcomes**, alongside observed
tool status counts and mean durations from execution logs. Pending booking confirmation
is shown separately from success. The dashboard does not trigger paid API calls.

A live baseline is saved in `reports/evaluation/latest.json`. Treat the measured scores
as a small synthetic regression baseline: the dataset is not independently clinician
reviewed, most cases are deliberately simple, and the judge may share a model family
with the generator. Rubric averages are not population accuracy estimates. Judge
controls check basic discrimination but do not establish grader reliability. Broader
representative data, human clinical review and separate retrieval/end-to-end evaluations
are needed before drawing deployment conclusions. Repeated runs can vary.

### Performance analysis dashboard — BRD §§6–7

Doctors and attendants can expand **Performance analysis dashboard** and choose an
inclusive UTC date range (default: last 30 days). It displays:

- Confirmed booking count, unique logged booking requests, success rate, and p95 latency.
- Booking outcome bars, daily confirmation/attempt counts and daily success-rate trends.
- Per-tool success rates, outcome distributions, and mean/p95 latency charts and tables.
- A JSON download of the operational aggregates, with no patient identifiers or clinical text.

**Model evaluation and tool outcomes** now includes charts for accuracy, relevance,
faithfulness, hallucination rates, generation latency and evaluation completion/failures
by task, drawn from the saved measured benchmark. These are labelled synthetic
benchmark scores, independent of the operational date filter; they are not ratings
of production conversations. Hover tooltips expose chart values.

Booking success rate is confirmed / unique logged confirmation requests for the
selected period. Discovery proposals are not booking attempts. Requests are deduplicated
by request key, subject and requester where available. The earliest confirmed outcome
wins; if none is confirmed, the latest non-success outcome is used. Deduplication occurs
before date filtering so repeated confirmations do not inflate a later period. A
successful idempotent retry can resolve an earlier uncertain remote outcome. A proposal
event count describes what discovery returned, not the number of currently unconfirmed
appointments. With no logged attempts, the success rate is N/A, never 100%.

Booking telemetry now records every call, including calls without a supplied request
ID, permission denials, invalid inputs, unavailable slots and exceptions. A remote
transport failure is `outcome_unknown`, since a booking may have reached the remote
server; retrying the same idempotency key can establish its outcome. Unknown outcomes
remain in the attempt denominator and are shown separately. A telemetry write failure
does not misreport an already-confirmed booking as failed. No retrospective events
are fabricated: older exceptions that were never logged remain outside the available
metrics. Metrics describe logged requests, not appointment attendance or cancellation
rates. Tool rates count completed execution events (success / all outcomes), and
missing/negative durations are excluded from latency statistics.

Implementation: `src/repositories/performance_repository.py`,
`app/performance_view.py`, and the existing evaluation panel/booking service.
Tests cover denominators, date windows, retry deduplication, unknown-outcome resolution,
empty data, exception logging, role gates and chart specifications. The visual tests
use a Streamlit double rather than a browser.

### Live appointment tracking — BRD §7

Patient appointment lists and doctor daily schedules now refresh automatically every
five seconds while **Appointment View** is active. Each panel shows a last-refreshed
UTC timestamp and a **Refresh appointments now** button. The doctor date selection
and selected dependent determine which records are refreshed. Changes made through
another session using the same SQLite database appear on the next successful poll,
including new bookings and changed statuses.

Only the read-only panel uses Streamlit's timed fragment; booking forms, calendar
editing, chat planning and LLM requests are outside it. Timer ticks do not submit
forms, book appointments, generate request keys, or consume model tokens. Profile
activation and patient/dependent permissions are checked on refresh. Revoked access,
logout, navigation away or query failure stops that render from showing old rows.
Viewing a dependent's appointments requires `view_appointments`; it does not require
the separate permission to book an appointment for them.

Configuration: `APPOINTMENT_REFRESH_SECONDS=5` in `.env` (allowed range 2–60 seconds).
Restart the app after changing this setting. This is near-real-time polling, not
instant server push; browser/session activity, network latency and query failures can
delay refresh. All app instances must use the same database. If using the optional
Schedule API, it must share that appointment database with the app; this feature does
not synchronize independent remote databases. Suggested booking slots remain proposals
and are revalidated at confirmation; the timer refreshes the tracking tables.

This feature upgrades the Streamlit pin from 1.31.1 to 1.55.0 for `st.fragment` support.
The new version is installed in this workspace's virtual environment. For another
existing installation, update Streamlit and restart the app:

```bash
.venv/bin/python -m pip install streamlit==1.55.0
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Tests exercise repeated ticks, separate writer connections, patient/doctor scoping,
permission revocation, navigation/logout, transient errors and preservation of form
state. Streamlit AppTest also renders the real decorated panels, observes separate
connection updates and clears data when accounts are deactivated. AppTest renders
are driven explicitly; these tests do not time a browser's automatic timer.

### Memory traces, planning breakdowns and scenario testing — BRD §8

Patients can expand **Memory traces and planning breakdowns** to inspect saved requests
for their selected patient (self or an authorized dependent). Each request shows the
explicit task query, tool, specialty/appointment preferences, prerequisites, execution
status, duration, and blocking dependency IDs. A dependency-edge table makes the order
of subgoals inspectable. These are observable tasks and outcomes, not hidden model
reasoning.

Memory traces capture actual conversation lookups at planning, medical-question context
and explicit recall stages. They identify the selected turn IDs/timestamps, selection
reason (recent continuity, keyword match or recent fallback), truncation flags and
question/answer character counts. Empty/denied lookups and context discarded because
a plan changed subjects are labelled. “Supplied” means passed to that stage, not proof
that the model relied on a particular turn. Users can reveal the referenced conversation
excerpts; each lookup rechecks access. Deleted memory is reported as unavailable, not
reconstructed from copied trace text.

Traces persist in the existing SQLite database's `request_traces` table, isolated by
requester and subject with `view_medical` checks. Task queries may contain patient
information, so detailed plans are stored here rather than in general event logs.
Operational `agent_events` retain their redacted plan metadata. Trace records contain
memory IDs and execution metadata, not duplicated conversation answers or generated
clinical summaries. **Delete saved traces for selected patient** removes those traces;
conversation deletion and trace deletion are separate controls. Historical requests
are not backfilled with invented memory provenance. Save failures are shown explicitly.

All signed-in roles can open **Scenario testing sandbox**. Choose a scenario, edit its
fictional request and click **Run sandbox scenario**. Available presets exercise father
appointment discovery, denied father-history access, a remembered follow-up and explicit
conversation recall. The sandbox creates and destroys a temporary synthetic database;
it overrides any configured remote schedule backend with its local synthetic calendar.
The OpenAI planner and real executor run, but reference answers and final summarization
are fixtures and external medical searches return no evidence. No production patient
records, conversations, appointments or reference indexes are changed. Scenario text
is sent to OpenAI; this uses API credits. Results disclose fixture boundaries and check
expected subgoals, expected permission/confirmation outcomes, and absence of booking writes.
Edited prompts are still compared against the selected preset's expectations.

The same scenarios can be run from the CLI:

```bash
.venv/bin/python -m scripts.run_scenarios
.venv/bin/python -m scripts.run_scenarios --scenario "Remembered follow-up"
```

Live planning results are written to `reports/scenarios/latest.json`; generation
failures and failed structural checks cause a nonzero exit. This is a planning and
orchestration exercise, not clinical-response evaluation. The initial live run is
retained separately for comparison; stochastic model failures should remain visible.
Tests cover memory selection/provenance, persistent scope checks, deleted memory,
blocked dependencies, sandbox isolation, and the real Streamlit trace panel.

Live scenario validation note: the first and full rerun reports contain rejected plans
and an API failure; these were preserved rather than replaced with claimed successes.
Separate targeted runs of appointment discovery and remembered follow-up passed and
are saved alongside the full report. Live planner reliability is therefore not claimed
as 100%. The CLI now records explicit structured planner attempts for synthetic
scenarios to help diagnose invalid outputs; it does not record private chain-of-thought.

Patient document uploads: Doctors and attendants can use **Patient documents — upload and download** to select a patient and save original PDFs (up to 20 MB / 200 pages) to the patient_documents table in SQLite. Attendants can access assigned patients only. Saved files can be downloaded from the selected patient's document list; duplicate PDFs are detected per patient. These files are stored as patient attachments and are not indexed into shared reference search. The separate Reference documents section remains available for general reference material.
