🤖 RAG Engineering & Agentic RAG
A practical Retrieval-Augmented Generation (RAG) engineering project covering the complete journey from embeddings and semantic search to chunking, indexing, RAG pipelines, evaluation, and Agentic RAG.
This project demonstrates how modern RAG systems are built step by step using Python, OpenAI, ChromaDB, LlamaIndex, LangChain, and LLM-based evaluation.
🚀 Project Overview
The project is organized into six progressive modules:
1. 🔢 Embeddings & Similarity Search
2. ✂️ Chunking & Vector Stores
3. 🗂️ Indexing Strategies
4. 🔎 Complete RAG Pipeline
5. 📊 RAG Evaluation
6. 🤖 Agentic RAG
Each module builds on the previous one and gradually moves from basic semantic search toward an intelligent tool-using RAG agent.
📚 Modules
1️⃣ Embeddings & Similarity Search
Learn how text is converted into numerical vectors and how semantic similarity can be used to retrieve relevant information.
Key Concepts
- 🔢 Text embeddings
- 🧠 Semantic representation
- 📐 Vector similarity
- 🔍 Semantic search
- 📊 Cosine similarity
- 📏 Embedding dimensions
- 🔎 Top-K retrieval
Workflow
Text
 │
 ▼
OpenAI Embedding Model
 │
 ▼
Vector Representation
 │
 ▼
Similarity Calculation
 │
 ▼
Top-K Similar Results
2️⃣ Chunking & Vector Stores
Large documents cannot always be passed directly to an LLM.
This module demonstrates how documents are divided into smaller chunks and stored in a vector database for efficient retrieval.
Chunking Strategies
- 📏 Fixed-size chunking
- 🔄 Recursive chunking
- 🧠 Semantic chunking
- 📝 Markdown chunking
- 🌐 HTML chunking
Vector Store
The project uses ChromaDB for storing and retrieving vector embeddings.
Workflow
Documents
    │
    ▼
Document Chunking
    │
    ▼
Text Chunks
    │
    ▼
Embeddings
    │
    ▼
ChromaDB
    │
    ▼
Semantic Retrieval
3️⃣ Indexing Strategies
Different RAG applications require different indexing strategies.
This module explores multiple approaches using LlamaIndex.
Indexing Strategies
Strategy	Purpose
Vector Index	Semantic similarity search
Summary Index	Retrieve information from summaries
Tree Index	Hierarchical retrieval
Keyword Index	Keyword-based retrieval
Hybrid Index	Combine multiple retrieval approaches


4️⃣ Complete RAG Pipeline
This module implements a complete Retrieval-Augmented Generation pipeline.
User Query
     │
     ▼
Query Processing
     │
     ▼
Retriever
     │
     ▼
Relevant Documents
     │
     ▼
Context Injection
     │
     ▼
LLM
     │
     ▼
Generated Answer
Core Components
- 🔍 Retriever
- 📝 Prompt template
- 📚 Retrieved context
- 🤖 LLM
- 🔗 LangChain chain
- 🛡️ Anti-hallucination instructions
5️⃣ RAG Evaluation
Building a RAG system is not enough. We also need to measure whether the system retrieves the correct information and generates a useful answer.
Retrieval Evaluation
- Precision
- Recall
- F1 Score
Generation Evaluation
- Groundedness
- Completeness
- Answer quality
- LLM-as-a-Judge
                 RAG System
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
        Retrieval         Generation
        Evaluation        Evaluation
             │                │
             ▼                ▼
      Precision/Recall    Groundedness
           /F1            Completeness
             │                │
             └───────┬────────┘
                     ▼
              Evaluation Report
6️⃣ 🤖 Agentic RAG
The final module extends traditional RAG into an Agentic RAG system.
Instead of always following a fixed retrieval pipeline, an LLM-powered agent decides which tool should be used based on the user's query.
Features
- 🔍 Semantic search using RAG
- 🤖 LLM-powered agent with tool calling
- 🎯 Intelligent tool selection
- 🎫 Retrieve specific tickets by ID
- 📂 Search tickets by category
- 📊 Ticket database statistics
- 🔄 Multi-step agent reasoning
- 💬 Conversational memory
- ⚠️ Error handling
- 📈 Agent tool-selection evaluation
Tools
Tool	Purpose
SearchSimilarTickets	Semantic search for similar support issues
GetTicketByID	Retrieve details of a specific ticket
SearchByCategory	Find tickets belonging to a category
GetTicketStatistics	Get ticket database statistics


Architecture
                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │   OpenAI LLM    │
                    │     Agent       │
                    └────────┬────────┘
                             │
                             ▼
                       Tool Selection
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
  RAG Search          Ticket Lookup       Category / Stats
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                       Tool Results
                             │
                             ▼
                       OpenAI LLM
                             │
                             ▼
                       Final Response
Example Queries
How do I fix login issues?
Show ticket TICK-001
Show me all payment tickets
How many tickets are there?
How many critical tickets are there?
🧠 Traditional RAG vs Agentic RAG
Feature	Traditional RAG	Agentic RAG
Retrieval	Fixed	Agent decides
Tool selection	No	Yes
Multi-step reasoning	Limited	Supported
Multiple tools	Usually fixed pipeline	Dynamic
Conversational interaction	Possible	Supported
Complexity	Lower	Higher
Latency	Generally lower	Can be higher
Best for	Simple predictable Q&A	Complex workflows


🏗️ Complete Project Architecture
                RAG ENGINEERING JOURNEY
                         │
                         ▼
              ┌─────────────────────┐
              │ 1. Embeddings       │
              │ Semantic Search     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 2. Chunking         │
              │ Vector Stores       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 3. Indexing         │
              │ Retrieval Strategies│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 4. RAG Pipeline     │
              │ Retrieve + Generate │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 5. RAG Evaluation   │
              │ Measure Quality     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 6. Agentic RAG      │
              │ Tools + Reasoning   │
              └─────────────────────┘
🛠️ Technologies Used
- 🐍 Python
- OpenAI
- LangChain
- LlamaIndex
- ChromaDB
- Embeddings
- Vector Search
- Retrieval-Augmented Generation
- LLM-based Evaluation
- Agentic AI / Tool Calling
📁 Project Structure
RAG-Engineering-Lab/
│
├── modules/
│   ├── 1_embeddings/
│   ├── 2_chunking/
│   ├── 3_indexing/
│   ├── 4_rag_pipeline/
│   ├── 5_evaluation/
│   └── 6_agentic_rag/
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
⚙️ Installation
Clone the repository:
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
Create a virtual environment:
python -m venv .venv
Activate it on Windows:
.venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
🔑 Environment Configuration
Create a .env file:
OPENAI_API_KEY=your_openai_api_key
Never commit API keys or other secrets to GitHub.
▶️ Running the Modules
Each module can be executed independently.
python modules/1_embeddings/demo.py
python modules/2_chunking/demo.py
python modules/3_indexing/demo.py
python modules/4_rag_pipeline/demo.py
python modules/5_evaluation/demo.py
python modules/6_agentic_rag/demo.py
Adjust the paths if your local filenames differ.
🔐 Security
Recommended .gitignore:
.env
.venv/
__pycache__/
*.pyc
chroma_db/
.vscode/
.idea/
🎯 Learning Outcomes
After completing this project, you will understand:
- How embeddings represent text as vectors
- How semantic similarity search works
- How documents are chunked for RAG
- How vector stores support retrieval
- Different indexing strategies
- How to build a complete RAG pipeline
- How retrieved context is injected into an LLM
- How to evaluate retrieval quality
- How to evaluate generated answers
- How LLM-as-a-Judge evaluation works
- How agents select and call tools
- How Agentic RAG differs from traditional RAG
- How conversational memory can be incorporated
- How to evaluate agent tool selection
- How error handling can be implemented in an Agentic RAG system
🚀 Key Takeaway
The project demonstrates the evolution of a RAG application:
Embeddings
     ↓
Semantic Search
     ↓
Chunking
     ↓
Vector Stores
     ↓
Indexing
     ↓
RAG Pipeline
     ↓
RAG Evaluation
     ↓
Agentic RAG
     ↓
Tool-Using AI Assistant
The final system moves beyond simple retrieval and demonstrates how an LLM can reason about a user's request, select the appropriate tool, retrieve information, and generate a final response.
👨‍💻 Author
Kaaleedass Manoharan
Data & AI Engineering | RAG | Agentic AI | Python | Automation
⭐ Project Focus
RAG Engineering
        +
Agentic AI
        +
LLM Applications
        +
Evaluation
        +
Production-Oriented AI Systems
