# -*- coding: utf-8 -*-
"""
Building the Complete RAG Pipeline Demo
================================================

This demo teaches:
1. Complete RAG architecture: retrieve → inject → generate
2. LangChain components (retrievers, prompts, chains)
3. Anti-hallucination strategies
4. Building a production-ready Q&A system

LEARNING RESOURCES:
- RAG Paper (Lewis et al.): https://arxiv.org/abs/2005.11401
- LangChain Documentation: https://python.langchain.com/docs/get_started/introduction
- LCEL Guide: https://python.langchain.com/docs/expression_language/
- Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- Chroma Vector DB: https://docs.trychroma.com/
"""

import json
import os
import time
from operator import itemgetter  # Used in exercises/solutions for LCEL key extraction
# LangChain is a framework for building LLM applications
# Reference: https://python.langchain.com/docs/get_started/introduction
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI integrations
from langchain_chroma import Chroma  # Vector database for similarity search
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart text chunking
from langchain_core.documents import Document  # Document abstraction
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt templates
from langchain_core.messages import HumanMessage, AIMessage  # Chat history message types
from langchain_core.output_parsers import StrOutputParser  # Parse LLM output
from langchain_core.runnables import RunnablePassthrough  # Pass data through pipeline

# Load environment variables (API keys, model names)
from dotenv import load_dotenv
load_dotenv()

print("="*80)
print("MODULE 4: RAG PIPELINE")
print("="*80)
print("""
STRUCTURE:

  PART 1: Data Ingestion & Vector Store
  PART 2: Retriever Setup
  PART 3: Prompt Engineering (Anti-Hallucination)
  PART 4: Language Model
  PART 5: LCEL Chain Assembly
  PART 6: Testing the RAG System
  PART 7: Validation & Fallback
  PART 8: Conversation with History (Multi-Turn RAG)
  PART 9: Interactive Demo
""")

# ============================================================================
# PART 1: Data Ingestion & Vector Store Setup
# ============================================================================
print("\n" + "="*80)
print("PART 1: Data Ingestion Pipeline")
print("="*80)

# Load tickets
with open("C:\\Users\\Kaale\\Desktop\\Projects\\RAG_PROJECT\\SupportDesk-RAG-Workshop-master\\SupportDesk-RAG-Workshop-master\\data\\synthetic_tickets.json", 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

# Convert to LangChain Document objects
# Documents are the core abstraction in LangChain - they combine content with metadata
# Reference: https://python.langchain.com/docs/modules/data_connection/document_loaders/
documents = []
for ticket in tickets:
    # Create rich document with all context
    # TIP: Structure your content logically - LLMs understand formatted text better
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Date: {ticket['created_date']} to {ticket['resolved_date']}

Problem Description:
{ticket['description']}

Resolution:
{ticket['resolution']}
    """.strip()
    
    # Create Document with metadata
    # Metadata is crucial for filtering, citation, and source tracking
    # Best practice: Include all information you might want to filter or display later
    doc = Document(
        page_content=content,  # The actual text content
        metadata={  # Structured data about the document
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'source': f"Ticket {ticket['ticket_id']}"
        }
    )
    documents.append(doc)

print(f"✓ Created {len(documents)} documents with metadata")

# Initialize OpenAI embeddings
# Embeddings convert text into vectors for semantic search
# Reference: https://platform.openai.com/docs/guides/embeddings
print("\nInitializing OpenAI embedding model...")
embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)
print("✓ OpenAI embedding model ready")

# Build vector store using Chroma
# Chroma is an open-source vector database optimized for AI applications
# It stores embeddings and enables fast similarity search
# Reference: https://docs.trychroma.com/
print("\nBuilding Chroma vector store...")
# Always rebuild from scratch so repeated runs don't accumulate duplicate documents.
import shutil
persist_directory = "./rag_vectorstore"
if os.path.exists(persist_directory):
    try:
        shutil.rmtree(persist_directory)
    except PermissionError:
        # Windows may keep Chroma files locked briefly from a prior run.
        # Fall back to a unique directory so the demo can continue.
        persist_directory = f"./rag_vectorstore_{int(time.time())}"
        print(f"⚠ Vector store directory is locked; using {persist_directory} instead")
vector_store = Chroma.from_documents(
    documents=documents,  # Our support ticket documents
    embedding=embeddings,  # Embedding function to use
    collection_name="supportdesk_rag",  # Name for this collection
    persist_directory=persist_directory,  # Where to save the database
    collection_metadata={"hnsw:space": "cosine"}  # Use cosine distance for similarity search
)
print("✓ Vector store created and persisted")

# ============================================================================
# PART 2: Create Retriever
# ============================================================================
print("\n" + "="*80)
print("PART 2: Setting Up Retriever")
print("="*80)

# Create a retriever from the vector store
# Retrievers are the interface for querying the vector store
# Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/
retriever = vector_store.as_retriever(
    search_type="similarity",  # Use cosine similarity for ranking
    search_kwargs={"k": 5}  # Retrieve top-3 most similar documents
    # Other options:
    # - "mmr" (Maximal Marginal Relevance): Balances relevance with diversity
    # - "similarity_score_threshold": Only return docs above a score threshold
)


# Try MMR for diverse results
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

print("✓ Retriever configured:")
print(f"  - Search type: similarity")
print(f"  - Top-K results: 5")
print("\nTIP: k=3-5 is usually optimal. Too few → missing context, too many → noise")

# Test retriever
test_query = "Users can't log in after changing passwords"
print(f"\nTest query: '{test_query}'")
retrieved_docs = retriever.invoke(test_query)

print(f"\nRetrieved {len(retrieved_docs)} documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n#{i} - {doc.metadata['ticket_id']}: {doc.metadata['title']}")
    print(f"  Category: {doc.metadata['category']}")

# ============================================================================
# PART 3: Create Prompt Template with Anti-Hallucination Rules
# ============================================================================
print("\n" + "="*80)
print("PART 3: Prompt Engineering for RAG")
print("="*80)

# Define strict grounding prompt
# Prompt engineering is CRUCIAL for RAG - it tells the LLM how to use the context
# Reference: https://platform.openai.com/docs/guides/prompt-engineering
# Key principles:
# 1. Be explicit about using ONLY the provided context
# 2. Define what to do when information is missing
# 3. Request citations for transparency and verification
# 4. Set the role/persona for appropriate tone
#prompt_template = """You are SupportDesk AI, a technical support assistant that helps engineers troubleshoot issues using historical support ticket data.

prompt_template = """Answer the question using only the ticket context below. Cite ticket IDs.

CRITICAL RULES:
1. Answer using ONLY information from the provided context.
2. If the question is broad or underspecified, provide the best matching known issue(s) from context and state any assumptions.
3. If context is partially relevant, still provide the most likely troubleshooting guidance from relevant tickets.
4. If the answer is truly not present in context, say "I don't have enough information in the ticket history to answer that question."
5. DO NOT make up information or use external knowledge.
6. Always cite ticket IDs for every issue/resolution you mention.
7. If multiple tickets are relevant, summarize each briefly.

Context from support tickets:
{context}

Question: {question}

Helpful Answer (with ticket citations):"""

# Convert string template to ChatPromptTemplate
# This creates a reusable template with variable placeholders
# Reference: https://python.langchain.com/docs/modules/model_io/prompts/


PROMPT = ChatPromptTemplate.from_template(prompt_template)

citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

    Example format:
    "Database connection timeouts occur when the pool is undersized [TICK-002]. Increase max_connections [TICK-002]."

    Context:
    {context}

    Question: {question}

    Answer with inline citations:"""

CITATION_PROMPT = ChatPromptTemplate.from_template(citation_prompt)

print("✓ Prompt template created with anti-hallucination rules:")
print("\n" + "-"*80)
print(prompt_template)
print("-"*80)

# ============================================================================
# PART 4: Initialize LLM
# ============================================================================
print("\n" + "="*80)
print("PART 4: Initializing Language Model")
print("="*80)

# Check if OpenAI key is available
if os.getenv("OPENAI_API_KEY"):
    print("✓ OpenAI API key found")
    # Initialize ChatOpenAI for generation
    # Reference: https://python.langchain.com/docs/integrations/chat/openai
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
        temperature=0,  # Temperature controls randomness (0 = deterministic, 2 = very creative)
        # For RAG, use temperature=0 to ensure consistent, factual responses
        # Reference: https://platform.openai.com/docs/guides/text-generation/how-should-i-set-the-temperature-parameter
        timeout=120,  # Increase timeout for slower connections
        max_retries=3,  # Retry on transient failures
    )
    print(f"✓ Using {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")
else:
    print("⚠ OpenAI API key not found!")
    print("  Please set OPENAI_API_KEY environment variable")
    print("  Or use Ollama: ollama pull llama2")
    print("\nFor this demo, we'll show the prompt without generating answers.")
    llm = None

# ============================================================================
# PART 5: Build RAG Chain using LCEL (LangChain Expression Language)
# ============================================================================
print("\n" + "="*80)
print("PART 5: Assembling RAG Chain")
print("="*80)

# Helper function to format retrieved documents
# This concatenates all retrieved document contents into a single context string
def format_docs(docs):
    """
    Convert a list of LangChain Document objects into a single context string.

    Why this helper exists:
    - Retrievers return `List[Document]` objects.
    - Prompt templates expect plain strings for `{context}`.
    - Joining with separators keeps boundaries visible to the LLM.
    """
    # Keep document boundaries explicit so the model can attribute facts by chunk.
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

if llm:
    # Build RAG chain using LCEL (LangChain Expression Language)
    # LCEL allows you to chain components using the | operator (like Unix pipes)
    # Reference: https://python.langchain.com/docs/expression_language/
    #
    # This chain does:
    # 1. Takes a question (string input)
    # 2. Retriever gets relevant docs, format_docs combines them
    # 3. PROMPT fills in {context} and {question} variables
    # 4. LLM generates answer based on filled prompt
    # 5. StrOutputParser extracts the string response
    #
    # The dict {"context": ..., "question": ...} creates the input for the prompt
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT 
        | llm
        | StrOutputParser()
    )
    print("✓ RAG chain assembled:")
    print("  Retriever → Context Injection → LLM → Answer")
    print("\nThis is the complete RAG pipeline! Query in → Answer out")
else:
    qa_chain = None
    print("⚠ LLM not available, showing architecture only")

# ============================================================================
# PART 6: Test the RAG System
# ============================================================================
print("\n" + "="*80)
print("PART 6: Testing the RAG System")
print("="*80)

test_queries = [
    "How do I fix authentication failures after password reset?",
    "What causes database connection timeouts?",
    "Why are emails not being delivered?",
    "How do I make the perfect pizza?"  # Should refuse to answer!
]

for query in test_queries:
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    # Show retrieved context
    docs = retriever.invoke(query)
    print(f"\nRetrieved {len(docs)} relevant tickets:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.metadata['ticket_id']}: {doc.metadata['title']}")
    
    if qa_chain:
        # Generate answer
        print("\nGenerating answer...")
        result = qa_chain.invoke(query)
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(result)
        
        print("\n" + "-"*80)
        print("SOURCE DOCUMENTS:")
        print("-"*80)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata['source']}")
    else:
        print("\n(LLM not configured - would generate answer here)")

# ============================================================================
# PART 7: Validation & Fallback
# ============================================================================
print("\n" + "="*80)
print("PART 7: Enhanced RAG with Answer Validation")
print("="*80)

def rag_with_validation(query, retriever, llm, min_similarity_score=0.5):
    """
        RAG pipeline with additional validation and fallback.

        Validation rule in this demo:
        - Use relevance score as a confidence proxy.
        - If the best document's relevance score is too low (< threshold),
            return a safe fallback instead of forcing an answer.

        Note:
        - similarity_search_with_relevance_scores() returns scores in [0, 1].
        - Higher = more similar (derived from cosine similarity).
        - This keeps things consistent with the cosine similarity concept
          taught in earlier modules (cosine similarity: -1 to 1).
        - This is a simple, practical guardrail for anti-hallucination behavior.
    """
    # Retrieve documents with relevance scores (0 = least relevant, 1 = most relevant)
    # Uses similarity_search_with_relevance_scores instead of similarity_search_with_score
    # because the latter returns raw cosine *distance* (0–2, lower=better), which is
    # confusing when we've been teaching cosine *similarity* (-1 to 1, higher=better).
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=3)

    print(f"\nQuery: {query}")
    print(f"\nRelevance scores (cosine similarity: 0=no match, 1=identical):")
    for doc, score in docs_with_scores:
        print(f"  - {doc.metadata['ticket_id']}: {score:.4f}")

    # Use the best retrieved document as the confidence anchor.
    # If even the best match is weak, the whole answer should be treated as risky.
    best_score = docs_with_scores[0][1]

    # Relevance score: higher = more similar. Below 0.7 means the match is too weak to answer.
    if best_score < min_similarity_score:
        print(f"\n⚠ Best match relevance ({best_score:.4f}) is below threshold ({min_similarity_score}) — too dissimilar to answer confidently")
        return "I don't have enough relevant information in the ticket history to answer that question confidently."
    
    # If we pass the confidence gate, build context and ask the model normally.
    docs = [doc for doc, score in docs_with_scores]
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""{prompt_template.replace('{context}', context).replace('{question}', query)}"""
    
    if llm:
        # Use chat-model invocation directly and normalize return type to string.
        # `ChatOpenAI.invoke(...)` returns an AIMessage object in modern LangChain.
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    else:
        return "(LLM not configured)"

print("\nTesting validation logic:")
print("\n1. Relevant query (should answer):")
rag_with_validation(
    "How to fix database connection timeouts?",
    retriever,
    llm,
    min_similarity_score=0.5
)

print("\n2. Irrelevant query (should refuse):")
rag_with_validation(
    "What is the capital of France?",
    retriever,
    llm,
    min_similarity_score=0.5
)


def smart_rag(query, vector_store, llm, min_score_threshold=0.7):
    """
    RAG with confidence-based fallbacks.

    Strategy:
    - High confidence (small distance): answer normally with full context.
    - Medium confidence: return a cautious clarification-style response.
    - Low confidence: refuse with a safe fallback.

    This pattern prevents overconfident hallucinations on weak retrieval.
    """
    # Retrieve candidate evidence + numeric distances from the vector store.
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    if not docs_with_scores:
        return "No relevant tickets found.", "no_results"
    
    # Lower distance = better match (cosine distance: 0=identical, 1=orthogonal).
    # We use the top distance as a lightweight proxy for answer reliability.
    best_distance = docs_with_scores[0][1]
    
    print(f"  Best match distance: {best_distance:.4f}")
    
    if best_distance < 0.5:  # Very relevant
        # High-confidence path: construct full grounded prompt and answer directly.
        docs = [doc for doc, score in docs_with_scores]
        context = format_docs(docs)
        
        prompt = f"""Answer using only this context. Cite ticket IDs.

Context: {context}

Question: {query}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content, "high_confidence"
    
    elif best_distance < 1.0:  # Somewhat relevant
        # Medium-confidence path: avoid overclaiming, ask for confirmation/context.
        ticket_id = docs_with_scores[0][0].metadata['ticket_id']
        return f"Found possibly relevant ticket ({ticket_id}), but confidence is moderate. Would you like me to show details?", "medium_confidence"
    
    else:  # Not relevant
        # Low-confidence path: safe refusal to avoid unsupported generation.
        return "I don't have relevant ticket history for this question.", "low_confidence"

# Test with different queries
test_queries = [
    ("authentication problems", "high confidence expected"),
    ("system performance", "medium confidence expected"),
    ("how to bake cookies", "low confidence expected")
]

for query, note in test_queries:
    print(f"\nQuery: '{query}' ({note})")
    answer, confidence = smart_rag(query, vector_store, llm)
    print(f"  Confidence: {confidence}")
    print(f"  Answer: {answer[:150]}...")



# ============================================================================
# PART 8: Conversation with History (Multi-Turn RAG)
# ============================================================================
print("\n" + "="*80)
print("PART 8: Conversation with History (Multi-Turn RAG)")
print("="*80)
print("""
Problem with single-turn RAG:
  Turn 1: "How do I fix authentication failures?"  → good answer
  Turn 2: "How long did it take to resolve?"       → loses context! "it" = ???

Solution: Two-part fix:
  1. MessagesPlaceholder injects prior HumanMessage / AIMessage objects into the prompt
     so the LLM can understand references like "that issue" or "it".
  2. Query reformulation rewrites follow-up questions into standalone queries
     BEFORE retrieval, so the retriever searches for the right documents.
     e.g. "How do I fix it?" + history → "How do I fix authentication failures?"
""")

if llm:
    # ── Step 1: Query Reformulation ────────────────────────────────────
    # The retriever only sees the raw question string.
    # "What was the resolution for that ticket?" → retriever gets vague query.
    # Fix: Use the LLM to rewrite follow-ups into standalone queries first.
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and a follow-up question, rephrase the "
         "follow-up as a standalone question that includes all necessary "
         "context from the history. If the question is already standalone, "
         "return it unchanged."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    
    # Chain that rewrites the question: dict → standalone question string
    condense_chain = condense_prompt | llm | StrOutputParser()
    
    
    # ── Step 2: Conversation-aware prompt ──────────────────────────────
    # MessagesPlaceholder expands the history list into the prompt at call time
    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are SupportDesk AI. Answer using the ticket context below and the chat history.
    Use chat history to resolve references like "that issue" or "that ticket".
    For factual claims, prioritize the retrieved context.
    If information is not available in context or history, say "I don't have that information."
    Always cite ticket IDs when available.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def ask_with_history(question, history):
        """Ask a question with query reformulation and history tracking.
        
        On the first turn (empty history), skip the condense step and send
        the question directly to the retriever. On follow-up turns, rewrite
        the question into a standalone query so the retriever finds the
        right documents (e.g. "that ticket" → "TICK-001").
        """
        if not history:
            # First turn: no history to condense, retrieve directly
            standalone = question
        else:
            # Follow-up turn: rewrite using history context
            # e.g. "What was the resolution for that ticket?" → 
            #      "What was the resolution for ticket TICK-001?"
            standalone = condense_chain.invoke({
                "question": question, "chat_history": history
            })

        # Retrieve docs using the standalone query and generate the answer
        context = format_docs(retriever.invoke(standalone))
        answer = (conv_prompt  | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": history,
            "question": question,  # Original question, not the rewritten one
        })

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        return answer

    # ── Demonstrate a 3-turn conversation ──────────────────────────────────
    print("Multi-turn conversation demo:\n")
    history = []

    q1 = "How do I fix authentication failures after a password reset?"
    print(f"Turn 1 — User: {q1}")
    a1 = ask_with_history(q1, history)
    print(f"         Assistant: {a1[:300]}{'...' if len(a1) > 300 else ''}")
    print(f"         [chat_history now has {len(history)} messages]")

    q2 = "What was the ticket ID for that issue?"
    print(f"\nTurn 2 — User: {q2}")
    a2 = ask_with_history(q2, history)
    print(f"         Assistant: {a2[:300]}{'...' if len(a2) > 300 else ''}")
    print(f"         [chat_history now has {len(history)} messages]")

    q3 = "What was the resolution for that ticket?"
    print(f"\nTurn 3 — User: {q3}")
    a3 = ask_with_history(q3, history)
    print(f"         Assistant: {a3[:300]}{'...' if len(a3) > 300 else ''}")
    print(f"         [chat_history now has {len(history)} messages]")

    print(f"\n✓ History contains {len(history)} messages ({len(history)//2} complete turns)")
    print("TIP: In production, cap history to avoid token bloat:")
    print("       history = history[-6:]  # keep last 3 turns")

else:
    print("(LLM not configured — would run multi-turn conversation here)")
    print("\nKey pattern:\n")
    print("  history = []")
    print()
    print("  # Step 1: Rewrite follow-up into standalone query (skip if no history)")
    print("  if history:")
    print("      standalone = condense_chain.invoke({'question': q, 'chat_history': history})")
    print("  else:")
    print("      standalone = q")
    print()
    print("  # Step 2: Retrieve using standalone query, generate with original question")
    print("  context = format_docs(retriever.invoke(standalone))")
    print("  answer = (conv_prompt | llm | StrOutputParser()).invoke({")
    print("      'context': context, 'chat_history': history, 'question': q")
    print("  })")
    print()
    print("  history.append(HumanMessage(content=q))")
    print("  history.append(AIMessage(content=answer))")
    print()
    print("Query reformulation ensures the retriever gets meaningful queries")
    print("instead of vague pronouns like 'it' or 'that ticket'.")


#When you have multiple retrieved docs, how do you combine them to generate an answer?

##**Strategy 1 — STUFF (default):** All docs concatenated into one prompt, one LLM call.
# Compare different retrieval strategies using LCEL
strategies = {
    "stuff": "Concatenate all documents into context (fast, simple)",
    "map_reduce": "Process each doc separately, then combine (parallel)",
    "refine": "Iteratively refine answer with each doc (highest quality)"
}

test_query = "How do I fix database timeouts?"

# ── STUFF strategy (default) ──────────────────────────────────────────
# All retrieved docs are "stuffed" into a single {context} variable
# and sent to the LLM in ONE call.
print("\nSTUFF strategy:")
start = time.time()
stuff_prompt = ChatPromptTemplate.from_template(
    "Answer using the context:\n\nContext: {context}\n\nQuestion: {question}"
)
stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | stuff_prompt
    | llm
    | StrOutputParser()
)
result = stuff_chain.invoke(test_query)
print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {result[:150]}...")

# ── MAP_REDUCE strategy ───────────────────────────────────────────────
# Phase 1 (Map):  Each doc → individual LLM call → summary
# Phase 2 (Reduce): All summaries → one LLM call → final answer
print("\nMAP_REDUCE strategy:")
start = time.time()
docs = retriever.invoke(test_query)

# Map phase: extract key info from each document independently
individual_answers = []
for doc in docs:
    single_prompt = ChatPromptTemplate.from_template(
        "Extract key info about this issue:\n{doc}\n\nKey points:"
    )
    chain = single_prompt | llm | StrOutputParser()
    individual_answers.append(chain.invoke({"doc": doc.page_content}))

# Reduce phase: combine all per-doc summaries into a single coherent answer
combine_prompt = ChatPromptTemplate.from_template(
    "Combine these points to answer: {question}\n\nPoints:\n{summaries}"
)
combine_chain = combine_prompt | llm | StrOutputParser()
result = combine_chain.invoke({"question": test_query, "summaries": "\n".join(individual_answers)})
print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {result[:150]}...")

# ── REFINE strategy ───────────────────────────────────────────────────
# Process docs sequentially. Start with the first doc to produce a draft,
# then feed each subsequent doc + the current draft to the LLM to refine.
#
#   Doc1 → LLM → Draft answer
#   Doc2 + Draft → LLM → Refined answer
#   Doc3 + Refined → LLM → Final answer
#
# Each step can incorporate new info AND correct earlier mistakes.
print("\nREFINE strategy:")
start = time.time()
docs = retriever.invoke(test_query)

# Step 1: Generate initial draft from the first document
initial_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only this context.\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)
initial_chain = initial_prompt | llm | StrOutputParser()
current_answer = initial_chain.invoke({
    "context": docs[0].page_content,
    "question": test_query
})

# Steps 2..N: Refine the draft with each remaining document
refine_prompt = ChatPromptTemplate.from_template(
    "Here is an existing answer to the question:\n\n"
    "Existing answer: {existing_answer}\n\n"
    "Now consider this additional context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Refine the existing answer using the new context. "
    "If the new context isn't useful, return the existing answer unchanged.\n\n"
    "Refined answer:"
)
refine_chain = refine_prompt | llm | StrOutputParser()

for doc in docs[1:]:  # Iterate over remaining docs (skip the first)
    current_answer = refine_chain.invoke({
        "existing_answer": current_answer,
        "context": doc.page_content,
        "question": test_query
    })

print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {current_answer[:150]}...")

print("\n→ 'stuff': Fastest (1 LLM call), concatenates all docs into one prompt")
print("→ 'map_reduce': Parallel processing (N+1 calls), good for many docs")
print("→ 'refine': Iterative (N sequential calls), highest quality but slowest")

# Add Metadata Filtering (Medium)

print("\n" + "=" * 80)
print("EXERCISE 6: Add Metadata Filtering")
print("=" * 80)

query = "system problem"

# Without filter
print("\nWithout filter:")
docs = vector_store.similarity_search(query, k=3)
for doc in docs:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")

# With category filter
print("\nWith category='Authentication' filter:")
docs_filtered = vector_store.similarity_search(
    query, 
    k=3,
    filter={"category": "Authentication"}
)
for doc in docs_filtered:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")

print("\nWith category='Database' filter:")
docs_filtered = vector_store.similarity_search(
    query, 
    k=3,
    filter={"category": "Database"}
)
for doc in docs_filtered:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")


# Exercise 7: Add Streaming Responses

from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Create streaming LLM
streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Use a detailed prompt so the response is long enough to see streaming in action
prompt = ChatPromptTemplate.from_template("""Answer using the context. Provide a detailed, thorough response.
Include step-by-step troubleshooting instructions, root causes, and preventive measures.
Cite ticket IDs for every fact.

Context: {context}

Question: {question}

Detailed Answer:""")

streaming_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | streaming_llm
    | StrOutputParser()
)

query = "What causes database connection issues and how do I troubleshoot and prevent them?"
result = streaming_chain.invoke(query)  # Will print token by token!


#step that rewrites follow-up questionsinto standalone queries before they reach the retriever.


# Store chat history
chat_history = []

condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rephrase the "
     "follow-up as a standalone question that includes all necessary "
     "context from the history. If the question is already standalone, "
     "return it unchanged."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Chain that rewrites the question: dict → standalone question string
condense_chain = condense_prompt | llm | StrOutputParser()

# Create conversational prompt with history placeholder
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are SupportDesk AI. Answer using ONLY the ticket context below.
Always cite ticket IDs. If the answer isn't in the context, say "I don't have that information."

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# ── Step 2: Build the full chain ──────────────────────────────────────
# Flow:
#   Input: {"question": str, "chat_history": List[Message]}
#     │
#     ├─→ condense_chain  → standalone question (str)
#     │                        │
#     │                        ├─→ retriever → format_docs → context (str)
#     │                        │
#     ├─→ (passes through chat_history and question unchanged)
#     │
#     ▼
#   conv_prompt  → receives {context, chat_history, question}
#     │
#     ▼
#   LLM → StrOutputParser → answer (str)
conv_chain = (
    # First, rewrite the question into a standalone form
    RunnablePassthrough.assign(
        standalone=condense_chain
    )
    # Then, use the standalone question to retrieve relevant docs
    | RunnablePassthrough.assign(
        context=itemgetter("standalone") | retriever | format_docs
    )
    | conv_prompt
    | llm
    | StrOutputParser()
)

def ask_with_history(question, history):
    """
    Execute one conversational turn and persist memory.

    If chat_history is empty (first turn), skip the condense step and
    send the question directly to the retriever — no LLM rewrite needed.
    For follow-up turns, condense the question first so the retriever
    gets a standalone query instead of vague references like "it" or "that".
    """
    if not history:
        # First turn: no history to condense, retrieve directly with original question
        standalone = question
    else:
        # Follow-up turn: rewrite the question using history context
        # e.g. "How do I fix it?" → "How do I fix authentication failures?"
        standalone = condense_chain.invoke({"question": question, "chat_history": history})

    # Retrieve docs using the standalone query and generate the answer
    context = format_docs(retriever.invoke(standalone))
    answer = (conv_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "chat_history": history,
        "question": question,  # Show the original question in the prompt, not the rewritten one
    })

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer

# Simulate a 3-turn conversation
# Use specific queries that match ticket content for reliable retrieval
conversation = [
    "Why are users unable to log in after a password reset?",
    "How do I fix it?",          # "it" → login issue (remembered from turn 1)
    "What about database issues?"  # new topic — no cross-contamination from history
]

print("\nSimulated conversation:")
for user_msg in conversation:
    print(f"\nUser: {user_msg}")
    result = ask_with_history(user_msg, chat_history)
    print(f"Assistant: {result[:200]}...")


# ============================================================================
# Bonus: Hallucination Detection (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Hallucination Detection")
print("=" * 80)

def detect_hallucination(query, answer, source_documents, llm):
    """
    Use LLM-as-judge to check if an answer is grounded in retrieved sources.

    Practical purpose:
    - Adds a post-generation QA gate.
    - Helps flag unsupported claims before returning responses in production.
    """
    source_text = "\n\n".join([doc.page_content for doc in source_documents])
    
    detection_prompt = f"""You are a fact-checker. Determine if the answer is fully grounded in the source documents.

SOURCE DOCUMENTS:
{source_text}

ANSWER TO CHECK:
{answer}

Is every claim in the answer supported by the source documents?
Respond with:
- "GROUNDED" if all claims are supported
- "HALLUCINATION" if any claims are not in sources
- Brief explanation

Response:"""

    response = llm.invoke(detection_prompt)
    return response.content

# Test hallucination detection using the LCEL chain already set up above
test_query = "How do I fix authentication issues?"

# Get answer and source docs using LCEL (no deprecated RetrievalQA)
source_docs = retriever.invoke(test_query)
answer_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. Cite ticket IDs.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
answer_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm | StrOutputParser()
)
answer = answer_chain.invoke(test_query)

print(f"\nQuery: {test_query}")
print(f"Answer: {answer[:200]}...")
print(f"\nHallucination check:")
check_result = detect_hallucination(test_query, answer, source_docs, llm)
print(check_result)

# ============================================================================
# PART 9: Interactive Demo
# ============================================================================
print("\n" + "="*80)
print("PART 9: Interactive SupportDesk Assistant")
print("="*80)

if qa_chain:
    print("\nSupportDesk RAG Assistant Ready!")
    print("Ask questions about support ticket history.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_query = input("You: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        print("\nAssistant: ", end="")
        answer = qa_chain.invoke(user_query)
        print(answer)
        
        docs = retriever.invoke(user_query)
        print(f"\n📎 Sources: {', '.join([doc.metadata['ticket_id'] for doc in docs])}")
        print()
else:
    print("\n⚠ Interactive mode requires OpenAI API key")
    print("Set OPENAI_API_KEY to try the interactive assistant!")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. RAG pipeline: Retrieve → Inject Context → Generate")
print("2. Strict prompt engineering prevents hallucinations")
print("3. Always return source documents for verification")
print("4. Implement fallbacks for low-confidence matches")
print("5. Temperature=0 for deterministic, grounded answers")
print("\nNext: Evaluation & Metrics")
