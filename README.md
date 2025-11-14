# SimpleAgenticRAGwithLangGraph

A lightweight Retrieval-Augmented Generation (RAG) system built using **LangGraph** for accurate and tool-aware document question answering.

This project demonstrates how to build a modular RAG pipeline with:
- PDF / document ingestion  
- Intelligent chunking  
- Vector retrieval  
- Context-aware LLM agent  
- Page-level fetching for complete information  
- Calculator tools for numerical reasoning  
- Conversation history windowing  

---

## 🚀 Features

- **LangGraph Agent** with multi-step tool orchestration  
- `context_retriever_tool` for fetching relevant text chunks  
- `page_retriever_tool` for retrieving full document pages  
- **Calculator tools** (`add`, `sub`, `mul`, `div`) for strict numeric reasoning  
- **Automatic page completion logic**  
- **Metadata-based chunking** (retains page numbers)  
- **Conversation history** capped to last 30 messages  
- Fully modular and easy to extend


---

## ▶️ How to Run

Follow these steps to set up the environment, prepare the vector database, and run the Meta 10-K agent.

### **1. Create a virtual environment using `uv`**
```bash
uv venv
```
### 2. Activate the virtual environment
```bash
source .venv/bin/activate
```

### 3. Install required dependencies
```bash
uv pip install -r requirements.txt
```

4. Prepare the vector database

Run the notebook that processes the PDF, chunks it, and builds the vector store:
```bash
jupyter notebook prepare_context.ipynb
```

5. Run the Meta 10-K RAG Agent
```bash
python meta_10k_agent.py
```



