# Modular RAG System with Llama 3.2 & LangGraph
A CPU-optimized Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph, designed to run efficiently on regular computers without requiring GPU acceleration.

## Features

- Modular Design: Separate components for document loading, embedding, retrieval, generation, and caching
- CPU Optimization: Designed to run efficiently on CPU environments
- Korean Language Support: Works with Korean language models and documents
- Vector Database: Persistent storage of document embeddings for faster processing
- Query Caching: Stores previous query results to improve response time
- Memory Management: Optimized memory usage with garbage collection and monitoring
- Visualization: Graph visualization of the RAG pipeline workflow


## Project Structure
```
llama_modular_rag/
├── config.py               # Environment settings and constants
├── data_loader.py          # PDF loading and vector store creation
├── embeddings.py           # Embedding model setup
├── caching.py              # Caching mechanism
├── llm_setup.py            # LLM model setup
├── query_processing.py     # Query processing
├── retrieval.py            # Document retrieval and filtering
├── generation.py           # Answer generation
├── state.py                # State management
├── graph_builder.py        # Graph configuration
└── main.py                 # Main execution file
```

## Installation


### 1.Clone the repository:
```
bash
 - git clone https://github.com/yourusername/llama-modular-rag.git
 - cd llama-modular-rag
```

### 2.Create a conda environment using the provided environment file:
```
bash
 - conda env create -f environment.yaml
 - conda activate rag-env
```

### 3.Or install required packages using pip:
```
bash
 - pip install -r requirements.txt
```

## Usage

### 1.Place your PDF documents in the project directory
### 2.Run the main script:

```
bash 
 - python llama_modular_rag/main.py
```

### 3.Enter the path to your PDF file and your query when prompted
### 4.View generated answers and supporting document excerpts

## Configuration
### Adjust settings in config.py to optimize for your specific hardware:
```
python
 # CPU thread settings - adjust based on your CPU
 - os.environ["OMP_NUM_THREADS"] = "8"       # Computation library threads
 - os.environ["MKL_NUM_THREADS"] = "8"       # Intel MKL library threads
```

## Models
### This project uses two main models:

- Llama 3.2 Korean GGACHI 1B Instruct: For text generation
- Ko-sRoBERTa-multitask: For embeddings

### You can replace these with other models by modifying the paths in config.py.
## Performance Optimization
### The system includes several optimizations for CPU environments:

- Reduced token generation limits
- Smaller document chunks
- Query result caching
- Memory management and monitoring
- Simplified workflow with minimal necessary steps

## Dependencies
### Main dependencies include:

- langchain & langgraph
- transformers
- sentence-transformers
- chromadb
- pypdf
- torch
- graphviz
- psutil

## License
- MIT License

## Acknowledgments
- This project uses LangChain and LangGraph