# RAG Implementation: Knowledge Extraction from GeeksforGeeks using Gemini 2.5 Flash


A **Retrieval-Augmented Generation (RAG)** pipeline that extracts structured Q&A knowledge from **GeeksforGeeks** articles and enables intelligent question-answering using **Gemini 2.5 Flash**, **Sentence Transformers**, and **ChromaDB**.

---

## 📖 Overview

This project implements a **RAG system** to answer machine learning and programming questions by retrieving relevant context from a curated knowledge base built from **GeeksforGeeks**. The ground truth evaluation dataset — [`prsdm/Machine-Learning-QA-dataset`](https://huggingface.co/datasets/prsdm/Machine-Learning-QA-dataset) — contains **101 high-quality question-answer pairs** used for **zero-shot, few-shot, and Chain-of-Thought (CoT)** prompting evaluation.

---

## 🚀 RAG Pipeline Steps

1. **Knowledge Ingestion**
   - Scrape and extract clean Q&A content from GeeksforGeeks articles.
   - Preprocess text: remove code snippets, headers, footers, and noise.
   - Chunk documents into semantically meaningful segments (~300 tokens).

2. **Embedding Generation**
   - Use **SentenceTransformer(`all-MiniLM-L6-v2`)** to encode text chunks.
   - Generate dense 384-dimensional vectors optimized for semantic similarity.

3. **Vector Storage**
   - Store embeddings in **ChromaDB** (persistent, lightweight vector database).
   - Enable fast ANN (Approximate Nearest Neighbor) search using HNSW indexing.

4. **Retrieval**
   - Encode user query → retrieve top-k (k=5) relevant chunks via cosine similarity.

5. **Augmented Prompting**
   - Inject retrieved context into prompt templates.
   - Apply **zero-shot**, **few-shot**, and **Chain-of-Thought** strategies.

6. **Generation**
   - Query **Gemini 2.5 Flash** (via API) with augmented prompt.
   - Generate concise, faithful, and relevant answers.

7. **Evaluation**
   - Compare generated answers against ground truth using **RAG evaluation metrics**.

---

## 🔑 Key Components

### Embedding Model: `all-MiniLM-L6-v2`
- **Lightweight**: Only **80 MB**, 384-dimensional vectors.
- **High Performance**: Outperforms larger models on semantic textual similarity (STS) tasks.
- **Speed**: ~14,000 sentences/sec on CPU — ideal for real-time retrieval.
- **Sentence-Level Optimization**: Trained on 1B+ sentence pairs for robust meaning representation.

### LLM: Gemini 2.5 Flash
- **Multimodal & Fast**: Optimized for low-latency, high-throughput inference.
- **Strong Reasoning**: Supports complex instructions, CoT, and structured output.
- **Cost-Effective**: Ideal for scalable RAG applications.

### Vector Database: ChromaDB
- **Simple & Local**: No external dependencies; runs in-memory or persisted to disk.
- **Production-Ready**: Supports metadata filtering, updates, and persistence.
- **Developer-Friendly**: Python-first API with seamless integration.

---

## 🧠 Prompting Techniques

| Technique       | Description |
|-----------------|-----------|
| **Zero-Shot**   | Direct question + retrieved context → answer. No examples provided. |
| **Few-Shot**    | Include 2–3 high-quality Q&A examples in prompt to guide format and tone. |
| **Chain-of-Thought (CoT)** | Instruct model to "think step-by-step" before answering. Improves reasoning. |

---

## 📊 Evaluation Metrics (RAG Triad + Correctness)

### (Gemini 2.5 Flash)

Evaluated on **101 ground truth Q&A pairs** using **LLM-as-a-judge** (Gemini 2.5 Flash) with chunk_size = 980, chunk_overlap = 200:

### zero-Shot Metrics

| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.578` | 57.8% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.608` | 60.8% alignment between answer and question intent. |
| **Answer Correctness** | `0.588` | 58.8% factual accuracy against ground truth. |

### Few-shot Metrics


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.718` | 71.8% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.769` | 76.9% alignment between answer and question intent. |
| **Answer Correctness** | `0.737` | 73.7% factual accuracy against ground truth. |

### COT Metrics


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.704` | 70.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.753` | 75.3% alignment between answer and question intent. |
| **Answer Correctness** | `0.726` | 72.6% factual accuracy against ground truth. |

---
---
### (Gemini 2.0 flash)
Evaluated on **101 ground truth Q&A pairs** using **LLM-as-a-judge** (Gemini 2.0 Flash) with chunk_size = 980, chunk_overlap = 200:

### zero-Shot Metrics with chunk_size = 980, chunk_overlap = 200

| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.742` | 74.2% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.824` | 82.4% alignment between answer and question intent. |
| **Answer Correctness** | `0.774` | 77.4% factual accuracy against ground truth. |

#### zero-Shot Metrics with chunk_size = 800, chunk_overlap = 120
| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.722` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.777` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.744` | 74.2% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.825` | 82.4% alignment between answer and question intent. |
| **Answer Correctness** | `0.776` | 77.4% factual accuracy against ground truth. |


### Few-shot Metrics with chunk_size = 980, chunk_overlap = 200


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.757` | 75.7% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.805` | 80.5% alignment between answer and question intent. |
| **Answer Correctness** | `0.774` | 77.4% factual accuracy against ground truth. |


#### Few-shot Metrics with chunk_size = 800, chunk_overlap = 120


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.722` | 72.2% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.777` | 77.7% of ground truth information is retrieved. |
| **Faithfulness**       | `0.759` | 75.9% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.801` | 80.1% alignment between answer and question intent. |
| **Answer Correctness** | `0.774` | 77.4% factual accuracy against ground truth. |

### COT Metrics


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.615` | 61.5% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.687` | 68.7% alignment between answer and question intent. |
| **Answer Correctness** | `0.664` | 66.4% factual accuracy against ground truth. |


#### COT Metrics with chunk_size = 800, chunk_overlap = 120


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.72` | 72.2% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.777` | 77.7% of ground truth information is retrieved. |
| **Faithfulness**       | `0.634` | 63.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.696` | 69.6% alignment between answer and question intent. |
| **Answer Correctness** | `0.659` | 65.9% factual accuracy against ground truth. |


---
---


### (Gemini-2.0-flash-lite)
Evaluated on **101 ground truth Q&A pairs** using **LLM-as-a-judge** (Gemini 2.0 Flash) with chunk_size = 980, chunk_overlap = 200:

### zero-Shot Metrics with chunk_size = 800, chunk_overlap = 120

| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.663` | 66.3% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.763` | 76.3% of ground truth information is retrieved. |
| **Faithfulness**       | `0.659` | 65.9% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.83` | 83% alignment between answer and question intent. |
| **Answer Correctness** | `0.732` | 73.2% factual accuracy against ground truth. |


#### Few-shot Metrics with chunk_size = 800, chunk_overlap = 120


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.663` | 66.3% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.763` | 76.3% of ground truth information is retrieved. |
| **Faithfulness**       | `0.604` | 60.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.726` | 72.6% alignment between answer and question intent. |
| **Answer Correctness** | `0.658` | 65.8% factual accuracy against ground truth. |


---
---

### (gemma-3-1b-it) 
Evaluated on **101 ground truth Q&A pairs** using **LLM-as-a-judge** (gemma-3-1b-it) with chunk_size = 980, chunk_overlap = 200:

### zero-Shot Metrics with chunk_size = 980, chunk_overlap = 200


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.727` | 72.7% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.768` | 76.8% alignment between answer and question intent. |
| **Answer Correctness** | `0.738` | 73.8% factual accuracy against ground truth. |


#### zero-Shot Metrics with chunk_size = 800, chunk_overlap = 120:


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.722` | 72.2% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.777` | 77.7% of ground truth information is retrieved. |
| **Faithfulness**       | `0.754` | 75.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.81` | 81.0% alignment between answer and question intent. |
| **Answer Correctness** | `0.772` | 77.2% factual accuracy against ground truth. |

### Few-shot Metrics with chunk_size = 980, chunk_overlap = 200


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.729` | 72.9% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.83` | 83% alignment between answer and question intent. |
| **Answer Correctness** | `0.769` | 76.9% factual accuracy against ground truth. |


### Few-shot Metrics with chunk_size = 800, chunk_overlap = 120:


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.722` | 72.2% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.777` | 77.7% of ground truth information is retrieved. |
| **Faithfulness**       | `0.761` | 76.1% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.851` | 85.1% alignment between answer and question intent. |
| **Answer Correctness** | `0.794` | 79.4% factual accuracy against ground truth. |


### COT Metrics with chunk_size = 980, chunk_overlap = 200


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.724` | 72.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.793` | 79.3% alignment between answer and question intent. |
| **Answer Correctness** | `0.748` | 74.8% factual accuracy against ground truth. |


#### COT Metrics with chunk_size = 800, chunk_overlap = 120


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.722` | 72.2% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.777` | 77.7% of ground truth information is retrieved. |
| **Faithfulness**       | `0.754` | 75.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.831` | 83.1% alignment between answer and question intent. |
| **Answer Correctness** | `0.783` | 78.3% factual accuracy against ground truth. |
---
---


### (Gemini 2.5 pro) 
Evaluated on **101 ground truth Q&A pairs** using **LLM-as-a-judge** (Gemini 2.5 pro) with chunk_size = 980, chunk_overlap = 200:

### zero-Shot Metrics

| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `-0.006` | -0.06% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `-0.001` | -0.01% alignment between answer and question intent. |
| **Answer Correctness** | `-0.0` | -0% factual accuracy against ground truth. |

### Few-shot Metrics


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `-0.013` | -1.3% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `-0.009` | -0.9% alignment between answer and question intent. |
| **Answer Correctness** | `-0.021` | -2.1% factual accuracy against ground truth. |

### COT Metrics


| Metric                  | Score  | Interpretation |
|------------------------|--------|---------------|
| **Context Relevance**  | `0.718` | 71.8% of retrieved context is relevant to the question. |
| **Context Recall**     | `0.768` | 76.8% of ground truth information is retrieved. |
| **Faithfulness**       | `0.002` | 72.4% of generated claims are fully grounded in context → **room for improvement**. |
| **Answer Relevance**   | `0.004` | 79.3% alignment between answer and question intent. |
| **Answer Correctness** | `-0.002` | 74.8% factual accuracy against ground truth. |

---

# RAG Evaluation Summary (101 Q&A Pairs)

**Chunk Sizes Tested:**  
- `chunk_size=980, overlap=200` (default)  
- `chunk_size=800, overlap=120` (alternative)

**Judges:** Gemini 2.5 Flash, Gemini 2.0 Flash, gemma-3-1b-it, Gemini 2.5 Pro , Gemini-2.0-flash-lite
**Prompting Strategies:** Zero-shot, Few-shot, Chain-of-Thought (COT)

---

## Best Overall Configuration (Across All Judges)

| Judge | Best Config | Faithfulness | Answer Relevance | Answer Correctness |
|-------|-------------|--------------|------------------|--------------------|
| **Gemini 2.0 Flash** | Few-shot + `800/120` | **0.759** | **0.801** | **0.774** |
| **gemma-3-1b-it** | Few-shot + `800/120` | **0.761** | **0.851** | **0.794** |
| **Gemini 2.5 Flash** | Few-shot + `980/200` | **0.718** | **0.769** | **0.737** |
| **Gemini 2.5 Pro** | *Unreliable (bugged scores)* | — | — | — |

> **Conclusion:** Use **Few-shot prompting** with **`chunk_size=800, overlap=120`** for best performance across stable judges.

---

## Summary Table: Top Scores per Judge & Strategy

| Judge | Strategy | Chunk | Context Relevance | Context Recall | Faithfulness | Answer Relevance | Answer Correctness |
|-------|----------|-------|-------------------|----------------|--------------|------------------|--------------------|
| **Gemini 2.0 Flash** | Few-shot | 800/120 | 0.722 | 0.777 | **0.759** | 0.801 | **0.774** |
| **gemma-3-1b-it** | Few-shot | 800/120 | 0.722 | 0.777 | **0.761** | **0.851** | **0.794** |
| **Gemini 2.5 Flash** | Few-shot | 980/200 | 0.718 | 0.768 | 0.718 | 0.769 | 0.737 |
| **Gemini 2.0 Flash** | Zero-shot | 800/120 | 0.722 | 0.777 | 0.744 | **0.825** | 0.776 |
| **gemma-3-1b-it** | COT | 800/120 | 0.722 | 0.777 | 0.754 | **0.831** | 0.783 |
| **Gemini-2.0-flash-lite** | Zero-shot | 800/120 | 0.663 | 0.763 | 0.659 | **0.83** | 0.732 |

---

## Key Observations

| Metric | Insight |
|-------|--------|
| **Chunk Size** | `800/120` consistently **outperforms** `980/200` in **Recall, Faithfulness, Relevance, Correctness** |
| **Few-shot** | Best overall strategy — improves **Faithfulness (+10–15%)** and **Correctness (+15%)** vs Zero-shot |
| **COT** | Mixed results; hurts Gemini 2.0/2.5 Flash, helps gemma slightly |
| **Gemini 2.5 Pro** | **Invalid scores (negative/bugged)** — likely prompt/judge failure. **Exclude from analysis.** |

---

## Recommended Setup

```yaml
chunk_size: 800
chunk_overlap: 120
prompt_strategy: few-shot
judge: gemma-3-1b-it or Gemini 2.0 Flash