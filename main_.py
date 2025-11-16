# %% [markdown]
# ## Data Ingestion

# %%

# ## GeeksforGeeks Loader – Dynamic Content Extraction

from langchain_community.document_loaders import WebBaseLoader
import bs4
import re

urls = [
    "https://www.geeksforgeeks.org/machine-learning/machine-learning/",
    "https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/",
    "https://www.geeksforgeeks.org/machine-learning/regularization-in-machine-learning/",
    "https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/",
    "https://www.geeksforgeeks.org/machine-learning/ml-bias-variance-trade-off/",
    "https://www.geeksforgeeks.org/machine-learning/underfitting-and-overfitting-in-machine-learning/",
    "https://www.geeksforgeeks.org/data-science/what-is-gradient-descent/",
    "https://www.geeksforgeeks.org/machine-learning/ml-common-loss-functions/",
    "https://www.geeksforgeeks.org/machine-learning/ml-classification-vs-regression/",
    "https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/",
    "https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/",
    "https://www.geeksforgeeks.org/machine-learning/what-is-feature-engineering/",
    "https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/",
]

def extract_main_article(html):
    """Dynamic extractor: Find div with most <p> tags (the article body)"""
    soup = bs4.BeautifulSoup(html, "html.parser")
    
    # Find all divs and score by number of <p> children
    candidates = soup.find_all("div")
    if not candidates:
        return soup.get_text()  # Fallback to full text
    
    best_div = max(candidates, key=lambda d: len(d.find_all("p")), default=soup)
    
    # Clean the best div
    for tag in best_div(["script", "style", "nav", "header", "footer", "aside", "iframe", "img"]):
        tag.decompose()
    
    # Extract text from key elements
    text_parts = []
    for elem in best_div.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]):
        text = elem.get_text(separator=" ", strip=True)
        if len(text) > 15:  # Meaningful length
            text_parts.append(text)
    
    full_text = "\n\n".join(text_parts)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text.strip())
    full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
    
    return full_text

# Load with custom extractor (no bs_kwargs needed)
loader = WebBaseLoader(
    web_paths=urls,
    # Custom post-load processing via a simple override
)

docs = loader.load()

# Apply extractor to each
clean_docs = []
for d in docs:
    extracted = extract_main_article(d.page_content)
    if len(extracted) > 500:  # Only keep substantial content
        d.page_content = extracted
        # Try to grab title
        soup = bs4.BeautifulSoup(d.page_content, "html.parser")  # Reuse for title
        title_elem = soup.find("h1") or soup.find("title")
        d.metadata["title"] = title_elem.get_text(strip=True) if title_elem else "ML Article"
        clean_docs.append(d)
    else:
        print(f"Skipped {d.metadata.get('source')}: too short ({len(extracted)} chars)")

print(f"Successfully loaded & cleaned {len(clean_docs)} documents")


# %% [markdown]
# ## Chunking Data

# %%
# Chunking the Data
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
documents = text_splitter.split_documents(clean_docs)

# %%
print(documents)

# %% [markdown]
# ### Using Gemini token from .env

# %%
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found in .env under 'GEMINI_API_KEY'")

# %% [markdown]
# ### Intializing Embedding model

# %%
# Creating Embeddings and Vector Store
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# Initialize the model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # 384 dimensions


# %% [markdown]
# ### Loading VectorDB

# %%
# Create the vector store
vectordb = Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=embedding_model,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# %% [markdown]
# ### Loading Multiple models from Gemini

# %%
# List available models (for debugging)
import google.generativeai as genai
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)

# %% [markdown]
# ### Configuring Gemini Model

# %%
# === Simple Gemini Wrapper for LangChain ===
import os
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Gemini API key not found in .env under 'GEMINI_API_KEY'")

genai.configure(api_key=api_key)

_GEMINI_MODEL_ID = "gemini-2.0-flash-lite"
_generation_config = genai.types.GenerationConfig(
    temperature=0.3,
    top_p=0.9,
    max_output_tokens=512,
)
_model = genai.GenerativeModel(_GEMINI_MODEL_ID)

def _gemini_call(inputs, **kwargs) -> str:
    if isinstance(inputs, dict):
        prompt_text = "\n\n".join(f"{k}: {v}" for k, v in inputs.items())
    else:
        prompt_text = str(inputs)
    try:
        resp = _model.generate_content(
            prompt_text,
            generation_config=_generation_config,
            safety_settings={  # Optional: relax safety if needed
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # === DEBUG: Print token usage ===
        if hasattr(resp, 'usage_metadata'):
            print(f"[DEBUG] Prompt tokens: {resp.usage_metadata.prompt_token_count}")
            print(f"[DEBUG] Output tokens: {resp.usage_metadata.candidates_token_count}")
            print(f"[DEBUG] Total tokens: {resp.usage_metadata.total_token_count}")

        # === CASE 1: Normal response with text ===
        if resp.candidates and resp.candidates[0].content.parts:
            text = resp.candidates[0].content.parts[0].text.strip()
            return text if text else "Empty response from model."

        # === CASE 2: No text generated (e.g., MAX_TOKENS) ===
        candidate = resp.candidates[0]
        finish_reason = candidate.finish_reason

        if finish_reason == 2:  # MAX_TOKENS
            return "[TRUNCATED] Response cut off due to max_output_tokens. Try reducing context size."
        elif finish_reason == 3:  # SAFETY
            return "[BLOCKED] Response blocked for safety."
        else:
            return f"[NO OUTPUT] Finish reason: {finish_reason}"

    except Exception as e:
        return f"[ERROR] {str(e)}"

# Rebuild LLM
llm = RunnableLambda(_gemini_call)

# %% [markdown]
# #### From each doc trying to extract metadata like if present page.no. to add to end of chunk

# %%
# ==== RAG Prompt & Chain ====
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(clean_docs):
    # Helpful for page-aware citations if metadata exists
    def one(d):
        pg = d.metadata.get("page", None)
        tag = f" [page {pg}]" if pg is not None else ""
        return d.page_content.strip() + tag
    return "\n\n---\n\n".join(one(d) for d in clean_docs)
    

# %% [markdown]
# ### Zero-shot Prompt

# %%

# Define a plain (non-chat) prompt template to avoid role tags in output
prompt = PromptTemplate.from_template(
    """You are an expert in machine learning. Use ONLY the provided context to answer the question.
If the answer is not in the context, respond exactly: I don't know.

Answer in a clear, concise paragraph. Do not use bullet points. Do not add external knowledge.

Question: {question}

Context: {context}

Answer:"""
)

# Keep your retriever the same; build a chain that accepts explicit context
generation_chain = prompt | llm | StrOutputParser()



# %% [markdown]
# ## Few-shot prompt

# %%

# --- Few-shot: two format exemplars, still "use only context" ---
prompt = PromptTemplate.from_template(
    """You are given a question and some context. Use ONLY the context.
If the answer is not in the context, respond exactly: I don't know.

Answer in a concise, natural paragraph (not bullets).

### EXAMPLE
Question: What is overfitting?

Context: Overfitting occurs when a model learns the training data too well, including noise. This leads to poor performance on new data.

Answer: Overfitting occurs when a model learns the training data too well, including noise, leading to poor performance on new data.

### TASK
Question: {question}

Context: {context}

Answer:"""
)
generation_chain_fewshot = prompt | llm | StrOutputParser()


# %% [markdown]
# ## Chain-of-thought (CoT) prompt

# %%

# --- CoT: brief step-by-step, but return a clean <final>...</final> block we can parse ---
prompt = PromptTemplate.from_template(
    """You are an expert in machine learning. Use ONLY the context to answer the question.
If the answer is not in the context, respond exactly: I don't know.

Think step-by-step VERY briefly (1-2 sentences max), then give the final answer inside <final>...</final>.
The final answer must be a concise paragraph — no bullets, no extra text.

Question: {question}

Context: {context}

Reasoning (brief):
1) Identify key facts from context.
2) Formulate a direct, natural answer.

<final>
<!-- Final answer only -->
</final>
"""
)
generation_chain_cot = prompt | llm | StrOutputParser()




# %% [markdown]
# #### This is mainly extracting content between final tags during  COT since it provides reasoning and with final tag.

# %%
import re
def extract_final(text: str) -> str:
    m = re.search(r"<final>(.*?)</final>", text, flags=re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()  # fallback if tags missing

def generate_answer(chain, q: str) -> str:
    # Retrieve & format context exactly like your base flow
    docs = retriever.invoke(q)
    ctx = format_docs(docs)
    out = chain.invoke({"question": q, "context": ctx})
    return out, ctx


# %%
# Helper to run once, show the exact retrieved context, and generate
def answer_with_trace(q: str):
    docs = retriever.invoke(q)         # same retriever as before
    ctx = format_docs(docs)                            # exactly what goes into the prompt
    ans = generation_chain.invoke({"question": q, "context": ctx})

    print("\nuser question:\n")
    print(q)
    print("\nretrieved context:\n")
    print(ctx if len(ctx) < 4000 else ctx[:4000] + "\n...[truncated]...")
    print("\nllm output:\n")
    print(ans)

# ==== Ask questions ====
question = "what is machine learning?"
answer_with_trace(question)

# %% [markdown]
# ### Loading Ground Truth Dataset from Huggingface

# %%
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
df = pd.read_csv("hf://datasets/prsdm/Machine-Learning-QA-dataset/ML-101-QandA.csv")

# %%

# === Load evaluation dataset ===

# Reference question/answer columns
questions = df["Question"].tolist()
gold_answers = df["Answer"].tolist()

# Embedding model for semantic comparison
eval_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# %% [markdown]
# ### Evaluation metrics

# %%
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === Load evaluation dataset (only first 10 rows) ===
sample_df = df.head(10)  #
questions = sample_df["Question"].tolist()
gold_answers = sample_df["Answer"].tolist()

# Embedding model for semantic comparison
eval_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Storage for metric results
results = []

for q, gold in tqdm(zip(questions, gold_answers), total=len(questions), desc="Evaluating RAG (1 samples)"):
    # --- Retrieve context ---
    retrieved_docs = retriever.invoke(q)
    retrieved_texts = [d.page_content for d in retrieved_docs]
    context = "\n\n".join(retrieved_texts)
    
    # --- Generate answer ---
    gen_answer = generation_chain_cot.invoke({"question": q, "context": context})

    # --- Compute embeddings ---
    q_emb = eval_embedder.encode([q])
    gold_emb = eval_embedder.encode([gold])
    gen_emb = eval_embedder.encode([gen_answer])
    ctx_embs = eval_embedder.encode(retrieved_texts)

    # --- Retrieval metrics ---
    ctx_rel = float(np.mean(cosine_similarity(q_emb, ctx_embs)))   # Context Relevance
    ctx_rec = float(np.max(cosine_similarity(gold_emb, ctx_embs))) # Context Recall

    # --- Generation metrics ---
    faith = float(np.mean(cosine_similarity(gen_emb, ctx_embs)))   # Faithfulness (Groundedness)
    ans_rel = float(cosine_similarity(gen_emb, gold_emb)[0][0])    # Answer Relevance

    # --- End-to-end metric ---
    ans_corr = (2 * faith * ans_rel) / (faith + ans_rel + 1e-9)    # Answer Correctness

    results.append({
        "Question": q,
        "Context Relevance": round(ctx_rel, 3),
        "Context Recall": round(ctx_rec, 3),
        "Faithfulness": round(faith, 3),
        "Answer Relevance": round(ans_rel, 3),
        "Answer Correctness": round(ans_corr, 3)
    })

# === Average metrics ===
import pandas as pd

results_df = pd.DataFrame(results)
avg_metrics = results_df.drop(columns=["Question"]).mean().round(3)

print("\n=== Average RAG Evaluation Metrics (10 samples) ===\n")
print("\n Metrics")
print(avg_metrics.to_frame().T)



