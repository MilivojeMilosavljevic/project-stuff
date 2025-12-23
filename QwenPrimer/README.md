# 📄 README – Qwen3-0.6B OpenVINO RAG Demo

This project demonstrates a **local Retrieval-Augmented Generation (RAG) pipeline** using:

- **Qwen/Qwen3-0.6B** for text generation  
- **OpenVINO** for GPU-accelerated inference  
- **FAISS + SentenceTransformers** for document retrieval  
- **Runtime selection via stdin** (CPU / GPU, RAG / normal mode)

The script is designed to be **simple, explicit, and educational**, showing how modern local LLM systems are built.

---

## 🧠 What this script does (high level)

At runtime, the user chooses:

1. **Which device to use**
   - CPU → Hugging Face PyTorch model
   - GPU → OpenVINO-optimized model
2. **Which mode to run**
   - Normal text generation
   - RAG (retrieval + generation)

Depending on these choices, the script dynamically loads the correct model and execution path.

---

## 🧩 Technologies used

- **Hugging Face Transformers**  
  Tokenization and CPU-based model execution.

- **OpenVINO**  
  GPU-accelerated inference on Intel hardware.

- **SentenceTransformers**  
  Vector embeddings for documents and queries.

- **FAISS**  
  Fast similarity search over embeddings.

---

## 🏗️ Script structure (step by step)

### 1️⃣ User input (stdin)

When the script starts, the user is asked:

#### Device selection
```
0 = CPU (Hugging Face)
1 = GPU (OpenVINO)
2 = NPU (not supported for LLMs)
```

> NPU is intentionally blocked because current NPUs do not support
> causal text generation models reliably.

#### Mode selection
```
0 = Normal generation
1 = RAG (Retrieval-Augmented Generation)
```

---

### 2️⃣ Model loading logic

#### CPU path
- Loads `Qwen/Qwen3-0.6B` using PyTorch
- Runs entirely on CPU
- Best for compatibility and debugging

#### GPU path
- Converts the model to OpenVINO **once**
- Saves it locally (`openvino_qwen3_0_6b/`)
- Reloads the optimized model on future runs
- Uses FP16 and Intel GPU acceleration

---

### 3️⃣ RAG setup (only if RAG mode is selected)

When RAG is enabled:

1. A list of documents is defined
2. Documents are embedded using SentenceTransformers (CPU)
3. Embeddings are stored in a FAISS index
4. The index is queried to retrieve relevant context

This separation is intentional:
- Retrieval runs on CPU
- Generation runs on GPU or CPU

This mirrors real production RAG systems.

---

### 4️⃣ Normal generation mode

If **normal mode** is selected:
- The user prompt is sent directly to the model
- No retrieval is performed
- The model generates a response based only on its training

---

### 5️⃣ RAG generation mode

If **RAG mode** is selected:

1. The user question is embedded
2. FAISS retrieves the top-K relevant documents
3. Retrieved text is injected into a structured prompt
4. The model generates an answer grounded in retrieved context

This improves factual accuracy and consistency.

---

## 🚫 Why NPU is not supported

Although Intel NPUs exist, they currently do not support:

- Autoregressive decoding
- Dynamic sequence lengths
- KV-cache used by LLMs

As a result, text-generation models like Qwen are explicitly blocked from running on NPU.

---

## 📌 Summary

This project demonstrates:

- Runtime-selectable inference backends
- OpenVINO GPU acceleration
- A clean and minimal RAG pipeline
- Best practices for local LLM deployment

It is intentionally written to be **readable, modifiable, and educational**.
