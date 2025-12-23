This project demonstrates a local Retrieval-Augmented Generation (RAG) pipeline using:

Qwen/Qwen3-0.6B for text generation

OpenVINO for GPU-accelerated inference

FAISS + SentenceTransformers for document retrieval

Runtime selection via stdin (CPU / GPU, RAG / normal mode)

The script is designed to be simple, explicit, and educational, showing how modern local LLM systems are built.

🧠 What this script does (high level)

At runtime, the user chooses:

Which device to use

CPU → Hugging Face PyTorch model

GPU → OpenVINO-optimized model

Which mode to run

Normal text generation

RAG (retrieval + generation)

Depending on these choices, the script dynamically loads the correct model and execution path.

🧩 Technologies used

Hugging Face Transformers
Used for tokenization and CPU-based model execution.

OpenVINO
Used to run the same model efficiently on Intel GPUs.

SentenceTransformers
Used to generate vector embeddings for documents and queries.

FAISS
Used to store and search document embeddings for retrieval.

🏗️ Script structure (step by step)
1️⃣ User input (stdin)

When the script starts, the user is asked:

Device selection
0 = CPU (Hugging Face)
1 = GPU (OpenVINO)
2 = NPU (not supported for LLMs)


⚠️ NPU is intentionally blocked because current NPUs do not support
causal text generation models reliably.

Mode selection
0 = Normal generation
1 = RAG (Retrieval-Augmented Generation)

2️⃣ Model loading logic
CPU path

Loads Qwen/Qwen3-0.6B using PyTorch

Runs entirely on CPU

Best for compatibility and debugging

GPU path

Converts the model to OpenVINO once

Saves it locally (openvino_qwen3_0_6b/)

Reloads the optimized model on future runs

Uses FP16 and Intel GPU acceleration

3️⃣ RAG setup (only if RAG mode is selected)

When RAG is enabled:

A small list of documents is defined

Documents are embedded using SentenceTransformers (CPU)

Embeddings are stored in a FAISS index

The index is queried at runtime to retrieve relevant context

This separation is intentional:

Retrieval runs on CPU

Generation runs on GPU or CPU

This mirrors real production RAG systems.

4️⃣ Normal generation mode

If normal mode is selected:

The user prompt is sent directly to the model

No retrieval is performed

The model generates a response based only on its training

This behaves like a standard chatbot.

5️⃣ RAG generation mode

If RAG mode is selected:

The user question is embedded

FAISS retrieves the top-K relevant documents

Retrieved text is injected into a structured prompt

The model generates an answer grounded in retrieved context

This improves:

Factual accuracy

Consistency

Control over model output

🔁 Why OpenVINO is used only for generation

OpenVINO excels at:

Dense tensor computation

Transformer inference

GPU acceleration

It is not useful for:

Embedding lookup

Vector search

Control flow logic

Therefore:

Retrieval stays on CPU

Generation is accelerated on GPU

This is the correct architectural split.

🚫 Why NPU is not supported

Although Intel NPUs exist, they currently do not support:

Autoregressive decoding

Dynamic sequence lengths

KV-cache used by LLMs

As a result, text-generation models like Qwen are explicitly blocked from running on NPU in this script.

✅ Why Qwen3-0.6B was chosen

Small and fast

Works well on CPU and GPU

Ideal for local RAG demos

Fast OpenVINO conversion

Low memory usage

It is a perfect model for experimentation.

🧪 Typical usage flow

Run the script

Select device (CPU or GPU)

Select mode (normal or RAG)

Enter a prompt

Receive generated output with timing information

📌 Summary

This script demonstrates:

Runtime-selectable inference backends

OpenVINO GPU acceleration

A clean, minimal RAG pipeline

Best practices for local LLM deployment

It is intentionally written to be:

Readable

Modifiable

Educational

🚀 Possible extensions

INT8 quantization with OpenVINO

Streaming token output

Larger document collections

File-based document ingestion

Hybrid CPU/GPU fallback logic
