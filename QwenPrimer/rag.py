# Filename: qwen_openvino_rag_selectable.py

import os
import time
import torch
import faiss

from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM
from sentence_transformers import SentenceTransformer


# =========================================================
# User selections
# =========================================================
print("Select device:")
print("0 = CPU (Hugging Face)")
print("1 = GPU (OpenVINO)")
print("2 = NPU (NOT supported for LLMs)")

device_choice = input("Enter device (0/1/2): ").strip()

if device_choice == "2":
    raise RuntimeError(
        "Intel NPU does not support causal text generation models yet."
    )

print("\nSelect mode:")
print("0 = Normal generation")
print("1 = RAG (Retrieval-Augmented Generation)")

mode_choice = input("Enter mode (0/1): ").strip()


# =========================================================
# Model config
# =========================================================
model_name = "Qwen/Qwen3-0.6B"
ov_model_dir = "openvino_qwen3_0_6b"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# =========================================================
# Load generation model
# =========================================================
if device_choice == "0":
    print("Loading Hugging Face model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

elif device_choice == "1":
    os.makedirs(ov_model_dir, exist_ok=True)

    if not os.path.exists(os.path.join(ov_model_dir, "config.json")):
        print("Converting model to OpenVINO (GPU)...")
        model = OVModelForCausalLM.from_pretrained(
            model_name,
            device="GPU"
        )
        model.save_pretrained(ov_model_dir)
    else:
        print("Loading OpenVINO model on GPU...")
        model = OVModelForCausalLM.from_pretrained(
            ov_model_dir,
            device="GPU"
        )


# =========================================================
# RAG setup (only if needed)
# =========================================================
if mode_choice == "1":
    documents = [
        "OpenVINO accelerates inference on Intel GPUs like Arc.",
        "Qwen3-0.6B is a compact language model optimized for local inference.",
        "RAG combines retrieval with generation to improve factual accuracy.",
        "Uros is a guy from Novi Sad in Serbia, he is twenty two years old. He is tall and has brown eyes."
    ]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)

    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    print(f"FAISS index ready with {index.ntotal} documents")


# =========================================================
# Generation functions
# =========================================================
def normal_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    if device_choice == "0":
        inputs = inputs.to(model.device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    elapsed = time.time() - start

    return tokenizer.decode(outputs[0], skip_special_tokens=True), elapsed


def rag_answer(question, top_k=2):
    q_embedding = embed_model.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    context = "\n".join(retrieved_docs)
    prompt = f"""
You are an assistant. Use the context below to answer.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    if device_choice == "0":
        inputs = inputs.to(model.device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )
    elapsed = time.time() - start

    return tokenizer.decode(outputs[0], skip_special_tokens=True), elapsed


# =========================================================
# Run
# =========================================================
user_prompt = input("\nEnter your prompt: ").strip()

if mode_choice == "0":
    answer, elapsed = normal_answer(user_prompt)
else:
    answer, elapsed = rag_answer(user_prompt)

print("\n=== Output ===")
print(answer)
print(f"\n⏱️ Generation took {elapsed:.3f} seconds")
