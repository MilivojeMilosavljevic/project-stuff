import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM

# =========================
# Configuration
# =========================
model_name = "Qwen/Qwen3-0.6B"
ov_model_dir = "openvino_qwen3_0_6b"
prompt = "what is 2+2?"

# =========================
# Ask user for device
# =========================
print("Select device:")
print("0 = CPU (original Hugging Face model)")
print("1 = GPU (OpenVINO)")
print("2 = NPU (OpenVINO)")

device_choice = input("Enter choice (0/1/2): ").strip()

# =========================
# Load tokenizer (shared)
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_name)

# =========================
# Device logic
# =========================
if device_choice == "0":
    # ---------------------------------
    # CPU → Original Hugging Face model
    # ---------------------------------
    print("Loading Hugging Face model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

elif device_choice in ("1", "2"):
    # ---------------------------------
    # GPU / NPU → OpenVINO
    # ---------------------------------
    ov_device = "GPU" if device_choice == "1" else "NPU"

    if not os.path.exists(os.path.join(ov_model_dir, "config.json")):
        print(f"Converting model to OpenVINO ({ov_device})...")
        model = OVModelForCausalLM.from_pretrained(
            model_name,
            device=ov_device
        )
        model.save_pretrained(ov_model_dir)
    else:
        print(f"Loading OpenVINO model on {ov_device}...")
        model = OVModelForCausalLM.from_pretrained(
            ov_model_dir,
            device=ov_device
        )

else:
    raise ValueError("Invalid selection. Choose 0, 1, or 2.")

# =========================
# Tokenize input
# =========================
inputs = tokenizer(prompt, return_tensors="pt")

# Hugging Face CPU model needs tensors on same device
if device_choice == "0":
    inputs = inputs.to(model.device)

# =========================
# Generate
# =========================
start_time = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

elapsed = time.time() - start_time

# =========================
# Decode output
# =========================
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Generated Text ===")
print(generated_text)
print(f"\n⏱️ Generation took {elapsed:.3f} seconds")
