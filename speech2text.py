import tensorflow as tf
from transformers import AutoProcessor, TFWav2Vec2ForCTC
import os

# --- 1. Settings for the ASR Model ---
# This is a standard base model for English ASR (Automatic Speech Recognition)
MODEL_NAME = "facebook/wav2vec2-base-960h" 
TFLITE_FILE = "speech_to_text_asr_quant.tflite"

# Wav2Vec2 needs a fixed input shape for TFLite. (4 seconds @ 16kHz)
MAX_AUDIO_SAMPLES = 64000 

print(f"Loading {MODEL_NAME} and processor...")

# 2. Load the Processor (Feature Extractor + Tokenizer) and Model
# Note: For ASR, we need a tokenizer/processor to handle the text output.
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = TFWav2Vec2ForCTC.from_pretrained(MODEL_NAME, from_pt=True)

# 3. Define the Concrete Function for TFLite Export
@tf.function(input_signature=[
    # The input is the raw audio signal: [Batch Size, Audio Samples]
    tf.TensorSpec(shape=[1, MAX_AUDIO_SAMPLES], dtype=tf.float32, name='input_values')
])
def serving_fn(input_values):
    # Wav2Vec2 uses Connectionist Temporal Classification (CTC) for ASR
    # The model outputs a sequence of logits representing predicted tokens.
    outputs = model(input_values=input_values)
    return {'logits': outputs.logits} # Outputs shape: [1, sequence_length, vocab_size]

print("Converting to TFLite with Quantization...")

# 4. Conversion with Optimization (Mandatory)
converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])

# --- Mandatory Optimization: Quantization for size reduction ---
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
# Note: Full integer quantization often works better for ASR accuracy, 
# but we stick to default dynamic range quantization for simplicity here.

tflite_model = converter.convert()

# 5. Save TFLite file and Tokenizer
with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

# 6. Save the Vocab/Tokenizer for Android Post-Processing
# The Android app needs to know how to map the model's output logits back to characters.
processor.tokenizer.save_pretrained("./asr_vocab")

print(f"\nSUCCESS! Move {TFLITE_FILE} to your 'src/main/assets' folder.")
print(f"Also, move the generated vocabulary files in the './asr_vocab' folder to your assets folder.")