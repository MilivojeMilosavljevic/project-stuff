import tensorflow as tf
from transformers import AutoFeatureExtractor, TFWav2Vec2ForSequenceClassification
import os

# --- 1. Settings for the NEW Model ---
MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
TFLITE_FILE = "emotion_model_quant.tflite"

# Wav2Vec2 takes a variable length input, but TFLite needs a fixed shape.
# Define a maximum duration (e.g., 4 seconds @ 16kHz = 64000 samples)
MAX_AUDIO_SAMPLES = 64000 

print(f"Loading {MODEL_NAME}...")

# 2. Load the Feature Extractor and Model
# Note the different class names: TFWav2Vec2ForSequenceClassification is needed for TF export
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = TFWav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, from_pt=True)

# 3. Define the Concrete Function for TFLite
@tf.function(input_signature=[
    # The input is a single tensor: [Batch Size, Audio Samples], 
    # and the data type MUST be tf.float32 (raw audio samples).
    tf.TensorSpec(shape=[1, MAX_AUDIO_SAMPLES], dtype=tf.float32, name='input_values')
])
def serving_fn(input_values):
    # Wav2Vec2 only needs the raw audio signal (input_values). 
    # No attention mask is typically needed for fixed-length/classification export.
    outputs = model(input_values=input_values)
    return {'logits': outputs.logits}

print("Converting to TFLite with Quantization...")

# 4. Conversion with Optimization (Crucial for a 1.3 GB model!)
converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])

# --- Mandatory Optimization for Wav2Vec2 ---
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
# Post-training quantization (8-bit) will drastically reduce size and is essential.

tflite_model = converter.convert()

# 5. Save TFLite file
with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

print(f"\nSUCCESS! Move {TFLITE_FILE} to your Android project's 'src/main/assets' folder.")
# NOTE: No vocab file is needed for the input, but save the labels for Android post-processing!