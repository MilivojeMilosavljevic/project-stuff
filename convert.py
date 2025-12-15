import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import os

# Settings
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TFLITE_FILE = "sentiment_model.tflite"
SEQ_LEN = 128  # Fixed sequence length for Android

print(f"Loading {MODEL_NAME}...")

# 1. Load the Tokenizer and Model
# We use from_pt=True to load PyTorch weights into TensorFlow if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, from_pt=True)

# 2. Define the Concrete Function for TFLite
# TFLite needs to know exactly what input shape to expect.
# We define a function that takes [1, 128] int32 inputs.
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name='input_ids'),
    tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name='attention_mask')
])
def serving_fn(input_ids, attention_mask):
    # Pass inputs to the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Return the raw logits (scores)
    return {'logits': outputs.logits}

print("Converting to TFLite...")

# 3. Convert
converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Optimize for mobile size
tflite_model = converter.convert()

# 4. Save TFLite file
with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

# 5. Save Vocab file (Crucial for Android)
# Your Android app needs this to tokenize text!
tokenizer.save_pretrained("android_assets")

print(f"\nSUCCESS!")
print(f"1. Model saved: {TFLITE_FILE}")
print(f"2. Vocab saved: android_assets/vocab.txt")
print("Move BOTH files to your Android project's 'src/main/assets' folder.")