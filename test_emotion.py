
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, TFWav2Vec2ForSequenceClassification


# --- Model & File Settings ---
MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
# REPLACE THIS with the path to your 16kHz WAV audio file
AUDIO_FILE_PATH = "C:\\Users\\mmilivoje\\Downloads\\negative.wav" 
TARGET_SAMPLING_RATE = 16000 # Wav2Vec2 standard

# 1. Load the Feature Extractor and Model
print(f"Loading {MODEL_NAME}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
# Use TFWav2Vec2ForSequenceClassification if you want to test the TF version
# or Wav2Vec2ForSequenceClassification for the PyTorch version (recommended for ease)
model = TFWav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, from_pt=True) 

# Get the label map from the model configuration
id2label = model.config.id2label
print(f"Model Labels: {id2label}")

# 2. Load and Preprocess Audio
def preprocess_audio(audio_path, target_sr):
    # Load audio, ensuring it's resampled to the target rate
    speech, sr = librosa.load(audio_path, sr=target_sr)
    
    # Ensure the audio is exactly 4 seconds (64000 samples) by padding or cutting
    max_samples = 64000
    if len(speech) > max_samples:
        speech = speech[:max_samples]
    elif len(speech) < max_samples:
        # Pad with zeros to meet the fixed TFLite input size
        padding_needed = max_samples - len(speech)
        speech = np.pad(speech, (0, padding_needed), 'constant')
        
    # The model expects a single channel (mono), which librosa.load provides by default
    return speech.astype(np.float32)

try:
    audio_input = preprocess_audio(AUDIO_FILE_PATH, TARGET_SAMPLING_RATE)
    print(f"Audio loaded and processed. Shape: {audio_input.shape}")
except Exception as e:
    print(f"Error loading audio file: {e}")
    exit()

# 3. Prepare Model Input (TensorFlow/Keras format)
# The feature extractor handles normalization and converts the NumPy array to a tensor
inputs = feature_extractor(
    audio_input, 
    sampling_rate=TARGET_SAMPLING_RATE, 
    return_tensors="tf" # Requesting a TensorFlow tensor
)

# 4. Run Inference
# The input tensor needs to be in the shape [1, 64000]
input_tensor = inputs['input_values']

# Run the prediction
outputs = model(input_tensor)
logits = outputs.logits.numpy() # Convert back to NumPy array

# 5. Post-Process: Find the predicted class
predicted_index = np.argmax(logits, axis=1)[0]
predicted_emotion = id2label[predicted_index]
confidence_score = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
max_confidence = confidence_score[predicted_index] * 100

# 6. Print Result
print("\n--- Model Prediction ---")
print(f"Predicted Class Index: {predicted_index}")
print(f"Predicted Emotion: {predicted_emotion.upper()}")
print(f"Confidence: {max_confidence:.2f}%")
print("------------------------")

# Optional: Print all logits/scores for debugging
# print("All Confidence Scores:")
# for i, score in enumerate(confidence_score):
#     print(f"  {id2label[i]}: {score*100:.2f}%")