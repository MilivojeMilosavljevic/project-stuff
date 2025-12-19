
ON-DEVICE AI MODELS FOR ANDROID (TensorFlow Lite)

OVERVIEW
------------------------------------------------------------
This project investigates running multiple AI models fully
offline on an Android device using TensorFlow Lite.

The goal was to understand the complete workflow of:
- selecting pretrained AI models
- converting them for mobile use
- deploying and running inference directly on Android
without internet access or cloud services.

This work was done as an investigation / feasibility study,
not as a production-ready system.


SUPPORTED FEATURES
------------------------------------------------------------
The Android application supports three AI modes:

1) TEXT SENTIMENT ANALYSIS
   - Classifies text as POSITIVE or NEGATIVE
   - Uses a DistilBERT-based model

2) AUDIO EMOTION RECOGNITION
   - Detects emotion from recorded voice
   - Outputs labels such as Angry, Happy, Neutral, etc.

3) SPEECH-TO-TEXT (ASR)
   - Converts spoken English into text
   - Uses a CTC-based speech recognition model

All processing is performed locally on the device.


MODEL SOURCES
------------------------------------------------------------
All pretrained models were obtained from Hugging Face:

- Sentiment Analysis:
  distilbert-base-uncased-finetuned-sst-2-english

- Speech Emotion Recognition:
  r-f/wav2vec-english-speech-emotion-recognition

- Speech-to-Text:
  facebook/wav2vec2-base-960h

Original models were trained in PyTorch and converted
for Android usage.


PYTHON MODEL CONVERSION
------------------------------------------------------------
Python scripts are used to convert pretrained models into
TensorFlow Lite (.tflite) format.

Conversion steps include:
- Loading Hugging Face models
- Converting PyTorch weights to TensorFlow
- Defining fixed input shapes (required by TFLite)
- Applying quantization to reduce model size
- Exporting .tflite files and vocabulary data

Important constraints:
- TensorFlow Lite requires static input sizes
- Tokenization must run outside the model for text models
- Quantization is mandatory for large audio models


AUDIO PROCESSING DETAILS
------------------------------------------------------------
Audio-based models use fixed-length input.

- Sample rate: 16 kHz
- Recording duration: 4 seconds
- Total samples: 64,000
  (16,000 samples/sec × 4 sec)

Audio processing steps:
1) Record microphone input
2) Convert to mono PCM
3) Normalize to float values in range [-1, 1]
4) Pad or truncate to exactly 64,000 samples


ANDROID APPLICATION ARCHITECTURE
------------------------------------------------------------
The Android app uses TensorFlow Lite for on-device inference.

Key components:
- TensorFlow Lite Interpreter
- Background threads for inference
- Manual preprocessing and postprocessing
- No network or cloud dependency


ASSETS DIRECTORY (CRITICAL)
------------------------------------------------------------
All models must be placed in the Android assets folder:

app/src/main/assets/

Required files:
- sentiment_model.tflite
- emotion_model_quant.tflite
- speech_to_text_asr_quant.tflite
- vocab.txt
- emotion_vocab/
- speech_to_text/vocab.json

The application will fail to start if any required
asset file is missing or renamed.


ANDROID APP MODES
------------------------------------------------------------

MODE 1: TEXT SENTIMENT
- User enters text
- Text is tokenized on-device
- Model outputs class logits
- Highest logit determines sentiment label

MODE 2: AUDIO EMOTION
- App records exactly 4 seconds of audio
- Audio is preprocessed into a fixed buffer
- Model outputs emotion probabilities

MODE 3: SPEECH-TO-TEXT
- User starts and stops recording manually
- Audio is converted to a fixed-size buffer
- Model outputs CTC logits
- Custom decoder converts logits to text


CUSTOM IMPLEMENTATIONS
------------------------------------------------------------
- AudioPreprocessor:
  Handles recording, normalization, and padding

- CTCDecoder:
  Converts speech model logits into readable text
  (TensorFlow Lite does not provide built-in CTC decoding)

- ModelConfig:
  Centralized configuration for all models

- TfLitePredictor / AsrPredictor:
  Lightweight wrappers around the TFLite interpreter


KEY DESIGN DECISIONS
------------------------------------------------------------
- Fully offline operation
- Quantized models for mobile CPUs
- Fixed input sizes for TensorFlow Lite compatibility
- Manual tokenization and decoding
- Background inference to keep UI responsive


LIMITATIONS
------------------------------------------------------------
- Large model sizes (audio models are still ~100–300 MB)
- Fixed 4-second audio input
- No streaming speech recognition
- CPU-only inference (no GPU acceleration)


PURPOSE OF THIS WORK
------------------------------------------------------------
This project was created as an investigation into
mobile AI deployment, focusing on:

- Model conversion challenges
- Mobile hardware constraints
- Offline inference feasibility
- Android + AI integration

It demonstrates the full pipeline from pretrained
research models to real-world mobile execution.
