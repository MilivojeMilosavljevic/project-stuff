import tensorflow as tf
from transformers import AutoFeatureExtractor, TFWav2Vec2ForSequenceClassification

MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
TFLITE_FILE = "emotion_model_quant.tflite"

# Fixed input length required by TFLite
MAX_AUDIO_SAMPLES = 64000

# Feature extractor handles audio normalization
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# SequenceClassification head outputs class logits (emotions)
model = TFWav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME, from_pt=True
)

@tf.function(input_signature=[
    tf.TensorSpec(
        shape=[1, MAX_AUDIO_SAMPLES],
        dtype=tf.float32,
        name="input_values"
    )
])
def serving_fn(input_values):
    # Only raw audio input is required
    outputs = model(input_values=input_values)
    return {"logits": outputs.logits}

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [serving_fn.get_concrete_function()]
)

# Without quantization this model is ~1.3 GB
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

