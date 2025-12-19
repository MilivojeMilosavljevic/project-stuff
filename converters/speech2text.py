import tensorflow as tf
from transformers import AutoProcessor, TFWav2Vec2ForCTC

# Base English ASR model trained with CTC loss
MODEL_NAME = "facebook/wav2vec2-base-960h"
TFLITE_FILE = "speech_to_text_asr_quant.tflite"

# TFLite requires a FIXED input shape, 4 seconds * 16 kHz = 64,000 samples
MAX_AUDIO_SAMPLES = 64000

# Processor = feature extractor + tokenizer
# Required to decode logits back to text
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load TF model (converted from PyTorch weights)
model = TFWav2Vec2ForCTC.from_pretrained(MODEL_NAME, from_pt=True)

# Concrete function defines the exact graph TFLite will freeze
@tf.function(input_signature=[
    tf.TensorSpec(
        shape=[1, MAX_AUDIO_SAMPLES],   # batch=1, fixed-length audio
        dtype=tf.float32,               # raw waveform
        name="input_values"
    )
])
def serving_fn(input_values):
    # Outputs CTC logits: [batch, time_steps, vocab_size]
    outputs = model(input_values=input_values)
    return {"logits": outputs.logits}

# TFLite converter works ONLY on concrete functions
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [serving_fn.get_concrete_function()]
)

# Dynamic range quantization mandatory to reduce size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save model
with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

# Tokenizer/vocab is REQUIRED on Android to decode logits → text
processor.tokenizer.save_pretrained("./asr_vocab")
