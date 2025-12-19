import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Tokenizer runs OUTSIDE the model (not TFLite compatible)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TF model loaded from PyTorch weights
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, from_pt=True
)

# Wrapper is required because:
# - TFLite cannot run HuggingFace tokenizers
# - tf.py_function bridges Python → TensorFlow
class SentimentModel(tf.keras.Model):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def call(self, inputs):
        # Tokenization must happen in Python
        def tokenize(texts):
            return self.tokenizer(
                list(texts.numpy()),
                padding=True,
                truncation=True,
                return_tensors="tf"
            )

        # tf.py_function breaks graph purity but allows export
        tokens = tf.py_function(
            func=tokenize,
            inp=[inputs],
            Tout=[tf.int32, tf.int32]
        )

        input_dict = {
            "input_ids": tokens[0],
            "attention_mask": tokens[1],
        }

        return self.model(**input_dict).logits

wrapper = SentimentModel(model, tokenizer)

# String input is required for text models
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.string)
])
def serving_fn(inputs):
    return wrapper(inputs)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [serving_fn.get_concrete_function()]
)

# Reduces size and improves mobile performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("sentiment_model.tflite", "wb") as f:
    f.write(tflite_model)
