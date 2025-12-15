package com.example.myapplication;

/**
 * Configuration class to hold all unique, hardcoded parameters
 * for a specific TFLite model and its task.
 */
public class ModelConfig {
    // File names
    public final String modelFileName;
    public final String vocabFileName; // Null if no vocabulary is needed (e.g., Audio Model)

    // Input configuration
    public final int sequenceLength; // For text models (MAX_LEN)
    public final int numberOfInputs; // Number of input tensors (e.g., 2 for BERT: IDs, Mask)
    // Note: Assumes all inputs are INT32 for text models.

    // Output configuration
    public final int outputTensorIndex; // The index of the primary output tensor (usually 0)
    public final int outputClasses;     // Size of the final output vector (e.g., 2 for sentiment)
    public final String[] outputLabels; // Labels mapped to the output indices

    // --- NEW FIELD FOR AUDIO MODELS ---
    public final int sampleRate; // Wav2Vec2 requires 16000 Hz

    public ModelConfig(String modelFile, String vocabFile, int seqLen, int numInputs, int outIndex, int outClasses, String[] labels, int sampleRate) {
        this.modelFileName = modelFile;
        this.vocabFileName = vocabFile;
        this.sequenceLength = seqLen;
        this.numberOfInputs = numInputs;
        this.outputTensorIndex = outIndex;
        this.outputClasses = outClasses;
        this.outputLabels = labels;
        // Initialize new field
        this.sampleRate = sampleRate;
    }

    /**
     * Factory method for the DistilBERT Sentiment Model (Text Classification).
     * Passes -1 for sampleRate as it is not applicable.
     */
    public static ModelConfig getSentimentConfig() {
        String[] labels = {"NEGATIVE", "POSITIVE"};
        return new ModelConfig(
                "sentiment_model.tflite", // Model File
                "vocab.txt",              // Vocab File
                128,                      // Sequence Length (MAX_LEN)
                2,                        // Number of Inputs (IDs, Mask)
                0,                        // Output Tensor Index
                2,                        // Number of Output Classes
                labels,                   // Labels
                -1                        // Sample Rate (N/A for text)
        );
    }

    /**
     * Factory method for the Wav2Vec2 Speech Emotion Recognition Model (Audio Classification).
     * Passes the required 16000 sample rate.
     */
    public static ModelConfig getEmotionConfig() {
        String[] labels = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"};
        return new ModelConfig(
                "emotion_model_quant.tflite", // Model File (must be the quantized TFLite file)
                null,                         // Vocab File (Not needed for audio input)
                -1,                           // Sequence Length (Not applicable for audio)
                1,                            // Number of Inputs (Only the raw audio Float array)
                0,                            // Output Tensor Index
                7,                            // Number of Output Classes (7 emotions)
                labels,                       // Labels
                16000                         // Sample Rate (CRITICAL: Wav2Vec2 requirement)
        );
    }
}