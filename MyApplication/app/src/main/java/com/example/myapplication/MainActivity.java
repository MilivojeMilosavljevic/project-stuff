package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "DualModelApp";
    private static final int REQUEST_RECORD_AUDIO = 101;

    // --- Model Fields ---
    private TfLitePredictor predictor;
    private Vocabulary vocab;
    private ModelConfig config;
    private AudioPreprocessor audioPreprocessor;

    // --- UI Fields ---
    private EditText inputText;
    private Button analyzeButton;
    private TextView resultText;
    private Switch modeSwitch;
    private TextView modeLabel;
    private TextView statusText;

    private boolean isTextMode = true;

    // --- Lifecycle and Initialization ---

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1. Initialize UI Views
        inputText = findViewById(R.id.input_text);
        analyzeButton = findViewById(R.id.analyze_button);
        resultText = findViewById(R.id.result_text);
        modeSwitch = findViewById(R.id.mode_switch);
        modeLabel = findViewById(R.id.mode_label);
        statusText = findViewById(R.id.status_text);

        inputText.setText("The quick brown fox jumps over the lazy dog.");

        // 2. Set up Mode Switch Listener
        modeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            isTextMode = !isChecked; // isChecked=false -> Text Mode (isTextMode=true)
            statusText.setText("Status: Loading model...");

            if (isTextMode) {
                initializeTextModel();
            } else {
                initializeAudioModel();
            }
        });

        // Ensure the switch reflects the initial state
        modeSwitch.setChecked(!isTextMode);

        // 3. Initial Model Load (Default is Text Mode)
        initializeTextModel();

        // 4. Set up the Button Click Listener
        analyzeButton.setOnClickListener(v -> {
            if (isTextMode) {
                runSentimentAnalysis(inputText.getText().toString());
            } else {
                checkAndRunEmotionAnalysis();
            }
        });
    }

    // --- Model Initializers ---

    private void initializeTextModel() {
        try {
            config = ModelConfig.getSentimentConfig();

            vocab = new Vocabulary(this, config.vocabFileName);
            predictor = new TfLitePredictor(this, config);

            isTextMode = true;
            modeLabel.setText("Current Mode: Text Analysis (Sentiment)");
            inputText.setVisibility(View.VISIBLE);
            analyzeButton.setText("Analyze Text Sentiment");
            statusText.setText("Status: Ready for text input.");
            analyzeButton.setEnabled(true);

        } catch (IOException e) {
            Log.e(TAG, "FATAL: Failed to load Sentiment model files.", e);
            statusText.setText("ERROR: Text model files missing.");
            analyzeButton.setEnabled(false);
        }
    }

    private void initializeAudioModel() {
        try {
            config = ModelConfig.getEmotionConfig();
            predictor = new TfLitePredictor(this, config);

            // AudioPreprocessor constructor is now fixed to use config.sampleRate
            audioPreprocessor = new AudioPreprocessor(config.sampleRate);

            isTextMode = false;
            modeLabel.setText("Current Mode: Audio Analysis (Emotion)");
            inputText.setVisibility(View.GONE);
            analyzeButton.setText("RECORD and Analyze Emotion (4s)");
            statusText.setText("Status: Ready. Check for microphone permission.");
            analyzeButton.setEnabled(true);

        } catch (IOException e) {
            Log.e(TAG, "FATAL: Failed to load Emotion model files.", e);
            statusText.setText("ERROR: Audio model files missing. Check assets.");
            analyzeButton.setEnabled(false);
        }
    }

    // --- Audio Mode Logic and Permissions ---

    private void checkAndRunEmotionAnalysis() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            statusText.setText("Requesting Microphone Permission...");
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        } else {
            runEmotionAnalysis();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            runEmotionAnalysis();
        } else if (requestCode == REQUEST_RECORD_AUDIO) {
            statusText.setText("Microphone permission denied. Cannot record audio.");
            analyzeButton.setEnabled(true);
        }
    }

    private void runEmotionAnalysis() {
        statusText.setText("Recording audio (4s)... Do not move.");
        analyzeButton.setEnabled(false);

        // FIX: Replaced Future.whenComplete with standard Android threading pattern
        Executors.newSingleThreadExecutor().submit(new Runnable() {
            @Override
            public void run() {
                try {
                    // Audio processing and inference logic
                    ByteBuffer audioInputBuffer = audioPreprocessor.recordAndProcess();

                    // Post status update from the background thread
                    statusText.post(() -> statusText.setText("Status: Running inference..."));

                    float[][] logits = predictor.runInference(audioInputBuffer);
                    final String emotionResult = postProcessClassification(logits);

                    // Update UI on the main thread after successful inference
                    runOnUiThread(() -> {
                        resultText.setText("Emotion: " + emotionResult);
                        statusText.setText("Status: Analysis complete.");
                        analyzeButton.setEnabled(true);
                    });

                } catch (Exception e) {
                    // Handle errors and update UI on the main thread
                    Log.e(TAG, "Audio analysis failed.", e);

                    // getLocalizedMessage() is safe inside the background thread
                    final String errorMessage = "Analysis Error: " + e.getLocalizedMessage();

                    runOnUiThread(() -> {
                        statusText.setText(errorMessage);
                        resultText.setText("Emotion: ERROR");
                        analyzeButton.setEnabled(true);
                    });
                }
            }
        });
    }

    // --- TEXT SENTIMENT ANALYSIS ---

    private void runSentimentAnalysis(String text) {
        if (text.isEmpty() || vocab == null) {
            Toast.makeText(this, "Please enter text.", Toast.LENGTH_SHORT).show();
            return;
        }

        statusText.setText("Status: Analyzing text...");
        analyzeButton.setEnabled(false);

        Executors.newSingleThreadExecutor().submit(() -> {
            try {
                ByteBuffer[] inputs = TextPreprocessor.packageTextInputs(text, vocab, config.sequenceLength);
                float[][] logits = predictor.runInference(inputs);

                String result = postProcessClassification(logits); // Use generic post-processor

                runOnUiThread(() -> {
                    resultText.setText("Sentiment: " + result);
                    statusText.setText("Status: Analysis complete.");
                    analyzeButton.setEnabled(true);
                });
            } catch (Exception e) {
                Log.e(TAG, "Error during sentiment analysis.", e);
                runOnUiThread(() -> {
                    statusText.setText("Analysis Error.");
                    analyzeButton.setEnabled(true);
                });
            }
        });
    }

    /**
     * Generic post-processor to find the highest logit for ANY classification task.
     */
    private String postProcessClassification(float[][] output) {
        float maxLogit = Float.MIN_VALUE;
        int predictedClass = -1;

        for (int i = 0; i < config.outputClasses; i++) {
            if (output[0][i] > maxLogit) {
                maxLogit = output[0][i];
                predictedClass = i;
            }
        }

        if (predictedClass >= 0 && predictedClass < config.outputLabels.length) {
            return config.outputLabels[predictedClass];
        }
        return "UNKNOWN_CLASS";
    }

    // --- Resource Cleanup ---
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (predictor != null) {
            predictor.close();
        }
    }
}