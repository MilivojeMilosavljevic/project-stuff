package com.example.myapplication;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;
import android.annotation.SuppressLint;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Handles recording audio and converting it into the required TFLite input format (Float32 array).
 * Wav2Vec2 requires 16kHz, mono, Float32 input.
 */
public class AudioPreprocessor {
    private static final String TAG = "AudioProc";

    // Wav2Vec2 requires 16kHz, mono, 16-bit PCM
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    // We use a fixed duration for recording (4 seconds)
    private static final int RECORDING_DURATION_SECONDS = 4;
    private final int recordingLengthSamples; // Calculated based on sample rate
    private final int sampleRate;

    public AudioRecord recorder = null;
    private int bufferSize;

    public AudioPreprocessor(int sampleRate) {
        this.sampleRate = sampleRate;
        // Total samples needed for a fixed duration
        this.recordingLengthSamples = sampleRate * RECORDING_DURATION_SECONDS;

        bufferSize = AudioRecord.getMinBufferSize(sampleRate, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSize == AudioRecord.ERROR_BAD_VALUE || bufferSize == AudioRecord.ERROR) {
            // Default safe buffer size if minBufferSize fails
            bufferSize = recordingLengthSamples;
        }

        // Initialize AudioRecord - Needs @SuppressLint
        initializeRecorder();
    }

    // Ensures AudioRecord is created/released with proper state management
    @SuppressLint("MissingPermission")
    private void initializeRecorder() {
        if (recorder != null) {
            recorder.release();
        }
        try {
            // Set source to VOICE_RECOGNITION for cleaner audio
            recorder = new AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    sampleRate,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    bufferSize
            );
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize AudioRecord", e);
        }
    }

    /**
     * Records a fixed length of audio and converts it to a Float32 ByteBuffer.
     * @return ByteBuffer containing the raw Float32 audio samples.
     */
    @SuppressLint("MissingPermission")
    public ByteBuffer recordAndProcess() throws Exception {
        // --- PRE-CHECK ---
        if (recorder == null || recorder.getState() != AudioRecord.STATE_INITIALIZED) {
            initializeRecorder(); // Attempt re-initialization
            if (recorder == null || recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                throw new IllegalStateException("AudioRecord not initialized or failed to initialize.");
            }
        }

        short[] audioBufferShort = new short[recordingLengthSamples];
        int shortsRead = 0;

        try {
            // Check if recording state is correct before starting
            if (recorder.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
                recorder.startRecording();
            }
            Log.d(TAG, "Recording started for " + RECORDING_DURATION_SECONDS + " seconds...");

            // Read audio data from the microphone
            shortsRead = recorder.read(audioBufferShort, 0, recordingLengthSamples);

            // --- ERROR CHECK ---
            if (shortsRead <= 0) {
                Log.e(TAG, "Failed to read audio data. shortsRead=" + shortsRead);
                // Stop the recorder only if we failed to read (to prevent lockup)
                if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    recorder.stop();
                }
                throw new RuntimeException("Audio reading failed with result: " + shortsRead);
            }

            Log.d(TAG, "Recording finished. Samples read: " + shortsRead);

            // Stop recording immediately after reading
            recorder.stop();


            // 1. Allocate the direct ByteBuffer (4 bytes per float)
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(shortsRead * 4)
                    .order(ByteOrder.nativeOrder());

            // 2. Create the FloatBuffer view of the ByteBuffer
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            // 3. Convert 16-bit short to 32-bit float and normalize to [-1.0, 1.0]
            for (int i = 0; i < shortsRead; i++) {
                // Normalization: Divide by 2^15 (32768.0f)
                floatBuffer.put(audioBufferShort[i] / 32768.0f);
            }

            // --- DIAGNOSTIC LOGGING (CRUCIAL!) ---
            // Check if actual sound data was captured.
            Log.d(TAG, "--- Diagnostic Audio Sample Check ---");
            for (int i = 0; i < Math.min(10, shortsRead); i++) {
                // Log the raw short value and the normalized float value
                float sampleValue = floatBuffer.get(i);
                Log.d(TAG, String.format("Sample %d: Raw=%d, Float=%.4f", i, audioBufferShort[i], sampleValue));
                if (Math.abs(sampleValue) > 0.0001f) {
                    Log.d(TAG, "Non-zero sample detected. Audio capture likely OK.");
                }
            }
            Log.d(TAG, "-------------------------------------");

            // Rewind the position to the start of the data (position 0) for the TFLite interpreter
            byteBuffer.position(0);

            return byteBuffer;

        } catch (Exception e) {
            // Ensure recording is stopped in case of an exception during the read process
            try {
                if (recorder != null && recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    recorder.stop();
                }
            } catch (Exception ignore) {}
            Log.e(TAG, "Error during audio recording or processing.", e);
            throw e;
        } finally {
            // Ensures the recorder is released and re-initialized for the next use.
            initializeRecorder();
        }
    }
}