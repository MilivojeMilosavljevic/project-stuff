package com.example.myapplication;

import android.os.Bundle;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private long whisperHandle = 0;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1) Copy model from assets to internal storage (only once)
        String modelAssetName = "ggml-small-q5_0.bin";
        File modelFile = new File(getFilesDir(), modelAssetName);

        if (!modelFile.exists()) {
            try (InputStream is = getAssets().open(modelAssetName);
                 OutputStream os = new FileOutputStream(modelFile)) {

                byte[] buf = new byte[1024 * 1024];
                int r;
                while ((r = is.read(buf)) != -1) os.write(buf, 0, r);
                os.flush();
            } catch (Exception e) {
                Log.e("WHISPER", "Model copy failed", e);
            }
        }

        // 2) Read wav file
        String wavAssetName = "jfk.wav";
        File wavFile = new File(getFilesDir(), wavAssetName);

        if (!wavFile.exists()) {
            try (InputStream is = getAssets().open(wavAssetName);
                 OutputStream os = new FileOutputStream(wavFile)) {

                byte[] buf = new byte[1024 * 1024];
                int r;
                while ((r = is.read(buf)) != -1) os.write(buf, 0, r);
                os.flush();
            } catch (Exception e) {
                Log.e("WHISPER", "WAV copy failed", e);
            }
        }


        // 3) Init Whisper (put this on background thread too, to avoid ANR)
        executor.execute(() -> {
            whisperHandle = WhisperBridge.init(modelFile.getAbsolutePath());
            Log.d("WHISPER", "init handle=" + whisperHandle);

            if (whisperHandle == 0) {
                Log.e("WHISPER", "Whisper init failed");
                return;
            }

            // 3) Transcribe a WAV file (CHANGE THIS PATH)
            String wavPath = wavFile.getAbsolutePath();

            String text = WhisperBridge.transcribeWav(whisperHandle, wavPath);
            Log.d("WHISPER", "Result: " + text);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        executor.shutdown();

        if (whisperHandle != 0) {
            WhisperBridge.free(whisperHandle);
            whisperHandle = 0;
        }
    }
}
