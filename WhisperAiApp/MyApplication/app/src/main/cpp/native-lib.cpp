#include <jni.h>
#include <android/log.h>

#include <vector>     // std::vector
#include <string>     // std::string
#include <cstdio>     // FILE, fopen, fread, fclose
#include <cstdint>    // int16_t
#include <cstring>

#include "whisper.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "WHISPER_JNI", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "WHISPER_JNI", __VA_ARGS__)

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_myapplication_WhisperBridge_init(JNIEnv* env, jclass, jstring modelPathJ) {
    const char* modelPath = env->GetStringUTFChars(modelPathJ, nullptr);

    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false; // keep it simple

    whisper_context* ctx = whisper_init_from_file_with_params(modelPath, params);

    env->ReleaseStringUTFChars(modelPathJ, modelPath);

    if (!ctx) {
        LOGE("Failed to init whisper context");
        return 0;
    }

    LOGI("Whisper init OK");
    return reinterpret_cast<jlong>(ctx);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_WhisperBridge_free(JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<whisper_context*>(handle);
    if (ctx) whisper_free(ctx);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_WhisperBridge_transcribeWav(
        JNIEnv* env,
        jclass,
        jlong handle,
        jstring wavPathJ) {

    auto* ctx = reinterpret_cast<whisper_context*>(handle);
    if (!ctx) {
        return env->NewStringUTF("ERROR: ctx is null");
    }

    const char* wavPath = env->GetStringUTFChars(wavPathJ, nullptr);

    // ---- Load WAV file (16-bit PCM, mono, 16kHz) ----
    std::vector<float> pcmf32;
    {
        FILE* f = fopen(wavPath, "rb");
        if (!f) {
            env->ReleaseStringUTFChars(wavPathJ, wavPath);
            return env->NewStringUTF("ERROR: cannot open WAV file");
        }

        fseek(f, 44, SEEK_SET); // skip WAV header
        int16_t sample;
        while (fread(&sample, sizeof(int16_t), 1, f) == 1) {
            pcmf32.push_back(sample / 32768.0f);
        }
        fclose(f);
    }

    env->ReleaseStringUTFChars(wavPathJ, wavPath);

    // ---- Whisper parameters ----
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = "en";
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    // ---- Run transcription ----
    int ret = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());
    if (ret != 0) {
        return env->NewStringUTF("ERROR: whisper_full failed");
    }

    // ---- Collect text ----
    std::string result;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        result += whisper_full_get_segment_text(ctx, i);
    }

    return env->NewStringUTF(result.c_str());
}
