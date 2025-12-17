package com.example.myapplication;

public class WhisperBridge {
    static {
        System.loadLibrary("native-lib");
    }

    public static native long init(String modelPath);
    public static native void free(long handle);
    public static native String transcribeWav(long handle, String wavPath);
}
