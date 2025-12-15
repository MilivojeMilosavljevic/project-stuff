package com.example.myapplication;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class TfLitePredictor {

    private Interpreter tflite;
    private ModelConfig config;

    public TfLitePredictor(Context context, ModelConfig config) throws IOException {
        this.config = config;
        MappedByteBuffer tfliteModel = loadModelFile(context, config.modelFileName);
        tflite = new Interpreter(tfliteModel);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Runs inference on the model using a single pre-packaged input buffer (USED for Audio).
     */
    public float[][] runInference(ByteBuffer input) {
        float[][] output = new float[1][config.outputClasses];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(config.outputTensorIndex, output);
        Object[] inputsArray = {input};
        tflite.runForMultipleInputsOutputs(inputsArray, outputs);
        return output;
    }

    /**
     * Runs inference on the model using multiple input buffers (USED for Sentiment/Text).
     */
    public float[][] runInference(ByteBuffer[] inputs) {
        float[][] output = new float[1][config.outputClasses];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(config.outputTensorIndex, output);

        tflite.runForMultipleInputsOutputs(inputs, outputs);
        return output;
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
        }
    }
}