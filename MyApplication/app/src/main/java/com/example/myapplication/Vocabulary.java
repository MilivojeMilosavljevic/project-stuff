package com.example.myapplication;

import android.content.Context;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

/**
 * Handles loading and lookup for WordPiece vocabulary files (vocab.txt).
 * This is specific to BERT/DistilBERT family tokenization.
 */
public class Vocabulary {

    private final HashMap<String, Integer> token2id = new HashMap<>();
    private int padId = 0;
    private int unkId = 100;
    private final int clsId = 101;
    private final int sepId = 102;

    public Vocabulary(Context context, String vocabFile) throws IOException {
        if (vocabFile == null) return; // Skip initialization for models without vocab (e.g., Audio)

        // Load the vocab file from assets
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(vocabFile)));
        String line;
        int idx = 0;
        while ((line = reader.readLine()) != null) {
            token2id.put(line.trim(), idx++);
        }
        reader.close();

        if (token2id.containsKey("[PAD]")) padId = token2id.get("[PAD]");
        if (token2id.containsKey("[UNK]")) unkId = token2id.get("[UNK]");
    }

    public int getIdOrUnknown(String token) {
        return token2id.getOrDefault(token, unkId);
    }

    public int getPadId() { return padId; }
    public int getClsId() { return clsId; }
    public int getSepId() { return sepId; }
}