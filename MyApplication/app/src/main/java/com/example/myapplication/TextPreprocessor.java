package com.example.myapplication;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Utility class to handle text-specific pre-processing steps like tokenization and padding.
 */
public class TextPreprocessor {

    /**
     * Simple tokenization for BERT-like models.
     * It assumes the model expects [CLS]...[SEP] sequence.
     */
    private static int[] tokenizeSimple(String text, Vocabulary vocab, int maxLen) {
        int[] inputIds = new int[maxLen];

        inputIds[0] = vocab.getClsId();

        // Basic split using non-word characters as delimiters
        Pattern pattern = Pattern.compile("\\w+|[^\\w\\s]+");
        Matcher matcher = pattern.matcher(text.toLowerCase());

        int tokenIndex = 1;
        while (matcher.find() && tokenIndex < maxLen - 1) {
            String word = matcher.group();
            inputIds[tokenIndex++] = vocab.getIdOrUnknown(word);
        }

        if (tokenIndex < maxLen) {
            inputIds[tokenIndex++] = vocab.getSepId();
        }

        while (tokenIndex < maxLen) {
            inputIds[tokenIndex++] = vocab.getPadId();
        }

        return inputIds;
    }

    /**
     * Performs tokenization and packages the results into ByteBuffer inputs for BERT-like models.
     * @return An array of ByteBuffers containing Input IDs and Attention Mask.
     */
    public static ByteBuffer[] packageTextInputs(String text, Vocabulary vocab, int maxLen) {
        int[] inputIds = tokenizeSimple(text, vocab, maxLen);

        // Assumes all inputs are INT32 (4 bytes per int)
        int bufferSize = maxLen * 4;

        // Input IDs Buffer
        ByteBuffer inputIdsBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        for (int id : inputIds) {
            inputIdsBuffer.putInt(id);
        }
        inputIdsBuffer.rewind();

        // Attention Mask Buffer (1 for real tokens, 0 for PAD)
        ByteBuffer attentionMaskBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        for (int id : inputIds) {
            attentionMaskBuffer.putInt(id == vocab.getPadId() ? 0 : 1);
        }
        attentionMaskBuffer.rewind();

        return new ByteBuffer[]{inputIdsBuffer, attentionMaskBuffer};
    }
}