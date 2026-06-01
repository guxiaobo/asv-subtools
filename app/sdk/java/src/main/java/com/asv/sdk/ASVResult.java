package com.asv.sdk;

/**
 * Result of a speaker verification request.
 * <p>
 * Immutable data class holding the verification score, decision, and metadata.
 */
public class ASVResult {

    private final boolean success;
    private final boolean isSameSpeaker;
    private final double score;
    private final double thresholdUsed;
    private final double processingTimeMs;
    private final int embeddingADimension;
    private final String embeddingASource;
    private final int embeddingBDimension;
    private final String embeddingBSource;
    private final String scenario;
    private final String error;

    // Audio info (nullable)
    private final AudioInfo audioA;
    private final AudioInfo audioB;

    /**
     * Full constructor.
     */
    public ASVResult(boolean success, boolean isSameSpeaker, double score,
                     double thresholdUsed, double processingTimeMs,
                     int embeddingADimension, String embeddingASource,
                     int embeddingBDimension, String embeddingBSource,
                     String scenario, String error,
                     AudioInfo audioA, AudioInfo audioB) {
        this.success = success;
        this.isSameSpeaker = isSameSpeaker;
        this.score = score;
        this.thresholdUsed = thresholdUsed;
        this.processingTimeMs = processingTimeMs;
        this.embeddingADimension = embeddingADimension;
        this.embeddingASource = embeddingASource;
        this.embeddingBDimension = embeddingBDimension;
        this.embeddingBSource = embeddingBSource;
        this.scenario = scenario;
        this.error = error;
        this.audioA = audioA;
        this.audioB = audioB;
    }

    public boolean isSuccess() { return success; }
    public boolean isSameSpeaker() { return isSameSpeaker; }
    public double getScore() { return score; }
    public double getThresholdUsed() { return thresholdUsed; }
    public double getProcessingTimeMs() { return processingTimeMs; }
    public int getEmbeddingADimension() { return embeddingADimension; }
    public String getEmbeddingASource() { return embeddingASource; }
    public int getEmbeddingBDimension() { return embeddingBDimension; }
    public String getEmbeddingBSource() { return embeddingBSource; }
    public String getScenario() { return scenario; }
    public String getError() { return error; }
    public AudioInfo getAudioA() { return audioA; }
    public AudioInfo getAudioB() { return audioB; }

    @Override
    public String toString() {
        return String.format(
            "ASVResult{success=%s, isSameSpeaker=%s, score=%.4f, " +
            "threshold=%.4f, time=%.1fms, scenario=%s}",
            success, isSameSpeaker, score, thresholdUsed, processingTimeMs, scenario
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Nested AudioInfo
    // ────────────────────────────────────────────────────────────────────
    public static class AudioInfo {
        private final double durationSec;
        private final int sampleRate;
        private final double validSpeechSec;
        private final int channels;

        public AudioInfo(double durationSec, int sampleRate,
                         double validSpeechSec, int channels) {
            this.durationSec = durationSec;
            this.sampleRate = sampleRate;
            this.validSpeechSec = validSpeechSec;
            this.channels = channels;
        }

        public double getDurationSec() { return durationSec; }
        public int getSampleRate() { return sampleRate; }
        public double getValidSpeechSec() { return validSpeechSec; }
        public int getChannels() { return channels; }
    }

    // ────────────────────────────────────────────────────────────────────
    // Parser from JSON (org.json)
    // ────────────────────────────────────────────────────────────────────
    static ASVResult fromJson(String jsonText) {
        // Using basic string parsing to avoid adding org.json dependency
        // In production, use Jackson/Gson
        try {
            boolean success = findBool(jsonText, "success", true);
            boolean sameSpeaker = findBool(jsonText, "is_same_speaker", false);
            double score = findDouble(jsonText, "score", 0.0);
            double threshold = findDouble(jsonText, "threshold_used", 0.0);
            double timeMs = findDouble(jsonText, "processing_time_ms", 0.0);
            String scenario = findString(jsonText, "scenario", null);
            String error = findString(jsonText, "error", null);

            // Embedding A
            int embADim = findInt(jsonText, "embedding_a.dimension", 0);
            String embASrc = findString(jsonText, "embedding_a.source", "computed");
            int embBDim = findInt(jsonText, "embedding_b.dimension", 0);
            String embBSrc = findString(jsonText, "embedding_b.source", "computed");

            return new ASVResult(
                success, sameSpeaker, score, threshold, timeMs,
                embADim, embASrc, embBDim, embBSrc,
                scenario, error, null, null
            );
        } catch (Exception e) {
            throw new ASVException("Failed to parse response: " + e.getMessage(), e);
        }
    }

    // Simple JSON field extractors (no dependency)
    private static boolean findBool(String json, String key, boolean def) {
        String val = findString(json, key, null);
        if (val == null) return def;
        return "true".equalsIgnoreCase(val);
    }

    private static double findDouble(String json, String key, double def) {
        try {
            // Handle nested keys like "embedding_a.dimension"
            String search = "\"" + key + "\":";
            int idx = json.indexOf(search);
            if (idx < 0) {
                // Try nested path
                if (key.contains(".")) {
                    String[] parts = key.split("\\.");
                    String outer = "\"" + parts[0] + "\":\\{";
                    java.util.regex.Pattern p = java.util.regex.Pattern.compile(outer);
                    java.util.regex.Matcher m = p.matcher(json);
                    if (m.find()) {
                        int start = m.end();
                        // Find matching closing brace
                        int depth = 1;
                        int end = start;
                        while (depth > 0 && end < json.length()) {
                            char c = json.charAt(end);
                            if (c == '{') depth++;
                            else if (c == '}') depth--;
                            end++;
                        }
                        String inner = json.substring(start, end - 1);
                        return findDouble(inner, parts[1], def);
                    }
                }
                return def;
            }
            idx += search.length();
            // Skip whitespace
            while (idx < json.length() && json.charAt(idx) == ' ') idx++;
            // Read value until comma or closing brace
            StringBuilder sb = new StringBuilder();
            while (idx < json.length() && json.charAt(idx) != ',' && json.charAt(idx) != '}' && json.charAt(idx) != ']') {
                sb.append(json.charAt(idx));
                idx++;
            }
            return Double.parseDouble(sb.toString().trim());
        } catch (Exception e) {
            return def;
        }
    }

    private static int findInt(String json, String key, int def) {
        return (int) findDouble(json, key, def);
    }

    private static String findString(String json, String key, String def) {
        String search = "\"" + key + "\":\"";
        int idx = json.indexOf(search);
        if (idx < 0) return def;
        idx += search.length();
        StringBuilder sb = new StringBuilder();
        while (idx < json.length() && json.charAt(idx) != '"') {
            if (json.charAt(idx) == '\\') {
                idx++;
                if (idx >= json.length()) break;
            }
            sb.append(json.charAt(idx));
            idx++;
        }
        return sb.toString();
    }
}
