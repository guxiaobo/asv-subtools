package com.asv.sdk;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.UUID;

/**
 * Java SDK client for the ASV Speaker Verification API.
 * <p>
 * Uses {@link java.net.http.HttpClient} (Java 11+) with no external dependencies.
 * <p>
 * Usage:
 * <pre>{@code
 * ASVClient client = new ASVClient("http://localhost:8000");
 *
 * // Mode A: direct file upload
 * ASVResult result = client.verifyFiles(
 *     Path.of("/path/to/a.wav"),
 *     Path.of("/path/to/b.wav"),
 *     "debt_collection",
 *     null, null
 * );
 *
 * // Mode B: indirect by audio ID
 * ASVResult result2 = client.verifyIds(
 *     "recording-001", "recording-002",
 *     "nas", "nas",
 *     "customer_service", null, null
 * );
 *
 * client.close();
 * }</pre>
 */
public class ASVClient implements AutoCloseable {

    private static final String DEFAULT_BASE_URL = "http://localhost:8000";
    private static final int DEFAULT_TIMEOUT_SEC = 30;
    private static final int MAX_RETRIES = 2;
    private static final String BOUNDARY_PREFIX = "asv-java-sdk-";

    private final String baseUrl;
    private final String apiKey;
    private final int timeoutSec;
    private final HttpClient httpClient;

    // ────────────────────────────────────────────────────────────────────
    // Constructors
    // ────────────────────────────────────────────────────────────────────

    /**
     * Create a new ASV client with default settings.
     *
     * @param baseUrl API base URL (e.g. "http://localhost:8000").
     * @param apiKey  Optional API key (sent as Bearer token). Nullable.
     */
    public ASVClient(String baseUrl, String apiKey, int timeoutSec) {
        this.baseUrl = (baseUrl != null) ? baseUrl.replaceAll("/+$", "") : DEFAULT_BASE_URL;
        this.apiKey = apiKey;
        this.timeoutSec = (timeoutSec > 0) ? timeoutSec : DEFAULT_TIMEOUT_SEC;

        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(this.timeoutSec))
                .build();
    }

    /**
     * Create client with default timeout (30s).
     */
    public ASVClient(String baseUrl, String apiKey) {
        this(baseUrl, apiKey, DEFAULT_TIMEOUT_SEC);
    }

    /**
     * Create client with no API key and default timeout.
     */
    public ASVClient(String baseUrl) {
        this(baseUrl, null, DEFAULT_TIMEOUT_SEC);
    }

    /**
     * Create client with all defaults (http://localhost:8000).
     */
    public ASVClient() {
        this(DEFAULT_BASE_URL, null, DEFAULT_TIMEOUT_SEC);
    }

    // ────────────────────────────────────────────────────────────────────
    // Verify — Mode A: Direct file upload
    // ────────────────────────────────────────────────────────────────────

    /**
     * Verify two speakers by uploading audio files.
     *
     * @param audioA        Path to speaker A's audio file.
     * @param audioB        Path to speaker B's audio file.
     * @param scenario      Business scenario (nullable).
     * @param threshold     Decision threshold override (nullable).
     * @param scoringMethod Scoring method (nullable: "cosine", "euclidean", "dot_product").
     * @return ASVResult with score and decision.
     * @throws ASVException On network or server errors.
     */
    public ASVResult verifyFiles(
            Path audioA, Path audioB,
            String scenario, Double threshold, String scoringMethod
    ) {
        // Validate files
        if (!Files.exists(audioA)) {
            throw new ASVException("File not found: " + audioA);
        }
        if (!Files.exists(audioB)) {
            throw new ASVException("File not found: " + audioB);
        }

        try {
            String boundary = BOUNDARY_PREFIX + UUID.randomUUID();
            byte[] body = buildMultipartBody(
                boundary, audioA, audioB, scenario, threshold, scoringMethod
            );

            String response = doRequest(
                "POST",
                "/api/verify",
                "multipart/form-data; boundary=" + boundary,
                body
            );
            return ASVResult.fromJson(response);

        } catch (ASVException e) {
            throw e;
        } catch (Exception e) {
            throw new ASVException("File upload failed: " + e.getMessage(), e);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Verify — Mode B: Indirect by audio ID
    // ────────────────────────────────────────────────────────────────────

    /**
     * Verify two speakers by audio ID (indirect retrieval).
     *
     * @param audioIdA      Audio ID for speaker A.
     * @param audioIdB      Audio ID for speaker B.
     * @param backendA      Storage backend for A ("nas", "s3", "redis").
     * @param backendB      Storage backend for B.
     * @param scenario      Business scenario (nullable).
     * @param threshold     Decision threshold override (nullable).
     * @param scoringMethod Scoring method (nullable).
     * @return ASVResult with score and decision.
     */
    public ASVResult verifyIds(
            String audioIdA, String audioIdB,
            String backendA, String backendB,
            String scenario, Double threshold, String scoringMethod
    ) {
        if (audioIdA == null || audioIdA.isEmpty()) {
            throw new ASVException("audioIdA must not be empty");
        }
        if (audioIdB == null || audioIdB.isEmpty()) {
            throw new ASVException("audioIdB must not be empty");
        }

        StringBuilder json = new StringBuilder();
        json.append("{");
        json.append("\"mode\":\"indirect\",");
        json.append("\"audio_a\":{");
        json.append("\"audio_id\":\"").append(escapeJson(audioIdA)).append("\",");
        json.append("\"storage_backend\":\"").append(escapeJson(backendA != null ? backendA : "nas")).append("\"");
        json.append("},");
        json.append("\"audio_b\":{");
        json.append("\"audio_id\":\"").append(escapeJson(audioIdB)).append("\",");
        json.append("\"storage_backend\":\"").append(escapeJson(backendB != null ? backendB : "nas")).append("\"");
        json.append("}");
        if (scenario != null) {
            json.append(",\"scenario\":\"").append(escapeJson(scenario)).append("\"");
        }
        if (threshold != null) {
            json.append(",\"threshold\":").append(threshold);
        }
        if (scoringMethod != null) {
            json.append(",\"scoring_method\":\"").append(escapeJson(scoringMethod)).append("\"");
        }
        json.append("}");

        try {
            String response = doRequest(
                "POST",
                "/api/verify/indirect",
                "application/json",
                json.toString().getBytes(StandardCharsets.UTF_8)
            );
            return ASVResult.fromJson(response);
        } catch (ASVException e) {
            throw e;
        } catch (Exception e) {
            throw new ASVException("ID verification failed: " + e.getMessage(), e);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Health check
    // ────────────────────────────────────────────────────────────────────

    /**
     * Query the API health endpoint.
     *
     * @return JSON response string (parsing left to the caller).
     */
    public String health() {
        try {
            return doRequest("GET", "/health", null, null);
        } catch (Exception e) {
            throw new ASVException("Health check failed: " + e.getMessage(), e);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Internal HTTP handling
    // ────────────────────────────────────────────────────────────────────

    private String doRequest(String method, String path,
                             String contentType, byte[] body) {
        String url = baseUrl + path;

        ASVException lastError = null;

        for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            try {
                HttpRequest.Builder builder = HttpRequest.newBuilder()
                        .uri(URI.create(url))
                        .timeout(Duration.ofSeconds(timeoutSec))
                        .version(HttpClient.Version.HTTP_1_1);

                // Headers
                builder.header("User-Agent", "asv-sdk-java/0.1.0");
                if (apiKey != null && !apiKey.isEmpty()) {
                    builder.header("Authorization", "Bearer " + apiKey);
                }
                if (contentType != null) {
                    builder.header("Content-Type", contentType);
                }

                // Method
                HttpRequest request;
                if ("GET".equalsIgnoreCase(method)) {
                    request = builder.GET().build();
                } else if ("POST".equalsIgnoreCase(method)) {
                    if (body != null) {
                        request = builder.POST(HttpRequest.BodyPublishers.ofByteArray(body)).build();
                    } else {
                        request = builder.POST(HttpRequest.BodyPublishers.noBody()).build();
                    }
                } else {
                    throw new ASVException("Unsupported method: " + method);
                }

                HttpResponse<String> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofString()
                );

                int code = response.statusCode();
                String respBody = response.body();

                if (code >= 200 && code < 300) {
                    return respBody;
                }

                // Server error
                lastError = new ASVException(
                    "Server returned HTTP " + code + ": " + extractErrorMessage(respBody),
                    code, null
                );

                // Don't retry on 4xx
                if (code < 500) {
                    throw lastError;
                }

            } catch (ASVException e) {
                throw e;
            } catch (IOException | InterruptedException e) {
                lastError = new ASVException("Network error: " + e.getMessage(), e);
                if (attempt < MAX_RETRIES) {
                    try {
                        Thread.sleep((attempt + 1) * 1000L);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new ASVException("Interrupted during retry", ie);
                    }
                    continue;
                }
            }
        }

        throw (lastError != null)
            ? lastError
            : new ASVException("Request failed after " + (MAX_RETRIES + 1) + " attempts");
    }

    // ────────────────────────────────────────────────────────────────────
    // Multipart body builder (no external deps)
    // ────────────────────────────────────────────────────────────────────

    private byte[] buildMultipartBody(
            String boundary,
            Path audioA, Path audioB,
            String scenario, Double threshold, String scoringMethod
    ) throws IOException {
        byte[] boundaryBytes = ("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8);
        byte[] boundaryEndBytes = ("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8);
        byte[] crlf = "\r\n".getBytes(StandardCharsets.UTF_8);

        byte[] aName = audioA.getFileName().toString().getBytes(StandardCharsets.UTF_8);
        byte[] bName = audioB.getFileName().toString().getBytes(StandardCharsets.UTF_8);
        byte[] aData = Files.readAllBytes(audioA);
        byte[] bData = Files.readAllBytes(audioB);

        java.io.ByteArrayOutputStream os = new java.io.ByteArrayOutputStream();

        // Field: audio_a
        os.write(boundaryBytes);
        writeHeader(os, "Content-Disposition: form-data; name=\"audio_a\"; filename=\"", aName, "\"");
        writeHeader(os, "Content-Type: " + guessMimeType(audioA));
        os.write(crlf);
        os.write(aData);
        os.write(crlf);

        // Field: audio_b
        os.write(boundaryBytes);
        writeHeader(os, "Content-Disposition: form-data; name=\"audio_b\"; filename=\"", bName, "\"");
        writeHeader(os, "Content-Type: " + guessMimeType(audioB));
        os.write(crlf);
        os.write(bData);
        os.write(crlf);

        // Optional fields
        if (scenario != null) {
            os.write(boundaryBytes);
            writeHeader(os, "Content-Disposition: form-data; name=\"scenario\"");
            os.write(crlf);
            os.write(scenario.getBytes(StandardCharsets.UTF_8));
            os.write(crlf);
        }
        if (threshold != null) {
            os.write(boundaryBytes);
            writeHeader(os, "Content-Disposition: form-data; name=\"threshold\"");
            os.write(crlf);
            os.write(Double.toString(threshold).getBytes(StandardCharsets.UTF_8));
            os.write(crlf);
        }
        if (scoringMethod != null) {
            os.write(boundaryBytes);
            writeHeader(os, "Content-Disposition: form-data; name=\"scoring_method\"");
            os.write(crlf);
            os.write(scoringMethod.getBytes(StandardCharsets.UTF_8));
            os.write(crlf);
        }

        // End boundary
        os.write(boundaryEndBytes);

        return os.toByteArray();
    }

    private void writeHeader(java.io.ByteArrayOutputStream os, String prefix,
                             byte[] name, String suffix) throws IOException {
        os.write(prefix.getBytes(StandardCharsets.UTF_8));
        os.write(name);
        os.write(suffix.getBytes(StandardCharsets.UTF_8));
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
    }

    private void writeHeader(java.io.ByteArrayOutputStream os,
                             String header) throws IOException {
        os.write(header.getBytes(StandardCharsets.UTF_8));
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static String guessMimeType(Path file) {
        String name = file.getFileName().toString().toLowerCase();
        if (name.endsWith(".wav")) return "audio/wav";
        if (name.endsWith(".mp3")) return "audio/mpeg";
        if (name.endsWith(".flac")) return "audio/flac";
        if (name.endsWith(".ogg")) return "audio/ogg";
        if (name.endsWith(".ulaw") || name.endsWith(".alaw")) return "audio/basic";
        return "application/octet-stream";
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String extractErrorMessage(String jsonBody) {
        try {
            // Simple extraction: look for "message" or "error" field
            String msg = extractJsonString(jsonBody, "message");
            if (msg != null) return msg;
            msg = extractJsonString(jsonBody, "error");
            if (msg != null) return msg;
            msg = extractJsonString(jsonBody, "detail");
            if (msg != null) return msg;
            return jsonBody.length() > 200 ? jsonBody.substring(0, 200) + "..." : jsonBody;
        } catch (Exception e) {
            return jsonBody;
        }
    }

    private static String extractJsonString(String json, String key) {
        String search = "\"" + key + "\":\"";
        int idx = json.indexOf(search);
        if (idx < 0) return null;
        idx += search.length();
        StringBuilder sb = new StringBuilder();
        while (idx < json.length() && json.charAt(idx) != '"') {
            if (json.charAt(idx) == '\\') { idx++; if (idx >= json.length()) break; }
            sb.append(json.charAt(idx));
            idx++;
        }
        return sb.toString();
    }

    @Override
    public void close() {
        // HttpClient has a dedicated executor — we let GC handle it.
        // Override if using a custom executor that needs cleanup.
    }
}
