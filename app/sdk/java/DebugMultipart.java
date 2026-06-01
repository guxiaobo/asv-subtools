import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

/**
 * Debug: test multipart upload manually
 */
public class DebugMultipart {
    static final String BOUNDARY_PREFIX = "test-boundary-";

    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newBuilder().build();

        Path audioA = Path.of("test_data/public/us_0010.wav");
        Path audioB = Path.of("test_data/public/us_0010.wav");

        String boundary = BOUNDARY_PREFIX + UUID.randomUUID().toString().substring(0, 8);
        byte[] aData = Files.readAllBytes(audioA);
        byte[] bData = Files.readAllBytes(audioB);
        byte[] crlf = "\r\n".getBytes(StandardCharsets.UTF_8);

        String part1 = "--" + boundary + "\r\n" +
            "Content-Disposition: form-data; name=\"audio_a\"; filename=\"us_0010.wav\"\r\n" +
            "Content-Type: audio/wav\r\n\r\n";

        String part2 = "\r\n--" + boundary + "\r\n" +
            "Content-Disposition: form-data; name=\"audio_b\"; filename=\"us_0010.wav\"\r\n" +
            "Content-Type: audio/wav\r\n\r\n";

        String part3 = "\r\n--" + boundary + "--\r\n";

        byte[] header1 = part1.getBytes(StandardCharsets.UTF_8);
        byte[] header2 = part2.getBytes(StandardCharsets.UTF_8);
        byte[] footer = part3.getBytes(StandardCharsets.UTF_8);

        // Print header format
        System.out.print("=== Part1 header (" + header1.length + " bytes) ===\n");
        System.out.print(part1);
        System.out.print("[WAV DATA: " + aData.length + " bytes]\n");
        System.out.print("=== End preview ===\n");

        java.io.ByteArrayOutputStream os = new java.io.ByteArrayOutputStream();
        os.write(header1);
        os.write(aData);
        os.write(header2);
        os.write(bData);
        os.write(footer);
        byte[] body = os.toByteArray();

        System.out.println("Total body: " + body.length + " bytes");
        System.out.println("Content-Type: multipart/form-data; boundary=" + boundary);

        // Send request
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:8000/api/verify"))
            .header("Content-Type", "multipart/form-data; boundary=" + boundary)
            .POST(HttpRequest.BodyPublishers.ofByteArray(body))
            .build();

        HttpResponse<String> response = client.send(request,
            HttpResponse.BodyHandlers.ofString());

        System.out.println("HTTP " + response.statusCode());
        System.out.println("Response: " + response.body().substring(0, Math.min(200, response.body().length())));
    }
}
