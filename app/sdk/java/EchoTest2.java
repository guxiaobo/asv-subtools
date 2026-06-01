import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Test HttpClient with HTTP/1.1 explicitly set.
 * Send both audio_a and audio_b correctly.
 */
public class EchoTest2 {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newBuilder().build();
        Path audioA = Path.of("test_data/public/us_0010.wav");
        Path audioB = Path.of("test_data/public/us_0010.wav");
        byte[] aData = Files.readAllBytes(audioA);
        byte[] bData = Files.readAllBytes(audioB);
        String boundary = "test-boundary-raw";
        
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        
        // audio_a
        os.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        os.write("Content-Disposition: form-data; name=\"audio_a\"; filename=\"us_0010.wav\"\r\n".getBytes(StandardCharsets.UTF_8));
        os.write("Content-Type: audio/wav\r\n".getBytes(StandardCharsets.UTF_8));
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
        os.write(aData);
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
        
        // audio_b
        os.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        os.write("Content-Disposition: form-data; name=\"audio_b\"; filename=\"us_0010.wav\"\r\n".getBytes(StandardCharsets.UTF_8));
        os.write("Content-Type: audio/wav\r\n".getBytes(StandardCharsets.UTF_8));
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
        os.write(bData);
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
        
        // End
        os.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        
        byte[] body = os.toByteArray();
        System.out.println("Body: " + body.length + " bytes");
        
        // Try with explicit HTTP/1.1
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:8000/api/verify"))
            .header("Content-Type", "multipart/form-data; boundary=" + boundary)
            .version(HttpClient.Version.HTTP_1_1)
            .POST(HttpRequest.BodyPublishers.ofByteArray(body))
            .build();
        
        HttpResponse<String> response = client.send(request,
            HttpResponse.BodyHandlers.ofString());
        System.out.println("HTTP " + response.statusCode());
        System.out.println("Version: " + response.version());
        String respBody = response.body();
        System.out.println("Response: " + (respBody.length() > 300 ? respBody.substring(0, 300) : respBody));
        
        // Also check response headers
        System.out.println("Response headers:");
        response.headers().map().forEach((k, v) -> System.out.println("  " + k + ": " + v));
    }
}
