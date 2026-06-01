import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Test multipart using classic HttpURLConnection.
 */
public class EchoTest {
    static final String BOUNDARY = "-----------------------asv-test-" + System.currentTimeMillis();

    public static void main(String[] args) throws Exception {
        Path audioA = Path.of("test_data/public/us_0010.wav");
        Path audioB = Path.of("test_data/public/us_0010.wav");
        byte[] aData = Files.readAllBytes(audioA);
        byte[] bData = Files.readAllBytes(audioB);

        ByteArrayOutputStream os = new ByteArrayOutputStream();
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(os, StandardCharsets.UTF_8), true);

        // Part 1: audio_a
        pw.append("--").append(BOUNDARY).append("\r\n");
        pw.append("Content-Disposition: form-data; name=\"audio_a\"; filename=\"us_0010.wav\"\r\n");
        pw.append("Content-Type: audio/wav\r\n");
        pw.append("\r\n").flush();
        os.write(aData);
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
        pw.flush();

        // Part 2: audio_b
        pw.append("--").append(BOUNDARY).append("\r\n");
        pw.append("Content-Disposition: form-data; name=\"audio_b\"; filename=\"us_0010.wav\"\r\n");
        pw.append("Content-Type: audio/wav\r\n");
        pw.append("\r\n").flush();
        os.write(bData);
        os.write("\r\n".getBytes(StandardCharsets.UTF_8));
        pw.flush();

        // End
        pw.append("--").append(BOUNDARY).append("--\r\n").flush();

        byte[] body = os.toByteArray();
        System.out.println("Body: " + body.length + " bytes");
        System.out.println("Boundary: " + BOUNDARY);

        // Send via HttpURLConnection
        HttpURLConnection conn = (HttpURLConnection) new URL("http://localhost:8000/api/verify").openConnection();
        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + BOUNDARY);
        conn.setRequestProperty("Content-Length", String.valueOf(body.length));
        conn.connect();

        OutputStream out = conn.getOutputStream();
        out.write(body);
        out.flush();
        out.close();

        int code = conn.getResponseCode();
        System.out.println("HTTP " + code);

        BufferedReader reader = new BufferedReader(
            new InputStreamReader(
                code >= 200 && code < 300 ? conn.getInputStream() : conn.getErrorStream(),
                StandardCharsets.UTF_8
            )
        );
        String line;
        StringBuilder resp = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            resp.append(line).append("\n");
        }
        reader.close();
        System.out.println("Response: " + resp.toString().substring(0, Math.min(300, resp.length())));
    }
}
