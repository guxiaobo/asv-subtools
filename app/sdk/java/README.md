# ASV Java SDK

Java client for the ASV Speaker Verification API. Zero external dependencies — uses `java.net.http.HttpClient` (JDK 11+).

## Requirements

- Java 11 or higher
- Maven 3.6+ (for building)

## Build

```bash
cd sdk/java
mvn clean package
```

This produces `target/asv-sdk-0.1.0.jar`.

## Usage

```java
import com.asv.sdk.ASVClient;
import com.asv.sdk.ASVResult;
import java.nio.file.Path;

public class Example {
    public static void main(String[] args) {
        // Create client
        ASVClient client = new ASVClient("http://localhost:8000");

        // Mode A: Direct file upload
        ASVResult result = client.verifyFiles(
            Path.of("/path/to/speaker_a.wav"),
            Path.of("/path/to/speaker_b.wav"),
            "debt_collection",   // scenario (nullable)
            0.7,                 // threshold (nullable)
            "cosine"             // scoring method (nullable)
        );

        System.out.println("Same speaker: " + result.isSameSpeaker());
        System.out.println("Score: " + result.getScore());
        System.out.println("Time: " + result.getProcessingTimeMs() + "ms");

        // Mode B: Indirect by audio ID
        ASVResult result2 = client.verifyIds(
            "recording-001",
            "recording-002",
            "nas",       // backend A
            "nas",       // backend B
            "customer_service",
            null,        // threshold (use server default)
            null         // scoring method (use server default)
        );

        // Health check
        String healthJson = client.health();
        System.out.println(healthJson);

        client.close();
    }
}
```

## API Reference

### `ASVClient`

| Constructor | Description |
|-------------|-------------|
| `ASVClient()` | Default: `http://localhost:8000` |
| `ASVClient(baseUrl)` | Custom base URL |
| `ASVClient(baseUrl, apiKey)` | With Bearer token auth |
| `ASVClient(baseUrl, apiKey, timeoutSec)` | Full config |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `verifyFiles(audioA, audioB, scenario, threshold, scoringMethod)` | `ASVResult` | Upload two audio files |
| `verifyIds(audioIdA, audioIdB, backendA, backendB, scenario, threshold, scoringMethod)` | `ASVResult` | Verify by audio ID |
| `health()` | `String` | Query health endpoint (raw JSON) |

### `ASVResult` fields

| Method | Type | Description |
|--------|------|-------------|
| `isSuccess()` | `boolean` | Request succeeded |
| `isSameSpeaker()` | `boolean` | Whether same speaker |
| `getScore()` | `double` | Similarity score |
| `getThresholdUsed()` | `double` | Decision threshold |
| `getProcessingTimeMs()` | `double` | Processing time |
| `getEmbeddingASource()` | `String` | "computed" or "cached" |
| `getEmbeddingADimension()` | `int` | Embedding dimension |
| `getError()` | `String` | Error message (nullable) |

## Maven Dependency

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>com.asv</groupId>
    <artifactId>asv-sdk</artifactId>
    <version>0.1.0</version>
</dependency>
```

Then build locally:

```bash
mvn install -DskipTests
```
