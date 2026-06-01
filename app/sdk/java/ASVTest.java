import com.asv.sdk.ASVClient;
import com.asv.sdk.ASVResult;
import java.nio.file.Path;

/**
 * ASV Java SDK 测试程序
 * 测试同说话人验证 + 不同说话人验证
 */
public class ASVTest {
    public static void main(String[] args) throws Exception {
        ASVClient client = new ASVClient("http://localhost:8000");

        // Test 1: Health check
        System.out.println("=== Health Check ===");
        String health = client.health();
        System.out.println("API health: " + health);
        System.out.println();

        // Test 2: Same speaker (same file)
        System.out.println("=== Same Speaker Test ===");
        ASVResult sameResult = client.verifyFiles(
            Path.of("test_data/public/us_0010.wav"),
            Path.of("test_data/public/us_0010.wav"),
            "debt_collection",
            null,
            null
        );
        System.out.println("Score: " + sameResult.getScore());
        System.out.println("Same speaker: " + sameResult.isSameSpeaker());
        System.out.println("Threshold: " + sameResult.getThresholdUsed());
        System.out.println("Time: " + sameResult.getProcessingTimeMs() + "ms");
        System.out.println();

        // Test 3: Cross speaker (different files)
        System.out.println("=== Cross Speaker Test ===");
        ASVResult crossResult = client.verifyFiles(
            Path.of("test_data/public/us_0017.wav"),
            Path.of("test_data/public/us_0038.wav"),
            "debt_collection",
            null,
            null
        );
        System.out.println("Score: " + crossResult.getScore());
        System.out.println("Same speaker: " + crossResult.isSameSpeaker());
        System.out.println("Threshold: " + crossResult.getThresholdUsed());
        System.out.println("Time: " + crossResult.getProcessingTimeMs() + "ms");
        System.out.println();

        // Test 4: Cross speaker (likely similar)
        System.out.println("=== Cross Speaker Test 2 ===");
        ASVResult cross2 = client.verifyFiles(
            Path.of("test_data/public/us_0010.wav"),
            Path.of("test_data/public/us_0011.wav"),
            "debt_collection",
            null,
            null
        );
        System.out.println("Score: " + cross2.getScore());
        System.out.println("Same speaker: " + cross2.isSameSpeaker());
        System.out.println("Threshold: " + cross2.getThresholdUsed());
        System.out.println("Time: " + cross2.getProcessingTimeMs() + "ms");
        System.out.println();

        // Test 5: With custom threshold
        System.out.println("=== Cross with high threshold (0.85) ===");
        ASVResult highThresh = client.verifyFiles(
            Path.of("test_data/public/us_0010.wav"),
            Path.of("test_data/public/us_0011.wav"),
            "debt_collection",
            0.85,  // higher threshold
            null
        );
        System.out.println("Score: " + highThresh.getScore());
        System.out.println("Same speaker: " + highThresh.isSameSpeaker());
        System.out.println("Threshold: " + highThresh.getThresholdUsed());
        System.out.println("    (with high threshold, cross-pair should be DIFF)");
        System.out.println();

        // Test 6: Batch - verify multiple pairs
        System.out.println("=== Batch Test (5 pairs) ===");
        String[][] pairs = {
            {"test_data/public/us_0012.wav", "test_data/public/us_0012.wav"},
            {"test_data/public/us_0013.wav", "test_data/public/us_0013.wav"},
            {"test_data/public/us_0013.wav", "test_data/public/us_0034.wav"},
            {"test_data/public/us_0014.wav", "test_data/public/us_0035.wav"},
            {"test_data/public/us_0017.wav", "test_data/public/us_0038.wav"},
        };
        for (int i = 0; i < pairs.length; i++) {
            ASVResult res = client.verifyFiles(
                Path.of(pairs[i][0]), Path.of(pairs[i][1]),
                null, null, null
            );
            System.out.printf("  [%d] %s vs %s: score=%.4f same=%s [%.0fms]%n",
                i+1,
                Path.of(pairs[i][0]).getFileName(),
                Path.of(pairs[i][1]).getFileName(),
                res.getScore(), res.isSameSpeaker(), res.getProcessingTimeMs());
        }

        client.close();
        System.out.println();
        System.out.println("=== Java SDK 测试完成 ===");
    }
}
