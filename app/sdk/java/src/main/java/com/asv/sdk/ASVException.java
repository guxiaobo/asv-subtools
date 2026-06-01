package com.asv.sdk;

/**
 * Exception thrown by the ASV SDK on API errors or network failures.
 */
public class ASVException extends RuntimeException {

    private final int statusCode;

    /**
     * Create a new ASVException.
     *
     * @param message   Human-readable error description.
     * @param statusCode HTTP status code (0 if network error).
     * @param cause     Root cause (nullable).
     */
    public ASVException(String message, int statusCode, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }

    /**
     * Create an ASVException with no HTTP status code (network error).
     */
    public ASVException(String message, Throwable cause) {
        this(message, 0, cause);
    }

    /**
     * Create an ASVException from a server error message.
     */
    public ASVException(String message) {
        this(message, 0, null);
    }

    /**
     * HTTP status code of the error response, or 0 if network-related.
     */
    public int getStatusCode() {
        return statusCode;
    }

    /**
     * Whether this was a server-side error (HTTP 4xx/5xx).
     */
    public boolean isServerError() {
        return statusCode >= 400;
    }

    /**
     * Whether this was a client-side network error.
     */
    public boolean isNetworkError() {
        return statusCode == 0;
    }

    @Override
    public String toString() {
        if (statusCode > 0) {
            return String.format("ASVException[HTTP %d]: %s", statusCode, getMessage());
        }
        return String.format("ASVException[network]: %s", getMessage());
    }
}
