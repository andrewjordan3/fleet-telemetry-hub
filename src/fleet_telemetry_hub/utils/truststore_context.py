# fleet_telemetry_hub/utils/truststore_context.py

from __future__ import annotations

import ssl
from ssl import SSLContext


def build_truststore_ssl_context() -> SSLContext:
    """
    Create an SSLContext using truststore for system certificate validation.

    Explicitly uses PROTOCOL_TLS_CLIENT for secure client-side TLS with
    automatic protocol negotiation and certificate verification.

    Returns:
        SSLContext: Configured SSLContext using OS trust store.

    Raises:
        RuntimeError: If truststore is not installed when use_truststore=True.

    Notes:
        - PROTOCOL_TLS_CLIENT: Secure defaults for client connections
        - Safe for library code (no global monkey-patching)
        - Lazy import keeps truststore optional
    """
    try:
        import truststore  # noqa: PLC0415
    except ImportError as import_error:
        raise RuntimeError(
            'truststore is required when use_truststore=True; '
            'install it with: pip install truststore'
        ) from import_error

    # Explicit protocol for clarity and security
    ssl_context: SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl_context
