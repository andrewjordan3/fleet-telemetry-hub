# fleet_telemetry_hub/utils/truststore_context.py
"""
SSL Context Factory using System Trust Store.

This module provides a factory function to create SSL contexts that verify
certificates using the operating system's native trust store, rather than
Python's default bundled certificates (certifi).

Primary Use Case:
    Corporate environments using aggressive proxies (like Zscaler) that act as
    Man-in-the-Middle (MITM) inspectors. These proxies re-encrypt traffic using
    a private Root CA that is installed in the Windows/macOS system store but
    is unknown to standard Python libraries.

    Without this module, requests fail with `SSLCertVerificationError`.
    With this module, Python trusts the Zscaler Root CA, enabling secure
    connectivity without disabling SSL verification.

Design Benefit: Optional Dependency
    The `truststore` library is imported lazily inside the factory function.
    This design ensures that the package remains usable in standard environments
    (or Linux servers) where `truststore` might not be installed. The package
    will only raise an ImportError if this specific function is explicitly
    called without the dependency present.

Dependencies:
    - truststore: Optional. Required only if using this factory function.
      Install with: `pip install truststore`
"""

import ssl
from ssl import SSLContext

__all__: list[str] = ['build_truststore_ssl_context']

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
