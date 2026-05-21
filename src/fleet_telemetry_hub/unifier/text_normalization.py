"""Pure text-normalization helpers shared by the unifier transforms.

These helpers are I/O-free and side-effect-free. They handle the
recurring shape "API string in, normalized string or None out" so
the per-provider transform layers can stay focused on the
provider-specific structural mapping.
"""

import unicodedata

__all__: list[str] = ['nfkc_strip', 'normalize_driver_name', 'nullify_tokens']

# Driver-name token nullification: the empty string already maps to
# None via nfkc_strip, so this set only needs the textual sentinels
# Motive and Samsara emit for "no driver attached".
_DRIVER_NAME_NULL_TOKENS: frozenset[str] = frozenset({'unknown'})


def nfkc_strip(value: str | None) -> str | None:
    """
    Apply NFKC normalization, strip surrounding whitespace, nullify empty.

    NFKC is the compatibility-decomposition + canonical-composition
    Unicode normalization form. It folds compatibility variants like
    full-width digits or precomposed/decomposed accented characters
    to a single representative, so downstream equality and hashing
    do the right thing across cosmetic input variants.

    Args:
        value: Raw string from an API response, possibly ``None``.

    Returns:
        The NFKC-normalized, whitespace-stripped string, or ``None``
        if the input was ``None`` or normalized to an empty string.
    """
    if value is None:
        return None
    normalized = unicodedata.normalize('NFKC', value).strip()
    return normalized if normalized else None


def nullify_tokens(
    value: str | None,
    null_tokens: frozenset[str],
) -> str | None:
    """
    Return ``None`` if ``value`` matches any null token (case-insensitively).

    Comparison uses ``str.casefold()`` on both sides, so it tolerates
    Unicode case variants (Turkish dotless i, German sharp s, etc.).
    The caller is responsible for any prior normalization (NFKC
    fold, whitespace strip). A ``None`` input passes through as
    ``None``. The original case of ``value`` is preserved when no
    null token matches.

    Args:
        value: String to check, possibly ``None``.
        null_tokens: Set of token strings that should map to ``None``.

    Returns:
        ``None`` if ``value`` is ``None`` or case-insensitively
        matches a null token; the original ``value`` otherwise.
    """
    if value is None:
        return None
    folded = value.casefold()
    if any(folded == token.casefold() for token in null_tokens):
        return None
    return value


def normalize_driver_name(value: str | None) -> str | None:
    """
    Driver-name-specific normalization: ``nfkc_strip`` then nullify ``unknown``.

    Returns ``None`` for ``None`` input, empty input, whitespace-only
    input, or any case-insensitive NFKC variant of ``"unknown"``.
    Internal whitespace in real names (e.g. ``"Sam Snowflake"``) is
    preserved.

    Args:
        value: Raw driver-name string from an API response.

    Returns:
        The normalized driver name, or ``None`` if the input
        normalized to empty or to the ``"unknown"`` sentinel.
    """
    return nullify_tokens(nfkc_strip(value), _DRIVER_NAME_NULL_TOKENS)
