"""Tests for ``unifier.text_normalization`` helpers.

Covers NFKC compatibility folding, whitespace stripping, empty-to-
None mapping, case-insensitive token nullification, and the
composed driver-name normalization.
"""

import pytest

from fleet_telemetry_hub.unifier.text_normalization import (
    nfkc_strip,
    normalize_driver_name,
    nullify_tokens,
)


class TestNfkcStrip:
    """``nfkc_strip`` folds compatibility variants and nullifies empty input."""

    def test_none_input_passes_through(self) -> None:
        """None input round-trips as None."""

        assert nfkc_strip(None) is None

    def test_empty_string_becomes_none(self) -> None:
        """Empty input normalizes to None."""

        assert nfkc_strip('') is None

    def test_whitespace_only_becomes_none(self) -> None:
        """Whitespace-only input strips to empty and normalizes to None."""

        assert nfkc_strip('   ') is None

    def test_strips_surrounding_whitespace(self) -> None:
        """Leading and trailing whitespace are stripped."""

        assert nfkc_strip(' Sam ') == 'Sam'

    def test_already_clean_input_is_idempotent(self) -> None:
        """Already-clean input is returned unchanged."""

        assert nfkc_strip('Sam') == 'Sam'

    def test_full_width_digits_fold_to_ascii(self) -> None:
        """NFKC folds U+FF11..U+FF13 to ASCII digits."""

        assert nfkc_strip('１２３') == '123'  # noqa: RUF001 -- intentional full-width digits

    @pytest.mark.parametrize(
        ('precomposed', 'decomposed'),
        [
            ('café', 'café'),
            ('über', 'über'),
        ],
    )
    def test_precomposed_and_decomposed_match(
        self,
        precomposed: str,
        decomposed: str,
    ) -> None:
        """NFKC produces the same string from precomposed and decomposed input."""

        assert nfkc_strip(precomposed) == nfkc_strip(decomposed)


class TestNullifyTokens:
    """``nullify_tokens`` maps configured tokens to None case-insensitively."""

    def test_none_input_passes_through(self) -> None:
        """None input round-trips as None regardless of token set."""

        assert nullify_tokens(None, frozenset({'unknown'})) is None

    @pytest.mark.parametrize(
        'value',
        ['unknown', 'Unknown', 'UNKNOWN', 'uNkNoWn'],
    )
    def test_case_insensitive_match_returns_none(self, value: str) -> None:
        """All case variants of a configured token map to None."""

        assert nullify_tokens(value, frozenset({'unknown'})) is None

    def test_non_token_value_preserves_original_case(self) -> None:
        """A non-matching value is returned unchanged."""

        assert nullify_tokens('Sam', frozenset({'unknown'})) == 'Sam'

    def test_empty_token_set_is_no_op(self) -> None:
        """An empty null-token set never nullifies."""

        assert nullify_tokens('Sam', frozenset()) == 'Sam'

    @pytest.mark.parametrize('value', ['unknown', 'N/A', 'tbd'])
    def test_multiple_tokens_each_match_case_insensitively(self, value: str) -> None:
        """Each token in a multi-token set matches case-insensitively."""

        tokens = frozenset({'unknown', 'n/a', 'tbd'})
        assert nullify_tokens(value, tokens) is None


class TestNormalizeDriverName:
    """``normalize_driver_name`` composes ``nfkc_strip`` with ``unknown`` nullification."""

    @pytest.mark.parametrize(
        'value',
        [None, '', '   '],
    )
    def test_empty_and_whitespace_become_none(self, value: str | None) -> None:
        """Empty, whitespace-only, and None all yield None."""

        assert normalize_driver_name(value) is None

    def test_real_name_is_returned_unchanged(self) -> None:
        """A non-token name is returned unchanged."""

        assert normalize_driver_name('Sam') == 'Sam'

    def test_surrounding_whitespace_is_stripped(self) -> None:
        """Surrounding whitespace is stripped from a real name."""

        assert normalize_driver_name(' Sam ') == 'Sam'

    @pytest.mark.parametrize('value', ['unknown', 'Unknown', ' UNKNOWN '])
    def test_unknown_variants_become_none(self, value: str) -> None:
        """Case variants of ``unknown`` (with optional whitespace) all nullify."""

        assert normalize_driver_name(value) is None

    def test_nfkc_runs_before_nullification(self) -> None:
        """A full-width spelling of ``unknown`` folds to ASCII and then nullifies."""

        assert normalize_driver_name('ｕｎｋｎｏｗｎ') is None  # noqa: RUF001 -- intentional full-width chars

    def test_internal_whitespace_preserved(self) -> None:
        """Internal whitespace in a real name is preserved verbatim."""

        assert normalize_driver_name('Sam Snowflake') == 'Sam Snowflake'
