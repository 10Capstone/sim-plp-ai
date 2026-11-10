import pytest

from core.filters import build_query_filter


def test_empty_filters_are_unrestricted():
    assert build_query_filter(None) is None
    assert build_query_filter({}) is None


def test_native_qdrant_filter_preserves_ranges():
    filters = {"must": [{"key": "page", "range": {"gte": 2, "lte": 5}}]}
    result = build_query_filter(filters)
    assert result.must[0].range.gte == 2
    assert result.must[0].range.lte == 5


def test_flat_filters_accept_scalars_and_match_any_lists():
    result = build_query_filter({"chapter": ["I", "II"], "page": 3, "active": True})
    assert result.must[0].match.any == ["I", "II"]
    assert result.must[1].match.value == 3
    assert result.must[2].match.value is True


@pytest.mark.parametrize("value", [None, {}, [], ["I", 1]])
def test_unsupported_flat_filters_fail(value):
    with pytest.raises(ValueError):
        build_query_filter({"chapter": value})
