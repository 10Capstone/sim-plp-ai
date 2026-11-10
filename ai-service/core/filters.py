from typing import Any

from qdrant_client import models


def build_query_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Accept flat payload matches or Qdrant's native filter JSON."""
    if not filters:
        return None
    if {"must", "should", "must_not", "min_should"}.intersection(filters):
        return models.Filter.model_validate(filters)

    conditions = []
    for key, value in filters.items():
        if isinstance(value, (str, bool, int)):
            condition = models.FieldCondition(
                key=key, match=models.MatchValue(value=value)
            )
        elif isinstance(value, list) and value:
            # MatchAny validates that the list contains only strings or only integers.
            condition = models.FieldCondition(key=key, match=models.MatchAny(any=value))
        elif isinstance(value, float):
            condition = models.FieldCondition(
                key=key, range=models.Range(gte=value, lte=value)
            )
        else:
            raise ValueError(f"Unsupported filter value for '{key}'")
        conditions.append(condition)
    return models.Filter(must=conditions)
