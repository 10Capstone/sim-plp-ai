import math
import threading
from unittest.mock import Mock

import pytest

from services.embedder import QueryEmbedderService


async def test_embedding_matches_notebook_and_runs_off_event_loop(settings):
    main_thread = threading.get_ident()

    def encode(query, **kwargs):
        assert threading.get_ident() != main_thread
        return [1.0 / math.sqrt(384)] * 384

    model = Mock()
    model.encode.side_effect = encode
    service = QueryEmbedderService(settings, model)
    vector = await service.embed_query("  Apa prosedur PLP?  ")
    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)
    assert math.isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-5)
    model.encode.assert_called_once_with(
        "Apa prosedur PLP?",
        normalize_embeddings=True,
        prompt="",
        convert_to_numpy=True,
    )


@pytest.mark.parametrize("query", ["", "  ", None])
async def test_invalid_queries_do_not_load_model(settings, query):
    model = Mock()
    service = QueryEmbedderService(settings, model)
    with pytest.raises(ValueError, match="non-empty"):
        await service.embed_query(query)
    model.encode.assert_not_called()


@pytest.mark.parametrize("vector", [[1.0] * 768, [0.0] * 384, [float("nan")] * 384])
async def test_incompatible_embeddings_are_rejected(settings, vector):
    model = Mock()
    model.encode.return_value = vector
    with pytest.raises(ValueError):
        await QueryEmbedderService(settings, model).embed_query("pertanyaan")
