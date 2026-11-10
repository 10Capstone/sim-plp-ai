from core.config import Settings


def test_defaults_match_existing_collection(settings):
    assert settings.QDRANT_COLLECTION_NAME == "plp_indoe5_cs96_ov32"
    assert settings.QDRANT_VECTOR_NAME is None
    assert settings.PAYLOAD_TEXT_KEY == "text"
    assert settings.EMBEDDING_MODEL_NAME == "LazarusNLP/all-indo-e5-small-v4"
    assert settings.EMBEDDING_DIMENSION == 384
    assert settings.LLM_MODEL == "gemini-2.5-flash"
    assert (
        settings.GEMINI_API_KEY == settings.OPENAI_API_KEY == settings.LLM_API_KEY == ""
    )


def test_env_file_and_environment_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "QDRANT_URL=http://localhost:6333\nQDRANT_VECTOR_NAME=\n"
        "PAYLOAD_TEXT_KEY=content\nEXTRA_NOTEBOOK_SETTING=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    config = Settings(_env_file=env_file)
    assert config.QDRANT_VECTOR_NAME is None
    assert config.PAYLOAD_TEXT_KEY == "content"
    assert config.EMBEDDING_DIMENSION == 768


def test_default_env_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    env_path = Settings.model_config["env_file"]
    monkeypatch.chdir(tmp_path)
    assert env_path.is_absolute()
    assert env_path.parent.name == "ai-service"
    assert env_path.name == ".env"
