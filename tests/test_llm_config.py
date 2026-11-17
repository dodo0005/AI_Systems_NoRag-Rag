"""LLM_URL / LLM_API_KEY / LLM_MODEL: the app is not tied to one provider."""

from backend import llm


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "abc12345")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    s = llm.settings()
    assert (s.url, s.keys, s.model) == ("http://localhost:11434/v1", ["abc12345"], "my-model")


def test_gemini_is_only_the_fallback(monkeypatch):
    for name in ("LLM_URL", "LLM_API_KEY", "LLM_MODEL", "GEMINI_MODEL", "GOOGLE_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "old-google-key")
    s = llm.settings()
    assert s.url == llm.GEMINI_URL
    assert s.keys == ["old-google-key"]


def test_missing_key_is_a_configuration_error(monkeypatch):
    for name in ("LLM_API_KEY", "GOOGLE_API_KEY", "GOOGLE_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    try:
        llm.chat("hello")
    except llm.LLMNotConfigured:
        pass
    else:
        raise AssertionError("expected LLMNotConfigured")


def test_cost_uses_the_prices_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_M", "1.0")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_M", "2.0")
    assert llm.cost_usd(1_000_000, 500_000) == 2.0


def test_describe_never_shows_the_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key")
    assert "super-secret-key" not in str(llm.describe())
