"""Hosted providers, against synthetic vendor responses.

These providers can't be called for real without paid keys, so the untested part
was never "does the vendor work" — it was *our* half of the contract: the URL and
auth header we send, and whether we parse the reply and classify the failure
correctly. That is all exercisable with a stubbed transport, and it is where the
bugs would actually be.

Payload shapes are taken from each vendor's documented response. What this does
NOT prove: that the vendor still returns this shape. A contract test against a
recorded live response is the upgrade path.
"""
from __future__ import annotations

import json
import sys
import types

import httpx
import pytest

from app.providers import ProviderConfigError, ProviderTransientError, get_provider


class FakeResponse:
    def __init__(self, json_data=None, content=b"", status=200):
        self._json, self.content, self.status_code = json_data, content, status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=httpx.Request("POST", "http://stub"),
                response=httpx.Response(self.status_code))


def capture(monkeypatch, response, method="post"):
    """Swap one httpx verb for a stub and record what the provider sent."""
    sent = {}

    def fake(url, **kw):
        sent.update(url=url, **kw)
        return response if not callable(response) else response(url, **kw)

    monkeypatch.setattr(httpx, method, fake)
    return sent


def get_provider_for(settings_env, kind, name):
    settings_env(**{f"PROVIDERS__{kind.upper()}__NAME": name,
                    f"PROVIDERS__{kind.upper()}__OPTIONS": "{}"})
    return get_provider(kind)


# --- speech to text -------------------------------------------------------

def test_openai_whisper_parses_segments(settings_env, tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    settings_env(OPENAI_API_KEY="sk-test")
    stt = get_provider_for(settings_env, "stt", "openai_whisper")
    sent = capture(monkeypatch, FakeResponse({
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.5, "text": " Hello there. "},
                     {"start": 1.6, "end": 2.0, "text": "   "}]}))

    out = stt.transcribe(wav)
    assert out.language == "en"
    assert [(s.start, s.end, s.text) for s in out.segments] == [(0.0, 1.5, "Hello there.")]
    assert sent["url"].endswith("/v1/audio/transcriptions")
    assert sent["headers"]["Authorization"] == "Bearer sk-test"


def test_deepgram_parses_utterances_and_detected_language(settings_env, tmp_path,
                                                          monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    settings_env(DEEPGRAM_API_KEY="dg-test")
    stt = get_provider_for(settings_env, "stt", "deepgram")
    sent = capture(monkeypatch, FakeResponse({"results": {
        "utterances": [{"start": 0.2, "end": 1.1, "transcript": "Hola"}],
        "channels": [{"detected_language": "es"}]}}))

    out = stt.transcribe(wav)
    assert out.language == "es"
    assert [(s.start, s.text) for s in out.segments] == [(0.2, "Hola")]
    assert sent["headers"]["Authorization"] == "Token dg-test"
    assert sent["params"]["detect_language"] == "true"      # no language pinned


def test_google_stt_parses_protobuf_style_offsets(settings_env, tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    settings_env(GOOGLE_API_KEY="g-test", GOOGLE_CLOUD_PROJECT="proj")
    stt = get_provider_for(settings_env, "stt", "google_stt")
    capture(monkeypatch, FakeResponse({"results": [{
        "languageCode": "en-US",
        "alternatives": [{"transcript": "Hello", "words": [
            {"startOffset": "0.500s", "endOffset": "1.250s"}]}]}]}))

    out = stt.transcribe(wav)
    assert out.language == "en-US"
    assert (out.segments[0].start, out.segments[0].end) == (0.5, 1.25)


def test_google_stt_maps_5xx_to_transient(settings_env, tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    settings_env(GOOGLE_API_KEY="g-test", GOOGLE_CLOUD_PROJECT="proj")
    stt = get_provider_for(settings_env, "stt", "google_stt")
    capture(monkeypatch, FakeResponse({}, status=503))
    with pytest.raises(ProviderTransientError):
        stt.transcribe(wav)


def test_google_stt_refuses_audio_over_the_inline_limit(settings_env, tmp_path,
                                                        monkeypatch):
    """The inline recognize endpoint caps at ~10MB — fail with the fix, not a 400."""
    wav = tmp_path / "big.wav"
    wav.write_bytes(b"\0" * 2048)
    settings_env(GOOGLE_API_KEY="g-test", GOOGLE_CLOUD_PROJECT="proj",
                 PROVIDERS__STT__NAME="google_stt",
                 PROVIDERS__STT__OPTIONS='{"max_inline_bytes": 1024}')
    with pytest.raises(Exception, match="batchRecognize"):
        get_provider("stt").transcribe(wav)


def test_assemblyai_uploads_polls_and_converts_milliseconds(settings_env, tmp_path,
                                                            monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    settings_env(ASSEMBLYAI_API_KEY="aai-test",
                 PROVIDERS__STT__NAME="assemblyai",
                 PROVIDERS__STT__OPTIONS='{"poll_sec": 0}')
    calls = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, **kw):
            calls.append(path)
            if path == "/v2/upload":
                return FakeResponse({"upload_url": "https://cdn/audio"})
            return FakeResponse({"id": "t-1"})

        def get(self, path, **kw):
            calls.append(path)
            # first poll still processing, second completes: proves we poll
            if calls.count("/v2/transcript/t-1") == 1:
                return FakeResponse({"status": "processing"})
            return FakeResponse({"status": "completed", "language_code": "en",
                                 "utterances": [{"start": 500, "end": 1500,
                                                 "text": "Hello"}]})

    monkeypatch.setattr(httpx, "Client", FakeClient)
    out = get_provider("stt").transcribe(wav)
    assert (out.segments[0].start, out.segments[0].end) == (0.5, 1.5)   # ms -> s
    assert calls.count("/v2/transcript/t-1") == 2                        # polled


# --- translation ----------------------------------------------------------

def test_openai_translation_parses_the_json_array(settings_env, monkeypatch):
    settings_env(OPENAI_API_KEY="sk-test")
    tr = get_provider_for(settings_env, "translation", "openai")
    sent = capture(monkeypatch, FakeResponse(
        {"choices": [{"message": {"content": '["Hola", "Adios"]'}}]}))

    assert tr.translate(["Hello", "Bye"], "en", "es") == ["Hola", "Adios"]
    assert sent["headers"]["Authorization"] == "Bearer sk-test"
    assert "es" in sent["json"]["messages"][0]["content"]


def test_gemini_strips_a_markdown_fence(settings_env, monkeypatch):
    """Models wrap JSON in ```json fences; the shared base has to survive it."""
    settings_env(GEMINI_API_KEY="g-test")
    tr = get_provider_for(settings_env, "translation", "gemini")
    capture(monkeypatch, FakeResponse({"candidates": [{"content": {"parts": [
        {"text": '```json\n["Hola"]\n```'}]}}]}))
    assert tr.translate(["Hello"], "en", "es") == ["Hola"]


def test_gemini_accepts_either_key_name(settings_env, monkeypatch):
    settings_env(GOOGLE_API_KEY="from-google-key")
    tr = get_provider_for(settings_env, "translation", "gemini")
    sent = capture(monkeypatch, FakeResponse({"candidates": [{"content": {"parts": [
        {"text": '["Hola"]'}]}}]}))
    tr.translate(["Hello"], "en", "es")
    assert sent["params"]["key"] == "from-google-key"


def test_deepl_parses_translations(settings_env, monkeypatch):
    settings_env(DEEPL_API_KEY="dl-test")
    tr = get_provider_for(settings_env, "translation", "deepl")
    sent = capture(monkeypatch, FakeResponse(
        {"translations": [{"text": "Hola"}, {"text": "Adios"}]}))

    assert tr.translate(["Hello", "Bye"], "en", "es") == ["Hola", "Adios"]
    assert sent["headers"]["Authorization"] == "DeepL-Auth-Key dl-test"
    assert sent["json"]["target_lang"] == "ES"          # DeepL wants upper case


def test_deepl_maps_5xx_to_transient(settings_env, monkeypatch):
    settings_env(DEEPL_API_KEY="dl-test")
    tr = get_provider_for(settings_env, "translation", "deepl")
    capture(monkeypatch, FakeResponse({}, status=502))
    with pytest.raises(ProviderTransientError):
        tr.translate(["Hello"], "en", "es")


def test_a_translator_returning_the_wrong_line_count_is_rejected(settings_env,
                                                                 monkeypatch):
    """Silently losing a line would desync every later subtitle and dub clip."""
    settings_env(OPENAI_API_KEY="sk-test")
    tr = get_provider_for(settings_env, "translation", "openai")
    capture(monkeypatch, FakeResponse(
        {"choices": [{"message": {"content": '["only one"]'}}]}))
    with pytest.raises(Exception, match="1 lines for 2 inputs"):
        tr.translate(["Hello", "Bye"], "en", "es")


def fake_anthropic(monkeypatch, *, text=None, error=None, stop_reason="end_turn"):
    class APIStatusError(Exception):
        def __init__(self, status_code, message="boom"):
            super().__init__(message)
            self.status_code, self.message = status_code, message

    class RateLimitError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    def create(**kw):
        if error:
            raise error(APIStatusError, RateLimitError)
        return types.SimpleNamespace(
            stop_reason=stop_reason,
            content=[types.SimpleNamespace(type="text", text=text)])

    module = types.SimpleNamespace(
        Anthropic=lambda api_key=None: types.SimpleNamespace(
            messages=types.SimpleNamespace(create=create)),
        APIStatusError=APIStatusError, RateLimitError=RateLimitError,
        APIConnectionError=APIConnectionError)
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_claude_parses_the_reply(settings_env, monkeypatch):
    fake_anthropic(monkeypatch, text=json.dumps(["Hola", "Adios"]))
    settings_env(ANTHROPIC_API_KEY="sk-ant")
    tr = get_provider_for(settings_env, "translation", "claude")
    assert tr.translate(["Hello", "Bye"], "en", "es") == ["Hola", "Adios"]


def test_claude_rate_limit_is_transient_but_a_refusal_is_not(settings_env, monkeypatch):
    fake_anthropic(monkeypatch, error=lambda status, rate: rate("slow down"))
    settings_env(ANTHROPIC_API_KEY="sk-ant")
    with pytest.raises(ProviderTransientError):
        get_provider_for(settings_env, "translation", "claude").translate(
            ["Hello"], "en", "es")

    fake_anthropic(monkeypatch, text="", stop_reason="refusal")
    settings_env(ANTHROPIC_API_KEY="sk-ant")
    with pytest.raises(Exception, match="declined"):
        get_provider_for(settings_env, "translation", "claude").translate(
            ["Hello"], "en", "es")


# --- voice generation -----------------------------------------------------

def test_elevenlabs_writes_the_audio_it_is_given(settings_env, tmp_path, monkeypatch):
    settings_env(ELEVENLABS_API_KEY="el-test")
    tts = get_provider_for(settings_env, "tts", "elevenlabs")
    sent = capture(monkeypatch, FakeResponse(content=b"ID3-audio-bytes"))

    out = tmp_path / "clip.mp3"
    tts.synthesize("Hola", "voice-1", out)
    assert out.read_bytes() == b"ID3-audio-bytes"
    assert sent["url"].endswith("/v1/text-to-speech/voice-1")
    assert sent["headers"]["xi-api-key"] == "el-test"


def test_elevenlabs_5xx_is_transient(settings_env, tmp_path, monkeypatch):
    settings_env(ELEVENLABS_API_KEY="el-test")
    tts = get_provider_for(settings_env, "tts", "elevenlabs")
    capture(monkeypatch, FakeResponse(content=b"", status=500))
    with pytest.raises(ProviderTransientError):
        tts.synthesize("Hola", "voice-1", tmp_path / "c.mp3")


def test_azure_sends_ssml_with_the_emotional_style(settings_env, tmp_path, monkeypatch):
    """Azure is the provider that carries requirement 5's 'emotional tone'."""
    settings_env(AZURE_SPEECH_KEY="az-test", AZURE_SPEECH_REGION="westeurope",
                 PROVIDERS__TTS__NAME="azure",
                 PROVIDERS__TTS__OPTIONS='{"style": "cheerful", "style_degree": 2.0}')
    tts = get_provider("tts")
    sent = capture(monkeypatch, FakeResponse(content=b"audio"))

    out = tmp_path / "clip.mp3"
    tts.synthesize("Hola & adios", "es-ES-ElviraNeural", out)
    ssml = sent["content"].decode()
    assert 'mstts:express-as style="cheerful"' in ssml and 'styledegree="2.0"' in ssml
    assert 'xml:lang="es-ES"' in ssml and "Hola &amp; adios" in ssml   # escaped
    assert "westeurope.tts.speech.microsoft.com" in sent["url"]
    assert out.read_bytes() == b"audio"


def test_azure_lists_voices_for_a_language(settings_env, monkeypatch):
    settings_env(AZURE_SPEECH_KEY="az-test", AZURE_SPEECH_REGION="westeurope")
    tts = get_provider_for(settings_env, "tts", "azure")
    capture(monkeypatch, FakeResponse([
        {"ShortName": "es-ES-Elvira", "Gender": "Female", "Locale": "es-ES"},
        {"ShortName": "de-DE-Katja", "Gender": "Female", "Locale": "de-DE"}]),
        method="get")

    voices = tts.voices("es")
    assert [v.id for v in voices] == ["es-ES-Elvira"] and voices[0].gender == "F"


# --- the local cloning providers -----------------------------------------

@pytest.mark.parametrize("name,env,expected", [
    ("xtts", {"COQUI_TOS_AGREED": "1"}, "coqui-tts"),
    ("openvoice", {"OPENVOICE_CKPT": "/ckpt"}, "myshell-openvoice"),
])
def test_cloning_providers_name_the_package_they_need(settings_env, name, env, expected):
    """Credential present but the optional package is not installed: the error
    has to say what to install, not raise ImportError from deep inside."""
    with pytest.raises(ProviderConfigError, match=expected):
        provider_env = {"PROVIDERS__TTS__NAME": name, "PROVIDERS__TTS__OPTIONS": "{}"}
        settings_env(**{**env, **provider_env})
        get_provider("tts")
