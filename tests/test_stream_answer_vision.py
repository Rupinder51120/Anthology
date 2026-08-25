"""
Regression tests for streaming vision support in
src/generation/generator.py::stream_answer().

Anthology's backend already had a vision-generation path
(_call_groq_vision, wired into the non-streaming generate_answer()), but it
was never reachable through /query/stream -- the endpoint the live chat UI
actually calls. These tests cover the change that wires stream_answer() to
the same, unmodified _call_groq_vision() function via a new
collect_image_paths() helper shared with the non-streaming path.

No real Groq/Ollama network calls are made -- _groq_enabled()/
_call_groq_vision()/the Groq SDK are monkeypatched, so these tests run
without API keys or network access.
"""
import pytest

from src.generation import generator as gen


def _sample_chunks():
    text_chunk = {
        "text": "Transformers use self-attention.",
        "metadata": {
            "content_type": "text", "chunk_type": "general",
            "title": "Paper A", "year": 2023, "section": "Intro",
            "authors": "Smith", "source": "a.pdf",
        },
    }
    figure_chunk = {
        "text": "Figure 1 shows the architecture diagram.",
        "metadata": {
            "content_type": "figure", "chunk_type": "figure_table",
            "image_path": "data/figures/a_fig1.png",
            "title": "Paper A", "year": 2023, "section": "Results",
            "authors": "Smith", "source": "a.pdf",
        },
    }
    return text_chunk, figure_chunk


class TestCollectImagePaths:
    def test_extracts_only_figure_chunks_with_image_path(self):
        text_chunk, figure_chunk = _sample_chunks()
        no_path_figure = {"text": "", "metadata": {"content_type": "figure"}}
        result = gen.collect_image_paths([text_chunk, figure_chunk, no_path_figure])
        assert result == ["data/figures/a_fig1.png"]

    def test_no_figures_returns_empty_list(self):
        text_chunk, _ = _sample_chunks()
        assert gen.collect_image_paths([text_chunk]) == []

    def test_multiple_figures_all_returned_unbounded_here(self):
        # collect_image_paths() itself does not cap the count -- the cap
        # lives in _call_groq_vision() (image_paths[:3]), exercised below.
        figures = [
            {"text": "", "metadata": {"content_type": "figure", "image_path": f"fig{i}.png"}}
            for i in range(5)
        ]
        assert gen.collect_image_paths(figures) == [f"fig{i}.png" for i in range(5)]


class TestStreamAnswerTextOnly:
    """A. Text-only query: no images retrieved -- existing streaming behavior is unchanged."""

    @pytest.mark.asyncio
    async def test_no_images_uses_existing_text_streaming_path(self, monkeypatch):
        monkeypatch.setattr(gen, "_groq_enabled", lambda: True)

        vision_called = {"value": False}
        monkeypatch.setattr(gen, "_call_groq_vision", lambda *a, **k: vision_called.__setitem__("value", True))

        class FakeStreamChunk:
            def __init__(self, text):
                self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})]

        class FakeStream:
            def __aiter__(self):
                self._items = iter([FakeStreamChunk("Hello"), FakeStreamChunk(" world")])
                return self
            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        class FakeCompletions:
            async def create(self, **kwargs):
                assert kwargs.get("stream") is True
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncGroq:
            def __init__(self, api_key):
                self.chat = FakeChat()
            async def close(self):
                pass

        monkeypatch.setattr("groq.AsyncGroq", FakeAsyncGroq)

        text_chunk, _ = _sample_chunks()
        events = [e async for e in gen.stream_answer("what is attention?", [text_chunk], image_paths=None)]

        token_texts = [e["text"] for e in events if e["type"] == "token"]
        assert "".join(token_texts) == "Hello world"
        assert events[-1]["type"] == "citations"
        assert vision_called["value"] is False


class TestStreamAnswerVision:
    """B. Figure-retrieving query: image_paths collected and passed to vision generation."""

    @pytest.mark.asyncio
    async def test_images_present_routes_to_call_groq_vision(self, monkeypatch):
        monkeypatch.setattr(gen, "_groq_enabled", lambda: True)

        received = {}
        def fake_vision(messages, image_paths):
            received["messages"] = messages
            received["image_paths"] = image_paths
            return "The figure shows a transformer block."
        monkeypatch.setattr(gen, "_call_groq_vision", fake_vision)

        text_chunk, figure_chunk = _sample_chunks()
        events = [
            e async for e in gen.stream_answer(
                "what does the figure show?", [text_chunk, figure_chunk],
                image_paths=["data/figures/a_fig1.png"],
            )
        ]

        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["text"] == "The figure shows a transformer block."
        assert received["image_paths"] == ["data/figures/a_fig1.png"]
        assert events[-1]["type"] == "citations"
        # citations/grounding still built from the retrieved chunks (requirement 9)
        assert any(c["title"] == "Paper A" for c in events[-1]["citations"])


class TestStreamAnswerVisionCallFails:
    """
    Observed live against the real Groq API during development: the
    configured GROQ_VISION_MODEL was not actually a vision-capable model
    accessible to the account, and the vision call raised a real
    groq.BadRequestError. This must degrade to the normal text-streaming
    path with an explicit status message, not error out the whole request.
    """

    @pytest.mark.asyncio
    async def test_vision_exception_falls_back_to_text_streaming(self, monkeypatch):
        monkeypatch.setattr(gen, "_groq_enabled", lambda: True)

        def failing_vision(messages, image_paths):
            raise RuntimeError("model does not support vision")
        monkeypatch.setattr(gen, "_call_groq_vision", failing_vision)

        class FakeStreamChunk:
            def __init__(self, text):
                self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})]

        class FakeStream:
            def __aiter__(self):
                self._items = iter([FakeStreamChunk("fallback"), FakeStreamChunk(" text")])
                return self
            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        class FakeCompletions:
            async def create(self, **kwargs):
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncGroq:
            def __init__(self, api_key):
                self.chat = FakeChat()
            async def close(self):
                pass

        monkeypatch.setattr("groq.AsyncGroq", FakeAsyncGroq)

        text_chunk, figure_chunk = _sample_chunks()
        events = [
            e async for e in gen.stream_answer(
                "what does the figure show?", [text_chunk, figure_chunk],
                image_paths=["data/figures/a_fig1.png"],
            )
        ]

        status_texts = [e["text"] for e in events if e["type"] == "status"]
        assert any("unavailable" in s for s in status_texts)
        token_texts = [e["text"] for e in events if e["type"] == "token"]
        assert "".join(token_texts) == "fallback text"
        assert events[-1]["type"] == "citations"


class TestStreamAnswerMissingImage:
    """C. Missing image file: must not crash the query."""

    @pytest.mark.asyncio
    async def test_missing_image_falls_back_without_crashing(self, monkeypatch):
        # Exercise the real _call_groq_vision (unmodified) with a path that
        # cannot be opened, but stub the underlying Groq client so no real
        # network call happens either for the vision attempt or its
        # text-only fallback.
        monkeypatch.setattr(gen, "_groq_enabled", lambda: True)

        class FakeResp:
            choices = [type("C", (), {"message": type("M", (), {"content": " fallback answer "})()})]

        class FakeCompletions:
            def create(self, **kwargs):
                return FakeResp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeGroqClient:
            chat = FakeChat()

        monkeypatch.setattr(gen, "_groq_client", lambda: FakeGroqClient())

        text_chunk, figure_chunk = _sample_chunks()
        events = [
            e async for e in gen.stream_answer(
                "what does the figure show?", [text_chunk, figure_chunk],
                image_paths=["/nonexistent/path/does_not_exist.png"],
            )
        ]

        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["text"] == "fallback answer"
        assert events[-1]["type"] == "citations"


class TestStreamAnswerBoundedImages:
    """D. Multiple retrieved figures: only a bounded number are actually sent, not the whole corpus."""

    @pytest.mark.asyncio
    async def test_only_up_to_three_images_are_opened(self, monkeypatch):
        monkeypatch.setattr(gen, "_groq_enabled", lambda: True)

        opened = []
        real_open = open
        def counting_open(path, *a, **k):
            opened.append(path)
            return real_open(__file__, *a, **k)  # open a real, harmless file instead
        monkeypatch.setattr("builtins.open", counting_open)

        class FakeResp:
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})]

        class FakeCompletions:
            def create(self, **kwargs):
                # confirm only <=3 image_url blocks were built into the vision message
                content = kwargs["messages"][-1]["content"]
                image_blocks = [c for c in content if c.get("type") == "image_url"]
                assert len(image_blocks) <= 3
                return FakeResp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeGroqClient:
            chat = FakeChat()

        monkeypatch.setattr(gen, "_groq_client", lambda: FakeGroqClient())

        text_chunk, _ = _sample_chunks()
        five_images = [f"fig{i}.png" for i in range(5)]
        events = [
            e async for e in gen.stream_answer(
                "describe all figures", [text_chunk], image_paths=five_images,
            )
        ]
        assert len(opened) == 3
        assert [e for e in events if e["type"] == "token"][0]["text"] == "ok"


class TestStreamAnswerOllamaWithImages:
    """Provider that can't do vision: images must not be silently pretended to work."""

    @pytest.mark.asyncio
    async def test_ollama_provider_ignores_images_with_explicit_status(self, monkeypatch):
        monkeypatch.setattr(gen, "_groq_enabled", lambda: False)

        class FakeStreamResp:
            def raise_for_status(self):
                pass
            async def aiter_lines(self):
                import json as _json
                yield _json.dumps({"message": {"content": "local answer"}, "done": False})
                yield _json.dumps({"message": {"content": ""}, "done": True})

        class FakeStreamCtx:
            async def __aenter__(self):
                return FakeStreamResp()
            async def __aexit__(self, *a):
                return False

        class FakeAsyncClient:
            def __init__(self, timeout=None):
                pass
            def stream(self, method, url, json=None):
                return FakeStreamCtx()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

        text_chunk, figure_chunk = _sample_chunks()
        events = [
            e async for e in gen.stream_answer(
                "what does the figure show?", [text_chunk, figure_chunk],
                image_paths=["data/figures/a_fig1.png"],
            )
        ]

        status_events = [e["text"] for e in events if e["type"] == "status"]
        assert any("does not support vision" in s for s in status_events)
        token_texts = [e["text"] for e in events if e["type"] == "token"]
        assert "".join(token_texts) == "local answer"
