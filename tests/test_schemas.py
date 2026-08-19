"""Unit tests for Pydantic request/response schemas and DTOs."""
import pytest
from pydantic import ValidationError
from app.schemas.payload import ChatRequest, ChatResponse, HealthResponse


class TestChatRequestSchema:
    """Test suite for ChatRequest Pydantic model validation."""

    def test_valid_chat_request_default_stream(self) -> None:
        """Verifies that a valid payload defaults stream to True."""
        data = {
            "agent_id": "portfolio",
            "session_id": "sess_12345",
            "message": "Hola, ¿cuáles son tus habilidades técnicas?",
        }
        req = ChatRequest(**data)
        assert req.agent_id == "portfolio"
        assert req.session_id == "sess_12345"
        assert req.message == "Hola, ¿cuáles son tus habilidades técnicas?"
        assert req.stream is True

    def test_valid_chat_request_explicit_stream_false(self) -> None:
        """Verifies that stream=False is respected."""
        data = {
            "agent_id": "ecommerce",
            "session_id": "sess_999",
            "message": "Consultar catálogo",
            "stream": False,
        }
        req = ChatRequest(**data)
        assert req.stream is False

    @pytest.mark.parametrize("missing_field", ["agent_id", "session_id", "message"])
    def test_chat_request_missing_required_fields(self, missing_field: str) -> None:
        """Verifies that omitting any required field raises a ValidationError."""
        data = {
            "agent_id": "portfolio",
            "session_id": "sess_123",
            "message": "Test prompt",
        }
        del data[missing_field]

        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(**data)

        errors = exc_info.value.errors()
        assert any(e["loc"] == (missing_field,) for e in errors)

    @pytest.mark.parametrize("empty_field", ["agent_id", "session_id", "message"])
    def test_chat_request_empty_string_validation(self, empty_field: str) -> None:
        """Verifies that empty strings fail min_length=1 constraint."""
        data = {
            "agent_id": "portfolio",
            "session_id": "sess_123",
            "message": "Test prompt",
        }
        data[empty_field] = ""

        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(**data)

        errors = exc_info.value.errors()
        assert any(e["loc"] == (empty_field,) for e in errors)

    def test_chat_request_invalid_types(self) -> None:
        """Verifies that incompatible data types raise ValidationError."""
        # agent_id as a dictionary/list
        with pytest.raises(ValidationError):
            ChatRequest(agent_id=["portfolio"], session_id="sess_1", message="hola")

        # stream as invalid non-boolean string
        with pytest.raises(ValidationError):
            ChatRequest(agent_id="portfolio", session_id="sess_1", message="hola", stream="not-a-bool")

    def test_chat_request_unicode_and_emojis(self) -> None:
        """Verifies handling of UTF-8, accented characters, and emojis."""
        req = ChatRequest(
            agent_id="portfolio",
            session_id="sess_🚀_123",
            message="¿Qué servicios de Inteligencia Artificial ofrecen? 🤖✨",
        )
        assert "🤖✨" in req.message
        assert "🚀" in req.session_id

    def test_chat_request_large_payload(self) -> None:
        """Verifies handling of large message payload (boundary test)."""
        large_text = "Python AI Microservice " * 1000
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_large_01",
            message=large_text,
        )
        assert len(req.message) > 20000

    def test_chat_request_serialization(self) -> None:
        """Verifies model_dump and model_dump_json produce correct dictionaries and JSON strings."""
        req = ChatRequest(
            agent_id="ecommerce",
            session_id="sess_abc",
            message="Comprar producto",
            stream=True,
        )
        dumped = req.model_dump()
        assert dumped["agent_id"] == "ecommerce"
        assert dumped["stream"] is True
        assert isinstance(req.model_dump_json(), str)


class TestChatResponseSchema:
    """Test suite for ChatResponse Pydantic model validation."""

    def test_valid_chat_response_default_metadata(self) -> None:
        """Verifies that ChatResponse defaults metadata to an empty dict."""
        res = ChatResponse(
            agent_id="portfolio",
            session_id="sess_123",
            message="Respuesta completa del agente",
        )
        assert res.agent_id == "portfolio"
        assert res.session_id == "sess_123"
        assert res.message == "Respuesta completa del agente"
        assert res.metadata == {}

    def test_valid_chat_response_with_custom_metadata(self) -> None:
        """Verifies ChatResponse with detailed metadata dictionary."""
        metadata = {
            "model": "gemini-2.5-flash",
            "prompt_tokens": 50,
            "candidate_tokens": 120,
            "latency_ms": 312.4,
            "tools_called": ["search_portfolio", "get_skills"],
        }
        res = ChatResponse(
            agent_id="portfolio",
            session_id="sess_123",
            message="Detalle de servicios",
            metadata=metadata,
        )
        assert res.metadata["model"] == "gemini-2.5-flash"
        assert res.metadata["latency_ms"] == 312.4
        assert len(res.metadata["tools_called"]) == 2

    @pytest.mark.parametrize("missing_field", ["agent_id", "session_id", "message"])
    def test_chat_response_missing_required_fields(self, missing_field: str) -> None:
        """Verifies that missing required fields in ChatResponse trigger ValidationError."""
        data = {
            "agent_id": "portfolio",
            "session_id": "sess_123",
            "message": "Hola",
        }
        del data[missing_field]

        with pytest.raises(ValidationError):
            ChatResponse(**data)


class TestHealthResponseSchema:
    """Test suite for HealthResponse Pydantic model validation."""

    def test_valid_health_response_default_status(self) -> None:
        """Verifies default status is 'ok'."""
        res = HealthResponse(
            app_name="AI Agent Gateway",
            environment="production",
            version="1.0.0",
        )
        assert res.status == "ok"
        assert res.app_name == "AI Agent Gateway"
        assert res.environment == "production"
        assert res.version == "1.0.0"

    def test_health_response_custom_status(self) -> None:
        """Verifies custom status override."""
        res = HealthResponse(
            status="degraded",
            app_name="AI Agent Gateway",
            environment="staging",
            version="1.0.0-rc1",
        )
        assert res.status == "degraded"

    def test_health_response_missing_fields(self) -> None:
        """Verifies that missing app_name, environment, or version fails validation."""
        with pytest.raises(ValidationError):
            HealthResponse(app_name="AI Gateway")
