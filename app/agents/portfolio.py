"""Portfolio Specialized Agent implementation."""
import json
import logging
from typing import AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse

logger = logging.getLogger("ai_gateway.agent.portfolio")


class PortfolioAgent(BaseAgent):
    """Specialized AI Agent handling inquiries regarding professional CV,

    experience, technical skills, architecture projects, and contact information.
    """

    def __init__(self, agent_id: str = "portfolio") -> None:
        super().__init__(
            agent_id=agent_id,
            name="Portfolio & CV Agent",
            description="Responde sobre CV, experiencia, habilidades técnicas, proyectos y contacto.",
            capabilities=[
                "cv_inquiry",
                "skills_overview",
                "projects_showcase",
                "architecture_consulting",
                "contact_info",
            ],
        )

    async def get_system_instruction(self, request: ChatRequest) -> str:
        """Returns specialized persona and constraints for the Portfolio Agent."""
        return (
            "Eres el Asistente Virtual Profesional y de Portafolio de Facundo, un Senior Fullstack & AI Engineer. "
            "Tu objetivo es representar su experiencia, stack tecnológico y proyectos de manera profesional, "
            "clara, concisa y persuasiva.\n\n"
            "Pautas de respuesta:\n"
            "1. Responde de forma amable, estructurada y en el mismo idioma en que te escriben (español por defecto).\n"
            "2. Destaca el dominio en arquitecturas de backend distribuidas con Python, FastAPI, Django, "
            "sistemas Multi-Agente con Google GenAI, Redis y React/TypeScript.\n"
            "3. Cuando te pregunten por proyectos, menciona detalles técnicos relevantes como desacoplamiento, "
            "streaming SSE, microservicios y escalabilidad.\n"
            "4. Sé honesto: si te preguntan por información que no se encuentra en el contexto proporcionado, "
            "invita amablemente al usuario a enviar un mensaje de contacto directo."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Fetches up-to-date developer profile, skills and project highlights."""
        try:
            profile_data = await self.django_service.get_portfolio_data()
            return f"Portfolio Profile Data:\n{json.dumps(profile_data, ensure_ascii=False, indent=2)}"
        except Exception as exc:
            logger.warning("Error fetching portfolio context: %s", exc)
            return None

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        return await self._execute_process(request)

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        async for token in self._execute_process_stream(request):
            yield token
