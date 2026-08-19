"""Analytics Specialized Agent implementation with metric query capabilities."""
import json
import logging
from typing import AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse

logger = logging.getLogger("ai_gateway.agent.analytics")


class AnalyticsAgent(BaseAgent):
    """Specialized AI Agent for querying business metrics, KPIs, sales summaries,

    and data analytics with token-based access validation.
    """

    def __init__(self, agent_id: str = "analytics") -> None:
        super().__init__(
            agent_id=agent_id,
            name="Analytics & Business Metrics Agent",
            description="Ejecuta consultas analíticas, métricas de ventas, KPIs y resúmenes de rendimiento.",
            capabilities=[
                "kpi_metrics",
                "sales_reporting",
                "user_traffic_analysis",
                "conversion_funnel",
                "executive_summaries",
            ],
        )

    async def get_system_instruction(self, request: ChatRequest) -> str:
        """Returns specialized persona and constraints for the Analytics Agent."""
        return (
            "Eres el Asistente Analista de Datos y Métricas de Negocio (Business Intelligence & Analytics Agent). "
            "Tu misión es interpretar consultas sobre KPIs, ventas, volumen de usuarios, tasas de conversión y rendimiento del sistema.\n\n"
            "Pautas de respuesta:\n"
            "1. Presenta las métricas de forma estructurada, utilizando tablas en Markdown, listas ordenadas y destacados en negrita.\n"
            "2. Proporciona resúmenes ejecutivos con observaciones clave y tendencias observadas a partir de los datos proporcionados.\n"
            "3. Si los datos indican que el token de usuario no es válido o falta autenticación para métricas restringidas, "
            "explica amablemente que se requiere un token JWT con privilegios analíticos.\n"
            "4. Sé riguroso y objetivo: no inventes cifras fuera de los datos provistos en el contexto."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Validates token and queries analytical metrics from Django backend."""
        user_token = request.user_token
        auth_status = {"authenticated": False, "role": "anonymous"}

        if user_token:
            validation = await self.django_service.validate_user_token(user_token)
            if validation.get("valid"):
                auth_status = {
                    "authenticated": True,
                    "user_id": validation.get("user_id", "unknown"),
                    "role": validation.get("role", "viewer"),
                }
            else:
                auth_status = {
                    "authenticated": False,
                    "error": validation.get("error", "Invalid authentication token"),
                }

        # Determine metric focus from message
        msg_lower = request.message.lower()
        metric_type = "overview"
        if "top" in msg_lower or "más vendido" in msg_lower or "mas vendido" in msg_lower:
            metric_type = "top_products"
        elif "categoria" in msg_lower or "categoría" in msg_lower or "distribución" in msg_lower or "distribucion" in msg_lower:
            metric_type = "category_distribution"
        elif "pronóstico" in msg_lower or "pronostico" in msg_lower or "forecast" in msg_lower or "tendencia" in msg_lower:
            metric_type = "forecast"
        elif "kpi" in msg_lower or "kpis" in msg_lower or "indicador" in msg_lower or "indicadores" in msg_lower:
            metric_type = "kpis"
        elif "todo" in msg_lower or "completo" in msg_lower or "all" in msg_lower or "general" in msg_lower:
            metric_type = "all"
        elif "venta" in msg_lower or "ingreso" in msg_lower or "sales" in msg_lower or "revenue" in msg_lower:
            metric_type = "kpis"

        try:
            analytics_data = await self.django_service.query_analytics(
                metric_type=metric_type,
                timeframe="30d",
                user_token=user_token,
            )
            context_payload = {
                "auth_context": auth_status,
                "target_metric_type": metric_type,
                "analytics_data": analytics_data,
            }
            return f"Analytics System Data:\n{json.dumps(context_payload, ensure_ascii=False, indent=2)}"
        except Exception as exc:
            logger.warning("Error querying analytics data: %s", exc)
            return f"Auth Status: {json.dumps(auth_status)}"

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        return await self._execute_process(request)

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        async for token in self._execute_process_stream(request):
            yield token
