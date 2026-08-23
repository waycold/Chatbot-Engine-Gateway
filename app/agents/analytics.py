"""Analytics Specialized Agent implementation with multi-tool analytical query capabilities."""
import json
import logging
import re
from typing import Any, AsyncGenerator, Optional
from app.agents.base import BaseAgent
from app.schemas.payload import ChatRequest, ChatResponse
from app.services.django_api import DjangoAPIService
from app.services.llm_client import LLMClientService
from app.services.memory import RedisMemoryService

logger = logging.getLogger("ai_gateway.agent.analytics")


class AnalyticsAgent(BaseAgent):
    """Specialized AI Agent for querying business metrics, KPIs, sales summaries,
    inventory health, product profitability, customer segmentation, funnel conversion,
    and safe SQL sandbox with token-based access validation.
    """

    def __init__(
        self,
        agent_id: str = "analytics",
        llm_service: Optional[LLMClientService] = None,
        memory_service: Optional[RedisMemoryService] = None,
        django_service: Optional[DjangoAPIService] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name="Analytics & Business Metrics Agent",
            description="Ejecuta consultas analíticas avanzadas, métricas de ventas, rentabilidad, inventario, segmentación de clientes y consultas SQL seguras.",
            capabilities=[
                "sales_analytics",
                "inventory_health",
                "profitability_margins",
                "conversion_funnel",
                "customer_segmentation",
                "reviews_sentiment",
                "sql_sandbox",
                "kpi_reporting",
            ],
            llm_service=llm_service,
            memory_service=memory_service,
            django_service=django_service,
        )

    async def get_system_instruction(self, request: ChatRequest) -> str:
        """Returns specialized persona and constraints for the Analytics Agent."""
        return (
            "Eres el Asistente Analista de Datos y Métricas de Negocio (Business Intelligence & Analytics Agent). "
            "Tu misión es interpretar consultas sobre KPIs, ventas, rentabilidad/márgenes, salud de inventario, "
            "embudos de conversión (funnel), segmentación de clientes (RFM), reseñas y consultas SQL.\n\n"
            "Pautas de respuesta:\n"
            "1. Presenta las métricas de forma estructurada, utilizando tablas en Markdown, listas ordenadas y destacados en negrita.\n"
            "2. Proporciona resúmenes ejecutivos con observaciones clave, tendencias y recomendaciones prácticas accionables.\n"
            "3. En consultas de SQL sandbox, muestra los resultados tabulados y aclara que opera en modo solo lectura de seguridad.\n"
            "4. Si los datos indican que el token de usuario no es válido o falta autenticación para métricas restringidas, "
            "explica amablemente que se requiere un token JWT con privilegios analíticos.\n"
            "5. Sé riguroso, objetivo y exacto: no inventes cifras fuera de los datos provistos en el contexto."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Validates token and queries analytical tools from Django backend based on query intent."""
        user_token = request.user_token
        auth_status: dict[str, Any] = {"authenticated": False, "role": "anonymous"}

        if user_token:
            validation = await self.django_service.validate_user_token(user_token)
            if validation.get("valid"):
                auth_status = {
                    "authenticated": True,
                    "user_id": validation.get("user_id", "unknown"),
                    "role": validation.get("role", "analyst"),
                }
            else:
                auth_status = {
                    "authenticated": False,
                    "error": validation.get("error", "Invalid authentication token"),
                }

        msg = request.message.strip()
        msg_lower = msg.lower()
        context_data: dict[str, Any] = {"auth_context": auth_status}

        try:
            # 1. Check for Safe SQL Sandbox query
            if any(k in msg_lower for k in ["select ", "sql:", "sql ", "drop ", "delete ", "insert ", "update ", "alter ", "truncate "]):
                cleaned_sql = re.sub(r'^(?:ejecuta|consulta|query|sql|run|execute)\s*[:\s]*', '', msg, flags=re.IGNORECASE).strip()
                sql_res = await self.django_service.execute_raw_sql_sandbox(sql_query=cleaned_sql, user_token=user_token)
                context_data["tool_invoked"] = "execute_raw_sql_sandbox"
                context_data["sql_results"] = sql_res

            # 2. Check for Inventory Health
            elif any(k in msg_lower for k in ["stock", "inventario", "agotado", "agotados", "crítico", "critico", "runout", "cobertura"]):
                status_filter = "critical" if "crítico" in msg_lower or "critico" in msg_lower else "all"
                inv_res = await self.django_service.get_inventory_health(status_filter=status_filter, user_token=user_token)
                context_data["tool_invoked"] = "get_inventory_health"
                context_data["inventory_health"] = inv_res

            # 3. Check for Margins & Profitability
            elif any(k in msg_lower for k in ["margen", "márgenes", "margenes", "rentabilidad", "ganancia", "profit", "markup"]):
                group_by = "category" if "categoría" in msg_lower or "categoria" in msg_lower else "product"
                margin_res = await self.django_service.get_product_profitability(group_by=group_by, user_token=user_token)
                context_data["tool_invoked"] = "get_product_profitability"
                context_data["margins_and_profitability"] = margin_res

            # 4. Check for Conversion Funnel & Abandoned Carts
            elif any(k in msg_lower for k in ["funnel", "embudo", "carrito", "carritos", "abandono", "checkout", "cupon", "cupón", "cupones"]):
                funnel_res = await self.django_service.get_funnel_and_cart_metrics(timeframe="30d", user_token=user_token)
                context_data["tool_invoked"] = "get_funnel_and_cart_metrics"
                context_data["funnel_and_cart_metrics"] = funnel_res

            # 5. Check for Reviews & Customer Sentiment
            elif any(k in msg_lower for k in ["reseña", "reseñas", "review", "reviews", "calificación", "calificacion", "estrellas", "satisfacción", "satisfaccion", "opinion", "opiniones"]):
                reviews_res = await self.django_service.get_customer_reviews_summary(user_token=user_token)
                context_data["tool_invoked"] = "get_customer_reviews_summary"
                context_data["reviews_summary"] = reviews_res

            # 6. Check for Customer Insights & RFM Segmentation
            elif any(k in msg_lower for k in ["cliente", "clientes", "rfm", "vip", "ltv", "churn", "en riesgo", "segmento", "segmentación", "segmentacion"]):
                segment = "vip" if "vip" in msg_lower else ("at_risk" if "riesgo" in msg_lower else "all")
                rfm_res = await self.django_service.get_customer_segmentation(segment=segment, user_token=user_token)
                context_data["tool_invoked"] = "get_customer_segmentation"
                context_data["customer_segmentation"] = rfm_res

            # 7. Dynamic Sales & Revenue Query
            elif any(k in msg_lower for k in ["venta", "ventas", "ingreso", "ingresos", "revenue", "facturación", "facturacion"]):
                dimension = "category" if "categoría" in msg_lower or "categoria" in msg_lower else "month"
                sales_res = await self.django_service.query_sales_analytics(dimension=dimension, user_token=user_token)
                context_data["tool_invoked"] = "query_sales_analytics"
                context_data["sales_analytics"] = sales_res

            # 8. General KPI Overview fallback
            else:
                analytics_data = await self.django_service.query_analytics(metric_type="all", timeframe="30d", user_token=user_token)
                context_data["tool_invoked"] = "query_analytics"
                context_data["general_metrics"] = analytics_data

            return f"=== ANALYTICS & BI LIVE DATA ===\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"

        except Exception as exc:
            logger.warning("Error querying analytics data: %s", exc)
            return f"=== ANALYTICS DATA ===\nAuth Status: {json.dumps(auth_status)}"

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        return await self._execute_process(request)

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        async for token in self._execute_process_stream(request):
            yield token
