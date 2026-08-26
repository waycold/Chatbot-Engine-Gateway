"""Analytics Specialized Agent implementation with multi-tool analytical query capabilities."""
import contextvars
import json
import logging
import re
from typing import Any, AsyncGenerator, Optional
from app.agents.base import BaseAgent, EventSink
from app.agents.tools import ANALYTICS_TOOL_DECLARATIONS, SQL_SANDBOX_TOOL_NAME
from app.schemas.payload import ChatRequest, ChatResponse
from app.services.django_api import DjangoAPIService
from app.services.llm_client import LLMClientService
from app.services.memory import RedisMemoryService

logger = logging.getLogger("ai_gateway.agent.analytics")

# Request-scoped cache of the resolved authorization status. `get_tool_declarations` is
# synchronous while token validation is async, so the verdict has to be handed across.
#
# It is deliberately NOT stashed on `request.context`: that field is part of the public
# ChatRequest payload, so a client could simply POST
# `{"context": {"auth_status": {"authenticated": true, "is_staff": true}}}` and forge its
# own staff privileges. A ContextVar is server-side only, is copied into any task
# spawned from the request (so the SSE pump task inherits it), and cannot be reached from
# the wire at all.
_auth_status_var: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "analytics_auth_status", default=None,
)


def set_auth_status(auth_status: dict[str, Any]) -> None:
    """Publishes the resolved authorization status for the current request scope.

    Args:
        auth_status: The auth status dict assembled from `validate_user_token`.
    """
    _auth_status_var.set(auth_status)


def get_auth_status() -> dict[str, Any]:
    """Reads the authorization status resolved earlier in this request scope.

    Returns:
        The cached auth status, or an empty dict when none was resolved — which
        `_is_staff` treats as not-staff, so the unknown case fails closed.
    """
    return _auth_status_var.get() or {}


def _is_staff(auth_status: dict[str, Any]) -> bool:
    """Determines whether a resolved auth status grants privileged analytics access.

    Privilege comes exclusively from Django's two native `auth_user` booleans, which
    `DjangoAPIService.validate_user_token` normalizes and this module carries through
    unchanged. The expected shape is::

        {"authenticated": bool, "user_id": Optional[int], "username": Optional[str],
         "is_staff": bool, "is_superuser": bool}

    The comparison is deliberately `is True`, not truthiness: a validator response that
    leaked a string (`"false"`) or any other non-boolean must never grant privilege. An
    unknown, malformed or absent auth status fails closed.

    Note:
        Known tech debt: reusing `is_staff` to authorize the read-only SQL console
        conflates two different permissions — access to the Django admin is not the same
        thing as permission to run SQL through a chat agent — an accepted simplification
        at this project's scope, to revisit if this grows past a portfolio system.

    Args:
        auth_status: The auth status dict assembled from `validate_user_token`.

    Returns:
        True only when the request is authenticated AND Django reports the identity as
        `is_staff` or `is_superuser`.
    """
    if not isinstance(auth_status, dict):
        return False

    return bool(auth_status.get("authenticated")) and (
        auth_status.get("is_staff") is True or auth_status.get("is_superuser") is True
    )


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

    # ------------------------------------------------------------------
    # Date extraction helper
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_date_range(text: str) -> tuple[Optional[str], Optional[str]]:
        """Parses natural-language date references from a user message and
        returns an (ISO date_from, ISO date_to) tuple.

        Supported patterns (Spanish + English):
          - "diciembre de/del 2025" / "december 2025"
          - "enero 2024" / "january 2024"
          - "el año pasado" / "last year"
          - "este mes" / "this month"
          - explicit "2025-12-01" / "2025/12/01"
          - "Q1 2025", "primer trimestre 2025"
        """
        from datetime import date, timedelta
        import calendar

        MONTHS_ES = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
        }
        MONTHS_EN = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        MONTHS = {**MONTHS_ES, **MONTHS_EN}

        t = text.lower()
        today = date.today()

        # Explicit ISO date range: "desde 2025-01-01 hasta 2025-03-31"
        iso_range = re.search(
            r'(\d{4}[-/]\d{2}[-/]\d{2})\s*(?:hasta|to|al|a)\s*(\d{4}[-/]\d{2}[-/]\d{2})', t
        )
        if iso_range:
            d1 = iso_range.group(1).replace("/", "-")
            d2 = iso_range.group(2).replace("/", "-")
            return d1, d2

        # "el año pasado" / "last year"
        if re.search(r'\baño\s+pasado\b|\blast\s+year\b', t):
            y = today.year - 1
            # Check if a specific month also mentioned (handled below); skip bare "año pasado" here
            # only if no month keyword is present
            has_month = any(re.search(rf'\b{m}\b', t) for m in MONTHS)
            if not has_month:
                return f"{y}-01-01", f"{y}-12-31"

        # "este año" / "this year"
        if re.search(r'\beste\s+año\b|\bthis\s+year\b', t):
            y = today.year
            return f"{y}-01-01", f"{y}-12-31"

        # "este mes" / "this month"
        if re.search(r'\beste\s+mes\b|\bthis\s+month\b', t):
            y, m = today.year, today.month
            last_day = calendar.monthrange(y, m)[1]
            return f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last_day}"

        # "mes pasado" / "last month"
        if re.search(r'\bmes\s+pasado\b|\blast\s+month\b', t):
            first_this = today.replace(day=1)
            last_prev = first_this - timedelta(days=1)
            y, m = last_prev.year, last_prev.month
            last_day = calendar.monthrange(y, m)[1]
            return f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last_day}"

        # Quarter detection: "Q1 2025", "primer trimestre 2025", etc.
        q_map = {
            "q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12),
            "primer trimestre": (1, 3), "segundo trimestre": (4, 6),
            "tercer trimestre": (7, 9), "cuarto trimestre": (10, 12),
            "first quarter": (1, 3), "second quarter": (4, 6),
            "third quarter": (7, 9), "fourth quarter": (10, 12),
        }
        for label, (qm_start, qm_end) in q_map.items():
            pattern = rf'\b{re.escape(label)}\b\s*(?:de\s+|del\s+)?(\d{{4}})'
            m_q = re.search(pattern, t)
            if m_q:
                y = int(m_q.group(1))
                last_day = calendar.monthrange(y, qm_end)[1]
                return f"{y}-{qm_start:02d}-01", f"{y}-{qm_end:02d}-{last_day}"

        # Named month + year: "diciembre del año pasado", "december of last year", "diciembre 2025", "december 2024"
        for month_name, m_num in MONTHS.items():
            # "mes_name del año pasado", "month_name of last year", "last year month_name"
            if re.search(
                rf'\b{month_name}\b.*(?:\baño\s+(?:pasado|anterior)\b|\b(?:last|previous)\s+year\b)'
                rf'|(?:\baño\s+(?:pasado|anterior)\b|\b(?:last|previous)\s+year\b).*\b{month_name}\b', t
            ):
                y = today.year - 1
                import calendar as _cal
                last_day = _cal.monthrange(y, m_num)[1]
                return f"{y}-{m_num:02d}-01", f"{y}-{m_num:02d}-{last_day}"

            # "mes_name de este año", "month_name of this year"
            if re.search(
                rf'\b{month_name}\b.*(?:\beste\s+año\b|\bthis\s+year\b)'
                rf'|(?:\beste\s+año\b|\bthis\s+year\b).*\b{month_name}\b', t
            ):
                y = today.year
                import calendar as _cal
                last_day = _cal.monthrange(y, m_num)[1]
                return f"{y}-{m_num:02d}-01", f"{y}-{m_num:02d}-{last_day}"

            # With explicit year: "diciembre 2025" / "diciembre de 2025" / "december 2025"
            explicit = re.search(
                rf'\b{month_name}\b\s*(?:(?:de(?:l)?|of)\s+)?(\d{{4}})', t
            )
            if explicit:
                y = int(explicit.group(1))
                import calendar as _cal
                last_day = _cal.monthrange(y, m_num)[1]
                return f"{y}-{m_num:02d}-01", f"{y}-{m_num:02d}-{last_day}"

        # Bare "año pasado" already handled above but may have reached here if a month was found yet not matched
        if re.search(r'\baño\s+pasado\b|\blast\s+year\b', t):
            y = today.year - 1
            return f"{y}-01-01", f"{y}-12-31"

        # No date range detected
        return None, None

    def get_tool_declarations(self, request: ChatRequest) -> list[dict[str, Any]]:
        """Returns the analytics tool schemas, exposing the SQL console to staff only.

        The privileged `execute_raw_sql_sandbox` schema is only included when this turn's
        token has ALREADY validated as staff — that is, when Django reported `is_staff` or
        `is_superuser` for it. `get_context_augmentation` runs first in both execution
        paths and publishes that verdict into the request-scoped ContextVar; when it is
        absent (unknown caller, out-of-order invocation) this fails closed and the console
        is withheld. Withholding the schema is layer one — `execute_tool`'s allowlist
        independently refuses the same call as layer two.

        Args:
            request: The incoming chat request.

        Returns:
            Six read-only analytics declarations, plus the SQL console for staff.
        """
        if _is_staff(get_auth_status()):
            return list(ANALYTICS_TOOL_DECLARATIONS)

        return [
            declaration
            for declaration in ANALYTICS_TOOL_DECLARATIONS
            if declaration["name"] != SQL_SANDBOX_TOOL_NAME
        ]

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
            "5. Sé riguroso, objetivo y exacto: no inventes cifras fuera de los datos provistos en el contexto.\n"
            "6. IMPORTANTE: cuando el usuario especifique un periodo temporal (mes, trimestre, año, rango de fechas), "
            "los datos que presentes DEBEN corresponder EXCLUSIVAMENTE a ese periodo — nunca al acumulado histórico general. "
            "Si el contexto de datos muestra 'date_from' y 'date_to', esos son los límites del análisis. "
            "No mezcles ni confundas ingresos acumulados históricos con ingresos del periodo solicitado."
        )

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Validates token and queries analytical tools from Django backend based on query intent."""
        user_token = request.user_token
        auth_status: dict[str, Any] = {
            "authenticated": False,
            "user_id": None,
            "username": None,
            "is_staff": False,
            "is_superuser": False,
        }

        if user_token:
            validation = await self.django_service.validate_user_token(user_token)
            if validation.get("valid"):
                # Privilege comes from Django's native `auth_user` booleans and nothing
                # else — `auth_user` has no `role` column — so the six normalized keys
                # returned by the validator are carried through verbatim.
                auth_status = {
                    "authenticated": True,
                    "user_id": validation.get("user_id"),
                    "username": validation.get("username"),
                    "is_staff": validation.get("is_staff"),
                    "is_superuser": validation.get("is_superuser"),
                }
            else:
                auth_status = {
                    "authenticated": False,
                    "user_id": None,
                    "username": None,
                    "is_staff": False,
                    "is_superuser": False,
                    "error": validation.get("error", "Invalid authentication token"),
                }

        is_staff = _is_staff(auth_status)

        # Publish the resolved status so the synchronous `get_tool_declarations` can read
        # it later in this same turn. Server-side only — never taken from the payload.
        set_auth_status(auth_status)

        msg = request.message.strip()
        msg_lower = msg.lower()
        context_data: dict[str, Any] = {"auth_context": auth_status}

        # Extract temporal date range from the user's message once, reuse for all tool calls
        date_from, date_to = self._extract_date_range(msg_lower)
        if date_from and date_to:
            context_data["detected_date_range"] = {"date_from": date_from, "date_to": date_to}

        try:
            # 1. Check for Safe SQL Sandbox query (staff only)
            if any(k in msg_lower for k in ["select ", "sql:", "sql ", "drop ", "delete ", "insert ", "update ", "alter ", "truncate "]):
                context_data["tool_invoked"] = SQL_SANDBOX_TOOL_NAME
                if is_staff:
                    cleaned_sql = re.sub(r'^(?:ejecuta|consulta|query|sql|run|execute)\s*[:\s]*', '', msg, flags=re.IGNORECASE).strip()
                    sql_res = await self.django_service.execute_raw_sql_sandbox(sql_query=cleaned_sql, user_token=user_token)
                    context_data["sql_results"] = sql_res
                else:
                    # The sandbox is NOT called at all for non-staff requests. `tool_invoked`
                    # stays set so the model can explain what was refused and why.
                    logger.warning(
                        "Blocked SQL sandbox access for session '%s': caller is not staff.",
                        request.session_id,
                    )
                    context_data["sql_results"] = {
                        "status": "error",
                        "blocked": True,
                        "error": "Acceso denegado: la consola SQL requiere un token JWT de staff válido.",
                    }

            # 2. Check for Inventory Health
            elif any(k in msg_lower for k in ["stock", "inventario", "agotado", "agotados", "crítico", "critico", "runout", "cobertura"]):
                status_filter = "critical" if "crítico" in msg_lower or "critico" in msg_lower else "all"
                inv_res = await self.django_service.get_inventory_health(status_filter=status_filter, user_token=user_token)
                context_data["tool_invoked"] = "get_inventory_health"
                context_data["inventory_health"] = inv_res

            # 3. Check for Margins & Profitability
            elif any(k in msg_lower for k in ["margen", "márgenes", "margenes", "rentabilidad", "ganancia", "profit", "markup"]):
                group_by = "category" if "categoría" in msg_lower or "categoria" in msg_lower else "product"
                margin_res = await self.django_service.get_product_profitability(
                    group_by=group_by,
                    date_from=date_from,
                    date_to=date_to,
                    user_token=user_token,
                )
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
            elif any(k in msg_lower for k in ["venta", "ventas", "ingreso", "ingresos", "revenue", "facturación", "facturacion", "sales", "sale", "vendido", "vendidos", "facturado"]):
                # Determine dimension: category, brand, supplier, payment_method, country, day, week, month, quarter
                if any(c in msg_lower for c in ["categoría", "categoria", "category", "categories"]):
                    dimension = "category"
                elif any(b in msg_lower for b in ["marca", "marcas", "brand", "brands"]):
                    dimension = "brand"
                elif any(s in msg_lower for s in ["proveedor", "proveedores", "supplier", "suppliers"]):
                    dimension = "supplier"
                elif any(p in msg_lower for p in ["pago", "payment", "metodo de pago", "payment_method"]):
                    dimension = "payment_method"
                elif any(co in msg_lower for co in ["país", "pais", "country", "countries"]):
                    dimension = "country"
                elif "semana" in msg_lower or "week" in msg_lower:
                    dimension = "week"
                elif "trimestre" in msg_lower or "quarter" in msg_lower or "q1" in msg_lower or "q2" in msg_lower or "q3" in msg_lower or "q4" in msg_lower:
                    dimension = "quarter"
                elif "dia" in msg_lower or "día" in msg_lower or "day" in msg_lower:
                    dimension = "day"
                else:
                    dimension = "month"

                sales_res = await self.django_service.query_sales_analytics(
                    date_from=date_from,
                    date_to=date_to,
                    dimension=dimension,
                    user_token=user_token,
                )
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

    async def process_stream(
        self,
        request: ChatRequest,
        *,
        event_sink: Optional[EventSink] = None,
    ) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens.

        Args:
            request: The incoming chat request.
            event_sink: Optional tool-progress callback forwarded to the tool loop.
        """
        async for token in self._execute_process_stream(request, event_sink=event_sink):
            yield token
