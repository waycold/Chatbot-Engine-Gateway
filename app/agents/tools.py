"""Tool Calling and Function Calling declarations for Google GenAI agents."""
import logging
from typing import Any, Callable, Coroutine, Optional
from app.services.django_api import get_django_api_service

logger = logging.getLogger("ai_gateway.agents.tools")

# Schema definitions for Google GenAI function declarations
ANALYTICS_TOOL_DECLARATIONS = [
    {
        "name": "query_sales_analytics",
        "description": "Consulta métricas de ventas agregadas, ingresos totales, costos, ganancias brutas y márgenes % agrupados por dimensión.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date_from": {
                    "type": "STRING",
                    "description": "Fecha de inicio en formato YYYY-MM-DD (opcional)",
                },
                "date_to": {
                    "type": "STRING",
                    "description": "Fecha de fin en formato YYYY-MM-DD (opcional)",
                },
                "dimension": {
                    "type": "STRING",
                    "description": "Dimensión de agrupación: 'category', 'brand', 'supplier', 'payment_method', 'country', 'day', 'week', 'month', 'quarter'",
                    "enum": ["category", "brand", "supplier", "payment_method", "country", "day", "week", "month", "quarter"],
                },
            },
        },
    },
    {
        "name": "get_inventory_health",
        "description": "Consulta el estado de salud del inventario, stock crítico, productos agotados, valorización y velocidad de agotamiento (Runout Rate a 30 días).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "status_filter": {
                    "type": "STRING",
                    "description": "Filtro de estado: 'all', 'critical', 'out_of_stock', 'healthy'",
                    "enum": ["all", "critical", "out_of_stock", "healthy"],
                },
                "limit": {
                    "type": "INTEGER",
                    "description": "Cantidad máxima de productos a listar (máximo 50)",
                },
            },
        },
    },
    {
        "name": "get_product_profitability",
        "description": "Calcula y consulta el ranking de rentabilidad y margen bruto % (((Ventas - Costos)/Ventas)*100) por producto o categoría.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "group_by": {
                    "type": "STRING",
                    "description": "Agrupación: 'product', 'category', 'brand', 'supplier'",
                    "enum": ["product", "category", "brand", "supplier"],
                },
                "limit": {
                    "type": "INTEGER",
                    "description": "Cantidad máxima de elementos en el ranking",
                },
            },
        },
    },
    {
        "name": "get_funnel_and_cart_metrics",
        "description": "Obtiene métricas de conversión del funnel de ventas, tasa de abandono de carritos, productos más abandonados y ROI de cupones.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "timeframe": {
                    "type": "STRING",
                    "description": "Ventana temporal de análisis: '7d', '30d', '90d'",
                    "enum": ["7d", "30d", "90d"],
                },
            },
        },
    },
    {
        "name": "get_customer_reviews_summary",
        "description": "Obtiene el resumen de reseñas de clientes, rating promedio (1-5 estrellas), distribución de satisfacción y feedback crítico.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "product_id": {
                    "type": "STRING",
                    "description": "ID del producto para filtrar reseñas (opcional, por defecto 'all')",
                },
                "sentiment": {
                    "type": "STRING",
                    "description": "Filtro de sentimiento: 'all', 'positive', 'critical', 'negative'",
                    "enum": ["all", "positive", "critical", "negative"],
                },
            },
        },
    },
    {
        "name": "get_customer_segmentation",
        "description": "Consulta la segmentación de clientes mediante análisis RFM (VIPs, En Riesgo >60d, Nuevos <30d), Customer LTV y geolocalización.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "segment": {
                    "type": "STRING",
                    "description": "Segmento objetivo: 'all', 'vip', 'at_risk', 'new', 'churned'",
                    "enum": ["all", "vip", "at_risk", "new", "churned"],
                },
            },
        },
    },
    {
        "name": "semantic_catalog_search",
        "description": "Realiza búsqueda semántica y conceptual en el catálogo de productos según la intención o caso de uso del usuario.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Consulta o intención conceptual (ej: 'cursos para programadores backend')",
                },
                "category_hint": {
                    "type": "STRING",
                    "description": "Categoría opcional para acotar la búsqueda",
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Cantidad de resultados a retornar",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_raw_sql_sandbox",
        "description": "Ejecuta de forma segura una consulta SQL de solo lectura (SELECT) en la base de datos analítica con límite de 50 filas.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "sql_query": {
                    "type": "STRING",
                    "description": "Consulta SQL (únicamente sentencias SELECT)",
                },
                "max_rows": {
                    "type": "INTEGER",
                    "description": "Límite de filas a retornar (máximo 50)",
                },
            },
            "required": ["sql_query"],
        },
    },
]


async def execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    user_token: Optional[str] = None,
) -> dict[str, Any]:
    """Dispatches a tool call invocation to the corresponding Django API service method.

    Args:
        tool_name: The declared tool name identifier.
        tool_args: Dictionary of arguments passed by the model.
        user_token: Optional JWT token for authenticated operations.

    Returns:
        Dictionary containing the tool execution result.
    """
    django_service = get_django_api_service()
    logger.info("Executing tool '%s' with args: %s", tool_name, tool_args)

    try:
        if tool_name == "query_sales_analytics":
            return await django_service.query_sales_analytics(
                date_from=tool_args.get("date_from"),
                date_to=tool_args.get("date_to"),
                dimension=tool_args.get("dimension", "category"),
                user_token=user_token,
            )

        elif tool_name == "get_inventory_health":
            return await django_service.get_inventory_health(
                status_filter=tool_args.get("status_filter", "all"),
                limit=int(tool_args.get("limit", 10)),
                user_token=user_token,
            )

        elif tool_name == "get_product_profitability":
            return await django_service.get_product_profitability(
                group_by=tool_args.get("group_by", "product"),
                limit=int(tool_args.get("limit", 10)),
                user_token=user_token,
            )

        elif tool_name == "get_funnel_and_cart_metrics":
            return await django_service.get_funnel_and_cart_metrics(
                timeframe=tool_args.get("timeframe", "30d"),
                user_token=user_token,
            )

        elif tool_name == "get_customer_reviews_summary":
            return await django_service.get_customer_reviews_summary(
                product_id=tool_args.get("product_id"),
                sentiment=tool_args.get("sentiment", "all"),
                user_token=user_token,
            )

        elif tool_name == "get_customer_segmentation":
            return await django_service.get_customer_segmentation(
                segment=tool_args.get("segment", "all"),
                user_token=user_token,
            )

        elif tool_name == "semantic_catalog_search":
            return await django_service.semantic_catalog_search(
                query=tool_args.get("query", ""),
                category_hint=tool_args.get("category_hint"),
                top_k=int(tool_args.get("top_k", 4)),
            )

        elif tool_name == "execute_raw_sql_sandbox":
            return await django_service.execute_raw_sql_sandbox(
                sql_query=tool_args.get("sql_query", ""),
                max_rows=int(tool_args.get("max_rows", 50)),
                user_token=user_token,
            )

        else:
            logger.warning("Unknown tool '%s' requested.", tool_name)
            return {"status": "error", "error": f"Tool '{tool_name}' is not recognized."}

    except Exception as exc:
        logger.error("Error executing tool '%s': %s", tool_name, exc, exc_info=True)
        return {"status": "error", "error": f"Tool execution failed: {str(exc)}"}
