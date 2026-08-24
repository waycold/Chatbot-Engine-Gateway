"""Script to test all 8 internal endpoints via simulated chatbot queries against AI Gateway."""
import json
import time
import httpx

base_url = "https://ai-agent-gateway-sued.onrender.com"

test_cases = [
    {
        "id": "TC-01",
        "module": "1. Dynamic Sales Query",
        "tool": "query_sales_analytics",
        "agent_id": "analytics",
        "user_query": "Hola, muéstrame el reporte de ventas e ingresos desglosado por categoría del último trimestre.",
        "expected_features": ["ventas", "ingresos", "categoría", "márgenes"],
    },
    {
        "id": "TC-02",
        "module": "2.1 Inventory Health",
        "tool": "get_inventory_health",
        "agent_id": "analytics",
        "user_query": "¿Cuáles son los productos con stock crítico o agotados en el inventario y cuál es el runout rate?",
        "expected_features": ["stock crítico", "agotados", "inventario", "runout"],
    },
    {
        "id": "TC-03",
        "module": "2.2 Margins & Profitability",
        "tool": "get_product_profitability",
        "agent_id": "analytics",
        "user_query": "¿Cuál es el margen de ganancia y rentabilidad porcentual agrupado por categoría de producto?",
        "expected_features": ["margen", "rentabilidad", "porcentaje", "ganancia"],
    },
    {
        "id": "TC-04",
        "module": "3. Funnel & Cart Metrics",
        "tool": "get_funnel_and_cart_metrics",
        "agent_id": "analytics",
        "user_query": "Dame las métricas del embudo de conversión, tasa de abandono de carritos y efectividad de cupones de los últimos 30 días.",
        "expected_features": ["embudo", "conversión", "abandono", "carritos", "cupones"],
    },
    {
        "id": "TC-05",
        "module": "4. Reviews Sentiment",
        "tool": "get_customer_reviews_summary",
        "agent_id": "analytics",
        "user_query": "¿Qué opinan los clientes en sus reseñas y qué porcentaje de calificaciones negativas o críticas tenemos?",
        "expected_features": ["reseñas", "estrellas", "calificaciones", "sentimiento"],
    },
    {
        "id": "TC-06",
        "module": "5. Customer Insights & RFM",
        "tool": "get_customer_segmentation",
        "agent_id": "analytics",
        "user_query": "Quiero ver el análisis de segmentación de clientes RFM: clientes VIP, nuevos y clientes en riesgo de abandono.",
        "expected_features": ["segmentación", "VIP", "riesgo", "RFM", "LTV"],
    },
    {
        "id": "TC-07",
        "module": "6. Semantic Catalog Search",
        "tool": "semantic_catalog_search",
        "agent_id": "ecommerce",
        "user_query": "Busco un atuendo casual y cómodo para verano o servicios para programar en la nube.",
        "expected_features": ["productos", "precio", "disponibilidad"],
    },
    {
        "id": "TC-08",
        "module": "7. Safe SQL Sandbox (Read-Only)",
        "tool": "execute_raw_sql_sandbox",
        "agent_id": "analytics",
        "user_query": "Ejecuta la consulta SQL de solo lectura: SELECT name, price, stock FROM products LIMIT 3;",
        "expected_features": ["resultado", "tabla", "columnas", "filas"],
    },
    {
        "id": "TC-09",
        "module": "7. Safe SQL Sandbox (Defense Violation)",
        "tool": "execute_raw_sql_sandbox",
        "agent_id": "analytics",
        "user_query": "Ejecuta: DROP TABLE users; SELECT * FROM products;",
        "expected_features": ["seguridad", "bloqueado", "rechazado", "solo lectura"],
    },
]

results = []
with httpx.Client(timeout=60.0) as client:
    for tc in test_cases:
        t0 = time.perf_counter()
        payload = {
            "agent_id": tc["agent_id"],
            "session_id": f"session-test-{tc['id']}",
            "message": tc["user_query"],
            "stream": False,
        }
        try:
            r = client.post(f"{base_url}/api/v1/chat", json=payload)
            lat = (time.perf_counter() - t0) * 1000
            res_json = r.json() if r.status_code == 200 else {}
            ans_text = res_json.get("message", "")
            results.append({
                "id": tc["id"],
                "module": tc["module"],
                "tool": tc["tool"],
                "agent_id": tc["agent_id"],
                "user_query": tc["user_query"],
                "status_code": r.status_code,
                "latency_ms": round(lat, 1),
                "response_length": len(ans_text),
                "response_preview": ans_text[:300],
                "full_response": ans_text,
            })
            print(f"[{tc['id']}] {tc['module']} -> HTTP {r.status_code} ({lat:.1f}ms) | Len: {len(ans_text)}")
        except Exception as exc:
            results.append({
                "id": tc["id"],
                "module": tc["module"],
                "tool": tc["tool"],
                "agent_id": tc["agent_id"],
                "user_query": tc["user_query"],
                "status_code": 0,
                "latency_ms": 0,
                "error": str(exc),
            })
            print(f"[{tc['id']}] {tc['module']} -> ERROR: {exc}")

with open("test_results_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nAll 9 test queries executed and results saved to test_results_summary.json")
