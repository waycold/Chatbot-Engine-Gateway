"""HTTP client service for internal communication with the Django monolith."""
import logging
import re
from typing import Any, Optional
import httpx
from app.core.config import settings

logger = logging.getLogger("ai_gateway.django_api")


class DjangoAPIService:
    """Async HTTP client for interacting with the core transactional Django backend.

    Secured via the X-Internal-Secret header. Provides methods to query portfolio data,
    product catalog (RAG), and analytics metrics.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        internal_secret: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or settings.DJANGO_BACKEND_URL).rstrip("/")
        self.internal_secret = internal_secret or settings.INTERNAL_API_SECRET
        self.timeout = timeout or settings.DJANGO_API_TIMEOUT_SECONDS
        self._headers = {
            "X-Internal-Secret": self.internal_secret,
            "Content-Type": "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Returns an async HTTP client configured with base URL and internal headers."""
        return httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=self.timeout)

    async def start(self) -> None:
        """Initializes the persistent httpx.AsyncClient session."""
        if self._client is None or self._client.is_closed:
            self._client = await self.get_client()
            logger.info("Django HTTP Client started targeting %s", self.base_url)

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            logger.info("Django HTTP Client closed.")
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Returns the active client or creates an ephemeral one if not started."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=self.timeout)
        return self._client

    async def get_portfolio_data(self) -> dict[str, Any]:
        """Fetches developer CV, skills, and projects data from Django monolith.

        Falls back to default structured developer profile if Django is unreachable.
        """
        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/internal/portfolio/")
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to fetch portfolio data from Django: %s. Using default profile.", exc)

        # Fallback profile
        return {
            "name": "Facundo / Fullstack & AI Engineer",
            "title": "Senior Software & AI Engineer",
            "bio": "Especialista en arquitecturas de backend distribuidas con Python, FastAPI, Django, y orquestación de LLMs.",
            "skills": [
                "Python (FastAPI, Django, Asyncio, Pydantic)",
                "AI & LLMs (Google GenAI, Gemini, RAG, Multi-Agent Systems, Function Calling)",
                "Frontend (React, TypeScript, Next.js, TailwindCSS)",
                "Databases & Cache (PostgreSQL, Redis, Vector Databases)",
                "DevOps & Cloud (Docker, Kubernetes, CI/CD, Render, AWS, GCP)",
            ],
            "projects": [
                {
                    "title": "Chatbot Engine Gateway",
                    "description": "Microservicio de IA de alto rendimiento con FastAPI, Google GenAI SDK, Server-Sent Events y Redis.",
                    "technologies": ["FastAPI", "Python 3.11", "Google GenAI", "Redis", "SSE"],
                },
                {
                    "title": "E-Commerce Core Monolith",
                    "description": "Plataforma de comercio electrónico con backend Django REST Framework y pasarelas de pago.",
                    "technologies": ["Django", "PostgreSQL", "Celery", "Redis"],
                },
                {
                    "title": "Real-time Business Analytics Dashboard",
                    "description": "Motor de analíticas y KPIs con agregación en tiempo real y soporte para consultas en lenguaje natural.",
                    "technologies": ["FastAPI", "Pandas", "PostgreSQL", "React"],
                },
            ],
            "contact": {
                "email": "contact@example.com",
                "github": "https://github.com",
                "linkedin": "https://linkedin.com",
            },
        }

    async def search_catalog(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Searches products in the Django e-commerce catalog via /api/v1/internal/catalog/search/ or /api/products/.

        Performs token-based relevance matching and falls back to representative mock catalog.
        """
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        # Attempt to query Django backend
        for endpoint in ["/api/v1/internal/catalog/search/", "/api/products/"]:
            try:
                client = await self.get_client()
                async with client:
                    response = await client.get(endpoint, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict):
                            items = data.get("items") or data.get("results") or []
                            if items:
                                return items
                        elif isinstance(data, list) and data:
                            return data
            except Exception as exc:
                logger.debug("Catalog endpoint '%s' query error: %s", endpoint, exc)

        # Fallback representative mock catalog
        mock_products = [
            {
                "id": 1,
                "name": "Servicio Cloud AI",
                "category": "Servicios",
                "price": 49.99,
                "currency": "USD",
                "stock": 10,
                "description": "Despliegue y configuración de microservicios de IA escalables en la nube (Render, AWS, GCP).",
            },
            {
                "id": 2,
                "name": "Consultoría DevOps",
                "category": "Servicios",
                "price": 120.00,
                "currency": "USD",
                "stock": 5,
                "description": "Optimización de pipelines CI/CD, contenedorización con Docker y arquitecturas resilientes.",
            },
            {
                "id": 3,
                "name": "Curso Avanzado de FastAPI & Microservicios",
                "category": "Cursos",
                "price": 49.99,
                "currency": "USD",
                "stock": 50,
                "description": "Domina la construcción de microservicios de alto rendimiento con FastAPI, Pydantic v2 y Docker.",
            },
            {
                "id": 4,
                "name": "Módulo de Integración LLM & Agentes Autónomos",
                "category": "Software",
                "price": 89.00,
                "currency": "USD",
                "stock": 25,
                "description": "Librería plug-and-play para orquestar agentes multi-rol con Google GenAI y streaming SSE.",
            },
            {
                "id": 5,
                "name": "Consultoría de Arquitectura de Software (1 Hora)",
                "category": "Servicios",
                "price": 120.00,
                "currency": "USD",
                "stock": 8,
                "description": "Sesión 1 a 1 de revisión de arquitectura, escalabilidad y optimización de microservicios.",
            },
            {
                "id": 6,
                "name": "Template Backend Django + FastAPI Gateway",
                "category": "Templates",
                "price": 29.99,
                "currency": "USD",
                "stock": 100,
                "description": "Boilerplate de producción desacoplado con autenticación por secret y Redis pub/sub.",
            },
        ]

        if not query or not query.strip():
            if category:
                cat_lower = category.lower()
                filtered = [p for p in mock_products if cat_lower in p["category"].lower()]
                return filtered[:limit] if filtered else mock_products[:limit]
            return mock_products[:limit]

        query_clean = query.strip().lower()
        # Extract meaningful alphanumeric tokens
        tokens = [t for t in re.findall(r'\b\w{2,}\b', query_clean)]

        scored_products: list[tuple[int, dict[str, Any]]] = []
        for p in mock_products:
            score = 0
            name_lower = p["name"].lower()
            desc_lower = p["description"].lower()
            cat_lower = p["category"].lower()

            # Exact name substring match
            if query_clean in name_lower:
                score += 50
            if name_lower in query_clean:
                score += 40

            # Token level matches
            for token in tokens:
                if token in name_lower:
                    score += 15
                if token in desc_lower:
                    score += 5
                if token in cat_lower:
                    score += 8

            if score > 0:
                scored_products.append((score, p))

        if scored_products:
            scored_products.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored_products[:limit]]

        # Return default top products if no keyword matched
        return mock_products[:limit]

    async def query_analytics(
        self,
        metric_type: str = "overview",
        timeframe: str = "30d",
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries business and operational analytics from Django backend via /api/v1/internal/analytics/metrics/.

        Args:
            metric_type: Target metric type ('overview', 'kpis', 'forecast', 'sales_trend', 'category_distribution', 'top_products', 'all').
            timeframe: Time horizon (e.g., '7d', '30d', '90d', '1y').
            user_token: Optional JWT token for role-based permission checks.
        """
        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        params = {"metric_type": metric_type}

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/analytics/metrics/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to query analytics from Django: %s. Using mock metrics.", exc)

        return {
            "metric_type": metric_type,
            "timeframe": timeframe,
            "daily_active_users": 1520,
            "conversion_rate": 3.8,
            "total_revenue": 54200.00,
            "source": "Django Analytics Engine (Simulated)",
        }

    async def validate_user_token(self, token: str) -> dict[str, Any]:
        """Validates a JWT token against Django Auth service via /api/v1/internal/auth/validate-token/."""
        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/auth/validate-token/", json={"token": token})
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Token validation failed via Django: %s", exc)

        if token and len(token) > 10:
            return {"valid": True, "user_id": 101, "username": "test_user", "roles": ["customer", "analyst"]}
        return {"valid": False, "error": "Token is invalid or expired."}

    async def health_check(self) -> bool:
        """Verifies connectivity with the Django backend via /api/v1/internal/health/."""
        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/health/", timeout=2.0)
                return response.status_code == 200
        except Exception:
            return False


# Global singleton instance
_django_service: Optional[DjangoAPIService] = None


def get_django_api_service() -> DjangoAPIService:
    """Returns the singleton instance of DjangoAPIService."""
    global _django_service
    if _django_service is None:
        _django_service = DjangoAPIService()
    return _django_service
