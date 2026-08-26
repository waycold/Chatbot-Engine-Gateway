"""HTTP client service for internal communication with the Django monolith."""
import hashlib
import json
import logging
import re
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import httpx
from app.core.config import settings

logger = logging.getLogger("ai_gateway.django_api")

# ==============================================================================
# Shared catalog fixture
# ==============================================================================
# Every development fallback in this module reads its products from
# `data/catalog_fixture.json`, the single source of truth shared with the Django
# team. Nothing here may hand-invent a product any more, and above all nothing may
# derive a `slug`: the chat widget builds links as `/product/<slug>/`, so a guessed
# slug renders a 404 for the customer. The fixture carries the real slug.
CATALOG_FIXTURE_PATH = "data/catalog_fixture.json"

# Last-resort catalog used ONLY when the fixture file is missing or malformed. The
# service must never fail to start because a data file was moved or truncated, so we
# degrade to a tiny hardcoded list and log a warning instead of raising.
_FALLBACK_CATALOG_ITEMS: list[dict[str, Any]] = [
    {
        "id": 1,
        "title": "Servicio Cloud AI",
        "slug": "servicio-cloud-ai",
        "price": 49.99,
        "currency": "USD",
        "stock": 10,
        "brand": "Cloud Ops Studio",
        "category": "Servicios",
        "description": "Despliegue y configuración de microservicios de IA escalables en la nube.",
    },
    {
        "id": 3,
        "title": "Curso Avanzado de FastAPI & Microservicios",
        "slug": "curso-avanzado-de-fastapi-microservicios",
        "price": 49.99,
        "currency": "USD",
        "stock": 50,
        "brand": "Academy Pro",
        "category": "Cursos",
        "description": "Microservicios de alto rendimiento con FastAPI, Pydantic v2 y Docker.",
    },
    {
        "id": 6,
        "title": "Template Backend Django + FastAPI Gateway",
        "slug": "template-backend-django-fastapi-gateway",
        "price": 29.99,
        "currency": "USD",
        "stock": 100,
        "brand": "DevKit",
        "category": "Templates",
        "description": "Boilerplate de producción desacoplado con autenticación por secret y Redis pub/sub.",
    },
]

# Module-level cache: the fixture is read from disk at most once per process.
_catalog_fixture_cache: Optional[list[dict[str, Any]]] = None


def _slugify(value: str) -> str:
    """Builds a URL-safe slug: lowercase, accents stripped, punctuation removed.

    Only used as a last resort for payloads that arrive without a slug. Catalog items
    must take their slug from the fixture (or from Django), never from this function.

    Args:
        value: Arbitrary product title.

    Returns:
        A hyphenated ASCII slug (e.g. "Consultoría DevOps" -> "consultoria-devops").
    """
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_only = ascii_only.encode("ascii", "ignore").decode("ascii").lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "-", ascii_only)
    return ascii_only.strip("-")


def _resolve_fixture_path(target_path_str: str) -> Path:
    """Resolves a data file path against cwd, the repository root and /app.

    Mirrors `KnowledgeBaseService._resolve_path` so both data files behave identically
    whether the process is started from the repo root, from a subdirectory, or inside
    the Docker image where the code lives at /app.

    Args:
        target_path_str: Relative or absolute path to the data file.

    Returns:
        The first existing candidate path, or the unresolved original when none exist.
    """
    candidate = Path(target_path_str)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / target_path_str).resolve()
        if cwd_candidate.is_file():
            return cwd_candidate

        repo_root = Path(__file__).resolve().parent.parent.parent
        repo_candidate = (repo_root / target_path_str).resolve()
        if repo_candidate.is_file():
            return repo_candidate

        docker_candidate = Path("/app") / target_path_str
        if docker_candidate.is_file():
            return docker_candidate.resolve()

    return candidate


def _load_catalog_fixture() -> list[dict[str, Any]]:
    """Loads the shared catalog fixture once and caches it at module level.

    Returns:
        The raw fixture items, or `_FALLBACK_CATALOG_ITEMS` when the file is missing,
        unreadable, or does not carry a non-empty `items` list.
    """
    global _catalog_fixture_cache
    if _catalog_fixture_cache is not None:
        return _catalog_fixture_cache

    resolved_path = _resolve_fixture_path(CATALOG_FIXTURE_PATH)
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        items = document.get("items") if isinstance(document, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("fixture carries no non-empty 'items' list")
        parsed = [item for item in items if isinstance(item, dict)]
        if not parsed:
            raise ValueError("fixture 'items' contains no objects")
    except Exception as exc:
        logger.warning(
            "Catalog fixture unavailable at '%s' (resolved as '%s'): %s. "
            "Falling back to the built-in minimal catalog.",
            CATALOG_FIXTURE_PATH, resolved_path, exc,
        )
        _catalog_fixture_cache = [dict(item) for item in _FALLBACK_CATALOG_ITEMS]
        return _catalog_fixture_cache

    logger.info("Loaded catalog fixture from '%s' (%d items)", resolved_path, len(parsed))
    _catalog_fixture_cache = parsed
    return _catalog_fixture_cache


def _coerce_optional_str(value: Any) -> Optional[str]:
    """Returns a stripped string, or None when the value is absent or empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _shape_catalog_item(product: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Emits the canonical catalog item contract agreed with the architecture team.

    This is the ONLY place in the gateway where a catalog item is shaped, so every
    method that returns products — search, vector search, similarity, verification,
    facets, ingestion — emits exactly the same fields. The canonical fields are
    `id`, `title`, `slug`, `price`, `stock`, `brand` and `category`; `currency`,
    `description` and `in_stock` travel alongside them because the agents still use
    them. The function is idempotent: re-shaping an already shaped item is a no-op.

    Args:
        product: A raw fixture item or a payload returned by Django.
        **extra: Score fields merged into the result (`similarity`, `match_score`,
            `semantic_score`), kept out of the canonical block on purpose.

    Returns:
        The canonical item dict.
    """
    title = _coerce_optional_str(product.get("title")) or _coerce_optional_str(product.get("name")) or ""

    # The slug is authoritative data, never a guess. Deriving it is only acceptable
    # for a legacy payload that carries no slug at all, and it is logged as such.
    slug = _coerce_optional_str(product.get("slug"))
    if slug is None:
        slug = _slugify(title)
        logger.debug("Catalog payload for '%s' carried no slug; derived '%s'.", title, slug)

    try:
        item_id: Optional[int] = int(product.get("id"))
    except (TypeError, ValueError):
        item_id = None

    try:
        price = float(product.get("price") or 0.0)
    except (TypeError, ValueError):
        price = 0.0

    try:
        stock = int(product.get("stock") or 0)
    except (TypeError, ValueError):
        stock = 0

    shaped: dict[str, Any] = {
        "id": item_id,
        "title": title,
        # DEPRECATED: `name` is a mirror of `title`, emitted only so the current chat
        # widget and the older test assertions keep working while they still read
        # `name`. Django's field is `title`. Remove this key once the widget migrates.
        "name": title,
        "slug": slug,
        "price": price,
        "stock": stock,
        "brand": _coerce_optional_str(product.get("brand")),
        "category": _coerce_optional_str(product.get("category")),
        "currency": _coerce_optional_str(product.get("currency")) or "USD",
        "in_stock": stock > 0,
        "description": str(product.get("description") or ""),
    }
    shaped.update(extra)
    return shaped


# Keys owned by `_shape_catalog_item`. Anything else riding on an incoming item — the
# `similarity` / `match_score` / `semantic_score` the engines attach, plus any future
# metadata — is passed through untouched.
_CANONICAL_ITEM_KEYS = frozenset({
    "id", "title", "name", "slug", "price", "stock", "brand",
    "category", "currency", "in_stock", "description",
})


def _shape_response_items(payload: Any) -> Any:
    """Reshapes the `items` of a catalog response into the canonical item contract.

    Applied to real Django responses as well as to the local fallbacks, so callers see
    exactly one item shape no matter which side produced the payload.

    Args:
        payload: A response dict that may carry an `items` list.

    Returns:
        The same payload, with `items` reshaped in place when it was a list.
    """
    if not isinstance(payload, dict):
        return payload

    items = payload.get("items")
    if isinstance(items, list):
        payload["items"] = [
            _shape_catalog_item(
                item,
                **{key: value for key, value in item.items() if key not in _CANONICAL_ITEM_KEYS},
            )
            if isinstance(item, dict)
            else item
            for item in items
        ]
    return payload


def _filter_catalog_items(
    items: list[dict[str, Any]],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    in_stock_only: bool = True,
) -> list[dict[str, Any]]:
    """Applies price/category/brand/stock filters to catalog items in pure Python.

    The development fallbacks must honour the same filters as the real endpoint,
    otherwise any test asserting filter behaviour would be vacuous.

    Args:
        items: Items already shaped by `_shape_catalog_item`.
        min_price: Inclusive lower price bound.
        max_price: Inclusive upper price bound.
        category: Case-insensitive substring match against the item category.
        brand: Case-insensitive substring match against the item brand.
        in_stock_only: When True, drops items with zero stock.

    Returns:
        The filtered list, preserving input order.
    """
    filtered: list[dict[str, Any]] = []
    cat_needle = str(category).strip().lower() if category else None
    brand_needle = str(brand).strip().lower() if brand else None

    for item in items:
        try:
            price = float(item.get("price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0

        if min_price is not None and price < float(min_price):
            continue
        if max_price is not None and price > float(max_price):
            continue
        if cat_needle and cat_needle not in str(item.get("category", "")).lower():
            continue
        if brand_needle and brand_needle not in str(item.get("brand", "")).lower():
            continue
        if in_stock_only and not item.get("in_stock", False):
            continue
        filtered.append(item)

    return filtered


# ==============================================================================
# Auth identity normalization
# ==============================================================================
# Canonical response of POST /api/v1/internal/auth/validate-token/:
#     {"valid": true, "user": {"id": 1, "username": "admin_user",
#                              "is_staff": true, "is_superuser": false}}
# and {"valid": false, "error": "..."} when the token is invalid, expired, or the
# auth service is unavailable.
#
# Django's `auth_user` model has NO `role` column. Privilege is expressed by the two
# native booleans and by nothing else, so this gateway must never invent role strings
# ("analyst", "admin", "staff") and must never compare against a configurable list of
# them: that would be a second source of truth that silently diverges from Django.
TOKEN_INVALID_ERROR = "Token is invalid, expired, or the auth service is unavailable."


def _denied_identity(error: str) -> dict[str, Any]:
    """Builds the normalized identity of a rejected token, with no privileges.

    Args:
        error: Diagnostic message stored in the `error` key.

    Returns:
        The internal identity dict with `valid`, `is_staff` and `is_superuser` False.
    """
    return {
        "valid": False,
        "user_id": None,
        "username": None,
        "is_staff": False,
        "is_superuser": False,
        "error": error,
    }


def _normalize_auth_identity(payload: Any) -> dict[str, Any]:
    """Normalizes Django's validate-token envelope into the internal identity dict.

    Args:
        payload: The parsed JSON body returned by Django. May be anything at all;
            this function is the trust boundary.

    Returns:
        The internal identity dict. Every shape that does not match the canonical
        envelope is normalized into a denied identity: unrecognized is not privileged.
    """
    if not isinstance(payload, dict):
        return _denied_identity("Auth response body was not a JSON object.")

    # `is True`, never truthiness. A body carrying the string "false" is truthy in
    # Python and would otherwise be read as a grant. Only a real boolean counts.
    if payload.get("valid") is not True:
        return _denied_identity(_coerce_optional_str(payload.get("error")) or TOKEN_INVALID_ERROR)

    raw_user = payload.get("user")
    if not isinstance(raw_user, dict):
        # A `valid: true` with no usable user block identifies nobody. The only safe
        # reading of a malformed success is failure.
        return _denied_identity("Auth response marked the token valid but carried no user object.")

    try:
        user_id: Optional[int] = int(raw_user.get("id"))
    except (TypeError, ValueError):
        user_id = None

    return {
        "valid": True,
        "user_id": user_id,
        "username": _coerce_optional_str(raw_user.get("username")),
        "is_staff": raw_user.get("is_staff") is True,
        "is_superuser": raw_user.get("is_superuser") is True,
        "error": None,
    }


def _build_embedding_text(product: dict[str, Any]) -> str:
    """Mirrors Django's `build_embedding_text()`: title + category + description."""
    return (
        f"{product.get('title') or product.get('name') or ''}. "
        f"Categoría: {product.get('category') or ''}. "
        f"{product.get('description') or ''}"
    ).strip()


class DjangoAPIService:
    """Async HTTP client for interacting with the core transactional Django backend.

    Secured via the X-Internal-Secret header. Provides methods for portfolio data,
    product catalog (RAG & Semantic Search), analytics, inventory health, margins,
    funnel metrics, reviews, customer RFM insights, and read-only Safe SQL sandbox.
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
        # Per-instance (deliberately NOT module-global) TTL cache of successful token
        # validations, keyed by the SHA-256 of the token. A fresh DjangoAPIService()
        # therefore always starts empty, which keeps tests deterministic.
        self._token_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

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

    # ==============================================================================
    # Core Portfolio & Legacy Endpoints
    # ==============================================================================

    async def get_portfolio_data(self) -> dict[str, Any]:
        """Fetches developer CV, skills, and projects data from Django monolith."""
        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/internal/portfolio/")
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to fetch portfolio data from Django: %s. Using default profile.", exc)

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

    def _catalog_items(self) -> list[dict[str, Any]]:
        """Returns every product of the shared fixture in the canonical item shape.

        Returns:
            A fresh list of canonical items; callers may mutate it safely.
        """
        return [_shape_catalog_item(product) for product in _load_catalog_fixture()]

    async def search_catalog(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Searches products in the Django e-commerce catalog via /api/v1/internal/catalog/search/ or /api/products/.

        Args:
            query: Free-text query; when empty the catalog is returned unranked.
            category: Case-insensitive category filter.
            limit: Maximum number of items to return.

        Returns:
            A list of canonical catalog items (see `_shape_catalog_item`), whether they
            came from Django or from the shared fixture.
        """
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if category:
            params["category"] = category

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
                                return [_shape_catalog_item(item) for item in items if isinstance(item, dict)]
                        elif isinstance(data, list) and data:
                            return [_shape_catalog_item(item) for item in data if isinstance(item, dict)]
            except Exception as exc:
                logger.debug("Catalog endpoint '%s' query error: %s", endpoint, exc)

        products = self._catalog_items()

        if not query or not query.strip():
            if category:
                cat_lower = category.lower()
                filtered = [p for p in products if cat_lower in str(p.get("category") or "").lower()]
                return filtered[:limit] if filtered else products[:limit]
            return products[:limit]

        query_clean = query.strip().lower()
        tokens = [t for t in re.findall(r'\b\w{2,}\b', query_clean)]

        scored_products: list[tuple[int, dict[str, Any]]] = []
        for p in products:
            score = 0
            title_lower = str(p.get("title") or "").lower()
            desc_lower = str(p.get("description") or "").lower()
            cat_lower = str(p.get("category") or "").lower()

            if query_clean in title_lower:
                score += 50
            if title_lower and title_lower in query_clean:
                score += 40

            for token in tokens:
                if token in title_lower:
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

        return products[:limit]

    # ==============================================================================
    # 8 Specialized Internal Endpoints (Ticket BE-05)
    # ==============================================================================

    # 1. Dynamic Sales Query (query_sales_analytics)
    async def query_sales_analytics(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        dimension: str = "category",
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries dynamic sales aggregations, revenue, units, and gross margins from Django.

        Endpoint: GET /api/v1/internal/analytics/query/
        """
        params: dict[str, Any] = {"dimension": dimension}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/analytics/query/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/analytics/query/: %s", exc)

        # Realistic mock fallback
        return {
            "status": "success",
            "date_from": date_from or "2026-07-01",
            "date_to": date_to or "2026-08-23",
            "dimension": dimension,
            "aggregates": {
                "total_revenue_usd": 148500.00,
                "total_units_sold": 1840,
                "total_cost_usd": 59400.00,
                "gross_profit_usd": 89100.00,
                "gross_margin_pct": 60.0,
            },
            "breakdown": [
                {"dimension_value": "Cursos & Software", "revenue_usd": 85000.00, "units": 1100, "margin_pct": 72.5},
                {"dimension_value": "Servicios Cloud", "revenue_usd": 45000.00, "units": 450, "margin_pct": 55.0},
                {"dimension_value": "Consultoría", "revenue_usd": 18500.00, "units": 290, "margin_pct": 48.2},
            ],
            "source": "Django Sales Analytics Engine (Fallback)",
        }

    # 2. Inventory Health (get_inventory_health)
    async def get_inventory_health(
        self,
        status_filter: str = "all",
        limit: int = 10,
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries critical stock, out of stock, inventory valuation and runout velocity.

        Endpoint: GET /api/v1/internal/inventory/health/
        """
        params = {"status_filter": status_filter, "limit": limit}
        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/inventory/health/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/inventory/health/: %s", exc)

        # Fallback inventory report
        return {
            "status": "success",
            "status_filter": status_filter,
            "total_products_tracked": 120,
            "critical_stock_count": 3,
            "out_of_stock_count": 1,
            "total_inventory_valuation_usd": 32540.00,
            "items": [
                {"id": 1, "name": "Servicio Cloud AI", "stock": 10, "status": "healthy", "runout_days_est": 45, "unit_cost": 20.00, "price": 49.99},
                {"id": 2, "name": "Consultoría DevOps", "stock": 2, "status": "critical", "runout_days_est": 5, "unit_cost": 50.00, "price": 120.00},
                {"id": 3, "name": "Curso Avanzado de FastAPI & Microservicios", "stock": 50, "status": "healthy", "runout_days_est": 120, "unit_cost": 10.00, "price": 49.99},
                {"id": 4, "name": "Módulo de Integración LLM", "stock": 0, "status": "out_of_stock", "runout_days_est": 0, "unit_cost": 30.00, "price": 89.00},
            ][:limit],
            "source": "Django Inventory Health Engine (Fallback)",
        }

    # 3. Margins & Profitability (get_product_profitability)
    async def get_product_profitability(
        self,
        group_by: str = "product",
        limit: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries profitability and gross margins aggregated by product, category, brand or supplier.

        Endpoint: GET /api/v1/internal/analytics/margins/
        """
        params: dict[str, Any] = {"group_by": group_by, "limit": limit}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/analytics/margins/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/analytics/margins/: %s", exc)

        return {
            "status": "success",
            "group_by": group_by,
            "overall_gross_margin_pct": 58.4,
            "ranking": [
                {"name": "Curso Avanzado de FastAPI & Microservicios", "revenue_usd": 42500.00, "cost_usd": 8500.00, "gross_profit_usd": 34000.00, "margin_pct": 80.0},
                {"name": "Módulo de Integración LLM & Agentes Autónomos", "revenue_usd": 38000.00, "cost_usd": 11400.00, "gross_profit_usd": 26600.00, "margin_pct": 70.0},
                {"name": "Servicio Cloud AI", "revenue_usd": 45000.00, "cost_usd": 20250.00, "gross_profit_usd": 24750.00, "margin_pct": 55.0},
                {"name": "Consultoría DevOps", "revenue_usd": 24000.00, "cost_usd": 12000.00, "gross_profit_usd": 12000.00, "margin_pct": 50.0},
            ][:limit],
            "source": "Django Margins Analytics Engine (Fallback)",
        }


    # 4. Funnel & Cart Metrics (get_funnel_and_cart_metrics)
    async def get_funnel_and_cart_metrics(
        self,
        timeframe: str = "30d",
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries e-commerce conversion funnel, cart abandonment rates, and coupon ROI.

        Endpoint: GET /api/v1/internal/analytics/funnel/
        """
        params = {"timeframe": timeframe}
        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/analytics/funnel/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/analytics/funnel/: %s", exc)

        return {
            "status": "success",
            "timeframe": timeframe,
            "funnel_stages": {
                "product_views": 18450,
                "cart_additions": 3620,
                "checkout_initiated": 1420,
                "completed_orders": 890,
            },
            "conversion_rate_pct": 4.82,
            "cart_abandonment_rate_pct": 60.77,
            "top_abandoned_products": [
                {"name": "Consultoría de Arquitectura de Software", "abandon_count": 142, "lost_revenue_usd": 17040.00},
                {"name": "Curso Avanzado de FastAPI", "abandon_count": 98, "lost_revenue_usd": 4899.02},
            ],
            "coupon_roi": {
                "active_coupons": 2,
                "discount_granted_usd": 1250.00,
                "revenue_generated_usd": 14200.00,
                "roi_multiplier": 11.36,
            },
            "source": "Django Funnel Analytics Engine (Fallback)",
        }

    # 5. Reviews Sentiment (get_customer_reviews_summary)
    async def get_customer_reviews_summary(
        self,
        product_id: Optional[Union[str, int]] = None,
        sentiment: str = "all",
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries customer reviews sentiment summary, 1-5 star distribution, and feedback.

        Endpoint: GET /api/v1/internal/catalog/reviews-summary/
        """
        params: dict[str, Any] = {"sentiment": sentiment}
        if product_id:
            params["product_id"] = str(product_id)

        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/catalog/reviews-summary/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/reviews-summary/: %s", exc)

        return {
            "status": "success",
            "product_id": str(product_id) if product_id else "all",
            "total_reviews": 328,
            "average_rating": 4.85,
            "rating_distribution": {"5_star": 280, "4_star": 36, "3_star": 8, "2_star": 3, "1_star": 1},
            "sentiment_summary": {"positive_pct": 96.3, "neutral_pct": 2.5, "critical_pct": 1.2},
            "highlights": [
                "Excelente nivel técnico y claridad en los cursos de FastAPI y LLMs.",
                "Soporte rápido y resolución efectiva de dudas de arquitectura.",
                "Plantillas bien estructuradas y listas para despliegue.",
            ],
            "critical_alerts": [],
            "source": "Django Customer Reviews Engine (Fallback)",
        }

    # 6. Customer Insights & RFM (get_customer_segmentation)
    async def get_customer_segmentation(
        self,
        segment: str = "all",
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries RFM customer segmentation (VIP, At-Risk, New), LTV, and regional stats.

        Endpoint: GET /api/v1/internal/customers/insights/
        """
        params = {"segment": segment}
        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/customers/insights/", params=params, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/customers/insights/: %s", exc)

        return {
            "status": "success",
            "segment_filter": segment,
            "total_customers": 2450,
            "segments": {
                "vip": {"count": 185, "avg_ltv_usd": 1250.00, "description": "Top 10% compradores recurrentes"},
                "at_risk": {"count": 94, "avg_days_inactive": 72, "description": "Inactivos > 60 días"},
                "new": {"count": 310, "avg_order_value_usd": 85.00, "description": "Registrados últimos 30 días"},
                "standard": {"count": 1861, "avg_ltv_usd": 140.00, "description": "Compradores regulares"},
            },
            "top_regions": [
                {"country": "Argentina", "customer_count": 1120, "revenue_share_pct": 45.7},
                {"country": "España", "customer_count": 480, "revenue_share_pct": 21.3},
                {"country": "México", "customer_count": 390, "revenue_share_pct": 16.2},
                {"country": "Estados Unidos / Otros", "customer_count": 460, "revenue_share_pct": 16.8},
            ],
            "source": "Django Customer Insights Engine (Fallback)",
        }

    # 7. Semantic Search (semantic_catalog_search)
    async def semantic_catalog_search(
        self,
        query: str,
        category_hint: Optional[str] = None,
        top_k: int = 4,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        in_stock_only: bool = True,
    ) -> dict[str, Any]:
        """Performs conceptual intent/semantic search on product catalog with relevance scoring.

        Endpoint: POST /api/v1/internal/catalog/semantic-search/

        Deprecated:
            New code should call `app.services.catalog_search.semantic_catalog_search_with_fallback`,
            which embeds the query with `RETRIEVAL_QUERY`, hits the pgvector endpoint and
            degrades gracefully to the lexical engine. This method remains for the legacy
            function-calling tool contract.

        Args:
            query: Natural language search query.
            category_hint: Deprecated alias for `category`; still honoured.
            top_k: Maximum number of items to return.
            min_price: Inclusive lower price bound.
            max_price: Inclusive upper price bound.
            category: Category filter (takes precedence over `category_hint`).
            brand: Brand filter.
            in_stock_only: When True, only returns items with available stock.

        Returns:
            A search response dict whose items carry a `semantic_score` float.
        """
        effective_category = category or category_hint

        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "in_stock_only": in_stock_only,
        }
        if min_price is not None:
            payload["min_price"] = min_price
        if max_price is not None:
            payload["max_price"] = max_price
        if effective_category:
            payload["category"] = effective_category
            # Kept for backwards compatibility with the pre-RAG Django contract.
            payload["category_hint"] = effective_category
        if brand:
            payload["brand"] = brand

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/semantic-search/", json=payload)
                if response.status_code == 200:
                    return _shape_response_items(response.json())
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/semantic-search/: %s", exc)

        # Match semantics against catalog
        catalog = await self.search_catalog(query=query, category=effective_category, limit=50)
        shaped = _filter_catalog_items(
            catalog,
            min_price=min_price,
            max_price=max_price,
            category=effective_category,
            brand=brand,
            in_stock_only=in_stock_only,
        )

        items = []
        for idx, prod in enumerate(shaped[:top_k]):
            score = round(0.95 - (idx * 0.06), 2)
            items.append(_shape_catalog_item(prod, semantic_score=max(score, 0.65)))

        return {
            "status": "success",
            "query": query,
            "top_k": top_k,
            "count": len(items),
            "items": items,
            "source": "Django Semantic Search Engine (Fallback)",
        }

    # 8. Safe SQL Sandbox (execute_raw_sql_sandbox)
    async def execute_raw_sql_sandbox(
        self,
        sql_query: str,
        max_rows: int = 50,
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Executes defensive read-only SQL SELECT queries with automatic safety restrictions.

        Endpoint: POST /api/v1/internal/query/raw-read/
        """
        # Defensive client-side check
        forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "revoke", "exec"]
        sql_lower = sql_query.lower()
        if any(re.search(rf"\b{kw}\b", sql_lower) for kw in forbidden_keywords):
            return {
                "status": "error",
                "error": "Safety violation: Only read-only SELECT queries are permitted in the SQL sandbox.",
                "rows_returned": 0,
                "columns": [],
                "data": [],
            }

        payload = {"sql_query": sql_query, "max_rows": min(max_rows, 50)}
        headers: dict[str, str] = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/query/raw-read/", json=payload, headers=headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/query/raw-read/: %s", exc)

        return {
            "status": "success",
            "sql_query": sql_query,
            "rows_returned": 3,
            "columns": ["id", "name", "category", "price", "stock"],
            "data": [
                [1, "Servicio Cloud AI", "Servicios", 49.99, 10],
                [2, "Consultoría DevOps", "Servicios", 120.00, 5],
                [3, "Curso Avanzado de FastAPI", "Cursos", 49.99, 50],
            ],
            "execution_time_ms": 12.4,
            "sandbox_mode": True,
            "source": "Django Safe SQL Sandbox (Fallback)",
        }

    # ==============================================================================
    # RAG / pgvector Catalog Endpoints (Fase 1-3)
    # ==============================================================================
    # NOTE: every fallback below is our development mock while the Django team finishes
    # the 4.1.2 -> 5.2 upgrade; the endpoints do not exist yet. The mocks apply their
    # filters in Python so they behave like the real engine.

    async def _mock_rag_items(
        self,
        query: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        in_stock_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Builds the filtered canonical item list shared by the RAG fallbacks."""
        catalog = await self.search_catalog(query=query, category=None, limit=50)
        return _filter_catalog_items(
            catalog,
            min_price=min_price,
            max_price=max_price,
            category=category,
            brand=brand,
            in_stock_only=in_stock_only,
        )

    async def vector_search(
        self,
        query_vector: list[float],
        query_text: str = "",
        top_k: int = 8,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        in_stock_only: bool = True,
    ) -> dict[str, Any]:
        """Runs a pgvector cosine similarity search over the catalog embeddings.

        Endpoint: POST /api/v1/internal/catalog/vector-search/

        Args:
            query_vector: The `RETRIEVAL_QUERY` embedding of the user's text.
            query_text: The original user text, echoed back for logging/telemetry.
            top_k: Maximum number of items to return.
            min_price: Inclusive lower price bound.
            max_price: Inclusive upper price bound.
            category: Category filter.
            brand: Brand filter.
            in_stock_only: When True, only returns items with available stock.

        Returns:
            A dict with the ranked `items` (each carrying a `similarity` float),
            the applied filters and the engine name.
        """
        payload: dict[str, Any] = {
            "query_vector": query_vector,
            "query_text": query_text,
            "top_k": top_k,
            "in_stock_only": in_stock_only,
        }
        if min_price is not None:
            payload["min_price"] = min_price
        if max_price is not None:
            payload["max_price"] = max_price
        if category:
            payload["category"] = category
        if brand:
            payload["brand"] = brand

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/vector-search/", json=payload)
                if response.status_code == 200:
                    return _shape_response_items(response.json())
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/vector-search/: %s", exc)

        candidates = await self._mock_rag_items(
            query=query_text or None,
            min_price=min_price,
            max_price=max_price,
            category=category,
            brand=brand,
            in_stock_only=in_stock_only,
        )

        items = [
            _shape_catalog_item(item, similarity=max(round(0.95 - (idx * 0.05), 4), 0.5))
            for idx, item in enumerate(candidates[:top_k])
        ]

        return {
            "status": "success",
            "query": query_text,
            "top_k": top_k,
            "count": len(items),
            "items": items,
            "filters_applied": {
                "min_price": min_price,
                "max_price": max_price,
                "category": category,
                "brand": brand,
                "in_stock_only": in_stock_only,
            },
            "engine": "pgvector",
            "source": "Django pgvector Vector Search (Fallback)",
        }

    async def find_similar_products(
        self,
        item_id: int,
        top_k: int = 5,
        exclude_out_of_stock: bool = True,
    ) -> dict[str, Any]:
        """Finds catalog items whose embeddings are nearest to a reference item.

        Endpoint: POST /api/v1/internal/catalog/embeddings/similar/

        Args:
            item_id: Primary key of the reference product.
            top_k: Maximum number of neighbours to return.
            exclude_out_of_stock: When True, drops neighbours with zero stock.

        Returns:
            A dict with the neighbour `items` and the `reference_item_id`.
        """
        payload: dict[str, Any] = {
            "item_id": item_id,
            "top_k": top_k,
            "exclude_out_of_stock": exclude_out_of_stock,
        }

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/embeddings/similar/", json=payload)
                if response.status_code == 200:
                    return _shape_response_items(response.json())
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/embeddings/similar/: %s", exc)

        candidates = await self._mock_rag_items(in_stock_only=exclude_out_of_stock)
        # A product is never its own recommendation.
        neighbours = [item for item in candidates if item.get("id") != item_id]

        items = [
            _shape_catalog_item(item, similarity=max(round(0.95 - (idx * 0.05), 4), 0.5))
            for idx, item in enumerate(neighbours[:top_k])
        ]

        return {
            "status": "success",
            "reference_item_id": item_id,
            "top_k": top_k,
            "count": len(items),
            "items": items,
            "engine": "pgvector",
            "source": "Django Vector Similarity Engine (Fallback)",
        }

    async def get_pending_embeddings(self, limit: int = 20) -> dict[str, Any]:
        """Pulls pending embedding tasks from the Django outbox table.

        Endpoint: GET /api/v1/internal/catalog/embeddings/pending/

        Args:
            limit: Maximum number of pending tasks to pull in this ingestion run.

        Returns:
            A dict with the `tasks` list, each carrying `task_id`, `item_id`,
            the pre-built `text` and its `content_hash`. The source products come from
            the shared fixture through `_shape_catalog_item`, so the embedded text is
            built from the canonical `title`/`category`/`description` fields.
        """
        params = {"limit": limit}

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/catalog/embeddings/pending/", params=params)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/embeddings/pending/: %s", exc)

        catalog = self._catalog_items()
        tasks: list[dict[str, Any]] = []
        for prod in catalog[: max(0, limit)]:
            text = _build_embedding_text(prod)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            tasks.append({
                "task_id": f"emb_task_{int(prod.get('id', 0) or 0):03d}",
                "item_id": prod.get("id"),
                "text": text,
                "content_hash": f"sha256:{digest}",
            })

        return {
            "status": "success",
            "count": len(tasks),
            "tasks": tasks,
            "source": "Django Embeddings Outbox (Fallback)",
        }

    async def upsert_embedding(
        self,
        item_id: int,
        task_id: str,
        vector: list[float],
        content_hash: str,
        model_name: str,
    ) -> dict[str, Any]:
        """Writes a freshly computed embedding back into the pgvector index.

        Endpoint: POST /api/v1/internal/catalog/embeddings/upsert/

        Args:
            item_id: Primary key of the embedded product.
            task_id: Outbox task identifier being completed.
            vector: The embedding values.
            content_hash: Hash of the text the vector was produced from.
            model_name: Identifier of the model that produced the vector.

        Returns:
            The upsert confirmation dict, or an error dict when the vector has the
            wrong dimensionality (in which case NO HTTP call is made).
        """
        # A wrong-dimension vector must never reach the index: pgvector would either
        # reject the row or, worse, corrupt ranking silently. Validate before sending.
        expected_dimensions = settings.EMBEDDING_DIMENSIONS
        if not vector:
            return {
                "status": "error",
                "error": "El vector de embedding está vacío; no se envió al índice.",
                "task_id": task_id,
                "item_id": item_id,
            }
        if len(vector) != expected_dimensions:
            return {
                "status": "error",
                "error": (
                    f"Dimensión de vector inválida: se recibieron {len(vector)} valores "
                    f"y se esperaban {expected_dimensions}."
                ),
                "task_id": task_id,
                "item_id": item_id,
            }

        payload: dict[str, Any] = {
            "item_id": item_id,
            "task_id": task_id,
            "vector": vector,
            "content_hash": content_hash,
            "model_name": model_name,
        }

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/embeddings/upsert/", json=payload)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/embeddings/upsert/: %s", exc)

        return {
            "status": "success",
            "task_id": task_id,
            "item_id": item_id,
            "dimensions": len(vector),
            "model_name": model_name,
            "content_hash": content_hash,
            "source": "Django Embeddings Upsert (Fallback)",
        }

    async def mark_embedding_error(self, task_id: str, error: str) -> dict[str, Any]:
        """Marks an outbox embedding task as failed so it can be retried later.

        Endpoint: POST /api/v1/internal/catalog/embeddings/mark-error/

        Args:
            task_id: Outbox task identifier that failed.
            error: Human readable error description (truncated to 500 chars).

        Returns:
            The confirmation dict for the marked task.
        """
        truncated_error = str(error or "")[:500]
        payload = {"task_id": task_id, "error": truncated_error}

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/embeddings/mark-error/", json=payload)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/embeddings/mark-error/: %s", exc)

        return {
            "status": "success",
            "task_id": task_id,
            "marked": "error",
            "error": truncated_error,
            "source": "Django Embeddings Outbox (Fallback)",
        }

    async def verify_items(
        self,
        item_ids: Optional[list[int]] = None,
        slugs: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Re-checks live stock and price for items retrieved from the vector index.

        Endpoint: POST /api/v1/internal/catalog/items/verify/

        The vector index can serve stale rows, so anything shown to a customer is
        re-validated against the transactional database first.

        Args:
            item_ids: Primary keys to verify.
            slugs: Slugs to verify (alternative to `item_ids`). Resolved against the
                real slugs stored in the shared fixture, never against a slug derived
                from a title, so a caller cannot "verify" a URL that does not exist.

        Returns:
            A dict with the verified `items` — full canonical items — and the
            `not_found` identifiers, which keep the exact value the caller passed in.
        """
        if not item_ids and not slugs:
            return {"status": "error", "error": "Debe indicar item_ids o slugs."}

        payload: dict[str, Any] = {}
        if item_ids:
            payload["item_ids"] = item_ids
        if slugs:
            payload["slugs"] = slugs

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/items/verify/", json=payload)
                if response.status_code == 200:
                    return _shape_response_items(response.json())
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/items/verify/: %s", exc)

        shaped = self._catalog_items()
        by_id = {item["id"]: item for item in shaped}
        # Keyed on the real fixture slug. `_slugify` is NOT applied to the lookup key:
        # an identifier the catalog does not actually carry must come back as
        # `not_found` rather than be silently rewritten into a neighbouring product.
        by_slug = {str(item["slug"]).strip().lower(): item for item in shaped}

        verified: list[dict[str, Any]] = []
        not_found: list[Any] = []
        seen_ids: set[Any] = set()

        for raw_id in item_ids or []:
            try:
                lookup_id: Optional[int] = int(raw_id)
            except (TypeError, ValueError):
                lookup_id = None
            item = by_id.get(lookup_id) if lookup_id is not None else None
            if item is None:
                not_found.append(raw_id)
                continue
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                verified.append(item)

        for raw_slug in slugs or []:
            item = by_slug.get(str(raw_slug).strip().lower())
            if item is None:
                not_found.append(raw_slug)
                continue
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                verified.append(item)

        return {
            "status": "success",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            # Full canonical items: the caller re-renders price, stock and the product
            # link from this payload, so trimming fields here would force it to fall
            # back on the stale vector-index copy it was trying to replace.
            "items": verified,
            "not_found": not_found,
            "source": "Django Live Stock & Price Check (Fallback)",
        }

    async def get_catalog_facets(self, facet: str = "both") -> dict[str, Any]:
        """Lists the distinct categories and/or brands available in the catalog.

        Endpoint: GET /api/v1/internal/catalog/facets/

        Args:
            facet: One of "category", "brand" or "both".

        Returns:
            A dict containing only the requested facet key(s), or an error dict. The
            lists are derived from the same shared fixture the search methods return
            items from, so a facet can never offer a filter value that yields no hits.
        """
        valid_facets = {"category", "brand", "both"}
        if facet not in valid_facets:
            return {
                "status": "error",
                "error": f"Faceta inválida: '{facet}'. Use 'category', 'brand' o 'both'.",
            }

        params = {"facet": facet}

        try:
            client = await self.get_client()
            async with client:
                response = await client.get("/api/v1/internal/catalog/facets/", params=params)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/facets/: %s", exc)

        shaped = self._catalog_items()

        result: dict[str, Any] = {
            "status": "success",
            "facet": facet,
        }
        if facet in ("category", "both"):
            result["categories"] = sorted({str(item["category"]) for item in shaped if item.get("category")})
        if facet in ("brand", "both"):
            result["brands"] = sorted({str(item["brand"]) for item in shaped if item.get("brand")})
        result["source"] = "Django Catalog Facets (Fallback)"
        return result

    async def legacy_lexical_search(
        self,
        query: str,
        top_k: int = 8,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        in_stock_only: bool = True,
    ) -> dict[str, Any]:
        """Runs the pre-existing keyword search engine (degradation path for RAG).

        Endpoint: POST /api/v1/internal/catalog/semantic-search/

        Args:
            query: Natural language / keyword query.
            top_k: Maximum number of items to return.
            min_price: Inclusive lower price bound.
            max_price: Inclusive upper price bound.
            category: Category filter.
            brand: Brand filter.
            in_stock_only: When True, only returns items with available stock.

        Returns:
            A search response shaped like `vector_search`, but with `match_score`
            instead of `similarity` and `engine="lexical"`.
        """
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "mode": "lexical",
            "in_stock_only": in_stock_only,
        }
        if min_price is not None:
            payload["min_price"] = min_price
        if max_price is not None:
            payload["max_price"] = max_price
        if category:
            payload["category"] = category
        if brand:
            payload["brand"] = brand

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/catalog/semantic-search/", json=payload)
                if response.status_code == 200:
                    return _shape_response_items(response.json())
        except Exception as exc:
            logger.warning("Failed to call /api/v1/internal/catalog/semantic-search/ (lexical): %s", exc)

        candidates = await self._mock_rag_items(
            query=query or None,
            min_price=min_price,
            max_price=max_price,
            category=category,
            brand=brand,
            in_stock_only=in_stock_only,
        )

        items = [
            _shape_catalog_item(item, match_score=max(round(0.95 - (idx * 0.05), 4), 0.5))
            for idx, item in enumerate(candidates[:top_k])
        ]

        return {
            "status": "success",
            "query": query,
            "top_k": top_k,
            "count": len(items),
            "items": items,
            "filters_applied": {
                "min_price": min_price,
                "max_price": max_price,
                "category": category,
                "brand": brand,
                "in_stock_only": in_stock_only,
            },
            "engine": "lexical",
            "source": "Django Lexical Search Engine (Fallback)",
        }

    # ==============================================================================
    # Auth & Utility Methods
    # ==============================================================================

    async def query_analytics(
        self,
        metric_type: str = "overview",
        timeframe: str = "30d",
        user_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries business and operational analytics from Django backend via /api/v1/internal/analytics/metrics/."""
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

    # --------------------------------------------------------------------------
    # Token validation cache (a PERFORMANCE optimization, never a security one)
    # --------------------------------------------------------------------------

    @staticmethod
    def _token_cache_key(token: str) -> str:
        """Returns the SHA-256 hex digest used to key a token in the cache.

        The raw token is never stored. A credential that is still live must not be
        recoverable from a heap dump, a debugger session, or an accidental log of the
        cache contents.

        Args:
            token: The raw user token.

        Returns:
            The hex digest of the token.
        """
        return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()

    def _get_cached_validation(self, token: str) -> Optional[dict[str, Any]]:
        """Returns a still-fresh cached identity for `token`, or None.

        Args:
            token: The raw user token.

        Returns:
            A copy of the cached identity, or None on a miss, an expired entry, or a
            TTL of 0 (caching disabled).
        """
        ttl = float(settings.TOKEN_VALIDATION_CACHE_TTL_SECONDS or 0.0)
        if ttl <= 0.0:
            return None

        key = self._token_cache_key(token)
        entry = self._token_cache.get(key)
        if entry is None:
            return None

        expires_at, identity = entry
        if expires_at <= time.monotonic():
            self._token_cache.pop(key, None)
            return None

        self._token_cache.move_to_end(key)
        # Hand out a copy: the agents annotate the identity they receive with their own
        # bookkeeping, and that must not mutate the shared cached entry.
        return dict(identity)

    def _store_validation(self, token: str, identity: dict[str, Any]) -> None:
        """Caches a SUCCESSFUL validation, evicting expired entries then the oldest.

        Only `valid: True` identities are stored. Caching a failure would be actively
        harmful: a `valid: False` produced by a transient Django outage would be pinned
        for the whole TTL and keep denying a legitimate staff user long after Django
        recovered. It would also buy nothing — a genuinely bad token is cheap to
        reject and is rare in the hot path, unlike the good token that the dispatcher
        and the analytics agent both validate on every single turn.

        Args:
            token: The raw user token.
            identity: The normalized identity returned by `_normalize_auth_identity`.
        """
        ttl = float(settings.TOKEN_VALIDATION_CACHE_TTL_SECONDS or 0.0)
        if ttl <= 0.0 or identity.get("valid") is not True:
            return

        now = time.monotonic()
        key = self._token_cache_key(token)
        self._token_cache[key] = (now + ttl, dict(identity))
        self._token_cache.move_to_end(key)

        max_entries = max(0, int(settings.TOKEN_VALIDATION_CACHE_MAX_ENTRIES))
        if max_entries <= 0:
            self._token_cache.clear()
            return

        # Expired entries are worthless, so they are dropped first; only after that do
        # we evict the oldest still-valid entries to respect the bound.
        for expired_key in [k for k, (expires_at, _) in self._token_cache.items() if expires_at <= now]:
            self._token_cache.pop(expired_key, None)
        while len(self._token_cache) > max_entries:
            self._token_cache.popitem(last=False)

    def clear_token_cache(self) -> None:
        """Empties the in-process token validation cache."""
        self._token_cache.clear()

    async def validate_user_token(self, token: str) -> dict[str, Any]:
        """Validates a user token against Django and normalizes the resulting identity.

        Endpoint: POST /api/v1/internal/auth/validate-token/

        Django answers `{"valid": true, "user": {"id": ..., "username": ...,
        "is_staff": ..., "is_superuser": ...}}`, or `{"valid": false, "error": "..."}`.
        `auth_user` has no `role` column: privilege is the two native booleans and
        nothing else, so no role strings are produced or consumed here.

        Successful validations are cached in-process for
        `settings.TOKEN_VALIDATION_CACHE_TTL_SECONDS`, because the dispatcher and the
        analytics agent each validate the same token within one conversation turn.

        Args:
            token: The raw user token forwarded with the chat request.

        Returns:
            The normalized internal identity, always exactly these keys regardless of
            what came back over the wire:
            `{"valid": bool, "user_id": Optional[int], "username": Optional[str],
            "is_staff": bool, "is_superuser": bool, "error": Optional[str]}`.

        Note:
            Known tech debt: reusing `is_staff` to authorize the read-only SQL console
            conflates two different permissions — access to the Django admin is not the
            same thing as permission to run SQL through a chat agent. This is acceptable
            at this project's scope, but it must be revisited before this grows beyond a
            portfolio system.
        """
        cached = self._get_cached_validation(token)
        if cached is not None:
            return cached

        try:
            client = await self.get_client()
            async with client:
                response = await client.post("/api/v1/internal/auth/validate-token/", json={"token": token})
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except Exception as exc:
                        logger.warning("Token validation response was not valid JSON: %s", exc)
                        return _denied_identity(TOKEN_INVALID_ERROR)

                    identity = _normalize_auth_identity(payload)
                    # No-op for a denied identity: only successes are cached.
                    self._store_validation(token, identity)
                    return identity

                # Django answered, so its answer is authoritative and the dev escape
                # hatch below does not apply: a reachable auth service saying "no" is
                # a real "no", not the outage that hatch exists for.
                logger.warning(
                    "Django token validation returned HTTP %s; denying the token.",
                    response.status_code,
                )
                return _denied_identity(TOKEN_INVALID_ERROR)
        except Exception as exc:
            logger.warning("Token validation failed via Django: %s", exc)

        # SECURITY: this fallback MUST fail closed.
        #
        # Every other method in this class degrades to a realistic mock payload so the team
        # can develop against a Django backend that is not deployed yet. Auth is the one
        # place where that convention is dangerous: the analytics agent's staff check, the
        # dispatcher's `_authorize_agent()`, and the per-agent tool allowlist all derive
        # privilege from the `is_staff` / `is_superuser` booleans this method returns. An
        # earlier version granted a privileged identity to any token longer than ten
        # characters whenever Django was unreachable, which made every one of those layers
        # bypassable with an arbitrary junk string. Do not reintroduce a permissive
        # fallback here: an auth service we cannot reach is an auth service we cannot
        # trust, and the only identity we may invent is an unprivileged one.
        if settings.ENVIRONMENT == "development" and settings.DEBUG and token and len(token) > 10:
            logger.warning(
                "DEV-ONLY: Django auth unreachable; issuing a NON-STAFF identity for local work. "
                "This branch is disabled outside ENVIRONMENT=development with DEBUG=true."
            )
            # Deliberately NOT cached: this identity exists because Django was
            # unreachable, so it is exactly as transient as the outage that produced it
            # and must not survive the backend coming back up.
            return {
                "valid": True,
                "user_id": 101,
                "username": "dev_user",
                "is_staff": False,       # never privileged, whatever the environment says
                "is_superuser": False,   # idem
                "error": None,
            }

        return _denied_identity(TOKEN_INVALID_ERROR)

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
