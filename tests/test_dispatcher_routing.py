"""End-to-end coverage for `AgentDispatcher.classify_intent` (fix/agent-isolation-and-routing).

Background: `diagnostico-plan-agentes-multi-agente.md` documents that `classify_intent`
had zero test coverage before this branch (Subagente 4, "Estado actual verificado").
Subagente 2 replaced naive substring matching with word-boundary matching plus weighted
scoring (`dispatcher.py:70-179`) and expanded `ANALYTICS_KEYWORDS` with the EN/ES terms
that were previously missing. This module is the coverage that closes that gap: the six
Analytics test cases and five Ecommerce test cases specified verbatim in the diagnostic's
"Subagente 4 — Testing & Validation" section, plus the two concrete cross-routing
regressions the diagnostic verified by hand (causa raíz A3: "recurso" false-matching
"curso", and "revenue de mis productos" losing to ecommerce-first precedence).

This file does not touch production code. Where a parametrized case from the diagnostic
does NOT route the way the diagnostic assumed, it is captured as an `xfail` with a
precise citation rather than silently adjusted to match current behavior -- see
`TestClassifyIntentEcommerceRouting.test_shipping_only_phrasing_currently_misroutes`
below and the "bug real" note in this session's final report.
"""
from typing import Optional
import pytest

from app.agents.dispatcher import AgentDispatcher, get_strong_tool_hint


@pytest.fixture(scope="module")
def dispatcher() -> AgentDispatcher:
    """A dispatcher with no registered agents -- `classify_intent` is a pure function
    of the message and needs no agent instances."""
    return AgentDispatcher(auto_register=False)


# ==============================================================================
# Analytics routing -- the 6 cases from the diagnostic's Subagente 4 section
# ==============================================================================

class TestClassifyIntentAnalyticsRouting:
    """Protects EN/ES analytics phrasing from falling through to ecommerce/portfolio.

    Each of these previously fell through to the "portfolio" default (causa raíz A2)
    because `ANALYTICS_KEYWORDS` had no matching term in either language.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "What are the top 5 best-selling products?",
            "top 5 productos con más ventas",
            "margin by category",
            "márgenes por categoría",
            "customer RFM",
            "customer revenue segmentation",
        ],
    )
    def test_classify_intent_routes_to_analytics(self, dispatcher: AgentDispatcher, message: str) -> None:
        assert dispatcher.classify_intent(message) == "analytics"

    def test_revenue_de_mis_productos_prefers_analytics_over_generic_catalog_word(
        self, dispatcher: AgentDispatcher,
    ) -> None:
        """REGRESSION (causa raíz A3): a message with BOTH an analytics signal
        ("revenue") and a generic catalog word ("productos") must not lose to
        ecommerce-first precedence. Weighted scoring (`dispatcher.py:170-176`) makes
        analytics win any tie, not just an outright majority.
        """
        assert dispatcher.classify_intent("revenue de mis productos") == "analytics"

    def test_top_n_productos_mas_vendidos_routes_to_analytics(self, dispatcher: AgentDispatcher) -> None:
        """The canonical Spanish "best sellers" ranking phrasing from causa raíz D/A2."""
        assert dispatcher.classify_intent("top 5 productos más vendidos") == "analytics"


# ==============================================================================
# Ecommerce routing -- the 5 cases from the diagnostic's Subagente 4 section
# ==============================================================================

class TestClassifyIntentEcommerceRouting:
    """Protects catalog/pricing/stock phrasing from being pulled toward analytics by
    the Subagente 2 keyword expansion, and pins the analytics keyword growth against
    stealing traffic that belongs to the public storefront agent.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "Productos de marca X",
            "Stock de producto Y",
            "What's the price of product Z?",
        ],
    )
    def test_classify_intent_routes_to_ecommerce(self, dispatcher: AgentDispatcher, message: str) -> None:
        assert dispatcher.classify_intent(message) == "ecommerce"

    @pytest.mark.parametrize(
        "message",
        [
            "Dónde envían?",
            "Where do you ship?",
        ],
    )
    @pytest.mark.xfail(
        reason=(
            "BUG (found by Subagente 4, not fixed here per task scope -- test-only "
            "changes): ECOMMERCE_KEYWORDS in app/agents/dispatcher.py:19-23 has no "
            "shipping/delivery term in either language (no 'envío'/'envíos'/'shipping'/"
            "'ship'/'delivery'/'entrega'). A pure shipping question with no price or "
            "stock word in it scores 0-0 and falls through to the 'portfolio' default "
            "instead of 'ecommerce'. This is exactly the 'envíos' case the diagnostic's "
            "Subagente 4 section listed as an example that 'debe rutear correctamente'; "
            "it currently does not. 'cuánto cuesta el envío' still routes correctly "
            "because it also contains the 'cuánto cuesta' keyword -- the gap is "
            "shipping-only phrasing."
        ),
        strict=True,
    )
    def test_shipping_only_phrasing_currently_misroutes(self, dispatcher: AgentDispatcher, message: str) -> None:
        assert dispatcher.classify_intent(message) == "ecommerce"

    def test_shipping_question_with_a_price_word_still_routes_to_ecommerce(
        self, dispatcher: AgentDispatcher,
    ) -> None:
        """Confirms the gap above is specific to shipping-ONLY phrasing: a shipping
        question that also contains an existing ecommerce keyword ("cuánto cuesta")
        already routes correctly today.
        """
        assert dispatcher.classify_intent("cuánto cuesta el envío?") == "ecommerce"


# ==============================================================================
# Cross-regression: neither keyword set should ever hijack the other's clean cases
# ==============================================================================

class TestClassifyIntentCrossRoutingRegressions:
    """Regressions verified by hand in the diagnostic, previously untested."""

    def test_recurso_does_not_false_match_the_curso_keyword(self, dispatcher: AgentDispatcher) -> None:
        """REGRESSION (causa raíz A3): naive substring matching used to match "curso"
        inside "re-curso", misrouting an unrelated HR message to ecommerce. Word-boundary
        matching (`_matches_any`, `dispatcher.py:70-85`) closes this: "curso" does not
        match inside "recurso" because there is no word boundary between "re" and
        "curso". With zero hits in either keyword set, the message defaults to
        "portfolio", not "ecommerce".
        """
        assert dispatcher.classify_intent("necesito ayuda con un recurso humano") != "ecommerce"

    @pytest.mark.parametrize(
        "message",
        [
            "Hola, ¿cómo estás?",
            "Contame sobre tu experiencia laboral",
            "Tell me about your background",
        ],
    )
    def test_generic_greetings_and_profile_questions_default_to_portfolio(
        self, dispatcher: AgentDispatcher, message: str,
    ) -> None:
        """Protects the zero-hit default from being nudged by either keyword expansion."""
        assert dispatcher.classify_intent(message) == "portfolio"


# ==============================================================================
# get_strong_tool_hint -- reinforcement signal introduced alongside classify_intent
# ==============================================================================

class TestGetStrongToolHint:
    """Protects the STRONG_TOOL_HINTS reinforcement hook (dispatcher.py:62-109).

    This never gates or replaces Gemini's own function-calling; it is a fast,
    pre-model guess consumed optionally by AnalyticsAgent. Covered here because it
    ships in the same commit as the routing fix and had no direct test.
    """

    @pytest.mark.parametrize(
        "message,expected_tool",
        [
            ("What are the top 5 best-selling products?", "get_product_profitability"),
            ("top 5 productos más vendidos", "get_product_profitability"),
            ("cuáles son los productos mas vendidos", "get_product_profitability"),
            ("top 10 productos", "get_product_profitability"),
        ],
    )
    def test_strong_hints_match_expected_tool(self, message: str, expected_tool: str) -> None:
        assert get_strong_tool_hint(message) == expected_tool

    def test_no_hint_for_unrelated_message(self) -> None:
        assert get_strong_tool_hint("Dónde envían?") is None

    def test_no_hint_for_generic_analytics_message_without_a_strong_pattern(self) -> None:
        """A generic analytics question ("margin by category") has no STRONG_TOOL_HINTS
        pattern -- the hint hook is intentionally narrow, not a catch-all for the
        analytics intent as a whole.
        """
        assert get_strong_tool_hint("margin by category") is None
