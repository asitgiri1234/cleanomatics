"""The mock order-status service, covering all four required behaviours."""

import pytest

from app.schemas.orders import LookupOutcome
from app.tools.errors import OrderServiceError, ToolError
from app.tools.orders import (
    FAILING_ORDER_ID,
    get_order_status,
    known_order_ids,
    lookup_order,
    normalise_order_id,
)

SEEDED_IDS = ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004"]


# 1. Valid order IDs

@pytest.mark.parametrize("order_id", SEEDED_IDS)
def test_valid_order_returns_structured_information(order_id):
    result = get_order_status(order_id)

    assert result.outcome is LookupOutcome.FOUND
    assert result.found
    assert result.order is not None
    assert result.order.order_id == order_id
    assert result.order.status


def test_all_four_reference_orders_exist():
    assert set(SEEDED_IDS).issubset(set(known_order_ids()))


def test_shipped_order_carries_carrier_and_delivery_estimate():
    order = get_order_status("ORD-1001").order

    assert order.status == "Shipped"
    assert order.carrier
    assert order.tracking_number
    assert order.estimated_delivery


def test_unshipped_order_has_no_carrier_or_tracking():
    order = get_order_status("ORD-1003").order

    assert order.status == "Processing"
    assert order.carrier is None
    assert order.tracking_number is None


def test_delivered_order_has_a_delivery_date():
    order = get_order_status("ORD-1002").order

    assert order.status == "Delivered"
    assert order.delivered_at


def test_held_order_explains_why():
    order = get_order_status("ORD-1004").order

    assert order.status == "On Hold"
    assert order.note


def test_lookup_is_case_and_whitespace_insensitive():
    assert get_order_status("  ord-1001 ").order.order_id == "ORD-1001"
    assert normalise_order_id(" ord-1001 ") == "ORD-1001"


# 2. Missing or empty order ID

@pytest.mark.parametrize("order_id", [None, "", "   ", "\n"])
def test_missing_order_id_is_a_validation_error(order_id):
    result = get_order_status(order_id)

    assert result.outcome is LookupOutcome.INVALID_REQUEST
    assert not result.found
    assert result.order is None
    assert "order" in result.message.lower()


# 3. Unknown order ID

@pytest.mark.parametrize("order_id", ["ORD-9999", "ORD-0000", "XYZ-1", "not-an-order"])
def test_unknown_order_is_reported_not_invented(order_id):
    result = get_order_status(order_id)

    assert result.outcome is LookupOutcome.NOT_FOUND
    assert result.order is None
    assert result.order_id == order_id.upper()
    assert "no order found" in result.message.lower()


def test_unknown_order_is_never_approximated_to_a_real_one():
    """A near-miss must not resolve to a neighbouring order."""
    result = get_order_status("ORD-1005")

    assert result.outcome is LookupOutcome.NOT_FOUND
    assert result.order is None


# 4. Simulated service failure

def test_failure_id_raises_a_service_error():
    with pytest.raises(OrderServiceError) as caught:
        get_order_status(FAILING_ORDER_ID)

    assert caught.value.order_id == FAILING_ORDER_ID
    assert isinstance(caught.value, ToolError)


def test_failure_id_is_case_insensitive():
    with pytest.raises(OrderServiceError):
        get_order_status("fail-001")


def test_safe_wrapper_turns_failure_into_a_structured_result():
    result = lookup_order(FAILING_ORDER_ID)

    assert result.outcome is LookupOutcome.SERVICE_ERROR
    assert not result.found
    assert result.order is None
    assert "could not be reached" in result.message


def test_safe_wrapper_passes_normal_outcomes_through():
    assert lookup_order("ORD-1001").outcome is LookupOutcome.FOUND
    assert lookup_order("ORD-9999").outcome is LookupOutcome.NOT_FOUND
    assert lookup_order(None).outcome is LookupOutcome.INVALID_REQUEST


# Independence from the rest of the system

def test_tool_does_not_depend_on_the_llm_or_knowledge_base():
    import app.tools.orders as module

    imported = module.__doc__ + str(sorted(vars(module)))
    assert "groq" not in imported.lower()
    assert not hasattr(module, "get_llm_client")
    assert not hasattr(module, "get_retriever")
