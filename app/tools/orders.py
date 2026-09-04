"""Mock external order-status service.

Stands in for the live order system. It knows nothing about the LLM, the
knowledge base, or the agent — it takes an order ID and reports what it has,
exactly as a real service integration would.

This is the *dynamic* half of the assistant's knowledge. Policy lives in the
knowledge base; what is happening to one particular parcel lives here.

`FAIL-001` is a deliberate trapdoor: looking it up raises `OrderServiceError`,
so failure handling can be exercised without breaking anything real.
"""

from app.schemas.orders import Order, OrderLookupResult
from app.schemas.orders import LookupOutcome as Outcome
from app.tools.errors import OrderServiceError

FAILING_ORDER_ID = "FAIL-001"

# Keyed by uppercase order ID. In a real integration this is an HTTP call.
_ORDERS: dict[str, dict] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "status": "Shipped",
        "carrier": "FastPost",
        "tracking_number": "FP884213099US",
        "estimated_delivery": "2026-09-08",
        "shipped_at": "2026-09-03",
        "placed_at": "2026-09-02",
        "service_level": "Standard",
        "items": 2,
        "order_total": "$84.50",
        "destination": "Austin, TX",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "status": "Delivered",
        "carrier": "FastPost",
        "tracking_number": "FP884210441US",
        "estimated_delivery": "2026-08-27",
        "shipped_at": "2026-08-22",
        "delivered_at": "2026-08-26",
        "placed_at": "2026-08-21",
        "service_level": "Express",
        "items": 1,
        "order_total": "$42.00",
        "destination": "Portland, OR",
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "status": "Processing",
        "carrier": None,
        "tracking_number": None,
        "estimated_delivery": "2026-09-11",
        "placed_at": "2026-09-04",
        "service_level": "Standard",
        "items": 4,
        "order_total": "$129.95",
        "destination": "Chicago, IL",
        "note": "Not yet shipped, so no carrier or tracking number has been assigned.",
    },
    "ORD-1004": {
        "order_id": "ORD-1004",
        "status": "On Hold",
        "carrier": None,
        "tracking_number": None,
        "estimated_delivery": None,
        "placed_at": "2026-08-30",
        "service_level": "Overnight",
        "items": 3,
        "order_total": "$310.20",
        "destination": "Miami, FL",
        "note": "Held because the delivery address failed validation.",
    },
}


def normalise_order_id(order_id: str) -> str:
    """Put an order reference into the form the service stores."""
    return order_id.strip().upper()


def get_order_status(order_id: str | None) -> OrderLookupResult:
    """Look up one order.

    Returns a structured result for the three ordinary outcomes:

    - a valid, known ID returns the order,
    - a missing or empty ID returns `INVALID_REQUEST`,
    - an unknown ID returns `NOT_FOUND`.

    An unknown ID is never guessed at or approximated — if the service does
    not have that order, it says so.

    Raises `OrderServiceError` when the service itself fails, which is what
    `FAIL-001` simulates. That is raised rather than returned because it is a
    fault in the lookup, not an answer about an order.
    """
    if order_id is None or not order_id.strip():
        return OrderLookupResult(
            outcome=Outcome.INVALID_REQUEST,
            order_id=None,
            message="No order ID was supplied. An order reference is required, for example ORD-1001.",
        )

    reference = normalise_order_id(order_id)

    if reference == FAILING_ORDER_ID:
        raise OrderServiceError(
            reference, "Order service returned an unexpected error"
        )

    record = _ORDERS.get(reference)
    if record is None:
        return OrderLookupResult(
            outcome=Outcome.NOT_FOUND,
            order_id=reference,
            message=f"No order found with reference {reference}.",
        )

    return OrderLookupResult(
        outcome=Outcome.FOUND,
        order_id=reference,
        order=Order(**record),
        message=f"Order {reference} found.",
    )


def lookup_order(order_id: str | None) -> OrderLookupResult:
    """Look up an order without raising.

    The same call as `get_order_status`, but a service failure comes back as a
    `SERVICE_ERROR` result instead of an exception. The orchestrator will use
    this: one lookup failing should not take down the whole answer, but it must
    still be visible rather than silently treated as "no order".
    """
    try:
        return get_order_status(order_id)
    except OrderServiceError as error:
        return OrderLookupResult(
            outcome=Outcome.SERVICE_ERROR,
            order_id=error.order_id,
            message=f"The order service could not be reached: {error.detail}.",
        )


def known_order_ids() -> list[str]:
    """Every order reference the mock service knows. Useful in tests and demos."""
    return sorted(_ORDERS)
