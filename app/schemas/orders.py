"""Models for the external order-status service."""

from enum import Enum

from pydantic import BaseModel, Field


class LookupOutcome(str, Enum):
    """What happened when an order was looked up.

    Kept separate from the order's own status: `FOUND` says the service
    answered, the order's `status` field says what it answered.
    """

    FOUND = "found"
    NOT_FOUND = "not_found"
    INVALID_REQUEST = "invalid_request"
    SERVICE_ERROR = "service_error"


class Order(BaseModel):
    """One order as the external service reports it."""

    order_id: str
    status: str = Field(description="Order status, e.g. 'Shipped'")
    carrier: str | None = Field(default=None, description="Carrier, when shipped")
    tracking_number: str | None = None
    estimated_delivery: str | None = Field(
        default=None, description="Estimated delivery date, ISO 8601"
    )
    shipped_at: str | None = None
    delivered_at: str | None = None
    placed_at: str | None = None
    service_level: str | None = Field(default=None, description="e.g. 'Standard'")
    items: int | None = Field(default=None, description="Number of items")
    order_total: str | None = None
    destination: str | None = None
    note: str | None = Field(default=None, description="Anything unusual about the order")


class OrderLookupResult(BaseModel):
    """The structured result of one lookup.

    Always carries an outcome. `order` is populated only when the outcome is
    `FOUND`, so a caller can never mistake a failure for an empty order.
    """

    outcome: LookupOutcome
    order_id: str | None = None
    order: Order | None = None
    message: str = Field(description="Plain-language explanation of the outcome")

    @property
    def found(self) -> bool:
        return self.outcome is LookupOutcome.FOUND
