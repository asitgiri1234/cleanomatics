"""Errors raised by external tools.

Separate from the LLM errors so the orchestrator can tell "the order service
broke" apart from "the model broke" and react differently to each.
"""


class ToolError(Exception):
    """Base class for every external-tool failure."""


class OrderServiceError(ToolError):
    """The order service was reachable but failed to answer.

    Stands in for the real thing: a 500 from the upstream API, a dropped
    connection, a timeout. It is raised, not returned, because it says nothing
    about the order — only that the question could not be asked.
    """

    def __init__(self, order_id: str, detail: str = "Order service is unavailable"):
        self.order_id = order_id
        self.detail = detail
        super().__init__(f"{detail} (order_id={order_id})")
