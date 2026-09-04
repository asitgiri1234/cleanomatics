# Order Management

*ShipFlow policy documentation. This explains how orders behave on the platform
— what the statuses mean and what can be changed when.*

**The live status of a specific order is not in this documentation.** Individual
order details are looked up through the order lookup, not from these documents.

## Order references

Every order gets a reference in the form `ORD-` followed by four or more digits.
The reference appears on the confirmation email and is what support and the
order lookup use to find an order.

## Order statuses

| Status | Meaning |
|---|---|
| Pending | Received, payment not yet confirmed |
| Confirmed | Payment taken, waiting to be picked |
| Processing | Being picked and packed |
| Shipped | Label created, handed to the carrier |
| Out for Delivery | On the delivery vehicle |
| Delivered | Carrier has recorded delivery |
| On Hold | Paused — stock, payment review, or address problem |
| Cancelled | Stopped before shipping |
| Returned | Came back and was received |
| Refunded | Money returned to the customer |

Statuses move forward only. An order that has shipped cannot go back to
Processing — it goes through Returned instead.

## Editing an order

Items, quantities, and the delivery address can be changed while an order is
**Pending, Confirmed, or Processing** — that is, at any point before the
shipping label is created. Once the label exists, the order is locked.

## Cancelling an order

An order can be cancelled from the dashboard until the label is created, and
that produces a full refund with no fees. After the label is created, the order
must be returned instead.

## Orders on hold

An order goes On Hold for one of three reasons: an item is out of stock, the
payment was flagged for review, or the address failed validation. Held orders
resume automatically once resolved, and are cancelled after **10 days** on hold
with no resolution.

## Searching and exporting

Orders can be found by reference, customer email, or tracking number. Orders can
be exported to CSV, and bulk actions can be applied to a filtered selection.
Order records are retained for **7 years**.

## Related

Delivery times and tracking behaviour: see shipping and delivery. Refunds on
cancelled orders: see the refund policy.
