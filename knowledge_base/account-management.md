# Account Management

*ShipFlow policy documentation. This covers accounts, users, and access. It does
not contain details of any individual account.*

## Roles

| Role | Can do |
|---|---|
| Owner | Everything, including plan changes, ownership transfer, deletion |
| Admin | Manage orders, returns, users, and integrations |
| Member | Manage orders and returns |
| Viewer | Read-only access to orders and reports |

There is exactly **one Owner** per account. Only the Owner can change the plan,
cancel the subscription, transfer ownership, or delete the account.

## Adding users

Users are invited by email from **Settings → Team**. An invitation expires
after **7 days** and can be resent. The number of users allowed depends on the
plan: 1 on Starter, 5 on Growth, 20 on Scale. Removing a user frees the seat
immediately.

## Passwords and sign-in

- Passwords must be at least 12 characters.
- A password reset link is valid for **60 minutes** and can only be used once.
- Five failed sign-in attempts lock the account for **15 minutes**.

## Two-factor authentication

2FA is available on every plan and uses an authenticator app (TOTP). SMS codes
are not supported. Ten single-use recovery codes are issued when 2FA is
switched on. An Owner or Admin can reset 2FA for another user; if the Owner is
locked out and has no recovery codes, identity verification with support is
required.

## Changing the account email

Changing an email address requires confirmation from **both** the old and the
new address. The change is not applied until both links are clicked.

## Transferring ownership

The current Owner nominates an existing Admin as the new Owner. The nominee has
**7 days** to accept. Once accepted, the previous Owner becomes an Admin.

## Deleting an account

Account deletion is separate from cancelling a plan. It is Owner-only, requires
the subscription to be cancelled first, and permanently erases all data after a
**30-day** grace period during which it can still be undone.

Simply cancelling the plan without deleting follows the gentler timeline in the
subscription document: read-only for 60 days, deleted at 90 days.

## Related

Cancelling a plan: see subscription plans. Billing contact: see billing and
payments.
