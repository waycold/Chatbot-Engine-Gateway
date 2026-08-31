"""Exceptions raised by the agent routing/authorization layer."""


class AgentAuthorizationError(Exception):
    """Raised when a privileged agent is requested without sufficient authorization.

    Replaces the previous silent downgrade to the e-commerce agent: an unauthorized
    analytics request must surface as an explicit, structured rejection rather than be
    quietly re-executed by a different agent under the caller's original intent.

    `status_code` follows HTTP semantics:
      * 401 — no credentials were presented at all (`request.user_token` is falsy).
      * 403 — credentials were presented but do not carry the required privilege
        (invalid token, non-staff account, or a validation failure — every failure
        mode here fails closed, so it is reported as "insufficient privilege" rather
        than silently retried as anonymous).
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
