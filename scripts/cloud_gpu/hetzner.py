"""Rent and cancel a Hetzner dedicated GPU server through the Robot API.

Hetzner Cloud has no GPU servers; the GPU line (GEX45, GEX63, GEX131) is
dedicated hardware ordered through the Robot webservice
(https://robot.hetzner.com/doc/webservice/en.html): hourly billing up to a
monthly cap, a one-time setup fee per order, and immediate cancellation.
Ordering through the API must be switched on in Robot first
(Administration, Settings, Web Service Settings, Ordering). Whether the GEX
products can be ordered through the API was not verified (it needs an
account); `plan` checks it with a test order. See docs/cloud-gpu.md.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu.shell import log

API = os.environ.get("HETZNER_ROBOT_URL", "https://robot-ws.your-server.de")
DEFAULT_DIST = "Ubuntu 24.04 LTS base"


class RobotError(RuntimeError):
    """The Robot webservice answered with an error."""

    def __init__(self, status: int, message: str) -> None:
        """Keep the HTTP status for callers that accept 404."""
        self.status = status
        super().__init__(message)


def _credentials() -> str:
    user = os.environ.get("HETZNER_ROBOT_USER", "")
    password = os.environ.get("HETZNER_ROBOT_PASSWORD", "")
    if not user or not password:
        message = (
            "set HETZNER_ROBOT_USER and HETZNER_ROBOT_PASSWORD (Robot webservice user)"
        )
        raise RobotError(0, message)
    return base64.b64encode(f"{user}:{password}".encode()).decode("ascii")


def request(
    method: str,
    path: str,
    form: list[tuple[str, str]] | None = None,
    *,
    api: str | None = None,
) -> Any:
    """Call the Robot API with a form body and return the decoded JSON."""
    url = f"{api or API}{path}"
    if not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
        message = f"refusing a non-HTTPS Robot URL: {url}"
        raise RobotError(0, message)
    data = urllib.parse.urlencode(form).encode("ascii") if form is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Basic {_credentials()}",
            "Accept": "application/json",
            "User-Agent": "medbench-cloud-gpu",
            **({"Content-Type": "application/x-www-form-urlencoded"} if data else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        message = f"Robot API {method} {path}: {error.code} {detail}"
        raise RobotError(error.code, message) from error
    return json.loads(raw) if raw else None


def product(product_id: str) -> dict[str, Any]:
    """Return an orderable product (price, setup fee, locations, images)."""
    answer: dict[str, Any] = request(
        "GET", f"/order/server/product/{urllib.parse.quote(product_id)}"
    )["product"]
    return answer


def gpu_products() -> list[dict[str, Any]]:
    """Return the orderable products whose id or name mentions GEX."""
    answer = request("GET", "/order/server/product")
    products = [item["product"] for item in answer]
    return [
        item
        for item in products
        if "GEX" in str(item.get("id", "")).upper()
        or "GEX" in str(item.get("name", "")).upper()
    ]


def pick_dist(item: dict[str, Any]) -> str:
    """Return the Ubuntu 24.04 image name the product offers."""
    for dist in item.get("dist", []):
        if "Ubuntu 24.04" in dist:
            return str(dist)
    return DEFAULT_DIST


def upload_key(name: str, public_key: str) -> str:
    """Store the run's public SSH key in Robot and return its fingerprint."""
    answer = request("POST", "/key", [("name", name), ("data", public_key)])
    fingerprint = str(answer["key"]["fingerprint"])
    log(f"uploaded SSH key {name} ({fingerprint})")
    return fingerprint


def delete_key(fingerprint: str) -> None:
    """Remove the run's SSH key; ignore one that is already gone."""
    try:
        request("DELETE", f"/key/{urllib.parse.quote(fingerprint)}")
    except RobotError as error:
        if error.status != 404:
            raise


def order(
    config: cfg.RunConfig,
    fingerprint: str,
    *,
    test: bool,
) -> dict[str, Any]:
    """Order the server; with ``test`` Hetzner only validates the order."""
    item = product(config.instance_type)
    location = config.hetzner_location or config.region
    form = [
        ("product_id", config.instance_type),
        ("authorized_key[]", fingerprint),
        ("dist", pick_dist(item)),
        ("lang", "en"),
        ("location", location),
        ("addon[]", "primary_ipv4"),
        ("test", "true" if test else "false"),
    ]
    log(
        f"{'test ' if test else ''}order of {config.instance_type} in {location}: "
        f"{item.get('price_setup', {}) or 'setup fee unknown'}"
    )
    transaction: dict[str, Any] = request("POST", "/order/server/transaction", form)[
        "transaction"
    ]
    return transaction


def order_status(transaction_id: str) -> dict[str, Any]:
    """Return an order (status, and server_number once delivered)."""
    answer: dict[str, Any] = request(
        "GET", f"/order/server/transaction/{urllib.parse.quote(transaction_id)}"
    )["transaction"]
    return answer


def wait_ready(transaction_id: str, *, timeout_minutes: int) -> dict[str, Any]:
    """Wait until Hetzner has handed over the server; return the transaction."""
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        transaction = order_status(transaction_id)
        status = transaction.get("status")
        if status == "ready" and transaction.get("server_number"):
            log(f"server {transaction['server_number']} is ready")
            return transaction
        if status == "cancelled":
            message = f"Hetzner cancelled order {transaction_id}"
            raise RobotError(0, message)
        if time.monotonic() > deadline:
            message = (
                f"order {transaction_id} not ready after {timeout_minutes} min; "
                "`down` and the reaper cancel it once it is delivered"
            )
            raise TimeoutError(message)
        log(f"order {transaction_id}: {status}")
        time.sleep(60)


def server_name(config: cfg.RunConfig) -> str:
    """Name that lets the reaper find the server and its deadline."""
    return f"{cfg.MANAGED_BY}-{config.run_id}-{config.expires_at}"


def rename(server_number: int, name: str) -> None:
    """Set the server name shown in Robot."""
    request("POST", f"/server/{server_number}", [("server_name", name)])


def cancel(server_number: int) -> None:
    """Cancel the server now: billing stops and Hetzner wipes it."""
    try:
        request(
            "POST",
            f"/server/{server_number}/cancellation",
            [("cancellation_date", "now")],
        )
    except RobotError as error:
        if error.status == 409:
            log(f"server {server_number} is already cancelled")
            return
        raise
    log(f"server {server_number} cancelled")


def servers() -> list[dict[str, Any]]:
    """Return every dedicated server of the account."""
    answer = request("GET", "/server")
    return [item["server"] for item in answer or []]


def transactions() -> list[dict[str, Any]]:
    """Return the server orders of the last 30 days."""
    answer = request("GET", "/order/server/transaction")
    return [item["transaction"] for item in answer or []]


def ordered_by_us(transaction: dict[str, Any]) -> bool:
    """Return True if the order carried a run's key (named medbench-<run id>)."""
    for item in transaction.get("authorized_key") or []:
        name = str((item.get("key") or {}).get("name", ""))
        if name.startswith("medbench-") and cfg.RUN_ID_PATTERN.match(name[9:]):
            return True
    return False


def order_deadline(transaction: dict[str, Any]) -> float | None:
    """Return the latest expires-at a run could have given the order's server."""
    try:
        ordered = datetime.fromisoformat(str(transaction.get("date", "")))
    except ValueError:
        return None
    if ordered.tzinfo is None:
        ordered = ordered.replace(tzinfo=UTC)
    return ordered.timestamp() + cfg.MAX_TTL_MINUTES * 60


def expired_servers(now: float | None = None) -> list[dict[str, Any]]:
    """Return our servers whose deadline has passed.

    A server that `up` named carries its deadline in the name. A server that
    Hetzner delivered after `up` stopped waiting has no name; it is found by
    the run's SSH key in the order and expires a day after the order.
    """
    moment = now if now is not None else time.time()
    found = []
    prefix = f"{cfg.MANAGED_BY}-"
    late: dict[int, float] = {}
    for transaction in transactions():
        deadline = order_deadline(transaction)
        number = transaction.get("server_number")
        if number and deadline is not None and ordered_by_us(transaction):
            late[int(number)] = deadline
    for server in servers():
        if server.get("cancelled"):
            continue
        name = str(server.get("server_name") or "")
        if name.startswith(prefix):
            expires = name.rsplit("-", 1)[-1]
            if expires.isdigit() and int(expires) < moment:
                found.append(server)
        elif not name and late.get(int(server["server_number"]), moment) < moment:
            found.append(server)
    return found
