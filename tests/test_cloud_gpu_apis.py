"""Tests for the GitHub and Hetzner API clients and the reaper (issue #23).

The APIs are a small HTTP server on 127.0.0.1 with recorded answers; the
cloud command line tools (aws, az, gcloud) are fake programs that answer
from a table and log every call.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
import threading
import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu import github, hetzner, reaper

NOW = 1_800_000_000
PAST, FUTURE = str(NOW - 60), str(NOW + 3600)


class FakeApi:
    """Answers keyed by "METHOD /path?query"; records every request."""

    def __init__(self) -> None:
        self.answers: dict[str, tuple[int, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.url = ""

    def answer(self, key: str, body: Any, status: int = 200) -> None:
        self.answers[key] = (status, body)


@pytest.fixture
def api() -> Iterator[FakeApi]:
    fake = FakeApi()

    class Handler(BaseHTTPRequestHandler):
        def handle_one(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8")
            key = f"{self.command} {self.path}"
            fake.requests.append(
                {"key": key, "body": body, "headers": dict(self.headers)}
            )
            status, answer = fake.answers.get(key, (404, {"message": "Not Found"}))
            data = b"" if answer is None else json.dumps(answer).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_DELETE = handle_one  # noqa: N815

        def log_message(self, *args: object) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def github_api(api: FakeApi, monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    monkeypatch.setattr(github, "API", api.url)
    for name in github.TOKEN_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLOUD_GPU_RUNNER_TOKEN", "test-token")
    return api


@pytest.fixture
def robot_api(api: FakeApi, monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    monkeypatch.setattr(hetzner, "API", api.url)
    monkeypatch.setenv("HETZNER_ROBOT_USER", "robot-user")
    monkeypatch.setenv("HETZNER_ROBOT_PASSWORD", "robot-secret")
    return api


# GitHub


def test_jit_runner_registration(github_api: FakeApi) -> None:
    github_api.answer(
        "POST /repos/o/r/actions/runners/generate-jitconfig",
        {"runner": {"id": 23}, "encoded_jit_config": "SklUCg=="},
        201,
    )
    labels = ["self-hosted", "linux", "x64", "medbench-x"]
    assert github.generate_jit_config("o/r", "medbench-x", labels) == (23, "SklUCg==")
    sent = github_api.requests[0]
    assert sent["headers"]["Authorization"] == "Bearer test-token"
    assert json.loads(sent["body"]) == {
        "name": "medbench-x",
        "runner_group_id": 1,
        "labels": labels,
        "work_folder": "_work",
    }


def test_runner_status_wait_and_delete(github_api: FakeApi) -> None:
    assert github.runner_status("o/r", 5) is None  # 404: gone
    github.wait_runner("o/r", 5)  # gone at once: the job is done
    github.delete_runner("o/r", 5)  # 404 is fine
    github_api.answer("DELETE /repos/o/r/actions/runners/6", {"message": "x"}, 500)
    with pytest.raises(github.GitHubError, match="500"):
        github.delete_runner("o/r", 6)


def test_runner_that_never_comes_online_times_out(github_api: FakeApi) -> None:
    github_api.answer(
        "GET /repos/o/r/actions/runners/7", {"status": "offline", "busy": False}
    )
    with pytest.raises(TimeoutError, match="did not come online"):
        github.wait_runner("o/r", 7, online_timeout=0, interval=0)


def test_stale_runners_pages_and_filters(github_api: FakeApi) -> None:
    first = [{"id": n, "name": f"other-{n}", "status": "offline"} for n in range(98)]
    first += [
        {"id": 100, "name": "medbench-a", "status": "offline"},
        {"id": 101, "name": "medbench-b", "status": "online"},
    ]
    second = [{"id": 200, "name": "medbench-c", "status": "offline"}]
    path = "GET /repos/o/r/actions/runners?per_page=100&page="
    github_api.answer(f"{path}1", {"runners": first})
    github_api.answer(f"{path}2", {"runners": second})
    found = github.stale_runners("o/r")
    assert [r["id"] for r in found] == [100, 200]


def test_github_client_needs_a_token_and_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in github.TOKEN_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(github.GitHubError, match="CLOUD_GPU_RUNNER_TOKEN"):
        github.token()
    monkeypatch.setenv("GH_TOKEN", "t")
    with pytest.raises(github.GitHubError, match="non-HTTPS"):
        github.request("GET", "/x", api="http://example.com")


# Hetzner Robot


def form(request: dict[str, Any]) -> list[tuple[str, str]]:
    return urllib.parse.parse_qsl(request["body"])


def make_config(tmp_path: Path) -> cfg.RunConfig:
    config = cfg.RunConfig(
        provider="hetzner",
        mode="vm",
        model="medgemma-27b-it",
        run_id="20261020120000-abcd",
        workdir=str(tmp_path),
        created_at=NOW,
    )
    config.fill_defaults()
    return config


def test_robot_order_flow(robot_api: FakeApi, tmp_path: Path) -> None:
    robot_api.answer("POST /key", {"key": {"fingerprint": "aa:bb"}}, 201)
    robot_api.answer(
        "GET /order/server/product/GEX45",
        {
            "product": {
                "id": "GEX45",
                "dist": ["Rescue system", "Ubuntu 24.04 LTS base"],
                "price_setup": {"net": "209.0000"},
            }
        },
    )
    robot_api.answer(
        "POST /order/server/transaction",
        {"transaction": {"id": "B20261020-1", "status": "in process"}},
        201,
    )
    robot_api.answer(
        "GET /order/server/transaction/B20261020-1",
        {
            "transaction": {
                "id": "B20261020-1",
                "status": "ready",
                "server_number": 321,
                "server_ip": "198.51.100.5",
            }
        },
    )
    robot_api.answer("POST /server/321", {"server": {"server_number": 321}})
    config = make_config(tmp_path)
    assert hetzner.upload_key(config.name, "ssh-ed25519 AAAA") == "aa:bb"
    assert hetzner.order(config, "aa:bb", test=True)["id"] == "B20261020-1"
    ready = hetzner.wait_ready("B20261020-1", timeout_minutes=1)
    assert ready["server_ip"] == "198.51.100.5"
    hetzner.rename(321, hetzner.server_name(config))
    key, _, ordered, _, rename = robot_api.requests
    expected = base64.b64encode(b"robot-user:robot-secret").decode()
    assert key["headers"]["Authorization"] == f"Basic {expected}"
    assert form(key) == [("name", config.name), ("data", "ssh-ed25519 AAAA")]
    assert form(ordered) == [
        ("product_id", "GEX45"),
        ("authorized_key[]", "aa:bb"),
        ("dist", "Ubuntu 24.04 LTS base"),
        ("lang", "en"),
        ("location", "FSN1"),
        ("addon[]", "primary_ipv4"),
        ("test", "true"),
    ]
    assert form(rename) == [
        ("server_name", f"medbench-cloud-gpu-20261020120000-abcd-{NOW + 240 * 60}")
    ]


def test_robot_cancel_and_key_deletion_are_repeatable(robot_api: FakeApi) -> None:
    robot_api.answer("POST /server/9/cancellation", {"error": {}}, 409)
    hetzner.cancel(9)  # already cancelled
    hetzner.delete_key("aa:bb")  # 404: already gone
    robot_api.answer("POST /server/10/cancellation", {"error": {}}, 500)
    with pytest.raises(hetzner.RobotError) as error:
        hetzner.cancel(10)
    assert error.value.status == 500
    sent = robot_api.requests[0]
    assert form(sent) == [("cancellation_date", "now")]


def test_robot_needs_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HETZNER_ROBOT_USER", raising=False)
    with pytest.raises(hetzner.RobotError, match="HETZNER_ROBOT_USER"):
        hetzner.request("GET", "/server")


def test_expired_servers_by_name_and_by_late_order(robot_api: FakeApi) -> None:
    day = cfg.MAX_TTL_MINUTES * 60
    robot_api.answer(
        "GET /server",
        [
            {
                "server": {
                    "server_number": 1,
                    "server_name": f"medbench-cloud-gpu-r1-{PAST}",
                }
            },
            {
                "server": {
                    "server_number": 2,
                    "server_name": f"medbench-cloud-gpu-r2-{FUTURE}",
                }
            },
            {
                "server": {
                    "server_number": 3,
                    "server_name": f"medbench-cloud-gpu-r3-{PAST}",
                    "cancelled": True,
                }
            },
            {"server": {"server_number": 4, "server_name": "production-db"}},
            {"server": {"server_number": 5, "server_name": ""}},  # late, old order
            {"server": {"server_number": 6, "server_name": ""}},  # late, new order
            {"server": {"server_number": 7, "server_name": ""}},  # someone else's
        ],
    )

    def order(number: int, ordered: float, key: str) -> dict[str, Any]:
        date = datetime.fromtimestamp(ordered, tz=UTC).isoformat()
        return {
            "transaction": {
                "id": f"B-{number}",
                "date": date,
                "status": "ready",
                "server_number": number,
                "authorized_key": [{"key": {"name": key, "fingerprint": "f"}}],
            }
        }

    transactions = [
        order(5, NOW - day - 60, "medbench-20261018120000-abcd"),
        order(6, NOW - 60, "medbench-20261020115900-abcd"),
        order(7, NOW - 2 * day, "laptop"),
    ]
    robot_api.answer("GET /order/server/transaction", transactions)
    found = hetzner.expired_servers(NOW)
    assert [s["server_number"] for s in found] == [1, 5]


# The reaper with fake cloud command line tools


FAKE_CLI = """\
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_CLI_LOG"], "a") as log:
    log.write(json.dumps([os.path.basename(sys.argv[0]), *args]) + "\\n")
table = json.loads(open(os.environ["FAKE_CLI_TABLE"]).read())
for words, answer in table:
    if all(word in args for word in words):
        if answer == "fail":
            sys.exit(1)
        print(json.dumps(answer))
        break
"""


@pytest.fixture
def cli_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("aws", "az", "gcloud"):
        tool = bin_dir / name
        tool.write_text(f"#!{sys.executable}\n{FAKE_CLI}")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "calls.log"))
    table = tmp_path / "table.json"
    monkeypatch.setenv("FAKE_CLI_TABLE", str(table))
    return table


def calls(table: Path) -> list[list[str]]:
    log = table.parent / "calls.log"
    return [json.loads(line) for line in log.read_text().splitlines()]


def deletions(table: Path) -> list[list[str]]:
    words = ("delete", "terminate", "detach")
    return [
        call
        for call in calls(table)
        if any(word in part for part in call for word in words)
        and "list-attached-role-policies" not in call
        and "wait" not in call
    ]


def command(call: list[str]) -> str:
    """The aws subcommand and its arguments, without the global options."""
    words = call[1:]
    while words and words[0] in ("--region", "--output"):
        words = words[2:]
    return " ".join(words)


def test_reap_azure(cli_table: Path) -> None:
    groups = [
        {"name": "medbench-old-rg", "tags": {"expires-at": PAST}},
        {"name": "medbench-new-rg", "tags": {"expires-at": FUTURE}},
        {"name": "medbench-untagged-rg", "tags": None},
    ]
    cli_table.write_text(json.dumps([[["group", "list"], groups]]))
    sweeper = reaper.Reaper(dry_run=False, now=NOW)
    sweeper.azure()
    assert sweeper.deleted == ["Azure resource group medbench-old-rg"]
    assert deletions(cli_table) == [
        ["az", "group", "delete", "--name", "medbench-old-rg", "--yes"]
    ]
    assert calls(cli_table)[0][1:] == [
        "group",
        "list",
        "--tag",
        "managed-by=medbench-cloud-gpu",
        "--output",
        "json",
    ]


def test_reap_gcp(cli_table: Path) -> None:
    zone = "projects/p/zones/us-central1-a"
    region = "projects/p/regions/us-central1"
    old = f"medbench-cloud-gpu run x expires-at={PAST}"
    table = [
        [
            ["instances", "list"],
            [
                {"name": "medbench-a", "zone": zone, "labels": {"expires-at": PAST}},
                {"name": "medbench-b", "zone": zone, "labels": {"expires-at": FUTURE}},
            ],
        ],
        [
            ["clusters", "list"],
            [
                {
                    "name": "medbench-c",
                    "location": "us-central1-a",
                    "resourceLabels": {"expires-at": PAST},
                }
            ],
        ],
        [["firewall-rules", "list"], [{"name": "medbench-fw", "description": old}]],
        [
            ["subnets", "list"],
            [{"name": "medbench-sn", "description": old, "region": region}],
        ],
        [
            ["networks", "list"],
            [{"name": "medbench-net", "description": "medbench-cloud-gpu"}],
        ],
    ]
    cli_table.write_text(json.dumps(table))
    sweeper = reaper.Reaper(dry_run=False, now=NOW)
    sweeper.gcp("p")
    base = ["gcloud", "--project", "p", "--quiet"]
    assert deletions(cli_table) == [
        [*base, "compute", "instances", "delete", "medbench-a", "--zone=us-central1-a"],
        [
            *base,
            "container",
            "clusters",
            "delete",
            "medbench-c",
            "--location=us-central1-a",
        ],
        [*base, "compute", "firewall-rules", "delete", "medbench-fw"],
        [
            *base,
            "compute",
            "networks",
            "subnets",
            "delete",
            "medbench-sn",
            "--region=us-central1",
        ],
    ]


def test_reap_aws(cli_table: Path) -> None:
    def tags(expires: str) -> list[dict[str, str]]:
        return [
            {"Key": "managed-by", "Value": "medbench-cloud-gpu"},
            {"Key": "expires-at", "Value": expires},
        ]

    table = [
        [
            ["describe-instances"],
            {
                "Reservations": [
                    {
                        "Instances": [
                            {"InstanceId": "i-old", "Tags": tags(PAST)},
                            {"InstanceId": "i-new", "Tags": tags(FUTURE)},
                        ]
                    }
                ]
            },
        ],
        [["list-clusters"], {"clusters": ["medbench-eks", "someone-else"]}],
        [
            ["describe-cluster", "medbench-eks"],
            {
                "cluster": {
                    "tags": {"managed-by": "medbench-cloud-gpu", "expires-at": PAST}
                }
            },
        ],
        [["describe-cluster", "someone-else"], {"cluster": {"tags": {}}}],
        [["list-nodegroups"], {"nodegroups": ["gpu"]}],
        [["describe-vpcs"], {"Vpcs": [{"VpcId": "vpc-1", "Tags": tags(PAST)}]}],
        [
            ["describe-internet-gateways"],
            {"InternetGateways": [{"InternetGatewayId": "igw-1"}]},
        ],
        [["describe-subnets"], {"Subnets": [{"SubnetId": "subnet-1"}]}],
        [
            ["describe-route-tables"],
            {
                "RouteTables": [
                    {"RouteTableId": "rtb-main", "Associations": [{"Main": True}]},
                    {"RouteTableId": "rtb-1", "Associations": []},
                ]
            },
        ],
        [
            ["describe-security-groups"],
            {
                "SecurityGroups": [
                    {"GroupId": "sg-default", "GroupName": "default"},
                    {"GroupId": "sg-1", "GroupName": "medbench"},
                ]
            },
        ],
        [
            ["describe-key-pairs"],
            {
                "KeyPairs": [
                    {"KeyPairId": "key-1", "KeyName": "medbench-k", "Tags": tags(PAST)}
                ]
            },
        ],
        [
            ["list-roles"],
            {"Roles": [{"RoleName": "medbench-role"}, {"RoleName": "admin"}]},
        ],
        [["list-role-tags"], {"Tags": tags(PAST)}],
        [
            ["list-attached-role-policies"],
            {"AttachedPolicies": [{"PolicyArn": "arn:aws:iam::aws:policy/X"}]},
        ],
        [["delete-vpc"], "fail"],
    ]
    cli_table.write_text(json.dumps(table))
    sweeper = reaper.Reaper(dry_run=False, now=NOW)
    sweeper.aws("eu-west-1")
    assert sweeper.failed == ["VPC vpc-1"]
    assert sweeper.deleted == [
        "EC2 instance i-old",
        "EKS cluster medbench-eks",
        "key pair medbench-k",
        "IAM role medbench-role",
    ]
    assert [command(call) for call in deletions(cli_table)] == [
        "ec2 terminate-instances --instance-ids i-old",
        "eks delete-nodegroup --cluster-name medbench-eks --nodegroup-name gpu",
        "eks delete-cluster --name medbench-eks",
        "ec2 detach-internet-gateway --internet-gateway-id igw-1 --vpc-id vpc-1",
        "ec2 delete-internet-gateway --internet-gateway-id igw-1",
        "ec2 delete-subnet --subnet-id subnet-1",
        "ec2 delete-route-table --route-table-id rtb-1",
        "ec2 delete-security-group --group-id sg-1",
        "ec2 delete-vpc --vpc-id vpc-1",
        "ec2 delete-key-pair --key-pair-id key-1",
        "iam detach-role-policy --role-name medbench-role --policy-arn arn:aws:iam::aws:policy/X",
        "iam delete-role --role-name medbench-role",
    ]


def test_dry_run_deletes_nothing(cli_table: Path) -> None:
    groups = [{"name": "medbench-old-rg", "tags": {"expires-at": PAST}}]
    cli_table.write_text(json.dumps([[["group", "list"], groups]]))
    sweeper = reaper.Reaper(dry_run=True, now=NOW)
    sweeper.azure()
    assert sweeper.deleted == ["Azure resource group medbench-old-rg"]
    assert deletions(cli_table) == []


def test_reap_hetzner_and_github(
    robot_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(github, "API", robot_api.url)
    monkeypatch.setenv("CLOUD_GPU_RUNNER_TOKEN", "t")
    robot_api.answer(
        "GET /server",
        [
            {
                "server": {
                    "server_number": 1,
                    "server_name": f"medbench-cloud-gpu-r-{PAST}",
                }
            }
        ],
    )
    robot_api.answer("GET /order/server/transaction", [])
    robot_api.answer("POST /server/1/cancellation", {"cancellation": {}})
    robot_api.answer(
        "GET /repos/o/r/actions/runners?per_page=100&page=1",
        {"runners": [{"id": 8, "name": "medbench-x", "status": "offline"}]},
    )
    robot_api.answer("DELETE /repos/o/r/actions/runners/8", None, 204)
    sweeper = reaper.Reaper(dry_run=False, now=NOW)
    sweeper.hetzner()
    sweeper.github("o/r")
    assert sweeper.deleted == [
        f"Hetzner server 1 (medbench-cloud-gpu-r-{PAST})",
        "GitHub runner medbench-x",
    ]
    assert sweeper.failed == []
