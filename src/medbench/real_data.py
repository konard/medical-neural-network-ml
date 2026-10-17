"""Real, openly licensed medical files, downloaded at test time.

The benchmark does not draw fake scans or render fake handwriting. Every
image or document is a real file from a public archive (Wikimedia Commons,
Wikisource, open research datasets) whose license allows anyone to copy it:
public domain, CC0, CC BY or CC BY-SA. The repository stores only a manifest
(``data/real_samples.json``): where each file lives, its SHA-256, its license,
its author and the known answer. :func:`fetch` downloads a file once into a
local cache and refuses it if the bytes differ from the pinned hash.

Licenses that need attribution are honoured by :func:`attribution_lines`,
which every report and the ``medbench sources`` command print.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_LICENSES",
    "USER_AGENT",
    "ChecksumError",
    "Sample",
    "attribution_lines",
    "default_cache_dir",
    "fetch",
    "load_manifest",
    "prepare_image",
    "samples_for",
]

# Licenses that allow copying, changing and commercial use. Non-commercial
# (NC), no-derivatives (ND) and "no license" files are never added.
ALLOWED_LICENSES = (
    "Public domain",
    "CC0 1.0",
    "CC BY 2.0",
    "CC BY 2.5",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 2.0",
    "CC BY-SA 2.5",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
)

# Wikimedia asks every client to name itself and give a contact URL.
USER_AGENT = (
    "medbench/0.1 (https://github.com/konard/medical-neural-network-ml; "
    "test-time download of openly licensed samples)"
)

Opener = Callable[[urllib.request.Request, float], Any]


class ChecksumError(RuntimeError):
    """A downloaded file is not the file the manifest pinned."""


@dataclass(frozen=True)
class Sample:
    """One real file and what is known about it.

    Attributes:
        sample_id: Short stable name, used in reports.
        task: Benchmark task that uses it, e.g. ``imaging-xray``.
        url: Direct download URL of the file.
        sha256: Expected SHA-256 of the downloaded bytes.
        license: Short license name, one of :data:`ALLOWED_LICENSES`.
        license_url: Where the license text is.
        author: Who made the file (needed for attribution).
        source_url: Human page that shows the file, its author and license.
        truth: The known answer, e.g. ``{"finding": "pneumothorax"}``.
        notes: Anything a reader should know (source of the label, caveats).
        archive_member: For files published only inside a zip: the member
            name. ``url`` is then the zip, and ``sha256`` still pins the
            member itself.
        archive_sha256: SHA-256 of the whole zip (with ``archive_member``).
    """

    sample_id: str
    task: str
    url: str
    sha256: str
    license: str
    license_url: str
    author: str
    source_url: str
    truth: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    archive_member: str = ""
    archive_sha256: str = ""

    @property
    def filename(self) -> str:
        """Cache file name: the id plus the original extension."""
        source = self.archive_member or self.url.split("?", 1)[0]
        suffix = Path(source).suffix.lower() or ".bin"
        return f"{self.sample_id}{suffix}"

    def attribution(self) -> str:
        """One line that credits the author as the license asks."""
        return (
            f"{self.sample_id}: {self.author}, {self.license} "
            f"({self.license_url}), via {self.source_url}"
        )


def _manifest_text() -> str:
    return (
        resources.files("medbench")
        .joinpath("data/real_samples.json")
        .read_text(encoding="utf-8")
    )


def load_manifest(path: Path | None = None) -> list[Sample]:
    """Read the samples manifest (the packaged one by default)."""
    text = path.read_text(encoding="utf-8") if path else _manifest_text()
    samples = [Sample(**entry) for entry in json.loads(text)["samples"]]
    for sample in samples:
        if sample.license not in ALLOWED_LICENSES:
            msg = f"Unsupported license for {sample.sample_id}: {sample.license}"
            raise ValueError(msg)
    return samples


def samples_for(task: str, samples: Iterable[Sample] | None = None) -> list[Sample]:
    """Samples of one task, in manifest order."""
    pool = load_manifest() if samples is None else samples
    return [sample for sample in pool if sample.task == task]


def default_cache_dir() -> Path:
    """``$MEDBENCH_DATA_DIR``, else ``~/.cache/medbench/real-data``."""
    configured = os.environ.get("MEDBENCH_DATA_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "medbench" / "real-data"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Wikimedia's robot policy asks bots to go one request at a time and to wait
# as long as a 429 reply's Retry-After says. CI runners share IP addresses
# with other bots, so 429 does happen there.
# https://wikitech.wikimedia.org/wiki/Robot_policy
_PAUSE_SECONDS = 1.0
_MAX_WAIT_SECONDS = 120.0
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_last_request: dict[str, float] = {}


def _open(request: urllib.request.Request, timeout: float) -> Any:
    host = urllib.parse.urlsplit(request.full_url).netloc
    idle = time.monotonic() - _last_request.get(host, float("-inf"))
    if idle < _PAUSE_SECONDS:
        time.sleep(_PAUSE_SECONDS - idle)
    _last_request[host] = time.monotonic()
    return urllib.request.urlopen(request, timeout=timeout)


def _retry_after(error: urllib.error.HTTPError) -> float | None:
    value = str(error.headers.get("Retry-After", "") if error.headers else "")
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, float(moment.timestamp()) - time.time())


def _wait_before_retry(error: Exception, attempt: int) -> float | None:
    """Seconds to wait before the next try, or ``None`` if retrying is useless."""
    backoff = 5.0 * 2.0 ** (attempt - 1)
    if isinstance(error, urllib.error.HTTPError):
        if error.code not in _RETRY_STATUS:
            return None
        retry_after = _retry_after(error)
        if retry_after is not None:
            # Do not retry earlier than the source permits. Fail this run if
            # the server asks us to wait longer than our per-retry budget.
            if retry_after > _MAX_WAIT_SECONDS:
                return None
            backoff = retry_after
    return min(backoff, _MAX_WAIT_SECONDS)


def _download(url: str, opener: Opener, timeout: float, attempts: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with opener(request, timeout) as response:
                return bytes(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            wait = _wait_before_retry(error, attempt)
            if wait is None or attempt == attempts:
                raise
            time.sleep(wait)
    message = "attempts must be at least 1"
    raise ValueError(message)


def _verified(sample: Sample, data: bytes, expected: str, what: str) -> bytes:
    digest = _sha256(data)
    if digest != expected:
        message = (
            f"{sample.sample_id}: SHA-256 of the {what} {digest} does not match "
            f"the pinned {expected}. The file at {sample.url} changed; check its "
            f"license and label on {sample.source_url} again before updating "
            "the manifest."
        )
        raise ChecksumError(message)
    return data


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def _from_archive(
    sample: Sample, directory: Path, opener: Opener, timeout: float, attempts: int
) -> bytes:
    """Read ``sample.archive_member``; the zip is downloaded once and shared."""
    archive = directory / f"archive-{sample.archive_sha256[:16]}.zip"
    if not (
        archive.exists() and _sha256(archive.read_bytes()) == sample.archive_sha256
    ):
        data = _download(sample.url, opener, timeout, attempts)
        _save(archive, _verified(sample, data, sample.archive_sha256, "archive"))
    with zipfile.ZipFile(archive) as handle:
        return handle.read(sample.archive_member)


def fetch(
    sample: Sample,
    cache_dir: Path | None = None,
    opener: Opener = _open,
    timeout: float = 60.0,
    attempts: int = 5,
) -> Path:
    """Return the local path of ``sample``, downloading it if needed.

    A cached copy is reused only if its hash still matches. Set
    ``MEDBENCH_OFFLINE=1`` to forbid downloads (a missing file then fails).

    Raises:
        ChecksumError: If the bytes differ from the pinned SHA-256. The
            source changed or the download was tampered with; the file is
            not kept.
        FileNotFoundError: If offline mode is on and the file is missing.
    """
    directory = cache_dir or default_cache_dir()
    path = directory / sample.filename
    if path.exists() and _sha256(path.read_bytes()) == sample.sha256:
        return path
    if os.environ.get("MEDBENCH_OFFLINE") == "1":
        message = f"{sample.sample_id} is not cached in {directory} (offline mode)"
        raise FileNotFoundError(message)
    if sample.archive_member:
        data = _from_archive(sample, directory, opener, timeout, attempts)
    else:
        data = _download(sample.url, opener, timeout, attempts)
    _save(path, _verified(sample, data, sample.sha256, "file"))
    return path


def prepare_image(data: bytes, max_side: int = 1024) -> bytes:
    """Convert any image to PNG no larger than ``max_side`` (needs Pillow).

    Vision models resize inputs anyway (MedGemma to 896x896), so sending a
    4000-pixel photo only costs bandwidth and memory.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        converted = image.convert("L" if image.mode in {"1", "L", "I;16"} else "RGB")
    converted.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    converted.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def attribution_lines(samples: Sequence[Sample]) -> list[str]:
    """Credits for ``samples``, one line each, without duplicates."""
    seen: dict[str, None] = {}
    for sample in samples:
        seen.setdefault(sample.attribution(), None)
    return list(seen)
