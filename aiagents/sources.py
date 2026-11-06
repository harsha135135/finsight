"""Bounded source ingestion from local documents and explicitly allowed HTTPS hosts.

Documents are evidence, never instructions. Network connections are pinned to a
validated public IP address while certificate verification retains the URL host.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import queue
import re
import socket
import ssl
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

MAX_SOURCE_BYTES = 1_048_576
MAX_TEXT_CHARS = 100_000
MAX_URL_CHARS = 8_192
FETCH_TIMEOUT_SECONDS = 15.0
LOCAL_EXTENSIONS = {".txt", ".md", ".json"}


class SourceError(ValueError):
    """A source cannot be read within the evidence-ingestion policy."""


class _HTMLText(HTMLParser):
    _ignored = {"script", "style", "noscript", "template", "svg"}
    _blocks = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored:
            self.ignored.append(tag)
        elif not self.ignored and tag in self._blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.ignored:
            if tag == self.ignored[-1]:
                self.ignored.pop()
        elif tag in self._blocks:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.ignored and tag in self._blocks:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored:
            self.parts.append(data)


def extract_html_text(document: str) -> str:
    """Extract visible text without scripts, styles, or active document content."""
    parser = _HTMLText()
    parser.feed(document)
    parser.close()
    lines = [re.sub(r"[\t \f\v]+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def _json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SourceError("JSON numbers must be finite")
    return parsed


def _reject_constant(value: str) -> None:
    raise SourceError(f"JSON constant {value} is not permitted")


def _parse_document(raw: bytes, *, is_json: bool, is_html: bool = False, charset: str = "utf-8") -> tuple[str, Any]:
    if len(raw) > MAX_SOURCE_BYTES:
        raise SourceError(f"Source exceeds {MAX_SOURCE_BYTES} bytes")
    if charset.lower().replace("_", "-") not in {"utf-8", "utf8", "us-ascii", "ascii"}:
        raise SourceError("Only UTF-8 and ASCII documents are supported")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceError("Source is not valid UTF-8") from exc
    data = None
    if is_json:
        try:
            data = json.loads(text, parse_float=_json_float, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            raise SourceError(f"Invalid JSON document: {exc}") from exc
    if is_html:
        text = extract_html_text(text)
    text = text.strip()
    if not text:
        raise SourceError("Source contains no readable text")
    if len(text) > MAX_TEXT_CHARS:
        raise SourceError(f"Source exceeds {MAX_TEXT_CHARS} text characters")
    return text, data


def _local_document(path_value: Any, base_dir: Path) -> tuple[bytes, str, bool]:
    if not isinstance(path_value, str) or not path_value:
        raise SourceError("Local source path must be a nonempty string")
    relative_path = Path(path_value)
    if relative_path.is_absolute():
        raise SourceError("Local source path must be relative to the job directory")
    base = base_dir.resolve(strict=True)
    candidate = (base / relative_path).resolve(strict=True)
    if not candidate.is_relative_to(base):
        raise SourceError("Local source escapes the job directory")
    if candidate.suffix.lower() not in LOCAL_EXTENSIONS:
        raise SourceError("Local sources must be .txt, .md, or .json files")
    if not candidate.is_file():
        raise SourceError("Local source is not a regular file")
    with candidate.open("rb") as source_file:
        raw = source_file.read(MAX_SOURCE_BYTES + 1)
    return raw, candidate.as_uri(), candidate.suffix.lower() == ".json"


def _canonical_host(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceError("Allowed hosts must be nonempty host names")
    if any(character in value for character in "/\\@?#%") or any(ord(character) < 33 for character in value):
        raise SourceError("Host names cannot include credentials, paths, or URL syntax")
    host = value.rstrip(".").lower()
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if ":" in host:
            raise SourceError("Allowed hosts must not include ports")
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise SourceError("Invalid host name") from exc
        if len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) for part in host.split(".")):
            raise SourceError("Invalid host name")
        return host


def _public_address(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        return _public_address(str(parsed.ipv4_mapped))
    return parsed.is_global and not any((parsed.is_private, parsed.is_loopback, parsed.is_link_local, parsed.is_multicast, parsed.is_reserved, parsed.is_unspecified))


def _resolve_public_addresses(host: str, timeout: float) -> list[tuple]:
    """Bound DNS latency; inspect every answer before selecting one connection."""
    results: queue.Queue = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            results.put((socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP), None))
        except OSError as exc:
            results.put((None, exc))

    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses, error = results.get(timeout=max(timeout, 0.001))
    except queue.Empty as exc:
        raise SourceError("DNS lookup timed out") from exc
    if error:
        raise SourceError(f"DNS lookup failed: {error}") from error
    if not addresses or any(not _public_address(item[4][0]) for item in addresses):
        raise SourceError("Host must resolve exclusively to public internet addresses")
    # Prefer IPv4 when both families are available; there is no retry or fallback.
    return sorted(addresses, key=lambda item: item[0] != socket.AF_INET)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: tuple, timeout: float) -> None:
        super().__init__(host, port=443, timeout=timeout, context=ssl.create_default_context())
        self._address = address
        self._deadline = time.monotonic() + timeout
        self._active_socket = None
        self._expired = threading.Event()

    def abort(self) -> None:
        """Interrupt slow response headers or bodies at the absolute deadline."""
        self._expired.set()
        active_socket = self._active_socket
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if self._expired.is_set() or remaining <= 0:
            raise SourceError("HTTPS source fetch timed out")
        return remaining

    def connect(self) -> None:
        family, socktype, proto, _, sockaddr = self._address
        raw_socket = socket.socket(family, socktype, proto)
        self._active_socket = raw_socket
        try:
            raw_socket.settimeout(self._remaining())
            raw_socket.connect(sockaddr)
            # Expose the TLS socket before the handshake so the deadline can
            # interrupt the handshake as well as ordinary HTTP reads.
            tls_socket = self._context.wrap_socket(raw_socket, server_hostname=self.host, do_handshake_on_connect=False)
            self._active_socket = tls_socket
            tls_socket.settimeout(self._remaining())
            tls_socket.do_handshake()
            self.sock = tls_socket
        except BaseException:
            self._active_socket.close()
            raise


def _fetch_https(url: str, allowed_hosts: list[str]) -> tuple[bytes, str, str]:
    if not isinstance(url, str) or len(url) > MAX_URL_CHARS or any(ord(char) < 33 or ord(char) == 127 for char in url) or "\\" in url:
        raise SourceError("Invalid HTTPS source URL")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise SourceError("Invalid HTTPS source URL") from exc
    if parsed.scheme != "https" or not parsed.hostname:
        raise SourceError("Network sources require HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise SourceError("Source URLs cannot contain credentials")
    if port not in (None, 443):
        raise SourceError("Only HTTPS port 443 is permitted")
    if parsed.fragment:
        raise SourceError("Source URLs must not include fragments")
    host = _canonical_host(parsed.hostname)
    hosts = {_canonical_host(item) for item in allowed_hosts}
    if host not in hosts:
        raise SourceError(f"Host {host!r} is not explicitly allowed")
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not _public_address(str(literal_ip)):
        raise SourceError("Private and non-public IP addresses are not permitted")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise SourceError("Local host names are not permitted")

    deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise SourceError("HTTPS source fetch timed out")
        return value

    addresses = _resolve_public_addresses(host, remaining())
    connection = _PinnedHTTPSConnection(host, addresses[0], timeout=remaining())
    deadline_timer = threading.Timer(remaining(), connection.abort)
    deadline_timer.daemon = True
    deadline_timer.start()
    try:
        target = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
        if parsed.query:
            target += "?" + quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
        connection.request("GET", target, headers={"User-Agent": "ResearchPipeline/1.0", "Accept": "application/json, text/plain, text/markdown, text/html", "Accept-Encoding": "identity", "Connection": "close"})
        network_socket = connection.sock
        network_socket.settimeout(remaining())
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise SourceError("HTTPS redirects are disabled; specify the final allowed URL")
        if response.status != 200:
            raise SourceError(f"HTTPS source returned HTTP {response.status}")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise SourceError("Compressed HTTP responses are not supported")
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        if content_type not in {"application/json", "text/plain", "text/markdown", "text/html", "application/xhtml+xml"} and not content_type.endswith("+json"):
            raise SourceError(f"Unsupported HTTP content type: {content_type}")
        content_length = response.getheader("Content-Length")
        expected_size = None
        if content_length is not None:
            try:
                expected_size = int(content_length)
            except ValueError as exc:
                raise SourceError("Invalid HTTP Content-Length") from exc
            if expected_size < 0 or expected_size > MAX_SOURCE_BYTES:
                raise SourceError(f"Source exceeds {MAX_SOURCE_BYTES} bytes or has invalid size")
        chunks: list[bytes] = []
        size = 0
        while True:
            network_socket.settimeout(remaining())
            chunk = response.read1(min(65_536, MAX_SOURCE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_SOURCE_BYTES:
                raise SourceError(f"Source exceeds {MAX_SOURCE_BYTES} bytes")
        remaining()
        if expected_size is not None and size != expected_size:
            raise SourceError("HTTP response ended before its declared Content-Length")
        return b"".join(chunks), content_type, charset
    finally:
        deadline_timer.cancel()
        connection.close()


def collect_sources(specs: list[dict], base_dir: Path, allowed_hosts: list[str]) -> dict:
    """Collect independently; an unavailable source becomes an explicit error.

    Relative file paths are confined to ``base_dir``. Source IDs and uniqueness
    are job-validation concerns; this function never chooses or follows a URL.
    """
    sources: list[dict] = []
    errors: list[dict] = []
    for spec in specs:
        source_id = spec.get("id", "unknown")
        try:
            if ("path" in spec) == ("url" in spec):
                raise SourceError("A source must specify exactly one of path or url")
            if "path" in spec:
                raw, uri, is_json = _local_document(spec["path"], Path(base_dir))
                text, data = _parse_document(raw, is_json=is_json)
            else:
                raw, content_type, charset = _fetch_https(spec["url"], allowed_hosts)
                uri = spec["url"]
                is_json = content_type == "application/json" or content_type.endswith("+json")
                text, data = _parse_document(raw, is_json=is_json, is_html=content_type in {"text/html", "application/xhtml+xml"}, charset=charset)
            source = {"id": source_id, "title": spec.get("title") or str(source_id), "uri": uri, "text": text, "sha256": hashlib.sha256(raw).hexdigest(), "retrieved_at": datetime.now(timezone.utc).isoformat()}
            if is_json:
                source["data"] = data
            sources.append(source)
        except (OSError, ValueError, TypeError, http.client.HTTPException, RecursionError) as exc:
            errors.append({"id": source_id, "error": str(exc)})
    return {"sources": sources, "errors": errors}
