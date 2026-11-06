import hashlib
import json
import socket
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

from aiagents import sources


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def collect(self, spec, allowed_hosts=None):
        return sources.collect_sources([{"id": "evidence", "title": "Evidence", **spec}], self.base, allowed_hosts or [])

    def test_local_json_retains_data_and_raw_digest(self):
        raw = b'{"revenue": 12.5}\n'
        (self.base / "figures.json").write_bytes(raw)
        result = self.collect({"path": "figures.json"})
        self.assertEqual(result["errors"], [])
        evidence = result["sources"][0]
        self.assertEqual(evidence["data"], {"revenue": 12.5})
        self.assertEqual(evidence["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertTrue(evidence["uri"].startswith("file:///"))
        self.assertIn("+00:00", evidence["retrieved_at"])
        json.dumps(result, allow_nan=False)

    def test_absolute_paths_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as outer:
            outside = Path(outer) / "private.txt"
            outside.write_text("Not authorized")
            for value in (str(outside), f"../{Path(outer).name}/private.txt"):
                with self.subTest(value=value):
                    result = self.collect({"path": value})
                    self.assertFalse(result["sources"])
                    self.assertTrue(result["errors"])

    def test_symlink_cannot_escape_job_directory(self):
        with tempfile.TemporaryDirectory() as outer:
            outside = Path(outer) / "private.txt"
            outside.write_text("Not authorized")
            (self.base / "linked.txt").symlink_to(outside)
            result = self.collect({"path": "linked.txt"})
            self.assertIn("escapes", result["errors"][0]["error"])

    def test_missing_unsupported_and_conflicting_sources_are_errors(self):
        (self.base / "file.pdf").write_bytes(b"pdf")
        for spec in ({"path": "missing.md"}, {"path": "file.pdf"}, {}, {"path": "x.md", "url": "https://example.com"}):
            with self.subTest(spec=spec):
                result = self.collect(spec)
                self.assertFalse(result["sources"])
                self.assertEqual(len(result["errors"]), 1)

    def test_source_failures_do_not_discard_readable_sources(self):
        (self.base / "good.md").write_text("Usable evidence")
        result = sources.collect_sources([{"id": "bad", "path": "missing.txt"}, {"id": "good", "path": "good.md"}], self.base, [])
        self.assertEqual([item["id"] for item in result["sources"]], ["good"])
        self.assertEqual([item["id"] for item in result["errors"]], ["bad"])

    def test_byte_and_text_limits_are_enforced_without_truncation(self):
        (self.base / "bytes.txt").write_bytes(b"x" * (sources.MAX_SOURCE_BYTES + 1))
        (self.base / "text.md").write_text("x" * (sources.MAX_TEXT_CHARS + 1))
        for name, term in (("bytes.txt", "bytes"), ("text.md", "characters")):
            with self.subTest(name=name):
                result = self.collect({"path": name})
                self.assertFalse(result["sources"])
                self.assertIn(term, result["errors"][0]["error"])

    def test_empty_invalid_utf8_and_nonfinite_json_fail(self):
        for index, raw in enumerate((b"", b"\xff", b'{"value": NaN}', b'{"value": 1e9999}')):
            name = f"invalid-{index}.json"
            (self.base / name).write_bytes(raw)
            self.assertFalse(self.collect({"path": name})["sources"])

    def test_html_excludes_scripts_styles_and_template_content(self):
        text = sources.extract_html_text("<h1>Results &amp; risks</h1><script>steal()</script><style>secret</style><template><p>hidden</p></template><p>Revenue <b>increased</b>.</p><p>Risk remains.</p>")
        self.assertEqual(text, "Results & risks\nRevenue increased.\nRisk remains.")

    def test_url_policy_rejects_unsafe_targets_before_dns(self):
        urls = [
            "http://example.com/data", "https://user:secret@example.com/data",
            "https://example.com:8443/data", "https://other.example/data",
            "https://127.0.0.1/data", "https://169.254.169.254/latest/meta-data",
            "https://[::1]/data", "https://localhost/data",
            "https://example.com/data\r\nInjected: true", "https://example.com/#fragment",
        ]
        with patch.object(sources, "_resolve_public_addresses") as resolve:
            for url in urls:
                with self.subTest(url=url):
                    result = self.collect({"url": url}, ["example.com", "127.0.0.1", "169.254.169.254", "::1", "localhost"])
                    self.assertFalse(result["sources"])
                    self.assertTrue(result["errors"])
            resolve.assert_not_called()

    def test_dns_rejects_private_mixed_and_nonunicast_answers(self):
        for address in ("10.0.0.1", "127.0.0.1", "169.254.169.254", "224.0.0.1", "0.0.0.0", "::1", "fc00::1", "::ffff:127.0.0.1"):
            answers = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443)), (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))]
            with self.subTest(address=address), patch.object(socket, "getaddrinfo", return_value=answers):
                with self.assertRaisesRegex(sources.SourceError, "public internet"):
                    sources._resolve_public_addresses("example.com", 1)

    def test_pinned_socket_keeps_tls_server_name(self):
        address = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        fake_context = MagicMock()
        with patch.object(sources.ssl, "create_default_context", return_value=fake_context), patch.object(sources.socket, "socket") as socket_factory:
            connection = sources._PinnedHTTPSConnection("example.com", address, 2)
            connection.connect()
            socket_factory.return_value.connect.assert_called_once_with(("93.184.216.34", 443))
            fake_context.wrap_socket.assert_called_once_with(socket_factory.return_value, server_hostname="example.com", do_handshake_on_connect=False)
            fake_context.wrap_socket.return_value.do_handshake.assert_called_once()
            connection.close()

    def mock_connection(self, status=200, content_type="text/html", chunks=None, **headers):
        response = MagicMock()
        response.status = status
        response.headers = Message()
        response.headers["Content-Type"] = content_type
        response.getheader.side_effect = lambda name, default=None: headers.get(name, default)
        response.read1.side_effect = chunks or [b"<p>Verified source</p>", b""]
        connection = MagicMock()
        connection.getresponse.return_value = response
        address = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        return response, connection, address

    def test_https_html_is_ingested_with_no_extra_requests(self):
        response, connection, address = self.mock_connection()
        with patch.object(sources, "_resolve_public_addresses", return_value=[address]), patch.object(sources, "_PinnedHTTPSConnection", return_value=connection):
            result = self.collect({"url": "https://example.com/report"}, ["example.com"])
        self.assertEqual(result["sources"][0]["text"], "Verified source")
        self.assertEqual(result["errors"], [])
        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_redirect_compression_oversize_and_binary_are_rejected(self):
        cases = [({"status": 302}, "redirect"), ({"Content-Encoding": "gzip"}, "Compressed"), ({"Content-Length": str(sources.MAX_SOURCE_BYTES + 1)}, "bytes"), ({"Content-Length": "100"}, "declared Content-Length"), ({"content_type": "application/pdf"}, "content type")]
        for kwargs, term in cases:
            response, connection, address = self.mock_connection(**kwargs)
            with self.subTest(kwargs=kwargs), patch.object(sources, "_resolve_public_addresses", return_value=[address]), patch.object(sources, "_PinnedHTTPSConnection", return_value=connection):
                result = self.collect({"url": "https://example.com/report"}, ["example.com"])
                self.assertFalse(result["sources"])
                self.assertIn(term, result["errors"][0]["error"])
                connection.request.assert_called_once()

    def test_connection_deadline_interrupts_slow_network_reads(self):
        address = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        connection = sources._PinnedHTTPSConnection("example.com", address, 2)
        fake_socket = MagicMock()
        connection._active_socket = fake_socket
        connection.abort()
        fake_socket.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        with self.assertRaisesRegex(sources.SourceError, "timed out"):
            connection._remaining()


if __name__ == "__main__":
    unittest.main()
