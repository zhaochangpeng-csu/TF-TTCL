import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from scripts.fetch_textvqa_train_direct_images import ProxyRangeReader


class FakeResponse:
    def __init__(self, content, status_code, headers):
        self.content = content
        self.status_code = status_code
        self.headers = headers

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, content):
        self.content = content
        self.proxies = {}
        self.trust_env = True
        self.closed = False

    def head(self, url, **kwargs):
        return FakeResponse(b"", 200, {"Content-Length": str(len(self.content)), "ETag": '"fixed"'})

    def get(self, url, headers, **kwargs):
        start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        return FakeResponse(self.content[start:end + 1], 206,
                            {"Content-Range": f"bytes {start}-{end}/{len(self.content)}",
                             "ETag": '"fixed"'})

    def close(self):
        self.closed = True


class RangeReaderTests(unittest.TestCase):
    def test_seek_read_and_zip_member(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "one.jpg"
            Image.new("RGB", (3, 3), "blue").save(image_path)
            archive_data = io.BytesIO()
            with zipfile.ZipFile(archive_data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(image_path, "train_images/one.jpg")
            raw = archive_data.getvalue()
            session = FakeSession(raw)
            with ProxyRangeReader("https://example.org/archive.zip", "socks5h://proxy:1234",
                                  block_size=64, session=session) as reader:
                self.assertEqual(reader.read(12), raw[:12])
                self.assertEqual(reader.seek(-20, io.SEEK_END), len(raw) - 20)
                self.assertEqual(reader.read(), raw[-20:])
                with zipfile.ZipFile(reader) as archive:
                    self.assertEqual(archive.read("train_images/one.jpg"), image_path.read_bytes())
            self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
