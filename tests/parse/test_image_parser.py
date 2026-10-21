"""
Tests for the untrusted-input limits in ImageParser (#2067).

Covers:
    - Default formats (PNG, JPEG, WEBP, BMP) open; TIFF and GIF are refused
      unless enabled through allowed_formats
    - EPS can never be enabled
    - File size and pixel count limits
    - Every entry point (parse, extract_metadata, preprocess_image,
      extract_text) applies the limits
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from semantica.parse.image_parser import ImageParser
from semantica.utils.exceptions import ValidationError


class TestImageParserLimits(unittest.TestCase):
    def setUp(self):
        logger_patcher = patch("semantica.parse.image_parser.get_logger")
        tracker_patcher = patch("semantica.parse.image_parser.get_progress_tracker")
        logger_patcher.start()
        tracker_patcher.start()
        self.addCleanup(logger_patcher.stop)
        self.addCleanup(tracker_patcher.stop)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _image(self, name, fmt, size=(8, 8)):
        path = self.dir / name
        Image.new("RGB", size, "white").save(path, format=fmt)
        return path

    def test_default_formats_parse(self):
        parser = ImageParser()
        for name, fmt in [
            ("a.png", "PNG"),
            ("a.jpg", "JPEG"),
            ("a.webp", "WEBP"),
            ("a.bmp", "BMP"),
        ]:
            with self.subTest(fmt=fmt):
                result = parser.parse(self._image(name, fmt))
                self.assertEqual(result["metadata"]["format"], fmt)

    def test_tiff_and_gif_refused_by_default(self):
        parser = ImageParser()
        for name, fmt in [("a.tiff", "TIFF"), ("a.gif", "GIF")]:
            with self.subTest(fmt=fmt):
                with self.assertRaises(ValidationError):
                    parser.parse(self._image(name, fmt))

    def test_tiff_and_gif_open_when_allowed(self):
        parser = ImageParser(allowed_formats=["png", "tiff", "gif"])
        for name, fmt in [("a.tiff", "TIFF"), ("a.gif", "GIF")]:
            with self.subTest(fmt=fmt):
                metadata = parser.extract_metadata(self._image(name, fmt))
                self.assertEqual(metadata.format, fmt)

    def test_extension_does_not_bypass_format_check(self):
        # A TIFF renamed to .png is still decoded as TIFF, so it's refused.
        path = self.dir / "disguised.png"
        Image.new("RGB", (8, 8)).save(path, format="TIFF")
        with self.assertRaises(ValidationError):
            ImageParser().parse(path)

    def test_eps_cannot_be_enabled(self):
        with self.assertRaises(ValidationError):
            ImageParser(allowed_formats=["PNG", "eps"])

    def test_file_size_limit(self):
        path = self._image("a.png", "PNG")
        parser = ImageParser(max_file_size=path.stat().st_size - 1)
        with self.assertRaises(ValidationError):
            parser.parse(path)

    def test_pixel_limit(self):
        path = self._image("a.png", "PNG", size=(20, 20))
        parser = ImageParser(max_pixels=399)
        with self.assertRaises(ValidationError):
            parser.extract_metadata(path)
        self.assertEqual(ImageParser(max_pixels=400).extract_metadata(path).width, 20)

    def test_missing_file(self):
        with self.assertRaises(ValidationError):
            ImageParser().extract_metadata(self.dir / "missing.png")

    def test_preprocess_image_applies_limits(self):
        parser = ImageParser()
        with self.assertRaises(ValidationError):
            parser.preprocess_image(self._image("a.gif", "GIF"))
        output = parser.preprocess_image(self._image("a.png", "PNG"))
        self.assertTrue(output.exists())

    def test_extract_text_passes_checked_image_to_tesseract(self):
        fake_tesseract = MagicMock()
        fake_tesseract.image_to_string.return_value = "hello"
        fake_tesseract.image_to_data.return_value = {
            "level": [1],
            "conf": [90],
            "text": ["hello"],
            "left": [0],
            "top": [0],
            "width": [8],
            "height": [8],
        }
        with (
            patch("semantica.parse.image_parser.TESSERACT_AVAILABLE", True),
            patch("semantica.parse.image_parser.pytesseract", fake_tesseract),
        ):
            parser = ImageParser()
            with self.assertRaises(ValidationError):
                parser.extract_text(self._image("a.gif", "GIF"))
            fake_tesseract.image_to_string.assert_not_called()

            result = parser.extract_text(self._image("a.png", "PNG"))

        self.assertEqual(result.text, "hello")
        # Tesseract gets the opened image, never the path, so it can't
        # reopen the file with Pillow's full set of decoders.
        passed = fake_tesseract.image_to_string.call_args.args[0]
        self.assertIsInstance(passed, Image.Image)


if __name__ == "__main__":
    unittest.main()
