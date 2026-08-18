"""
Regression tests for the upload path-traversal fix (audit BUG-04):
api/routers/papers.py used unsanitized `file.filename` directly in a
filesystem write path (`Path("data/papers") / file.filename`), allowing a
crafted filename to write outside the intended upload directory.
"""
from pathlib import Path

import pytest

from api.routers.papers import UPLOAD_DIR, _safe_pdf_filename


class TestSafePdfFilename:
    def test_normal_filename_passes_through(self):
        assert _safe_pdf_filename("my_paper.pdf") == "my_paper.pdf"

    def test_normal_filename_with_spaces_and_unicode(self):
        assert _safe_pdf_filename("Attention Is All You Need (2017).pdf") == \
            "Attention Is All You Need (2017).pdf"

    def test_relative_traversal_is_reduced_to_basename(self):
        assert _safe_pdf_filename("../../../etc/evil.pdf") == "evil.pdf"

    def test_deeply_nested_traversal(self):
        assert _safe_pdf_filename("a/b/../../../../../c/d/evil.pdf") == "evil.pdf"

    def test_absolute_path_is_reduced_to_basename(self):
        assert _safe_pdf_filename("/etc/passwd.pdf") == "passwd.pdf"

    def test_absolute_path_outside_pdf_gets_extension_appended(self):
        assert _safe_pdf_filename("/etc/cron.d/malicious") == "malicious.pdf"

    def test_nested_path_without_traversal(self):
        assert _safe_pdf_filename("subdir/nested/paper.pdf") == "paper.pdf"

    def test_bare_dotdot_falls_back_to_generated_name(self):
        name = _safe_pdf_filename("..")
        assert name != ".."
        assert name.endswith(".pdf")

    def test_empty_filename_falls_back_to_generated_name(self):
        name = _safe_pdf_filename("")
        assert name.endswith(".pdf")
        assert len(name) > len(".pdf")

    def test_none_filename_falls_back_to_generated_name(self):
        name = _safe_pdf_filename(None)
        assert name.endswith(".pdf")

    def test_windows_style_separators_do_not_escape(self):
        # Backslash isn't a POSIX path separator, but must never be treated
        # as one -- confirm it can't be used to smuggle a traversal segment.
        name = _safe_pdf_filename("..\\..\\evil.pdf")
        assert "/" not in name
        joined = (UPLOAD_DIR / name).resolve()
        assert UPLOAD_DIR.resolve() in joined.parents


class TestUploadDestinationConfinement:
    """
    Mirrors the exact dest-path construction + confinement check used in
    upload_paper() (api/routers/papers.py) without needing the full FastAPI
    app / DB / ingestion pipeline wired up.
    """

    @pytest.mark.parametrize("malicious_filename", [
        "../../../../../../tmp/evil.pdf",
        "/etc/passwd.pdf",
        "../../etc/cron.d/evil",
        "a/../../../b/../../c/evil.pdf",
        "....//....//evil.pdf",
    ])
    def test_malicious_filenames_stay_confined_to_upload_dir(self, malicious_filename):
        safe_name = _safe_pdf_filename(malicious_filename)
        dest = (UPLOAD_DIR / safe_name).resolve()
        assert UPLOAD_DIR.resolve() in dest.parents

    def test_normal_filename_stays_confined_to_upload_dir(self):
        safe_name = _safe_pdf_filename("normal_paper.pdf")
        dest = (UPLOAD_DIR / safe_name).resolve()
        assert dest.parent == UPLOAD_DIR.resolve()
        assert dest.name == "normal_paper.pdf"
