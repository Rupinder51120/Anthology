import pytest
import json
import os
from pathlib import Path
import src.ingestion.utils as utils

@pytest.fixture
def mock_paths(monkeypatch, tmp_path):
    """
    Override the global paths in utils.py to use a temporary directory.
    """
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()
    ckpt_file = str(idx_dir / "checkpoint.json")
    
    monkeypatch.setattr(utils, "INDEXES_DIR", str(idx_dir))
    monkeypatch.setattr(utils, "CHECKPOINT_PATH", ckpt_file)
    return idx_dir

class TestPreserveMath:
    @pytest.mark.parametrize("input_text, expected", [
        (r"The result is $\sum_{i=1}^n i$", "The result is [MATH]\\sum_{i=1}^n i[/MATH]"),
        (r"Eq: $$E=mc^2$$", "Eq: [MATH_BLOCK]E=mc^2[/MATH_BLOCK]"),
        (r"Using \frac{1}{2}", "Using [MATH]\\frac{1}{2}[/MATH]"),
        (r"Normal text", "Normal text"),
    ])
    def test_math_patterns(self, input_text, expected):
        assert utils.preserve_math(input_text) == expected

class TestCheckpoint:
    def test_save_and_load_valid(self, mock_paths):
        data = {"processed_files": ["paper1.pdf"], "last_run": "some-date"}
        utils.save_checkpoint(data)
        
        loaded = utils.load_checkpoint()
        assert loaded["processed_files"] == ["paper1.pdf"]
        assert "last_run" in loaded

    def test_load_missing_file(self, mock_paths):
        # Path does not exist (mock_paths is clean)
        loaded = utils.load_checkpoint()
        assert loaded == {"processed_files": [], "last_run": None}

    def test_corrupted_json_recovery(self, mock_paths):
        with open(utils.CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            f.write("{ 'invalid': json }")
            
        loaded = utils.load_checkpoint()
        assert loaded == {"processed_files": [], "last_run": None}

    def test_invalid_schema_recovery(self, mock_paths):
        with open(utils.CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)
            
        loaded = utils.load_checkpoint()
        assert loaded == {"processed_files": [], "last_run": None}

    def test_missing_key_recovery(self, mock_paths):
        with open(utils.CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_run": "2024-01-01"}, f)
            
        loaded = utils.load_checkpoint()
        assert "processed_files" in loaded
        assert loaded["processed_files"] == []

    def test_utf8_handling(self, mock_paths):
        data = {"processed_files": ["paper_∑.pdf"]}
        utils.save_checkpoint(data)
        
        loaded = utils.load_checkpoint()
        assert "paper_∑.pdf" in loaded["processed_files"]

class TestFilterChunks:
    def test_standard_filtering(self):
        chunks = [
            {"text": "Good chunk", "metadata": {}},
            {"text": "Bad chunk", "metadata": {}},
        ]
        score_fn = lambda c: 0.9 if "Good" in c["text"] else 0.1
        
        kept, stats = utils.filter_chunks(chunks, score_fn, min_score=0.5)
        
        assert len(kept) == 1
        assert kept[0]["text"] == "Good chunk"
        assert stats["total"] == 2
        assert stats["kept"] == 1
        assert stats["removal_rate"] == "50.0%"

    def test_metadata_safety(self):
        chunks = [{"text": "No metadata here"}]
        score_fn = lambda c: 1.0
        
        kept, _ = utils.filter_chunks(chunks, score_fn)
        
        assert "metadata" in kept[0]
        assert kept[0]["metadata"]["quality_score"] == 1.0

    def test_empty_chunks(self):
        score_fn = lambda c: 1.0
        kept, stats = utils.filter_chunks([], score_fn)
        
        assert kept == []
        assert stats["total"] == 0
        assert stats["removal_rate"] == "0%"
