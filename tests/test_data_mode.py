"""Input provenance is explicit for both new and cached reporting inputs."""
import pandas as pd
import pytest
from src.data import build_interactions as builder
from src.eval import pipeline


def test_real_data_is_required_and_fixture_has_explicit_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "OULAD_DIR", str(tmp_path))
    items = tmp_path / "items.csv"
    pd.DataFrame({"item_id": ["example"]}).to_csv(items, index=False)
    monkeypatch.setattr(builder, "ITEMS_PATH", str(items))
    monkeypatch.setattr(builder, "OUTPUT_PATH", str(tmp_path / "interactions.csv"))
    with pytest.raises(FileNotFoundError, match="OULAD"):
        builder.build_interactions()
    assert not builder.build_interactions(data_mode="synthetic").empty


def test_cached_data_without_provenance_cannot_be_used_as_real(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_PROCESSED", str(tmp_path))
    for filename in pipeline.REQUIRED_DATA_FILES:
        (tmp_path / filename).write_text("example\n")
    with pytest.raises(ValueError, match="source record"):
        pipeline.run_pipeline("unused.yaml", str(tmp_path / "out"))
