"""Checks for the reproducible site-model result and served bundle."""

import hashlib
import json
from pathlib import Path

import joblib


def _sha256(path: Path) -> str:
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_site_report_matches_its_source_files():
    report = json.loads(Path("reports/site_model_evaluation.json").read_text(encoding="utf-8"))
    source = report["source"]

    assert not Path(source["training_script"]).is_absolute()
    assert source["training_script_sha256"] == _sha256(Path(source["training_script"]))
    assert source["feature_script_sha256"] == _sha256(Path("src/features/build_features_v3.py"))
    assert source["timeseries_script_sha256"] == _sha256(Path("src/features/make_time_series.py"))


def test_site_method_selection_only_uses_development_metrics():
    report = json.loads(Path("reports/site_model_evaluation.json").read_text(encoding="utf-8"))

    for site in report["dataset"]["sites"]:
        selection = report["sites"][site]["final_holdout"]["selection"]
        development = selection["development_mean_mae_kw"]
        expected = "residual_recent" if development["residual_model"] < development["baseline"] else "calendar_baseline"
        assert selection["method"] == expected

    aggregate = report["aggregate_holdout"]
    assert aggregate["selected_method"]["mae_kw"] < aggregate["baseline"]["mae_kw"]


def test_served_bundle_contains_all_selected_sites():
    report = json.loads(Path("reports/site_model_evaluation.json").read_text(encoding="utf-8"))
    bundle = joblib.load(report["served_bundle"]["path"])

    assert bundle["sites"] == report["dataset"]["sites"]
    assert bundle["methods"] == {
        site: report["sites"][site]["final_holdout"]["selection"]["method"] for site in report["dataset"]["sites"]
    }
