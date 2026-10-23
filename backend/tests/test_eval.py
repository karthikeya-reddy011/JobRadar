import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from label_jobs import _candidate_jobs, _candidate_pool, _render_pair, _sample_candidate_pool
from eval_utils import eligibility_summary
from eval_utils import (
    SNAPSHOT_VERSION,
    load_snapshot,
    snapshot_digest,
    uses_role_only_candidate_sources,
)
from run_eval import evaluate, load_labels, ndcg_at_k, precision_at_k


def test_precision_at_5_counts_partial_relevance_as_relevant():
    assert precision_at_k([2, 0, 1], 5) == pytest.approx(0.4)


def test_ndcg_at_10_uses_graded_relevance():
    assert ndcg_at_k([2, 1, 0], 10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2], 10) < 1.0


def test_evaluate_ranks_keywords_and_averages_profiles():
    snapshot = {
        "profiles": {"me": {"skills": ["Python"], "titles": ["Data Engineer"]}},
        "jobs": {
            "1": {"title": "Designer", "description": "Design"},
            "2": {"title": "Data Engineer", "description": "Python systems"},
        },
        "candidate_pool": {"me": ["1", "2"]},
    }
    labels = {("me", "1"): 0, ("me", "2"): 2}

    result = evaluate(snapshot, labels)

    baseline = result["role_relevance_keyword_baseline"]
    assert baseline["profiles"]["me"]["precision@5"] == pytest.approx(0.2)
    assert baseline["profiles"]["me"]["nDCG@10"] == pytest.approx(1.0)
    assert baseline["macro_average"] == baseline["profiles"]["me"]
    assert result["label_distribution"] == {"0": 1, "1": 0, "2": 1}


def test_candidate_pool_mixes_role_matches_and_random_jobs_deterministically():
    profile = {
        "skills": ["Python"],
        "titles": ["Analyst"],
        "locations": ["Seattle"],
        "remote_ok": False,
    }
    jobs = {
        "1": {"title": "Engineer", "description": "Python systems", "location": "Austin"},
        "2": {"title": "Analyst", "description": "SQL", "location": "Austin"},
        "3": {"title": "Coordinator", "description": "Operations", "location": "Seattle"},
        "4": {"title": "Designer", "description": "Visual design", "location": "Miami"},
    }

    selected = _candidate_pool(profile, jobs, size=4, seed=11)

    assert selected == _candidate_pool(profile, jobs, size=4, seed=11)
    assert selected[:2] == ["1", "2"]
    assert set(selected) == set(jobs)
    _, sources = _sample_candidate_pool(profile, jobs, size=4, seed=11)
    assert set(sources.values()) == {"skill_keyword", "title_match", "random"}
    assert "location_compatible" not in sources.values()


def test_candidate_pool_is_independent_of_location_and_remote_preferences():
    jobs = {
        "1": {"title": "Engineer", "description": "Python", "location": "Seattle"},
        "2": {"title": "Designer", "description": "Design", "location": "Austin"},
        "3": {"title": "Analyst", "description": "Data", "location": "Remote"},
    }
    profile = {"skills": ["Python"], "titles": [], "locations": ["Seattle"], "remote_ok": False}
    changed_logistics = {
        **profile,
        "locations": ["Miami"],
        "remote_ok": True,
    }

    assert _candidate_pool(profile, jobs, size=3, seed=11) == _candidate_pool(
        changed_logistics, jobs, size=3, seed=11
    )


def test_role_relevance_metrics_do_not_change_with_eligibility_fields():
    snapshot = {
        "profiles": {
            "me": {
                "skills": ["Python"],
                "titles": [],
                "locations": ["Seattle"],
                "remote_ok": False,
                "years_experience": 2,
            }
        },
        "jobs": {
            "1": {
                "title": "Engineer",
                "description": "Python role",
                "location": "Seattle",
                "is_remote": False,
            },
            "2": {
                "title": "Engineer",
                "description": "No matching skills",
                "location": "Austin",
                "is_remote": True,
            },
        },
        "candidate_pool": {"me": ["1", "2"]},
    }
    labels = {("me", "1"): 2, ("me", "2"): 0}
    first = evaluate(snapshot, labels)["role_relevance_keyword_baseline"]

    snapshot["profiles"]["me"]["locations"] = ["Austin"]
    snapshot["profiles"]["me"]["remote_ok"] = True
    snapshot["profiles"]["me"]["years_experience"] = 20
    snapshot["jobs"]["1"]["location"] = "Remote"
    snapshot["jobs"]["1"]["is_remote"] = True
    second = evaluate(snapshot, labels)["role_relevance_keyword_baseline"]

    assert first == second


def test_label_display_hides_scores_sources_and_extracted_job_skills():
    profile = {"skills": ["Python"], "titles": [], "locations": []}
    job = {
        "title": "Engineer",
        "description": "Build systems.",
        "location": "Hidden location",
        "is_remote": True,
        "seniority": "senior",
        "source": "secret-source",
        "skills": ["secret-extracted-skill"],
        "score": 99,
    }

    rendered = _render_pair("me", profile, job)

    assert "Engineer" in rendered
    assert "secret-source" not in rendered
    assert "secret-extracted-skill" not in rendered
    assert "Hidden location" not in rendered
    assert "senior" not in rendered
    assert "99" not in rendered


def test_eligibility_filters_are_reported_separately_from_relevance_metrics():
    snapshot = {
        "profiles": {
            "me": {
                "locations": ["Seattle"],
                "remote_ok": False,
                "years_experience": 4,
                "skills": ["Python"],
                "titles": [],
            }
        },
        "jobs": {
            "1": {
                "title": "Engineer",
                "description": "Python",
                "location": "Seattle",
                "is_remote": False,
            },
            "2": {
                "title": "Engineer",
                "description": "Python",
                "location": "Austin",
                "is_remote": False,
            },
        },
        "candidate_pool": {"me": ["1", "2"]},
    }
    labels = {("me", "1"): 2, ("me", "2"): 0}

    result = evaluate(snapshot, labels)
    filters = result["eligibility_filter_summary"]["me"]

    assert filters["criteria"]["location_remote"] == {
        "compatible": 1,
        "incompatible": 1,
        "unknown": 0,
    }
    assert filters["criteria"]["experience"]["unknown"] == 2
    assert filters["not_excluded_by_known_filters"] == 1
    assert result["role_relevance_keyword_baseline"]["profiles"]["me"]["nDCG@10"] == 1.0


def test_load_labels_rejects_unlabeled_pairs(tmp_path):
    snapshot = {
        "snapshot_id": "frozen",
        "candidate_pool": {"me": ["1", "2"]},
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        '{"snapshot_id":"frozen","labels":[{"profile":"me","job_id":1,"label":2}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unlabeled"):
        load_labels(labels_path, snapshot)


def test_load_snapshot_rejects_changed_snapshot_content(tmp_path):
    snapshot = {
        "version": 1,
        "created_at": "2026-10-20T00:00:00+00:00",
        "profiles": {"me": {}},
        "jobs": {"1": {"title": "Original"}},
        "candidate_pool": {"me": ["1"]},
    }
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    snapshot["jobs"]["1"]["title"] = "Modified"
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="changed"):
        load_snapshot(path)


def test_version_two_snapshot_requires_source_for_each_candidate(tmp_path):
    snapshot = {
        "version": 2,
        "created_at": "2026-10-20T00:00:00+00:00",
        "profiles": {"me": {}},
        "jobs": {"1": {"title": "Engineer"}},
        "candidate_pool": {"me": ["1"]},
        "candidate_sources": {"me": {"1": "random"}},
    }
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    assert load_snapshot(path)["candidate_sources"]["me"]["1"] == "random"

    snapshot["candidate_sources"]["me"] = {}
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="source metadata"):
        load_snapshot(path)


def test_new_snapshots_use_version_three_and_role_only_provenance():
    assert SNAPSHOT_VERSION == 3
    assert uses_role_only_candidate_sources({
        "candidate_sources": {"me": {"1": "skill_keyword", "2": "random"}}
    })
    assert not uses_role_only_candidate_sources({
        "candidate_sources": {"me": {"1": "location_compatible"}}
    })


def test_version_three_snapshot_rejects_logistics_based_sources(tmp_path):
    snapshot = {
        "version": 3,
        "created_at": "2026-10-20T00:00:00+00:00",
        "profiles": {"me": {}},
        "jobs": {"1": {"title": "Engineer"}},
        "candidate_pool": {"me": ["1"]},
        "candidate_sources": {"me": {"1": "location_compatible"}},
    }
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="non-role candidate source"):
        load_snapshot(path)


def test_snapshot_keeps_only_jobs_in_candidate_pools():
    jobs = {
        "1": {"title": "Sampled"},
        "2": {"title": "Not sampled"},
        "3": {"title": "Shared candidate"},
    }

    result = _candidate_jobs(jobs, {"me": ["1", "3"], "other": ["3"]})

    assert result == {"1": jobs["1"], "3": jobs["3"]}
