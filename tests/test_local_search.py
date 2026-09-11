"""Offline retrieval tests used by GitHub Actions and local development."""

from app.common.local_search import (
    calendar_documents,
    contact_documents,
    is_local_backend,
    local_calendar_search,
    local_contact_search,
    local_document_search,
    local_tuition_search,
    policy_documents,
    tuition_documents,
)


def test_bundled_datasets_load_with_expected_sizes():
    assert len(tuition_documents()) == 130
    assert len(calendar_documents()) == 118
    assert len(contact_documents()) == 1552
    assert len(policy_documents()) == 599


def test_local_backend_detection(monkeypatch):
    monkeypatch.setenv("SEARCH_BACKEND", "local")
    assert is_local_backend() is True

    monkeypatch.setenv("SEARCH_BACKEND", "es")
    assert is_local_backend() is False


def test_tuition_search_returns_es_compatible_hits():
    hits = local_tuition_search({"school": "Chicago-Kent", "level": "graduate"}, 5)
    assert hits
    assert all("_source" in hit and "amount_value" in hit["_source"] for hit in hits)


def test_calendar_search_finds_fall_start_event():
    hits = local_calendar_search(
        "When does the Fall 2026 semester start?",
        {"term": "Fall 2026"},
        5,
    )
    assert any(
        hit["_source"].get("start_date") == "2026-08-17"
        for hit in hits
    )


def test_contact_search_finds_financial_aid():
    hits = local_contact_search("Financial Aid", "office", "Financial Aid", 3, True)
    assert len(hits) == 1
    assert hits[0]["_source"]["name"] == "Financial Aid"


def test_policy_search_returns_withdrawal_sources():
    hits = local_document_search("withdraw from a course", 7, "course_withdrawal")
    assert hits
    assert all(hit["_source"].get("source_url") for hit in hits)
