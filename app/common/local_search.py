"""Small, dependency-light search backend for public/demo deployments.

The normal local setup uses Elasticsearch.  Streamlit Community Cloud does not
provide a long-running Elasticsearch process, so this module loads the bundled
datasets and performs deterministic in-process retrieval instead.  It keeps
the same hit shape as Elasticsearch so the domain handlers do not need a
second answer-generation path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[2]
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def is_local_backend() -> bool:
    """Return whether retrieval should use the bundled files instead of ES."""
    configured = os.getenv("SEARCH_BACKEND", "").strip().lower()
    if configured in {"local", "memory", "offline"}:
        return True
    if configured in {"es", "elasticsearch"}:
        return False
    # A cloud deployment normally has no ES_URL.  Local development does.
    return not bool(os.getenv("ES_URL"))


def _hit(index: str, position: int, source: dict, score: float) -> dict:
    """Build the ES-compatible hit shape consumed by the existing handlers."""
    identifier = source.get("chunk_id") or source.get("name") or str(position)
    if index == "iit_contacts":
        identifier = hashlib.md5(str(identifier).encode("utf-8")).hexdigest()
    return {"_id": f"local-{index}-{identifier}", "_score": float(score), "_source": source}


def _academic_year(section: str) -> str | None:
    match = re.search(r"(20\d{2})[^\d]+(20\d{2}|\d{2})", section or "")
    if not match:
        return None
    first, second = match.groups()
    if len(second) == 2:
        second = first[:2] + second
    return f"{first}-{second}"


@lru_cache(maxsize=1)
def tuition_documents() -> tuple[dict, ...]:
    path = ROOT / "data" / "curated" / "tuition_fees.bulk.ndjson"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = []
    for i in range(0, len(lines), 2):
        doc = json.loads(lines[i + 1])
        item = doc.get("item") or ""
        section = doc.get("section") or ""
        item_lower = item.lower()
        normalized = dict(doc)
        normalized["fee_name"] = item or section or "Tuition"
        normalized["academic_year"] = _academic_year(section)
        if "full-time" in item_lower:
            normalized["enrollment"] = "full_time"
        elif "part-time" in item_lower:
            normalized["enrollment"] = "part_time"
        normalized["chunk_text"] = " ".join(
            str(value)
            for value in [
                doc.get("school"), doc.get("level"), section,
                item, doc.get("amount_text"), doc.get("unit"),
            ]
            if value
        )
        documents.append(normalized)
    return tuple(documents)


@lru_cache(maxsize=1)
def calendar_documents() -> tuple[dict, ...]:
    path = ROOT / "data" / "processed" / "calendar_chunks.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for doc in source:
        normalized = dict(doc)
        normalized["semantic_text"] = " ".join(
            str(value)
            for value in [
                doc.get("event_name"), doc.get("term"),
                doc.get("start_date"), doc.get("end_date"),
            ]
            if value
        )
        documents.append(normalized)
    return tuple(documents)


@lru_cache(maxsize=1)
def contact_documents() -> tuple[dict, ...]:
    path = ROOT / "data" / "raw" / "Contacts data.csv"
    field_map = {
        "Name": "name", "Department": "department", "Category": "category",
        "Description": "description", "Phone": "phone", "Fax": "fax",
        "Email": "email", "Building": "building", "Address": "address",
        "City": "city", "State": "state", "Zip": "zip", "Source_url": "source_url",
    }
    documents = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            doc = {
                target: row.get(source, "").strip()
                for source, target in field_map.items()
                if row.get(source, "").strip()
            }
            if doc.get("name"):
                documents.append(doc)
    return tuple(documents)


@lru_cache(maxsize=1)
def policy_documents() -> tuple[dict, ...]:
    path = ROOT / "data" / "processed" / "Unstructured data" / "Unstructured chunks k.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for doc in source:
        normalized = dict(doc)
        if "topic" not in normalized:
            normalized["topic"] = normalized.get("Topic", "")
        documents.append(normalized)
    return tuple(documents)


def local_tuition_filter_values() -> dict:
    docs = tuition_documents()
    schools = sorted({d.get("school") for d in docs if d.get("school")})
    fee_names = sorted({d.get("fee_name") for d in docs if d.get("fee_name")})
    programs = sorted({d.get("program") for d in docs if d.get("program")})
    years = sorted({d.get("academic_year") for d in docs if d.get("academic_year")})
    school_fees = {
        school: sorted({d.get("fee_name") for d in docs if d.get("school") == school and d.get("fee_name")})
        for school in schools
    }
    return {
        "schools": schools,
        "fee_names": fee_names,
        "programs": programs,
        "years": years,
        "school_fees": school_fees,
    }


def local_tuition_search(known: dict, top_k: int) -> list:
    """Apply the same hard/soft filter semantics as the ES tuition query."""
    if not any(known.get(key) for key in (
        "school", "level", "fee_name", "academic_year", "enrollment", "term", "unit", "program",
    )):
        return []

    hard_fields = ("school", "level", "fee_name", "academic_year", "unit", "program")
    soft_fields = ("enrollment", "term")
    hits = []
    for position, doc in enumerate(tuition_documents()):
        if any(known.get(key) and doc.get(key) != known[key] for key in hard_fields):
            continue
        # Sparse fields mirror the ES query: a missing value is allowed.
        if any(known.get(key) and doc.get(key) not in (None, "", known[key]) for key in soft_fields):
            continue
        hits.append(_hit("tuition_fees", position, doc, 1.0))
    return hits[:top_k]


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall((value or "").lower()))


def _date_overlap(doc: dict, year: int, month: int) -> bool:
    start = str(doc.get("start_date") or "")
    end = str(doc.get("end_date") or start)
    prefix = f"{year:04d}-{month:02d}-"
    return start.startswith(prefix) or end.startswith(prefix) or (start < f"{year:04d}-{month:02d}-32" and end > f"{year:04d}-{month:02d}-00")


def local_calendar_search(query: str, slots: dict, top_k: int) -> list:
    has_term = bool(slots.get("term") or slots.get("terms"))
    requested_terms = [slots["term"]] if slots.get("term") else slots.get("terms", [])
    requested_terms = {term.lower() for term in requested_terms}
    query_tokens = _tokens(query)
    candidates = []

    for position, doc in enumerate(calendar_documents()):
        term = str(doc.get("term") or "")
        if not slots.get("include_coursera") and "coursera" in term.lower():
            continue
        if requested_terms and term.lower() not in requested_terms:
            continue
        if slots.get("month") and not _date_overlap(doc, slots.get("year", 2026), slots["month"]):
            continue
        document_tokens = _tokens(doc.get("semantic_text", ""))
        overlap = len(query_tokens & document_tokens)
        score = overlap / max(len(query_tokens), 1)
        if overlap == 0:
            score = 0.01
        candidates.append((_hit("iit_calendar", position, doc, score), score))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    hits = [hit for hit, _ in candidates]
    if has_term:
        # Match Elasticsearch's event-name de-duplication.
        unique = []
        seen = set()
        for hit in hits:
            name = hit["_source"].get("event_name", "")
            if name not in seen:
                seen.add(name)
                unique.append(hit)
        return unique
    return hits[:top_k]


def local_contact_search(
    query: str,
    entity_type: str | None,
    name: str | None,
    top_k: int,
    exact_name: bool,
) -> list:
    office_categories = {"Administrative", "Center", "Institute", "Academic"}
    person_categories = {"Faculty", "Staff"}
    search_text = name or query
    query_tokens = _tokens(search_text)
    scored = []

    for position, doc in enumerate(contact_documents()):
        category = doc.get("category", "")
        if entity_type == "office" and category not in office_categories:
            continue
        if entity_type == "person" and category not in person_categories:
            continue

        name_text = doc.get("name", "")
        department = doc.get("department", "")
        description = doc.get("description", "")
        name_tokens = _tokens(name_text)
        searchable = _tokens(f"{name_text} {department} {description}")
        overlap = len(query_tokens & searchable) / max(len(query_tokens), 1)
        similarity = SequenceMatcher(None, search_text.lower(), name_text.lower()).ratio()
        score = max(overlap, similarity * 0.85)
        if search_text.lower() in name_text.lower() or name_text.lower() in search_text.lower():
            score = max(score, 0.95)
        if score >= 0.20:
            scored.append((_hit("iit_contacts", position, doc, score), score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    hits = [hit for hit, _ in scored[:top_k]]
    if not hits:
        return []
    if exact_name:
        return hits[:1]
    gate = 0.80 if entity_type == "office" else 0.60
    threshold = hits[0]["_score"] * gate
    return [hit for hit in hits if hit["_score"] >= threshold]


@lru_cache(maxsize=1)
def _policy_vector_index() -> tuple[TfidfVectorizer, object]:
    docs = policy_documents()
    texts = [f"{d.get('topic', '')} {d.get('content', '')}" for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    return vectorizer, vectorizer.fit_transform(texts)


def local_document_search(query: str, top_k: int, document_scope: str | None = None) -> list:
    """Retrieve policy chunks with TF-IDF cosine similarity and a scope boost."""
    docs = policy_documents()
    vectorizer, matrix = _policy_vector_index()
    scores = linear_kernel(vectorizer.transform([query]), matrix).ravel()
    scope_pattern = {
        "holds": "hold-information", "co_terminal": "co-terminal", "grades": "grade",
        "fees": "mandatory-and-other-fees", "transcripts": "transcripts",
        "commencement": "commencement", "hardship_withdrawal": "hardship-withdrawal",
        "course_withdrawal": "withdrawing", "course_repeats": "course-repeat",
        "coursera": "coursera", "registration": "registration",
        "student_handbook": "student%20handbook",
    }.get(document_scope or "")
    ranked = []
    for position, score in enumerate(scores):
        adjusted = float(score)
        if scope_pattern and scope_pattern in str(docs[position].get("source_url", "")).lower():
            adjusted += 0.15
        ranked.append((adjusted, position))
    ranked.sort(reverse=True)
    return [
        _hit("iit_policies", position, docs[position], score)
        for score, position in ranked[:top_k]
        if score > 0
    ]


def local_calendar_terms() -> list[str]:
    return sorted({d.get("term") for d in calendar_documents() if d.get("term")})


def local_calendar_event_tokens() -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "into", "students", "undergraduate", "online",
        "published", "observed", "observance", "semester", "coursera", "university",
        "schedule", "schedules", "charges", "late", "early", "last", "starts", "begins",
    }
    words = set()
    for doc in calendar_documents():
        words.update(word for word in re.findall(r"\b[a-z]{3,}\b", str(doc.get("event_name", "")).lower()) if word not in stopwords)
    return sorted(words)

