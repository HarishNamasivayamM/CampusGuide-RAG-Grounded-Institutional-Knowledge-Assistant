#!/usr/bin/env python3
"""Create and load the structured local data indexes used by the chatbot."""

import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "")
VERIFY = os.getenv("ES_VERIFY_CERTS", "false").lower() in {"1", "true", "yes", "on"}

es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASS) if ES_USER else None,
    verify_certs=VERIFY,
    ssl_show_warn=VERIFY,
)


def recreate_index(index: str, body: dict) -> None:
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, body=body)


def bulk_index(index: str, docs: list[dict]) -> None:
    actions = [
        {"_index": index, "_id": str(i), "_source": doc}
        for i, doc in enumerate(docs)
    ]
    success, errors = helpers.bulk(
        es,
        actions,
        chunk_size=200,
        request_timeout=120,
        raise_on_error=False,
    )
    if errors:
        raise RuntimeError(f"{len(errors)} documents failed while loading {index}")
    es.indices.refresh(index=index)
    print(f"[CHECK] {index}: {success} documents")


def tuition_year(section: str) -> str | None:
    match = re.search(r"(20\d{2})[^\d]+(20\d{2}|\d{2})", section or "")
    if not match:
        return None
    first, second = match.groups()
    if len(second) == 2:
        second = first[:2] + second
    return f"{first}-{second}"


def load_tuition() -> list[dict]:
    path = ROOT / "data" / "curated" / "tuition_fees.bulk.ndjson"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = []
    for i in range(0, len(lines), 2):
        doc = json.loads(lines[i + 1])
        item = doc.get("item") or ""
        section = doc.get("section") or ""
        item_lower = item.lower()
        normalized = dict(doc)
        normalized["fee_name"] = item or section or "Tuition"
        normalized["academic_year"] = tuition_year(section)
        if "full-time" in item_lower:
            normalized["enrollment"] = "full_time"
        elif "part-time" in item_lower:
            normalized["enrollment"] = "part_time"
        normalized["chunk_text"] = " ".join(
            str(value) for value in [
                doc.get("school"), doc.get("level"), section,
                item, doc.get("amount_text"), doc.get("unit"),
            ] if value
        )
        docs.append(normalized)
    return docs


def load_calendar() -> list[dict]:
    path = ROOT / "data" / "processed" / "calendar_chunks.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for doc in source:
        normalized = dict(doc)
        normalized["semantic_text"] = " ".join(
            str(value) for value in [
                doc.get("event_name"), doc.get("term"),
                doc.get("start_date"), doc.get("end_date"),
            ] if value
        )
        docs.append(normalized)
    return docs


def load_contacts() -> list[dict]:
    path = ROOT / "data" / "raw" / "Contacts data.csv"
    field_map = {
        "Name": "name", "Department": "department", "Category": "category",
        "Description": "description", "Phone": "phone", "Fax": "fax",
        "Email": "email", "Building": "building", "Address": "address",
        "City": "city", "State": "state", "Zip": "zip", "Source_url": "source_url",
    }
    docs = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            doc = {
                target: row.get(source, "").strip()
                for source, target in field_map.items()
                if row.get(source, "").strip()
            }
            if doc.get("name"):
                docs.append(doc)
    return docs


TUITION_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "school": {"type": "keyword"}, "level": {"type": "keyword"},
        "category": {"type": "keyword"}, "section": {"type": "text"},
        "item": {"type": "text"}, "fee_name": {"type": "keyword"},
        "amount_text": {"type": "text"}, "amount_value": {"type": "float"},
        "currency": {"type": "keyword"}, "unit": {"type": "keyword"},
        "source_url": {"type": "keyword"}, "last_verified_at": {"type": "date"},
        "academic_year": {"type": "keyword"}, "term": {"type": "keyword"},
        "enrollment": {"type": "keyword"}, "program": {"type": "keyword"},
        "chunk_text": {"type": "text"},
    }},
}

CALENDAR_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "event_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "term": {"type": "keyword"}, "start_date": {"type": "date"},
        "end_date": {"type": "date"}, "source_urls": {"type": "keyword"},
        "semantic_text": {"type": "text"},
    }},
}

CONTACTS_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "department": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "category": {"type": "keyword"}, "description": {"type": "text"},
        "phone": {"type": "keyword"}, "fax": {"type": "keyword"},
        "email": {"type": "keyword"}, "building": {"type": "text"},
        "address": {"type": "text"}, "city": {"type": "keyword"},
        "state": {"type": "keyword"}, "zip": {"type": "keyword"},
        "source_url": {"type": "keyword"},
    }},
}


def main() -> None:
    es.info()
    datasets = [
        ("tuition_fees", TUITION_MAPPING, load_tuition),
        ("iit_calendar", CALENDAR_MAPPING, load_calendar),
        ("iit_contacts", CONTACTS_MAPPING, load_contacts),
    ]
    for index, mapping, loader in datasets:
        docs = loader()
        recreate_index(index, mapping)
        bulk_index(index, docs)


if __name__ == "__main__":
    main()
