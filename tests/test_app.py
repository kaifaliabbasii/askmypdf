"""Automatic tests for AskMyPDF.

The real AI is replaced by a fake one, and the test PDFs are generated in code,
so the tests need no API key, no internet and no sample files. Run:  pytest
"""
import io
from types import SimpleNamespace

import pytest

import app as app_module
import rag


# ---------- helpers ----------
def make_pdf(pages_text):
    """Build a tiny but valid text PDF from a list of page texts (no parentheses in text)."""
    n = len(pages_text)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages_text):
        objs.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return out


PAGES = [
    "Photosynthesis is the process plants use to turn sunlight into energy.",
    "The capital of France is Paris and it is famous for the Eiffel Tower.",
    "Python is a programming language created by Guido van Rossum in 1991.",
]


class FakeCompletions:
    def __init__(self):
        self.last_messages = None

    def create(self, **kwargs):
        self.last_messages = kwargs["messages"]
        message = SimpleNamespace(content="Paris is the capital (p. 2).")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def client(monkeypatch):
    app_module.app.config["TESTING"] = True
    app_module._requests.clear()
    app_module._docs.clear()
    fake = FakeClient()
    monkeypatch.setattr(app_module, "client", fake)
    c = app_module.app.test_client()
    c.fake = fake
    return c


def upload(client, data, name="test.pdf"):
    return client.post("/upload", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


@pytest.fixture
def doc_id(client):
    res = upload(client, make_pdf(PAGES))
    assert res.status_code == 200
    return res.get_json()["doc_id"]


# ---------- rag.py unit tests ----------
def test_extract_pages():
    pages = rag.extract_pages(make_pdf(PAGES))
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "Paris" in pages[1][1]


def test_extract_rejects_garbage():
    with pytest.raises(ValueError):
        rag.extract_pages(b"%PDF-1.4 this is not really a pdf")


def test_extract_page_limit():
    with pytest.raises(ValueError):
        rag.extract_pages(make_pdf(PAGES), max_pages=2)


def test_chunking_keeps_page_numbers_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = rag.chunk_pages([(7, text)], size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c["page"] == 7 for c in chunks)
    assert all(len(c["text"]) <= 300 for c in chunks)
    assert "word0" in chunks[0]["text"] and "word399" in chunks[-1]["text"]


def test_search_finds_the_right_chunk():
    index = rag.Index(rag.chunk_pages(rag.extract_pages(make_pdf(PAGES))))
    best = index.search("Who created Python?", k=1)[0]
    assert best["page"] == 3


def test_search_with_no_match_returns_nothing():
    index = rag.Index(rag.chunk_pages(rag.extract_pages(make_pdf(PAGES))))
    assert index.search("quantum entanglement", k=3) == []


def test_summary_question_detection():
    assert rag.is_summary_question("Please summarize this")
    assert not rag.is_summary_question("Who created Python?")


# ---------- web tests ----------
def test_home_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").get_json()["status"] == "ok"


def test_upload_ok(client):
    res = upload(client, make_pdf(PAGES))
    body = res.get_json()
    assert res.status_code == 200
    assert body["pages"] == 3 and body["doc_id"]


def test_upload_rejects_wrong_extension(client):
    assert upload(client, make_pdf(PAGES), name="notes.txt").status_code == 400


def test_upload_rejects_fake_pdf(client):
    assert upload(client, b"hello, I am not a pdf", name="fake.pdf").status_code == 400


def test_upload_rejects_damaged_pdf(client):
    assert upload(client, b"%PDF-1.4 broken", name="broken.pdf").status_code == 400


def test_upload_without_file(client):
    assert client.post("/upload", data={}).status_code == 400


def test_ask_success_with_sources(client, doc_id):
    res = client.post("/ask", json={"doc_id": doc_id, "question": "What is the capital of France?"})
    body = res.get_json()
    assert res.status_code == 200
    assert "Paris" in body["reply"]
    assert body["sources"][0]["page"] == 2
    # only the matching text was sent to the AI, and the system prompt came first
    sent = client.fake.completions.last_messages
    assert sent[0]["role"] == "system"
    assert "Eiffel" in sent[-1]["content"]
    assert "Guido" not in sent[-1]["content"]


def test_ask_no_match_does_not_call_the_ai(client, doc_id):
    client.fake.completions.last_messages = None
    res = client.post("/ask", json={"doc_id": doc_id, "question": "quantum entanglement"})
    assert res.status_code == 200
    assert "couldn't find" in res.get_json()["reply"]
    assert client.fake.completions.last_messages is None


def test_ask_summary_uses_whole_document(client, doc_id):
    res = client.post("/ask", json={"doc_id": doc_id, "question": "Summarize this document"})
    assert res.status_code == 200
    assert {s["page"] for s in res.get_json()["sources"]} == {1, 2, 3}


def test_ask_validation(client, doc_id):
    assert client.post("/ask", json={"doc_id": doc_id, "question": "  "}).status_code == 400
    too_long = "x" * (app_module.MAX_QUESTION_CHARS + 1)
    assert client.post("/ask", json={"doc_id": doc_id, "question": too_long}).status_code == 400


def test_ask_unknown_document(client):
    res = client.post("/ask", json={"doc_id": "nope", "question": "hi there"})
    assert res.status_code == 404


def test_ask_missing_api_key(client, doc_id, monkeypatch):
    monkeypatch.setattr(app_module, "client", None)
    res = client.post("/ask", json={"doc_id": doc_id, "question": "capital of France"})
    assert res.status_code == 500 and "API_KEY" in res.get_json()["error"]


def test_rate_limit(client, doc_id, monkeypatch):
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MIN", 2)
    q = {"doc_id": doc_id, "question": "capital of France"}
    assert client.post("/ask", json=q).status_code == 200
    assert client.post("/ask", json=q).status_code == 200
    assert client.post("/ask", json=q).status_code == 429


def test_unexpected_ai_error_returns_json(client, doc_id):
    def boom(**kwargs):
        raise RuntimeError("boom")

    client.fake.completions.create = boom
    res = client.post("/ask", json={"doc_id": doc_id, "question": "capital of France"})
    assert res.status_code == 500 and "error" in res.get_json()


def test_document_store_is_capped(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_DOCS", 2)
    ids = [upload(client, make_pdf(PAGES)).get_json()["doc_id"] for _ in range(3)]
    assert app_module.get_doc(ids[0]) is None  # oldest was removed
    assert app_module.get_doc(ids[2]) is not None
