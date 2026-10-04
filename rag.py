"""The "RAG" part of AskMyPDF: read a PDF, split it into chunks, find the best chunks.

RAG = Retrieval-Augmented Generation:
    1. Retrieval:  find the parts of the document that match the question
    2. Generation: give ONLY those parts to the AI so it answers from them

This file has no web or AI code, so it is easy to read and to test.
"""
import io
import math
import re
from collections import Counter

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Very common words are ignored when searching because they match everything.
STOPWORDS = set(
    """a an the and or but if of to in on at by for with from as is are was were be been
    being this that these those it its i you he she we they them his her our your their
    do does did done can could should would will shall may might must not no yes so than
    then there here what which who whom whose when where why how about into over under
    again very just also any all each more most other some such only own same too s t""".split()
)

# Words that mean "give me an overview" - these questions have no good keywords.
SUMMARY_WORDS = {"summarize", "summarise", "summary", "overview", "outline", "gist", "tldr"}


def extract_pages(data, max_pages=100):
    """Return a list of (page_number, text) for a PDF given as bytes.

    Raises ValueError with a friendly message if the PDF cannot be used.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty password just to block copying.
            if not reader.decrypt(""):
                raise ValueError("This PDF is password protected.")
        total = len(reader.pages)
        if total == 0:
            raise ValueError("This PDF has no pages.")
        if total > max_pages:
            raise ValueError(f"This PDF has {total} pages. The maximum is {max_pages}.")

        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if text:
                pages.append((number, text))
    except ValueError:
        raise
    except (PdfReadError, Exception) as e:  # damaged file, unsupported format, etc.
        raise ValueError("Could not read this PDF. The file may be damaged.") from e

    if not pages:
        raise ValueError(
            "No selectable text found. This looks like a scanned PDF (images only)."
        )
    return pages


def chunk_pages(pages, size=900, overlap=150):
    """Cut every page into overlapping pieces of about `size` characters.

    Small pieces make the search precise and keep the AI request cheap.
    The overlap stops a sentence from being cut in half and lost.
    """
    chunks = []
    for number, text in pages:
        if len(text) <= size:
            chunks.append({"page": number, "text": text})
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):  # try to end on a space, not in the middle of a word
                space = text.rfind(" ", start + size // 2, end)
                if space != -1:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": number, "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def tokenize(text):
    """Lower-case words without stopwords, e.g. 'The Cat sat' -> ['cat', 'sat']."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class Index:
    """A small search engine over the chunks, using the BM25 scoring formula.

    BM25 gives a higher score to chunks that contain the question's words,
    especially rare words, and it does not favour very long chunks.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks):
        self.chunks = chunks
        self.term_counts = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = [sum(tc.values()) for tc in self.term_counts]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        self.doc_freq = Counter()
        for tc in self.term_counts:
            self.doc_freq.update(tc.keys())

    def _score(self, query_terms, i):
        score = 0.0
        n = len(self.chunks)
        for term in query_terms:
            tf = self.term_counts[i].get(term, 0)
            if not tf:
                continue
            df = self.doc_freq[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = tf + self.K1 * (1 - self.B + self.B * self.lengths[i] / (self.avg_length or 1))
            score += idf * tf * (self.K1 + 1) / norm
        return score

    def search(self, question, k=4):
        """Return the k best chunks for the question (best first). Empty if nothing matches."""
        terms = list(dict.fromkeys(tokenize(question)))  # remove duplicates, keep order
        if not terms:
            return []
        scored = [(self._score(terms, i), i) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]

    def overview(self, k=4):
        """Return k chunks spread evenly across the document (used for summaries)."""
        n = len(self.chunks)
        if n <= k:
            return list(self.chunks)
        step = n / k
        return [self.chunks[int(i * step)] for i in range(k)]


def is_summary_question(question):
    return bool(SUMMARY_WORDS & set(re.findall(r"[a-z]+", question.lower())))
