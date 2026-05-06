from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RagDocument:
    source_id: str
    path: str
    trust: str
    text: str


@dataclass
class RagHit:
    source_id: str
    trust: str
    score: int
    snippet: str


class SimpleRagIndex:
    """Dependency-free keyword RAG for repeatable security tests.

    The goal is not semantic retrieval quality; the goal is deterministic retrieval of
    trusted and untrusted documents that may contain indirect prompt injection text.
    """

    def __init__(self, rag_dir: Path) -> None:
        self.rag_dir = Path(rag_dir)
        self.documents = self._load_documents()

    def _load_documents(self) -> list[RagDocument]:
        docs: list[RagDocument] = []
        for trust in ["trusted", "untrusted"]:
            folder = self.rag_dir / trust
            if not folder.exists():
                continue
            for path in sorted(folder.glob("**/*")):
                if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                    docs.append(
                        RagDocument(
                            source_id=f"{trust}/{path.name}",
                            path=str(path),
                            trust=trust,
                            text=path.read_text(encoding="utf-8", errors="replace"),
                        )
                    )
        return docs

    def search(self, query: str, top_k: int = 3) -> list[RagHit]:
        q_terms = set(_terms(query))
        hits: list[RagHit] = []
        for doc in self.documents:
            d_terms = _terms(doc.text + " " + doc.source_id)
            score = sum(1 for t in d_terms if t in q_terms)
            # Ensure direct filename/theme requests are retrieved even with tiny vocab overlap.
            if any(t in doc.source_id.lower() for t in q_terms):
                score += 3
            if score > 0:
                hits.append(
                    RagHit(
                        source_id=doc.source_id,
                        trust=doc.trust,
                        score=score,
                        snippet=doc.text.strip()[:1800],
                    )
                )
        hits.sort(key=lambda h: (h.score, 1 if h.trust == "trusted" else 0), reverse=True)
        return hits[:top_k]

    def search_text(self, query: str, top_k: int = 3) -> tuple[str, dict]:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "No RAG documents matched the query.", {"hits": []}
        blocks = []
        for hit in hits:
            blocks.append(
                f"[{hit.trust.upper()} RAG DOC: {hit.source_id}; score={hit.score}]\n{hit.snippet}"
            )
        return "\n\n---\n\n".join(blocks), {"hits": [asdict(h) for h in hits]}


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9_\-]{3,}", (text or "").lower())
