from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHUNK_CHARS = 1400
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True, slots=True)
class DocChunk:
    chunk_id: str
    start_char: int
    end_char: int
    title: str | None
    text: str

    def to_summary_dict(self, *, preview_chars: int = 240) -> dict[str, object]:
        preview = self.text[:preview_chars]
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "char_count": len(self.text),
            "preview": preview,
            "truncated": len(self.text) > preview_chars,
        }

    def to_full_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "char_count": len(self.text),
            "text": self.text,
        }


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _iter_markdown_sections(text: str) -> list[tuple[str | None, int, int, str]]:
    heading_pattern = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [(None, 0, len(text), text)]

    sections: list[tuple[str | None, int, int, str]] = []
    if matches[0].start() > 0:
        intro_text = text[: matches[0].start()]
        if intro_text.strip():
            sections.append((None, 0, matches[0].start(), intro_text))

    for index, match in enumerate(matches):
        section_start = match.start()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(2).strip()
        section_text = text[section_start:section_end]
        sections.append((title, section_start, section_end, section_text))
    return sections


def _split_section_into_windows(
    *,
    title: str | None,
    start_offset: int,
    text: str,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[DocChunk]:
    normalized_text = text.strip()
    if not normalized_text:
        return []

    chunks: list[DocChunk] = []
    relative_cursor = 0
    chunk_index = 0
    step = max(chunk_chars - chunk_overlap, 1)
    while relative_cursor < len(text):
        window = text[relative_cursor : relative_cursor + chunk_chars]
        if not window.strip():
            relative_cursor += step
            continue
        chunk_start = start_offset + relative_cursor
        chunk_end = chunk_start + len(window)
        chunks.append(
            DocChunk(
                chunk_id=f"chunk_{len(chunks):04d}",
                start_char=chunk_start,
                end_char=chunk_end,
                title=title,
                text=window.strip(),
            )
        )
        chunk_index += 1
        relative_cursor += step
    return chunks


def build_doc_chunks(
    path: Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocChunk]:
    text = _normalize_newlines(path.read_text(encoding="utf-8", errors="replace"))
    sections = _iter_markdown_sections(text)
    chunks: list[DocChunk] = []

    for title, start, _end, section_text in sections:
        section_chunks = _split_section_into_windows(
            title=title,
            start_offset=start,
            text=section_text,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        for section_chunk in section_chunks:
            chunks.append(
                DocChunk(
                    chunk_id=f"chunk_{len(chunks):04d}",
                    start_char=section_chunk.start_char,
                    end_char=section_chunk.end_char,
                    title=section_chunk.title,
                    text=section_chunk.text,
                )
            )
    return chunks


def list_doc_chunks(
    path: Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    preview_chars: int = 240,
) -> dict[str, object]:
    chunks = build_doc_chunks(path, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    return {
        "path": str(path),
        "chunk_count": len(chunks),
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "chunks": [chunk.to_summary_dict(preview_chars=preview_chars) for chunk in chunks],
    }


def read_doc_chunk(
    path: Path,
    chunk_id: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, object]:
    chunks = build_doc_chunks(path, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    for chunk in chunks:
        if chunk.chunk_id == chunk_id:
            return {
                "path": str(path),
                "chunk": chunk.to_full_dict(),
            }
    raise KeyError(f"Unknown chunk_id: {chunk_id}")


def _query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9_]+", query.lower()) if term]


def search_doc_chunks(
    path: Path,
    query: str,
    *,
    top_k: int = 5,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    preview_chars: int = 240,
) -> dict[str, object]:
    terms = _query_terms(query)
    chunks = build_doc_chunks(path, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    scored: list[tuple[int, int, DocChunk]] = []

    for index, chunk in enumerate(chunks):
        haystack = f"{chunk.title or ''}\n{chunk.text}".lower()
        score = 0
        for term in terms:
            term_count = haystack.count(term)
            if term_count:
                score += term_count * 10
                if chunk.title and term in chunk.title.lower():
                    score += 5
        if score > 0:
            scored.append((score, -index, chunk))

    scored.sort(reverse=True)
    matches = [item[2] for item in scored[: max(top_k, 1)]]
    return {
        "path": str(path),
        "query": query,
        "match_count": len(matches),
        "matches": [chunk.to_summary_dict(preview_chars=preview_chars) for chunk in matches],
    }
