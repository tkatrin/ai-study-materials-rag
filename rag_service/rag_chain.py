from typing import Iterable, List, Optional, Protocol

from .models import Answer, Chunk


class TextGenerator(Protocol):
    def __call__(self, prompt: str) -> str:
        ...


def build_answer(
    question: str,
    retrieved_chunks: Iterable[Chunk],
    generator: Optional[TextGenerator] = None,
) -> Answer:
    chunks = list(retrieved_chunks)
    if not chunks:
        return Answer(
            question=question,
            answer="Не нашел релевантных фрагментов в загруженных материалах.",
            sources=[],
        )

    prompt = build_prompt(question, chunks)
    if generator is None:
        answer_text = extractive_answer(question, chunks)
    else:
        answer_text = generator(prompt).strip()
    return Answer(question=question, answer=answer_text, sources=chunks)


def build_prompt(question: str, chunks: List[Chunk]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source = _source_label(chunk)
        chunk_id = chunk.metadata.get("chunk_id", index)
        context_blocks.append(f"[{index}] {source}, fragment {chunk_id}\n{chunk.text}")

    context = "\n\n".join(context_blocks)
    return (
        "Ответь на вопрос только по контексту ниже. "
        "Если ответа нет в контексте, честно скажи, что данных недостаточно. "
        "В конце укажи номера использованных фрагментов.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n"
        "Ответ:"
    )


def extractive_answer(question: str, chunks: List[Chunk], max_sentences: int = 5) -> str:
    question_terms = _terms(question)
    scored_sentences = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        for sentence in _sentences(chunk.text):
            sentence_terms = _terms(sentence)
            overlap = len(question_terms & sentence_terms)
            if overlap:
                scored_sentences.append((overlap, chunk_index, sentence))

    if not scored_sentences:
        lead = " ".join(chunk.text for chunk in chunks[:2])
        return f"По найденным фрагментам можно опираться на такой контекст: {lead[:700]}..."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = []
    used_refs = []
    for _, chunk_index, sentence in scored_sentences:
        if sentence not in selected:
            selected.append(sentence)
            used_refs.append(chunk_index)
        if len(selected) >= max_sentences:
            break

    refs = ", ".join(f"[{index}]" for index in sorted(set(used_refs)))
    return " ".join(selected) + f"\n\nИспользованные фрагменты: {refs}"


def format_sources(chunks: Iterable[Chunk], max_chars: int = 500) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        source = _source_label(chunk)
        chunk_id = chunk.metadata.get("chunk_id", index)
        preview = chunk.text[:max_chars].strip()
        if len(chunk.text) > max_chars:
            preview += "..."
        lines.append(f"[{index}] {source}, фрагмент {chunk_id}: {preview}")
    return "\n\n".join(lines)


def _sentences(text: str) -> List[str]:
    normalized = " ".join(text.split())
    sentences = []
    current = []
    for char in normalized:
        current.append(char)
        if char in ".!?":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _terms(text: str) -> set:
    return {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}\"'")) > 3
    }


def _source_label(chunk: Chunk) -> str:
    source = chunk.metadata.get("source", "unknown")
    page_number = chunk.metadata.get("page_number")
    if page_number is not None:
        return f"{source}, стр. {page_number}"
    return source
