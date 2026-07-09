from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

import streamlit as st

from rag_service.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder
from rag_service.llm import make_generator
from rag_service.pipeline import ask_question, build_index
from rag_service.rag_chain import format_sources
from rag_service.vector_store import FaissVectorStore


st.set_page_config(page_title="AI-поиск по учебным материалам", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


INDEX_DIR = Path("data/index/default")

if "store" not in st.session_state:
    st.session_state.store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


def save_uploaded_files(uploaded_files) -> List[Tuple[Path, str]]:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        safe_name = _safe_filename(original_name)
        destination = upload_dir / f"{uuid4().hex}_{safe_name}"
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append((destination, original_name))
    return saved_files


def _safe_filename(filename: str) -> str:
    allowed = []
    for char in filename:
        if char.isalnum() or char in {".", "-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    safe = "".join(allowed).strip("._")
    return safe or "uploaded_document"


def indexed_filenames(store: FaissVectorStore) -> List[str]:
    names = []
    for chunk in store.chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in names:
            names.append(source)
    return names


st.title("AI-поиск по учебным материалам")

with st.sidebar:
    st.header("Материалы")
    uploaded_files = st.file_uploader(
        "Загрузите PDF, TXT, MD или DOCX",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    model_name = st.text_input("Embedding-модель", DEFAULT_EMBEDDING_MODEL)
    chunk_size = st.slider("Размер фрагмента", 400, 1600, 900, 100)
    top_k = st.slider("Количество источников", 2, 8, 4)
    generation_mode = st.selectbox("Режим ответа", ["Extractive", "Ollama"])
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434", disabled=generation_mode != "Ollama")
    ollama_model = st.text_input("Ollama-модель", "llama3.1", disabled=generation_mode != "Ollama")
    build_button = st.button("Построить индекс", type="primary", use_container_width=True)
    save_index_button = st.button("Сохранить индекс", disabled=st.session_state.get("store") is None, use_container_width=True)
    load_index_button = st.button("Загрузить сохраненный индекс", disabled=not (INDEX_DIR / "index.faiss").exists(), use_container_width=True)

if build_button:
    if not uploaded_files:
        st.warning("Добавьте хотя бы один файл.")
    else:
        with st.spinner("Извлекаю текст, режу на фрагменты и строю FAISS-индекс..."):
            try:
                saved_files = save_uploaded_files(uploaded_files)
                embedder = get_embedder(model_name)
                st.session_state.store = build_index(saved_files, embedder, chunk_size=chunk_size)
                st.session_state.indexed_files = [source_name for _path, source_name in saved_files]
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Индекс готов: {len(st.session_state.store.chunks)} фрагментов.")

if save_index_button and st.session_state.store is not None:
    st.session_state.store.save(INDEX_DIR)
    st.success(f"Индекс сохранен в {INDEX_DIR}.")

if load_index_button:
    st.session_state.store = FaissVectorStore.load(INDEX_DIR)
    st.session_state.indexed_files = indexed_filenames(st.session_state.store)
    st.success(f"Индекс загружен: {len(st.session_state.store.chunks)} фрагментов.")

left, right = st.columns([0.58, 0.42], gap="large")

with left:
    st.subheader("Вопрос")
    question = st.text_area(
        "Задайте вопрос по загруженным материалам",
        placeholder="Например: чем отличается supervised learning от unsupervised learning?",
        height=120,
    )
    ask_button = st.button("Найти ответ", disabled=st.session_state.store is None)

    if ask_button:
        if not question.strip():
            st.warning("Введите вопрос.")
        else:
            with st.spinner("Ищу релевантные фрагменты..."):
                embedder = get_embedder(model_name)
                generator = make_generator(generation_mode, ollama_model, ollama_url)
                try:
                    answer, results = ask_question(
                        question,
                        st.session_state.store,
                        embedder,
                        top_k=top_k,
                        generator=generator,
                    )
                except RuntimeError as error:
                    st.warning(f"LLM недоступна, показываю extractive-ответ. Детали: {error}")
                    answer, results = ask_question(question, st.session_state.store, embedder, top_k=top_k)
            st.markdown("### Ответ")
            st.write(answer.answer)
            st.markdown("### Источники")
            for index, (chunk, score) in enumerate(results, start=1):
                source = chunk.metadata.get("source", "unknown")
                chunk_id = chunk.metadata.get("chunk_id", index)
                with st.expander(f"[{index}] {source}, фрагмент {chunk_id} · score {score:.3f}", expanded=index == 1):
                    st.write(chunk.text)

with right:
    st.subheader("Индекс")
    if st.session_state.store is None:
        st.info("Загрузите документы и построите индекс.")
    else:
        st.metric("Фрагментов", len(st.session_state.store.chunks))
        st.write("Файлы:")
        for filename in st.session_state.indexed_files:
            st.write(f"- {filename}")
        with st.expander("Все источники"):
            st.text(format_sources(st.session_state.store.chunks, max_chars=240))
