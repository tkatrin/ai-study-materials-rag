from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from rag_service.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder
from rag_service.pipeline import ask_question, build_index
from rag_service.rag_chain import format_sources


st.set_page_config(page_title="AI-поиск по учебным материалам", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


def save_uploaded_files(uploaded_files) -> list[Path]:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uploaded_file in uploaded_files:
        destination = upload_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        paths.append(destination)
    return paths


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
    build_button = st.button("Построить индекс", type="primary", use_container_width=True)

if "store" not in st.session_state:
    st.session_state.store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if build_button:
    if not uploaded_files:
        st.warning("Добавьте хотя бы один файл.")
    else:
        with st.spinner("Извлекаю текст, режу на фрагменты и строю FAISS-индекс..."):
            paths = save_uploaded_files(uploaded_files)
            embedder = get_embedder(model_name)
            st.session_state.store = build_index(paths, embedder, chunk_size=chunk_size)
            st.session_state.indexed_files = [path.name for path in paths]
        st.success(f"Индекс готов: {len(st.session_state.store.chunks)} фрагментов.")

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
