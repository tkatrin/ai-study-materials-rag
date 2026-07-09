import streamlit as st

from rag_service import config
from rag_service.embedder import SentenceTransformerEmbedder
from rag_service.file_utils import save_uploaded_files
from rag_service.index_manager import (
    DEFAULT_INDEX_DIR,
    DEFAULT_UPLOAD_DIR,
    has_saved_index,
    indexed_filenames,
    load_index,
    save_index,
)
from rag_service.llm import make_generator
from rag_service.pipeline import ask_question, build_index
from rag_service.rag_chain import format_sources


st.set_page_config(page_title="AI-поиск по учебным материалам", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


if "store" not in st.session_state:
    st.session_state.store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


st.title("AI-поиск по учебным материалам")

with st.sidebar:
    st.header("Материалы")
    uploaded_files = st.file_uploader(
        "Загрузите PDF, TXT, MD или DOCX",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    model_name = st.text_input("Embedding-модель", config.EMBEDDING_MODEL)
    chunk_size = st.slider("Размер фрагмента", 400, 1600, config.CHUNK_SIZE, 100)
    top_k = st.slider("Количество источников", 2, 8, config.TOP_K)
    min_score = st.slider("Минимальный score", 0.0, 1.0, config.MIN_SCORE, 0.05)
    generation_mode = st.selectbox("Режим ответа", ["Extractive", "Ollama"])
    ollama_url = st.text_input("Ollama URL", config.OLLAMA_URL, disabled=generation_mode != "Ollama")
    ollama_model = st.text_input("Ollama-модель", config.OLLAMA_MODEL, disabled=generation_mode != "Ollama")
    build_button = st.button("Построить индекс", type="primary", use_container_width=True)
    save_index_button = st.button("Сохранить индекс", disabled=st.session_state.get("store") is None, use_container_width=True)
    load_index_button = st.button("Загрузить сохраненный индекс", disabled=not has_saved_index(), use_container_width=True)

if build_button:
    if not uploaded_files:
        st.warning("Добавьте хотя бы один файл.")
    else:
        with st.spinner("Извлекаю текст, режу на фрагменты и строю FAISS-индекс..."):
            try:
                saved_files = save_uploaded_files(uploaded_files, DEFAULT_UPLOAD_DIR)
                embedder = get_embedder(model_name)
                st.session_state.store = build_index(saved_files, embedder, chunk_size=chunk_size)
                st.session_state.indexed_files = [source_name for _path, source_name in saved_files]
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Индекс готов: {len(st.session_state.store.chunks)} фрагментов.")

if save_index_button and st.session_state.store is not None:
    save_index(st.session_state.store)
    st.success(f"Индекс сохранен в {DEFAULT_INDEX_DIR}.")

if load_index_button:
    try:
        st.session_state.store = load_index(model_name)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(f"Не удалось загрузить индекс: {error}")
    else:
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
                        min_score=min_score,
                    )
                except RuntimeError as error:
                    st.warning(f"LLM недоступна, показываю extractive-ответ. Детали: {error}")
                    answer, results = ask_question(
                        question,
                        st.session_state.store,
                        embedder,
                        top_k=top_k,
                        min_score=min_score,
                    )
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
