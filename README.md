# AI-поиск по учебным материалам

Локальный RAG-сервис для конспектов, PDF, Markdown, TXT и DOCX. Пользователь загружает учебные материалы, система извлекает текст, режет его на фрагменты, строит эмбеддинги через Sentence Transformers, сохраняет в FAISS и отвечает на вопросы только по найденному контексту.

![Пример интерфейса](docs/screenshots/ui-example.svg)

## Возможности MVP

- загрузка файлов `.pdf`, `.txt`, `.md`, `.docx`;
- извлечение и нормализация текста;
- разбиение документов на перекрывающиеся фрагменты;
- построение эмбеддингов моделью `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- хранение и поиск похожих фрагментов в FAISS;
- Streamlit-интерфейс для загрузки файлов и вопросов;
- вывод ответа вместе с источниками: файл, номер фрагмента и цитата.

## Архитектура

```text
app.py
  -> document_loader  -> PDF/TXT/MD/DOCX text extraction
  -> chunker          -> overlapping text chunks
  -> embedder         -> Sentence Transformers vectors
  -> vector_store     -> FAISS index + chunk metadata
  -> rag_chain        -> answer from retrieved context + sources
```

Основные модули находятся в `rag_service/`:

- `document_loader.py` загружает документы и возвращает текст с метаданными;
- `chunker.py` режет текст на фрагменты с overlap;
- `embedder.py` создает эмбеддинги;
- `vector_store.py` добавляет, ищет, сохраняет и загружает FAISS-индекс;
- `rag_chain.py` формирует ответ по найденным фрагментам;
- `pipeline.py` связывает шаги индексации и вопроса-ответа.

## Быстрый запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

После запуска откройте локальный адрес, который покажет Streamlit, обычно `http://localhost:8501`.

## Пример набора документов

В папке `examples/` лежат небольшие учебные файлы:

- `machine_learning.md`;
- `databases.txt`;
- `neural_networks.md`.

Их можно загрузить в интерфейсе, нажать `Построить индекс` и задать вопрос.

## Пример вопроса и ответа

Вопрос:

```text
Чем supervised learning отличается от unsupervised learning?
```

Пример ответа:

```text
В supervised learning модель обучается на примерах, где для каждого объекта известен правильный ответ. В unsupervised learning модель получает данные без заранее известных меток и ищет скрытую структуру.

Использованные фрагменты: [1]
```

Источник:

```text
[1] machine_learning.md, фрагмент 1:
В supervised learning модель обучается на примерах, где для каждого объекта известен правильный ответ...
```

## Замечания

Первый запуск может занять время: Sentence Transformers скачает модель. После этого модель кешируется локально. Для полностью офлайн-режима заранее скачайте модель Hugging Face и укажите локальный путь в поле `Embedding-модель`.
