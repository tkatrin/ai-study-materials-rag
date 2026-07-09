# Demo

Use the three files from `examples/`:

- `examples/machine_learning.md`
- `examples/databases.txt`
- `examples/neural_networks.md`

Suggested settings:

- `Embedding-модель`: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `Количество источников`: `4`
- `Минимальный score`: `0.25`
- `Reranking`: `Keyword` or `None`
- `Режим ответа`: `Ollama` for generated answers, `Extractive` as fallback

Example question:

```text
Чем supervised learning отличается от unsupervised learning?
```

Expected source:

```text
machine_learning.md
```

Expected answer shape:

```text
Supervised learning uses labeled examples with known answers.
Unsupervised learning works without labels and searches for hidden structure.
```

Screenshot:

![UI example](screenshots/ui-example.svg)
