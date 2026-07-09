# Retrieval Evaluation

The project includes a tiny manual evaluation set in `examples/eval_questions.json`.
The simplest metric is hit@k: whether at least one retrieved source file matches an expected source.

| Question | Expected source | Metric |
| --- | --- | --- |
| Чем supervised learning отличается от unsupervised learning? | `machine_learning.md` | hit@k |
| Что такое внешний ключ? | `databases.txt` | hit@k |
| Как бороться с переобучением нейронной сети? | `neural_networks.md` | hit@k |

Run a local evaluation:

```bash
python scripts/evaluate_retrieval.py --top-k 4 --min-score 0.25
```

The script builds a temporary FAISS index from the example documents, asks every question
from `eval_questions.json`, and reports hit@k as a quick retrieval smoke test.
