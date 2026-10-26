# Eval run: before

```
{
  "embedder": "tfidf",
  "chunk_size": 500,
  "overlap": 0,
  "title_prefix": false,
  "heading_aware": false,
  "top_k": 4,
  "run_at": "2026-10-26T14:16:24",
  "mode": "retrieval-only"
}
```

| category | n | retrieval_hit | hit_at_1 | mrr |
|---|---|---|---|---|
| date | 1 | 1.00 | 1.00 | 1.00 |
| lookup | 8 | 1.00 | 0.88 | 0.94 |
| math | 4 | 1.00 | 1.00 | 1.00 |
| multi-hop | 3 | 0.67 | 0.33 | 0.50 |
| **ALL** | 16 | 0.94 | 0.81 | 0.88 |

## Failures

- `work-abroad`: evidence not in top 4
