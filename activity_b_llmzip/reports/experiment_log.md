# Activity B Experiment Log

Record durable results here. Keep raw JSON outputs in `activity_b_llmzip/results/`.

| Date | Model / Method | Text chars | Context / stride | Metric | Result | Notes |
|---|---|---:|---|---|---:|---|
| TBD | raw zlib/bz2/lzma | TBD | n/a | bits/character | TBD | Baseline compressors. |
| TBD | gpt2 | TBD | TBD | LLM ideal bits/character | TBD | Smoke test first. |

| 2026-05-06 | zlib_9 raw text8 | 100,000 | n/a | bits/character | 2.656400 | Smoke baseline. |
| 2026-05-06 | bz2_9 raw text8 | 100,000 | n/a | bits/character | 2.283840 | Smoke baseline. |
| 2026-05-06 | lzma_9 raw text8 | 100,000 | n/a | bits/character | 2.454080 | Smoke baseline. |
| 2026-05-06 | zlib_9 raw text8 | 1,000,000 | n/a | bits/character | 2.638112 | 1MB baseline. |
| 2026-05-06 | bz2_9 raw text8 | 1,000,000 | n/a | bits/character | 2.097872 | 1MB baseline. |
| 2026-05-06 | lzma_9 raw text8 | 1,000,000 | n/a | bits/character | 2.197696 | 1MB baseline. |
| 2026-05-06 | gpt2 ideal codelength | 20,000 | context 1024 / stride 512 | bits/character | 1.078974 | GPU smoke on L40S; 5.821279 bits/token, 3707 scored tokens. |
| 2026-05-06 | gpt2 rank varint + zlib | 20,000 | context 1024 / stride 512 | bits/character | 1.310400 | Rank-zero fraction 0.31454; LLMZip-style rank stream smoke. |