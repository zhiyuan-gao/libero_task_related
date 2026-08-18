# Experimental / superseded policy integrations

`validate_semantic_query_dependency_legacy.py` preserves the earlier P2
Semantic-Query engineering gate. It is not part of primary P2, must not be
included in training commands, and is retained only for provenance.

As of 2026-08-18, primary P2 contains only eight Grounding queries and eight
Geometry queries. Concrete semantic-subtask text is supervised through a
separate native VLM autoregressive language-modeling pass.
