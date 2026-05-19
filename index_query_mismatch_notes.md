# Index/Query Mismatch Notes

This document records query-side follow-up work after modifier-aware indexing.

## Current Index Behavior

- Atomic phrases still create semantic targets using the full phrase surface.
- Compositional phrases are routed to the head token for semantic discovery.
- Compositional phrase occurrences preserve the original phrase surface and modifier metadata on `TextEmbedding` and `SpanOccurrence`.

Example:

```text
American director
```

is indexed as:

```text
TokenNode: director
SpanOccurrence.span_text: director
SpanOccurrence.surface_text: American director
SpanOccurrence.modifier_texts: ["American"]
```

## Known Query-Side Mismatches

The query pipeline has not been updated yet.

- Query phrase extraction still looks up the full phrase text first. A query for `American director` may look for a `TokenNode` named `American director`, but compositional indexing now stores it under `director`.
- Phrase fuzzy lookup still uses `phrase_token_nodes` and `phrase_index`. Compositional phrases no longer create phrase token nodes, so they are not discoverable through the old phrase word-intersection path.
- Query embeddings for compositional phrases still pool the full phrase span. The index now pools only the head span for semantic discovery, so exact semantic comparison should later use the query head embedding.
- Query ranking does not yet use `SpanOccurrence.modifier_texts` or `modifier_texts_norm`. Modifier matches such as `American` in `American director` currently do not boost or filter chunks.
- Token de-duplication in query still treats phrase words as ordinary phrase components. After query routing is updated, the head token should be de-duplicated against the routed compositional phrase.

## Follow-Up Direction

- Apply the same phrase analysis to query phrases.
- Route compositional query phrases to their head token.
- Preserve query modifiers in the resolved match record.
- Add modifier-aware chunk scoring or filtering using occurrence-level modifier metadata.
