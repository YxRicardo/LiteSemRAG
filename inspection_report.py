"""检视 / HTML 报告 mixin —— 从 ``RAG_graph.py`` 抽出的纯展示方法。

这些方法只读取图数据、产出 text / HTML / CSV / 控制台输出，不修改实例状态。
通过 mixin 组合进 ``LiteSemRAG``（见 ``RAG_graph.py`` 的 class 声明）：运行期 ``self``
就是完整的 ``LiteSemRAG`` 实例，方法体内的 ``self.xxx`` 访问与抽离前完全一致。

本模块刻意不 import ``RAG_graph``，仅依赖标准库与 ``utils.print_size_mb``，因此
``RAG_graph`` 可以安全地 ``from inspection_report import _InspectionReportMixin``
而不形成循环依赖。Mixin 不引入任何实例字段，对 pickle / schema 向后兼容无影响。
"""
import os
import time
import json
from html import escape
from collections import Counter, defaultdict

from utils import print_size_mb


class _InspectionReportMixin:
    """检视 / HTML 报告方法集合（详见模块 docstring）。"""

    # Format semantic-description logs as plain text.
    # Format one operation event as a single line for the chronological timeline view.
    def _format_timeline_event(self, event):
        event_type = event.get("event_type", "unknown")
        sample_assignment_event_types = {
            "assign_sample_as_class_seed",
            "assign_sample_by_d1_d2",
            "assign_sample_by_single_safe_class",
            "assign_sample_by_model_judgment",
            "assign_sample_by_medoid_fallback",
        }
        if event_type in sample_assignment_event_types:
            return f"[{event_type}] " + self._format_sample_assignment(event)

        if event_type == "assign_description_by_consensus":
            return (
                f"[{event_type}] sem_node_id={event.get('sem_node_id')} | "
                f"token={event.get('token_text')!r} | description={event.get('description')!r} | "
                f"samples={event.get('sample_count')} | top_count={event.get('top_description_count')} | "
                f"top_ratio={self._safe_float(event.get('top_description_ratio')):.3f}"
            )
        if event_type == "assign_description_by_fallback_top_vote":
            return (
                f"[{event_type}] sem_node_id={event.get('sem_node_id')} | "
                f"token={event.get('token_text')!r} | description={event.get('description')!r} | "
                f"samples={event.get('sample_count')} | top_count={event.get('top_description_count')} | "
                f"top_ratio={self._safe_float(event.get('top_description_ratio')):.3f} | "
                f"valid_classes={event.get('valid_class_count')}"
            )
        if event_type == "split_sem_node_by_sample_labels":
            initial_sizes = event.get("initial_class_sizes") or {}
            final_sizes = event.get("description_counts") or {}
            initial_repr = (
                ", ".join(f"{desc!r}={count}" for desc, count in initial_sizes.items())
                if initial_sizes else "-"
            )
            final_repr = (
                ", ".join(f"{desc!r}={count}" for desc, count in final_sizes.items())
                if final_sizes else "-"
            )
            return (
                f"[{event_type}] source_sem_node_id={event.get('source_sem_node_id')} | "
                f"token={event.get('token_text')!r} | source_chunk_count={event.get('source_chunk_count')} | "
                f"cluster_samples={event.get('cluster_sample_count')} | "
                f"valid_classes={event.get('valid_class_count')} | "
                f"safe_classes={event.get('safe_class_count')} | "
                f"danger_zone={event.get('danger_zone_count')} | "
                f"singleton_classes={event.get('singleton_class_count')} | "
                f"initial_class_sizes={{{initial_repr}}} | final_description_counts={{{final_repr}}}"
            )
        if event_type == "create_split_sem_node":
            return (
                f"[{event_type}] sem_node_id={event.get('sem_node_id')} | "
                f"source_sem_node_id={event.get('source_sem_node_id')} | "
                f"description={event.get('description')!r} | sample_count={event.get('sample_count')}"
            )
        if event_type == "merge_sem_nodes_with_same_description":
            return (
                f"[{event_type}] token={event.get('token_text')!r} | "
                f"description={event.get('description')!r} | "
                f"kept_sem_node_id={event.get('kept_sem_node_id')} | "
                f"merged_sem_node_ids={event.get('merged_sem_node_ids')} | "
                f"merged_chunk_count={event.get('merged_chunk_count')} | "
                f"retained_embed_count={event.get('retained_embed_count')} | "
                f"retained_embed_source_count={event.get('retained_embed_source_count')}"
            )
        # Generic fallback: dump remaining payload fields.
        payload = " | ".join(
            f"{key}={value!r}"
            for key, value in event.items()
            if key != "event_type"
        )
        return f"[{event_type}] {payload}"

    def _format_sem_description_logs_text(self):
        operations = list(self.sem_description_operation_logs)
        event_counts = Counter(item.get("event_type", "unknown") for item in operations)

        # Chronological timeline: replay every operation in the order it was emitted so the
        # full processing flow (per-sem-node consensus, splits, per-cluster descriptions,
        # sample assignments, and merges) is visible step by step.
        timeline_lines = ["=== Processing timeline ===", ""]
        if not operations:
            timeline_lines.append("(no operations recorded)")
        else:
            timeline_lines.append(f"({len(operations)} events in chronological order)")
            timeline_lines.append("")
            for index, event in enumerate(operations, start=1):
                timeline_lines.append(f"#{index:04d} {self._format_timeline_event(event)}")
        timeline_text = "\n".join(timeline_lines)

        consensus_events = [e for e in operations if e.get("event_type") == "assign_description_by_consensus"]
        split_events = [e for e in operations if e.get("event_type") == "split_sem_node_by_sample_labels"]
        create_split_events = [e for e in operations if e.get("event_type") == "create_split_sem_node"]
        sample_assignment_event_types = {
            "assign_sample_as_class_seed",
            "assign_sample_by_d1_d2",
            "assign_sample_by_model_judgment",
            "assign_sample_by_medoid_fallback",
        }
        sample_events = [e for e in operations if e.get("event_type") in sample_assignment_event_types]
        merge_events = [e for e in operations if e.get("event_type") == "merge_sem_nodes_with_same_description"]

        create_split_by_source = defaultdict(list)
        for event in create_split_events:
            create_split_by_source[event.get("source_sem_node_id")].append(event)
        create_split_by_sem_id = {
            event.get("sem_node_id"): event
            for event in create_split_events
            if event.get("sem_node_id") is not None
        }
        sample_events_by_source = defaultdict(list)
        for event in sample_events:
            sample_events_by_source[event.get("source_sem_node_id")].append(event)
        sample_events_by_target = defaultdict(list)
        for event in sample_events:
            sample_events_by_target[event.get("target_sem_node_id")].append(event)
        consensus_by_sem_id = {
            event.get("sem_node_id"): event
            for event in consensus_events
            if event.get("sem_node_id") is not None
        }

        predicted_by_sem_id = {
            item["sem_node_id"]: item for item in self.predicted_sem_description_logs
        }

        lines = []
        timing_lines = self._format_index_session_timing()
        if timing_lines:
            lines.append("=== Index session running info ===")
            lines.append("")
            lines.extend(timing_lines)
            lines.append("")
        lines.append("=== Semantic description log ===")
        lines.append("")
        lines.append(timeline_text)
        lines.append("")
        lines.append("=== Summary ===")
        lines.append("")
        lines.append("Event summary:")
        if not operations:
            lines.append("  (no operations recorded)")
        else:
            for event_type, count in sorted(event_counts.items()):
                lines.append(f"  {event_type}: {count}")

        lines.append("")
        lines.append(f"[1] Consensus description assignments ({len(consensus_events)})")
        if not consensus_events:
            lines.append("  (none)")
        else:
            for event in consensus_events:
                sem_id = event.get("sem_node_id")
                lines.append(
                    "  sem_node_id={} | token={!r} | description={!r} | "
                    "samples={} | top_count={} | top_ratio={:.3f}".format(
                        sem_id,
                        event.get("token_text"),
                        event.get("description"),
                        event.get("sample_count"),
                        event.get("top_description_count"),
                        self._safe_float(event.get("top_description_ratio")),
                    )
                )
                predicted = predicted_by_sem_id.get(sem_id)
                if predicted is not None:
                    sample_predictions = predicted.get("sample_predictions", [])
                    for sample in sample_predictions:
                        lines.append(
                            "    " + self._format_consensus_sample_prediction(sample)
                        )

        lines.append("")
        lines.append(f"[2] Split sem nodes by sample label ({len(split_events)})")
        if not split_events:
            lines.append("  (none)")
        else:
            for event in split_events:
                source_id = event.get("source_sem_node_id")
                lines.append(
                    "  source_sem_node_id={} | token={!r} | source_chunk_count={} | "
                    "cluster_samples={} | initial_samples={}".format(
                        source_id,
                        event.get("token_text"),
                        event.get("source_chunk_count"),
                        event.get("cluster_sample_count"),
                        event.get("initial_prediction_sample_count"),
                    )
                )
                lines.append(
                    "    valid_classes={} | safe_classes={} | safe_seeds={} | "
                    "danger_seeds={} | danger_zone={} | singleton_classes={} | "
                    "class_seed={} | single_safe_class={} | d1_d2={} | "
                    "model_judgment={} | medoid_fallback={}".format(
                        event.get("valid_class_count"),
                        event.get("safe_class_count"),
                        event.get("safe_seed_count"),
                        event.get("danger_seed_count"),
                        event.get("danger_zone_count"),
                        event.get("singleton_class_count"),
                        event.get("class_seed_sample_count"),
                        event.get("single_safe_class_assigned_count"),
                        event.get("d1_d2_assigned_count"),
                        event.get("model_judgment_count"),
                        event.get("medoid_fallback_count"),
                    )
                )
                initial_sizes = event.get("initial_class_sizes") or {}
                if initial_sizes:
                    lines.append(
                        "    initial_class_sizes: "
                        + ", ".join(f"{desc!r}={count}" for desc, count in initial_sizes.items())
                    )
                final_sizes = event.get("description_counts") or {}
                if final_sizes:
                    lines.append(
                        "    final_description_counts: "
                        + ", ".join(f"{desc!r}={count}" for desc, count in final_sizes.items())
                    )
                safe_sizes = event.get("safe_class_sizes") or {}
                if safe_sizes:
                    lines.append(
                        "    safe_class_sizes: "
                        + ", ".join(f"{desc!r}={count}" for desc, count in safe_sizes.items())
                    )
                created_events = create_split_by_source.get(source_id, [])
                if created_events:
                    lines.append("    created sem nodes:")
                    for created in created_events:
                        lines.append(
                            "      sem_node_id={} | description={!r} | sample_count={}".format(
                                created.get("sem_node_id"),
                                created.get("description"),
                                created.get("sample_count"),
                            )
                        )
                assignment_events = sample_events_by_source.get(source_id, [])
                if assignment_events:
                    lines.append("    sample assignments:")
                    for entry in assignment_events:
                        lines.append("      " + self._format_sample_assignment(entry))

        orphan_create_events = [
            event for event in create_split_events
            if event.get("source_sem_node_id") not in {s.get("source_sem_node_id") for s in split_events}
        ]
        if orphan_create_events:
            lines.append("")
            lines.append("  (orphan create_split_sem_node events without a matching split record)")
            for event in orphan_create_events:
                lines.append(
                    "    sem_node_id={} | source_sem_node_id={} | description={!r} | sample_count={}".format(
                        event.get("sem_node_id"),
                        event.get("source_sem_node_id"),
                        event.get("description"),
                        event.get("sample_count"),
                    )
                )

        lines.append("")
        lines.append(f"[3] Same-description merges ({len(merge_events)})")
        if not merge_events:
            lines.append("  (none)")
        else:
            for event in merge_events:
                lines.append(
                    "  token={!r} | description={!r} | kept_sem_node_id={} | "
                    "merged_sem_node_ids={} | merged_chunk_count={} | "
                    "retained_embed_count={} | retained_embed_source_count={}".format(
                        event.get("token_text"),
                        event.get("description"),
                        event.get("kept_sem_node_id"),
                        event.get("merged_sem_node_ids"),
                        event.get("merged_chunk_count"),
                        event.get("retained_embed_count"),
                        event.get("retained_embed_source_count"),
                    )
                )
                merged_details = event.get("merged_sem_node_details") or []
                if not merged_details:
                    continue
                lines.append("    merged sem node details:")
                for detail in merged_details:
                    sem_id = detail.get("sem_node_id")
                    consensus_event = consensus_by_sem_id.get(sem_id)
                    create_split_event = create_split_by_sem_id.get(sem_id)
                    origin = self._describe_sem_node_origin(
                        sem_id,
                        consensus_event,
                        create_split_event,
                    )
                    lines.append(
                        "      sem_node_id={} | origin={} | chunk_count={} | "
                        "span_occurrence_count={} | retained_embed_count={} | "
                        "retained_embed_source_count={} | has_embed={} | pending_embed_rebuild={}".format(
                            sem_id,
                            origin,
                            detail.get("chunk_count"),
                            detail.get("span_occurrence_count"),
                            detail.get("retained_embed_count"),
                            detail.get("retained_embed_source_count"),
                            detail.get("has_embed"),
                            detail.get("pending_embed_rebuild"),
                        )
                    )
                    if consensus_event is not None:
                        lines.append(
                            "        consensus: samples={} | top_count={} | top_ratio={:.3f}".format(
                                consensus_event.get("sample_count"),
                                consensus_event.get("top_description_count"),
                                self._safe_float(consensus_event.get("top_description_ratio")),
                            )
                        )
                        predicted = predicted_by_sem_id.get(sem_id)
                        if predicted is not None:
                            sample_predictions = predicted.get("sample_predictions", [])
                            if sample_predictions:
                                lines.append("        consensus sample predictions:")
                                for sample in sample_predictions:
                                    lines.append(
                                        "          " + self._format_consensus_sample_prediction(sample)
                                    )
                    if create_split_event is not None:
                        lines.append(
                            "        created_by_split: source_sem_node_id={} | sample_count={}".format(
                                create_split_event.get("source_sem_node_id"),
                                create_split_event.get("sample_count"),
                            )
                        )
                    target_assignment_events = sample_events_by_target.get(sem_id, [])
                    if target_assignment_events:
                        assignment_counts = Counter(
                            entry.get("event_type", "unknown")
                            for entry in target_assignment_events
                        )
                        lines.append(
                            "        assignment_summary: class_seed={} | single_safe_class={} | "
                            "d1_d2={} | model_judgment={} | medoid_fallback={}".format(
                                assignment_counts.get("assign_sample_as_class_seed", 0),
                                assignment_counts.get("assign_sample_by_single_safe_class", 0),
                                assignment_counts.get("assign_sample_by_d1_d2", 0),
                                assignment_counts.get("assign_sample_by_model_judgment", 0),
                                assignment_counts.get("assign_sample_by_medoid_fallback", 0),
                            )
                        )
                        lines.append("        assignment_details:")
                        for assignment_event in target_assignment_events:
                            lines.append(
                                "          " + self._format_sample_assignment(assignment_event)
                            )

        lines.append("")
        lines.append(f"[4] Deleted sem nodes ({len(self.deleted_merged_sem_logs)})")
        if not self.deleted_merged_sem_logs:
            lines.append("  (none)")
        else:
            for item in self.deleted_merged_sem_logs:
                lines.append(
                    "  deleted_sem_node_id={} | kept_sem_node_id={} | token={!r} | "
                    "description={!r} | deleted_chunk_count={}".format(
                        item["deleted_sem_node_id"],
                        item["kept_sem_node_id"],
                        item["token_text"],
                        item["description"],
                        item["deleted_chunk_count"],
                    )
                )

        lines.append("")
        lines.append(f"[5] Wikidata lookups with no usable results ({len(self.wikidata_no_result_logs)})")
        if not self.wikidata_no_result_logs:
            lines.append("  (none)")
        else:
            for item in self.wikidata_no_result_logs:
                lines.append(
                    "  term={!r} | stage={} | reason={}".format(
                        item["term"],
                        item["stage"],
                        item["reason"],
                    )
                )

        return "\n".join(lines)

    # Format a per-sample assignment event into a single human-readable line.
    def _format_sample_assignment(self, event):
        event_type = event.get("event_type", "")
        span_repr = "chunk_node_id={} | span=({}, {}) | span_text={!r}".format(
            event.get("chunk_node_id"),
            event.get("span_start"),
            event.get("span_end"),
            event.get("span_text"),
        )
        description = event.get("description")
        target = event.get("target_sem_node_id")
        if event_type == "assign_sample_as_class_seed":
            return (
                f"[class_seed] target_sem_node_id={target} | description={description!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_d1_d2":
            return (
                f"[d1_d2]     target_sem_node_id={target} | description={description!r} | "
                f"d1={self._safe_float(event.get('d1')):.4f} | d2={self._safe_float(event.get('d2')):.4f} | "
                f"ratio={self._safe_float(event.get('ratio')):.4f} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_single_safe_class":
            return (
                f"[single]    target_sem_node_id={target} | description={description!r} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_model_judgment":
            return (
                f"[model]     target_sem_node_id={target} | description={description!r} | "
                f"predicted_label={event.get('predicted_label')!r} | "
                f"predicted_definition={event.get('predicted_definition')!r} | "
                f"score={self._safe_float(event.get('score')):.4f} | "
                f"d1={self._safe_float(event.get('d1')):.4f} | d2={self._safe_float(event.get('d2')):.4f} | "
                f"ratio={self._safe_float(event.get('ratio')):.4f} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_medoid_fallback":
            return (
                f"[fallback]  target_sem_node_id={target} | description={description!r} | "
                f"nearest_medoid_distance={self._safe_float(event.get('nearest_medoid_distance')):.4f} | "
                f"d1={self._safe_float(event.get('d1')):.4f} | d2={self._safe_float(event.get('d2')):.4f} | "
                f"ratio={self._safe_float(event.get('ratio')):.4f} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        return f"[{event_type}] target_sem_node_id={target} | description={description!r} | {span_repr}"

    # Format a sampled prediction used in consensus description assignment.
    def _format_consensus_sample_prediction(self, sample):
        return (
            "sample #{} | chunk_node_id={} | predicted_description={!r} | "
            "predicted_label={!r} | score={:.4f} | matched_text={!r}".format(
                sample["sample_index"],
                sample["chunk_node_id"],
                sample["predicted_description"],
                sample["predicted_label"],
                self._safe_float(sample["prediction_score"]),
                sample["matched_text"],
            )
        )

    # Classify where a sem node came from so merge logs can show the full history.
    def _describe_sem_node_origin(self, sem_node_id, consensus_event, create_split_event):
        if consensus_event is not None:
            return "consensus_assignment"
        if create_split_event is not None:
            return "split_sem_node"
        return "preexisting_sem_node"

    # Coerce a possibly-None numeric field to float for safe formatting.
    @staticmethod
    def _safe_float(value):
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    # Format semantic-description logs as a small HTML details block.
    def _format_sem_description_logs_html(self, open_details=False):
        open_attr = " open" if open_details else ""
        return (
            f"<details{open_attr}>"
            "<summary>Semantic description logs</summary>"
            f"<pre>{escape(self._format_sem_description_logs_text())}</pre>"
            "</details>"
        )

    # Render the LiteSemRAG configuration (self.* attributes) as a JSON-formatted block.
    # Scalar attributes are dumped verbatim; small JSON-serializable containers (config
    # dicts/lists such as anchor_prop_params) are expanded so the dump shows their real
    # values instead of "<dict>". Runtime handles, large graph containers, per-token
    # caches, and log buffers are skipped.
    def _format_self_config_text(self):
        # Imported lazily here (not at module top) so this mixin module does not import
        # RAG_graph at load time, which would create a circular import.
        from RAG_graph import RUNTIME_FIELD_NAMES

        skip_field_names = set(RUNTIME_FIELD_NAMES) | {
            "doc_nodes", "chunk_nodes", "token_nodes", "phrase_token_nodes", "sem_nodes",
            "token_node_query", "phrase_index", "modifier_postings", "query_database",
            "predicted_sem_description_logs", "deleted_merged_sem_logs",
            "sem_description_operation_logs", "wikidata_no_result_logs",
            "_wikidata_no_result_keys", "_sem_description_candidate_bank_cache",
            "_combined_merge_sample_sense_descriptions",
            "llm_candidate_filter_result_cache",
        }
        scalar_types = (str, int, float, bool, type(None))
        # Cap on the serialized length of an expanded container so large runtime
        # caches that slipped past skip_field_names still render as "<dict>".
        max_container_chars = 4000
        config = {}
        for name, value in sorted(vars(self).items()):
            if name in skip_field_names:
                continue
            if isinstance(value, scalar_types):
                config[name] = value
                continue
            if isinstance(value, (dict, list, tuple)):
                try:
                    serialized = json.dumps(value, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    serialized = None
                if serialized is not None and len(serialized) <= max_container_chars:
                    config[name] = json.loads(serialized)
                    continue
            config[name] = f"<{type(value).__name__}>"
        return json.dumps(config, indent=2, ensure_ascii=False, default=str)

    # Save semantic-description logs and return the output path.
    def save_sem_description_logs(self, output_path=None, as_html=False, open_details=False):
        output_path = self.sem_description_log_path if output_path is None else output_path
        config_text = self._format_self_config_text()
        body_text = self._format_sem_description_logs_text()
        if as_html:
            open_attr = " open" if open_details else ""
            log_text = (
                f"<details{open_attr}>"
                "<summary>LiteSemRAG configuration</summary>"
                f"<pre>{escape(config_text)}</pre>"
                "</details>\n"
                f"<details{open_attr}>"
                "<summary>Semantic description logs</summary>"
                f"<pre>{escape(body_text)}</pre>"
                "</details>"
            )
        else:
            log_text = (
                "===== LiteSemRAG configuration =====\n"
                f"{config_text}\n"
                "===== Semantic description logs =====\n"
                f"{body_text}"
            )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        return output_path

    # Backward-compatible wrapper for callers that still use the old name.
    def print_sem_description_logs(self, output_path=None):
        return self.save_sem_description_logs(output_path=output_path)

    # Backward-compatible wrapper that now honors as_html and open_details.
    def show_sem_description_logs(self, as_html=True, open_details=False, output_path=None):
        return self.save_sem_description_logs(
            output_path=output_path,
            as_html=as_html,
            open_details=open_details,
        )

    # Print Wikidata lookup failures collected during description prediction.
    def print_wikidata_no_result_logs(self):
        print("Wikidata lookups with no usable results:")
        if not self.wikidata_no_result_logs:
            print("  (none)")
            return

        for item in self.wikidata_no_result_logs:
            print(
                "  "
                f"term={item['term']!r}, "
                f"stage={item['stage']}, "
                f"reason={item['reason']}"
            )

    # Print the approximate memory size of the graph object.
    def print_memory_size(self):
        print_size_mb(self)

    # Extract a representative sentence containing a token from a chunk.
    def _extract_sentence_for_token(self, chunk_text, token_text):
        if not chunk_text:
            return ""

        token_text_lower = token_text.lower()
        doc = self.nlp(chunk_text)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            if token_text_lower in sent_text.lower():
                return sent_text

        if len(doc) > 0:
            first_sent = next(doc.sents, None)
            if first_sent is not None:
                return first_sent.text.strip()

        return chunk_text.strip()

    # Collect representative sentence examples for one semantic node.
    def _collect_sem_sentence_examples(self, sem_node, token_text, max_sentences_per_sem=3):
        examples = []
        seen_sentences = set()

        for chunk_node in sem_node.chunk_node_list:
            sentence_text = self._extract_sentence_for_token(chunk_node.chunk_text, token_text)
            if not sentence_text:
                continue
            sentence_key = sentence_text.lower()
            if sentence_key in seen_sentences:
                continue

            seen_sentences.add(sentence_key)
            examples.append({
                "doc_name": chunk_node.doc_node.doc_name,
                "chunk_node_id": chunk_node.chunk_node_id,
                "sentence_text": sentence_text,
            })

            if len(examples) >= max_sentences_per_sem:
                break

        return examples

    # Return token nodes that have at least a requested number of semantic nodes.
    def inspect_multi_sem_token_nodes(self, min_sem_count=2, max_sentences_per_sem=3):
        result = {}

        for token_node in self.token_nodes:
            sem_count = len(token_node.sem_node_list)
            if sem_count < min_sem_count:
                continue

            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "sem_count": sem_count,
                "s_mean": getattr(token_node, "s_mean", None),
                "sem_nodes": [],
            }

            for sem_node in token_node.sem_node_list:
                result[token_node.token_text]["sem_nodes"].append({
                    "sem_node_id": sem_node.sem_node_id,
                    "description": sem_node.description,
                    "chunk_count": len(sem_node.chunk_node_list),
                    "sentence_examples": self._collect_sem_sentence_examples(
                        sem_node,
                        token_node.token_text,
                        max_sentences_per_sem=max_sentences_per_sem,
                    ),
                })

        return result

    # Aggregate the persisted s_mean over all token nodes with >= min_sem_count sem nodes.
    def compute_multi_sem_smean_stats(self, min_sem_count=2):
        # Statistics are computed over the full graph for tokens with
        # >= min_sem_count sem nodes, independent of any display truncation.
        # s_mean is persisted by semantic_type_cls during finalize, so tokens
        # that went through multi-sense splitting must have a value. However, an
        # entity / atomic phrase (force_single) can end up with multiple sem
        # nodes after later merging while still having s_mean=None. Those cases
        # count toward the token total but not the average, and are tracked
        # separately through the missing count.
        total = 0
        smean_values = []
        for token_node in self.token_nodes:
            if len(token_node.sem_node_list) < min_sem_count:
                continue
            total += 1
            s_mean = getattr(token_node, "s_mean", None)
            if s_mean is not None:
                smean_values.append(float(s_mean))
        mean_smean = (sum(smean_values) / len(smean_values)) if smean_values else None
        return {
            "min_sem_count": min_sem_count,
            "token_count": total,
            "smean_available_count": len(smean_values),
            "smean_missing_count": total - len(smean_values),
            "mean_smean": mean_smean,
        }

    # Return token nodes with semantic nodes that have descriptions.
    def inspect_described_sem_token_nodes(self, max_sentences_per_sem=3):
        result = {}

        for token_node in self.token_nodes:
            described_sem_nodes = [
                sem_node for sem_node in token_node.sem_node_list
                if getattr(sem_node, "description", None)
            ]
            if not described_sem_nodes:
                continue

            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "sem_count": len(described_sem_nodes),
                "s_mean": getattr(token_node, "s_mean", None),
                "sem_nodes": [],
            }

            for sem_node in described_sem_nodes:
                result[token_node.token_text]["sem_nodes"].append({
                    "sem_node_id": sem_node.sem_node_id,
                    "description": sem_node.description,
                    "chunk_count": len(sem_node.chunk_node_list),
                    "sentence_examples": self._collect_sem_sentence_examples(
                        sem_node,
                        token_node.token_text,
                        max_sentences_per_sem=max_sentences_per_sem,
                    ),
                })

        return result

    # Filter, sort, and truncate semantic-node inspection results.
    def _select_multi_sem_token_nodes(
            self,
            inspect_data,
            token_contains=None,
            sort_by="sem_count",
            max_token_nodes=None,
            max_sems_per_token=None,
    ):
        items = list(inspect_data.items())

        if token_contains:
            keyword = token_contains.lower()
            items = [
                (token_text, token_info)
                for token_text, token_info in items
                if keyword in token_text.lower()
            ]

        if sort_by == "token_text":
            items.sort(key=lambda x: x[0].lower())
        elif sort_by == "total_chunk_count":
            items.sort(
                key=lambda x: sum(sem_info["chunk_count"] for sem_info in x[1]["sem_nodes"]),
                reverse=True,
            )
        else:
            items.sort(key=lambda x: x[1]["sem_count"], reverse=True)

        if max_token_nodes is not None:
            items = items[:max_token_nodes]

        selected_data = {}
        for token_text, token_info in items:
            sem_nodes = token_info["sem_nodes"]
            if max_sems_per_token is not None:
                sem_nodes = sorted(
                    sem_nodes,
                    key=lambda x: x["chunk_count"],
                    reverse=True,
                )[:max_sems_per_token]

            selected_data[token_text] = {
                "token_node_id": token_info["token_node_id"],
                "sem_count": token_info["sem_count"],
                "s_mean": token_info.get("s_mean"),
                "displayed_sem_count": len(sem_nodes),
                "total_chunk_count": sum(sem_info["chunk_count"] for sem_info in token_info["sem_nodes"]),
                "sem_nodes": sem_nodes,
            }

        return selected_data

    # Limit displayed sentence examples across inspection results.
    def _limit_sem_sentence_examples(self, inspect_data, max_examples_per_token=None):
        if max_examples_per_token is None:
            return inspect_data

        limited_data = {}
        for token_text, token_info in inspect_data.items():
            remaining = max_examples_per_token
            limited_sem_nodes = []

            for sem_info in token_info["sem_nodes"]:
                if remaining <= 0:
                    limited_examples = []
                else:
                    limited_examples = sem_info["sentence_examples"][:remaining]
                remaining -= len(limited_examples)

                limited_sem = dict(sem_info)
                limited_sem["sentence_examples"] = limited_examples
                limited_sem_nodes.append(limited_sem)

            limited_token_info = dict(token_info)
            limited_token_info["sem_nodes"] = limited_sem_nodes
            limited_token_info["displayed_example_count"] = (
                max_examples_per_token - max(remaining, 0)
            )
            limited_data[token_text] = limited_token_info

        return limited_data

    # Return token-grouped LLM assignments made for Anchor samples.
    def inspect_llm_anchor_sample_assignments(self):
        result = {}
        for token_node in self.token_nodes:
            records = list(getattr(token_node, "llm_anchor_sample_assignments", []) or [])
            if not records:
                continue
            description_counts = Counter(
                record.get("assigned_description") or "(unassigned)"
                for record in records
            )
            source_counts = Counter(
                record.get("source") or "unknown"
                for record in records
            )
            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "sem_count": len(token_node.sem_node_list),
                "s_mean": getattr(token_node, "s_mean", None),
                "record_count": len(records),
                "accepted_count": sum(
                    1 for record in records if record.get("accepted_as_anchor_label")
                ),
                "description_counts": dict(description_counts),
                "source_counts": dict(source_counts),
                "records": records,
            }
        return result

    # Filter, sort, and truncate LLM Anchor assignment inspection data.
    def _select_llm_anchor_sample_assignments(
            self,
            inspect_data,
            token_contains=None,
            sort_by="record_count",
            max_token_nodes=None,
            max_records_per_token=None,
    ):
        items = list(inspect_data.items())
        if token_contains:
            keyword = token_contains.lower()
            items = [
                (token_text, token_info)
                for token_text, token_info in items
                if keyword in token_text.lower()
            ]

        if sort_by == "token_text":
            items.sort(key=lambda x: x[0].lower())
        elif sort_by == "accepted_count":
            items.sort(key=lambda x: x[1]["accepted_count"], reverse=True)
        elif sort_by == "sem_count":
            items.sort(key=lambda x: x[1]["sem_count"], reverse=True)
        else:
            items.sort(key=lambda x: x[1]["record_count"], reverse=True)

        if max_token_nodes is not None:
            items = items[:max_token_nodes]

        selected = {}
        for token_text, token_info in items:
            records = list(token_info["records"])
            records.sort(
                key=lambda record: (
                    record.get("source_sem_node_id") if record.get("source_sem_node_id") is not None else -1,
                    record.get("sample_order") if record.get("sample_order") is not None else -1,
                    record.get("anchor_position") if record.get("anchor_position") is not None else -1,
                )
            )
            if max_records_per_token is not None:
                records = records[:max_records_per_token]
            selected[token_text] = {
                **token_info,
                "displayed_record_count": len(records),
                "records": records,
            }
        return selected

    # Format LLM Anchor assignment inspection results as plain text.
    def _format_llm_anchor_sample_assignments_text(self, inspect_data):
        lines = []
        for token_text, token_info in inspect_data.items():
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            lines.append(
                f"[Token] {token_text} | token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']} | s_mean={s_mean_text} | "
                f"records={token_info['record_count']} | accepted={token_info['accepted_count']} | "
                f"displayed={token_info.get('displayed_record_count', token_info['record_count'])}"
            )
            lines.append(f"  descriptions={token_info.get('description_counts', {})}")
            lines.append(f"  sources={token_info.get('source_counts', {})}")
            for idx, record in enumerate(token_info["records"], start=1):
                lines.append(
                    "  ({}) source={} | sample_order={} | anchor_position={} | "
                    "accepted={} | description={!r} | doc={!r} | chunk_id={} | span=({}, {})".format(
                        idx,
                        record.get("source"),
                        record.get("sample_order"),
                        record.get("anchor_position"),
                        record.get("accepted_as_anchor_label"),
                        record.get("assigned_description"),
                        record.get("doc_name"),
                        record.get("chunk_node_id"),
                        record.get("span_start"),
                        record.get("span_end"),
                    )
                )
                if record.get("judgment"):
                    lines.append(
                        "      judgment={} | confidence={} | reason={!r}".format(
                            record.get("judgment"),
                            record.get("confidence"),
                            record.get("reason"),
                        )
                    )
                if record.get("matched_text"):
                    lines.append(f"      matched_text={record.get('matched_text')!r}")
                if record.get("context_text"):
                    lines.append(f"      context={record.get('context_text')}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # Format LLM Anchor assignment inspection results as clickable HTML details.
    def _build_llm_anchor_sample_assignments_html(self, inspect_data, open_details=False):
        total_tokens = len(inspect_data)
        total_records = sum(info["record_count"] for info in inspect_data.values())
        total_displayed = sum(
            info.get("displayed_record_count", info["record_count"])
            for info in inspect_data.values()
        )
        html_parts = [
            "<div style='font-family:Arial, sans-serif; line-height:1.5;'>",
            "<div style='margin-bottom:12px; padding:10px 12px; background:#f3f6f9; "
            "border:1px solid #d8e0e8; border-radius:8px;'>"
            f"<strong>LLM Anchor sample assignment token nodes:</strong> {total_tokens} | "
            f"records: {total_records} | displayed records: {total_displayed}"
            "</div>",
        ]
        open_attr = " open" if open_details else ""

        for token_text, token_info in inspect_data.items():
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            desc_counts = ", ".join(
                f"{desc}: {count}"
                for desc, count in token_info.get("description_counts", {}).items()
            ) or "-"
            source_counts = ", ".join(
                f"{source}: {count}"
                for source, count in token_info.get("source_counts", {}).items()
            ) or "-"
            token_meta = (
                f"token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']}, "
                f"s_mean={s_mean_text}, "
                f"records={token_info['record_count']}, "
                f"accepted={token_info['accepted_count']}, "
                f"displayed={token_info.get('displayed_record_count', token_info['record_count'])}"
            )
            html_parts.append(
                "<details style='margin:10px 0; border:1px solid #ddd; "
                f"border-radius:8px; padding:8px 12px; background:#fafafa;'{open_attr}>"
                f"<summary style='cursor:pointer; font-weight:700;'>{escape(token_text)}</summary>"
                f"<div style='margin:6px 0 0 2px; font-size:12px; color:#666;'>{escape(token_meta)}</div>"
                f"<div style='margin:4px 0 0 2px; font-size:12px; color:#666;'>descriptions: {escape(desc_counts)}</div>"
                f"<div style='margin:4px 0 8px 2px; font-size:12px; color:#666;'>sources: {escape(source_counts)}</div>"
            )

            for record in token_info["records"]:
                description = record.get("assigned_description") or "(unassigned)"
                summary = (
                    f"sample_order={record.get('sample_order')} | "
                    f"anchor_position={record.get('anchor_position')} | "
                    f"description={description} | "
                    f"accepted={record.get('accepted_as_anchor_label')}"
                )
                meta = (
                    f"source={record.get('source')} | method={record.get('sem_assignment_method')} | "
                    f"doc={record.get('doc_name')} | chunk_id={record.get('chunk_node_id')} | "
                    f"span=({record.get('span_start')}, {record.get('span_end')})"
                )
                judgment = (
                    f"judgment={record.get('judgment')} | sense_id={record.get('sense_id')} | "
                    f"confidence={record.get('confidence')} | reason={record.get('reason')}"
                    if record.get("judgment") or record.get("reason")
                    else ""
                )
                candidate = (
                    f"entity_id={record.get('predicted_entity_id')} | "
                    f"label={record.get('predicted_label')} | "
                    f"definition={record.get('predicted_definition')}"
                    if record.get("predicted_entity_id") or record.get("predicted_label")
                    else ""
                )
                html_parts.append(
                    "<details style='margin:8px 0 8px 16px; padding:8px 10px; "
                    "border-left:4px solid #9467bd; background:#fff;'>"
                    f"<summary style='cursor:pointer; font-weight:600;'>{escape(summary)}</summary>"
                    f"<div style='font-size:12px; color:#666; margin-top:4px;'>{escape(meta)}</div>"
                )
                if judgment:
                    html_parts.append(
                        f"<div style='font-size:12px; color:#666; margin-top:4px;'>{escape(judgment)}</div>"
                    )
                if candidate:
                    html_parts.append(
                        f"<div style='font-size:12px; color:#666; margin-top:4px;'>{escape(candidate)}</div>"
                    )
                if record.get("matched_text"):
                    html_parts.append(
                        "<div style='margin-top:6px;'><strong>Matched text:</strong> "
                        f"{escape(str(record.get('matched_text')))}</div>"
                    )
                if record.get("context_text"):
                    html_parts.append(
                        "<div style='margin-top:6px; white-space:pre-wrap;'>"
                        f"{escape(str(record.get('context_text')))}</div>"
                    )
                html_parts.append("</details>")
            html_parts.append("</details>")

        html_parts.append("</div>")
        return "".join(html_parts)

    # Display LLM-labeled Anchor samples grouped by token node.
    def show_llm_anchor_sample_assignments(
            self,
            as_html=True,
            token_contains=None,
            sort_by="record_count",
            max_token_nodes=50,
            max_records_per_token=20,
            open_details=False,
    ):
        inspect_data = self.inspect_llm_anchor_sample_assignments()
        inspect_data = self._select_llm_anchor_sample_assignments(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_records_per_token=max_records_per_token,
        )

        if not inspect_data:
            empty_text = "No LLM Anchor sample assignment records were found."
            if as_html:
                try:
                    from IPython.display import HTML
                    return HTML(f"<div>{escape(empty_text)}</div>")
                except ImportError:
                    return empty_text
            return empty_text

        if as_html:
            try:
                from IPython.display import HTML
                return HTML(
                    self._build_llm_anchor_sample_assignments_html(
                        inspect_data,
                        open_details=open_details,
                    )
                )
            except ImportError:
                pass

        return self._format_llm_anchor_sample_assignments_text(inspect_data)

    # Column order: keep token-level metadata first and raw text
    # (matched_text/context_text/span_text) later, so the table is easy to scan
    # while still preserving full context text without rendering huge nested
    # <details> blocks in notebooks.
    _LLM_ANCHOR_CSV_COLUMNS = [
        "token_text", "token_node_id", "sem_count", "s_mean",
        "source_sem_node_id", "sem_assignment_method", "source",
        "sample_order", "anchor_position", "accepted_as_anchor_label",
        "assigned_description",
        "doc_name", "chunk_node_id", "span_start", "span_end",
        "predicted_entity_id", "predicted_label", "predicted_definition",
        "prediction_score", "prediction_method", "llm_cache_hit", "llm_reason",
        "judgment", "sense_id", "confidence", "reason",
        "matched_text", "context_text", "span_text",
    ]

    # Flatten all LLM Anchor sample assignment records into one row per record,
    # including the raw text, and write them to a local file (CSV by default).
    # Returns the output file path. token_contains/sort_by are only optional
    # filtering/sorting helpers; by default all records are exported.
    def save_llm_anchor_sample_assignments_csv(
            self,
            output_path=None,
            token_contains=None,
            sort_by="record_count",
            max_token_nodes=None,
            max_records_per_token=None,
    ):
        import csv

        inspect_data = self.inspect_llm_anchor_sample_assignments()
        inspect_data = self._select_llm_anchor_sample_assignments(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_records_per_token=max_records_per_token,
        )

        if output_path is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(project_root, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(logs_dir, f"llm_anchor_assignments_{timestamp}.csv")
        else:
            parent = os.path.dirname(os.path.abspath(output_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        columns = self._LLM_ANCHOR_CSV_COLUMNS
        row_count = 0
        with open(output_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for token_info in inspect_data.values():
                token_meta = {
                    "sem_count": token_info.get("sem_count"),
                    "s_mean": token_info.get("s_mean"),
                }
                for record in token_info["records"]:
                    row = {column: record.get(column) for column in columns}
                    row.update(token_meta)
                    writer.writerow(row)
                    row_count += 1

        print(
            f"Saved {row_count} LLM Anchor sample assignment records "
            f"({len(inspect_data)} token nodes) to {output_path}"
        )
        return output_path

    # Format semantic-node inspection results as plain text.
    # Render the s_mean aggregate as a one-line plain-text summary.
    def _format_multi_sem_smean_summary_text(self, smean_stats):
        mean_smean = smean_stats.get("mean_smean")
        mean_text = f"{mean_smean:.4f}" if mean_smean is not None else "N/A"
        return (
            f"[s_mean summary] token nodes with >= {smean_stats['min_sem_count']} sem nodes: "
            f"{smean_stats['token_count']} | mean s_mean = {mean_text} "
            f"(over {smean_stats['smean_available_count']} with s_mean, "
            f"{smean_stats['smean_missing_count']} missing)"
        )

    # Render the s_mean aggregate as an HTML summary box.
    def _build_multi_sem_smean_summary_html(self, smean_stats):
        mean_smean = smean_stats.get("mean_smean")
        mean_text = f"{mean_smean:.4f}" if mean_smean is not None else "N/A"
        return (
            "<div style='font-family:Arial, sans-serif; margin-bottom:12px; padding:10px 12px; "
            "background:#eef7ee; border:1px solid #cfe3cf; border-radius:8px;'>"
            f"<strong>s_mean summary</strong> — token nodes with &ge; {smean_stats['min_sem_count']} sem nodes: "
            f"<strong>{smean_stats['token_count']}</strong> | "
            f"mean s_mean = <strong>{escape(mean_text)}</strong> "
            f"<span style='color:#666;'>(over {smean_stats['smean_available_count']} with s_mean, "
            f"{smean_stats['smean_missing_count']} missing)</span>"
            "</div>"
        )

    def _format_multi_sem_token_nodes_text(self, inspect_data):
        lines = []

        for token_text, token_info in inspect_data.items():
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            lines.append(
                f"[Token] {token_text} | token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']} | "
                f"s_mean={s_mean_text} | "
                f"displayed_sem_count={token_info['displayed_sem_count']} | "
                f"total_chunk_count={token_info['total_chunk_count']} | "
                f"displayed_example_count={token_info.get('displayed_example_count', 'all')}"
            )

            for sem_info in token_info["sem_nodes"]:
                description = sem_info.get("description") or "(none)"
                lines.append(
                    f"  [Sem] sem_node_id={sem_info['sem_node_id']} | "
                    f"description={description!r} | "
                    f"chunk_count={sem_info['chunk_count']}"
                )

                for idx, example in enumerate(sem_info["sentence_examples"], start=1):
                    lines.append(
                        f"    ({idx}) doc={example['doc_name']} | chunk_id={example['chunk_node_id']}"
                    )
                    lines.append(f"        {example['sentence_text']}")

            lines.append("")

        return "\n".join(lines).rstrip()

    # Format semantic-node inspection results as HTML.
    def _build_multi_sem_token_nodes_html(self, inspect_data, open_details=False):
        total_tokens = len(inspect_data)
        html_parts = [
            "<div style='font-family:Arial, sans-serif; line-height:1.5;'>"
        ]
        html_parts.append(
            "<div style='margin-bottom:12px; padding:10px 12px; background:#f3f6f9; "
            "border:1px solid #d8e0e8; border-radius:8px;'>"
            f"<strong>Matched token nodes:</strong> {total_tokens}"
            "</div>"
        )

        for token_text, token_info in inspect_data.items():
            token_header = (
                f"{escape(token_text)}"
            )
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            token_meta = (
                f"<span style='color:#666;'>"
                f"token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']}, "
                f"s_mean={s_mean_text}, "
                f"displayed_sem_count={token_info['displayed_sem_count']}, "
                f"total_chunk_count={token_info['total_chunk_count']}, "
                f"displayed_example_count={token_info.get('displayed_example_count', 'all')}"
                f"</span>"
            )
            html_parts.append(
                "<details style='margin:10px 0; border:1px solid #ddd; "
                f"border-radius:8px; padding:8px 12px; background:#fafafa;' {'open' if open_details else ''}>"
                f"<summary style='cursor:pointer; font-weight:700;'>{token_header}</summary>"
                f"<div style='margin:6px 0 0 2px; font-size:12px;'>{token_meta}</div>"
            )

            for sem_info in token_info["sem_nodes"]:
                description = sem_info.get("description") or "(none)"
                sem_header = (
                    f"sem_node_id={sem_info['sem_node_id']} | "
                    f"description={description} | "
                    f"chunk_count={sem_info['chunk_count']}"
                )
                html_parts.append(
                    "<div style='margin:10px 0 6px 16px; padding:8px 10px; "
                    "border-left:4px solid #4c78a8; background:#fff;'>"
                    f"<div style='font-weight:600; margin-bottom:6px;'>{escape(sem_header)}</div>"
                )

                for example in sem_info["sentence_examples"]:
                    meta = (
                        f"doc={example['doc_name']} | "
                        f"chunk_id={example['chunk_node_id']}"
                    )
                    html_parts.append(
                        "<div style='margin:8px 0 10px 8px;'>"
                        f"<div style='font-size:12px; color:#666; margin-bottom:2px;'>{escape(meta)}</div>"
                        f"<div style='white-space:pre-wrap;'>{escape(example['sentence_text'])}</div>"
                        "</div>"
                    )

                html_parts.append("</div>")

            html_parts.append("</details>")

        html_parts.append("</div>")
        return "".join(html_parts)

    # Display token nodes that have multiple semantic nodes.
    def show_multi_sem_token_nodes(
            self,
            min_sem_count=2,
            max_sentences_per_sem=3,
            as_html=True,
            token_contains=None,
            sort_by="sem_count",
            max_token_nodes=20,
            max_sems_per_token=5,
            max_examples_per_token=10,
            open_details=False,
    ):
        inspect_data = self.inspect_multi_sem_token_nodes(
            min_sem_count=min_sem_count,
            max_sentences_per_sem=max_sentences_per_sem,
        )
        inspect_data = self._select_multi_sem_token_nodes(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_sems_per_token=max_sems_per_token,
        )
        inspect_data = self._limit_sem_sentence_examples(
            inspect_data,
            max_examples_per_token=max_examples_per_token,
        )

        # Compute the average s_mean over the full graph for tokens with
        # >= min_sem_count, independent of token_contains / max_token_nodes
        # truncation, and show it as summary information.
        smean_stats = self.compute_multi_sem_smean_stats(min_sem_count=min_sem_count)

        if not inspect_data:
            empty_text = "No token nodes matched the multi-semantic condition."
            if as_html:
                try:
                    from IPython.display import HTML
                    return HTML(f"<div>{escape(empty_text)}</div>")
                except ImportError:
                    return empty_text
            return empty_text

        if as_html:
            try:
                from IPython.display import HTML
                return HTML(
                    self._build_multi_sem_smean_summary_html(smean_stats)
                    + self._build_multi_sem_token_nodes_html(inspect_data, open_details=open_details)
                )
            except ImportError:
                pass

        summary_text = self._format_multi_sem_smean_summary_text(smean_stats)
        body_text = self._format_multi_sem_token_nodes_text(inspect_data)
        return f"{summary_text}\n\n{body_text}" if summary_text else body_text

    # Display token nodes whose semantic nodes have descriptions.
    def show_described_sem_token_nodes(
            self,
            max_sentences_per_sem=3,
            as_html=True,
            token_contains=None,
            sort_by="sem_count",
            max_token_nodes=20,
            max_sems_per_token=10,
            max_examples_per_token=10,
            open_details=False,
    ):
        inspect_data = self.inspect_described_sem_token_nodes(
            max_sentences_per_sem=max_sentences_per_sem,
        )
        inspect_data = self._select_multi_sem_token_nodes(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_sems_per_token=max_sems_per_token,
        )
        inspect_data = self._limit_sem_sentence_examples(
            inspect_data,
            max_examples_per_token=max_examples_per_token,
        )

        if not inspect_data:
            empty_text = "No sem nodes with descriptions were found."
            if as_html:
                try:
                    from IPython.display import HTML
                    return HTML(f"<div>{escape(empty_text)}</div>")
                except ImportError:
                    return empty_text
            return empty_text

        if as_html:
            try:
                from IPython.display import HTML
                return HTML(self._build_multi_sem_token_nodes_html(inspect_data, open_details=open_details))
            except ImportError:
                pass

        return self._format_multi_sem_token_nodes_text(inspect_data)

