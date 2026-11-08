"""Public cluster IDs follow final names; visual groups remain traceable.

Pure standard-library policy, shared by extraction exports and GeoLens's CSV
adapter. Call after resolving runner-up/blank labels. No inference or I/O occurs.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence


def _text(value) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ''
    return ' '.join(str(value).split())


def _true(value) -> bool:
    return _text(value).casefold() in ('true', '1', 'yes')


def _number(value, default=0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def consolidate_lithology_clusters(rows: Sequence[Mapping], source_clusters: Sequence[Mapping] = ()
                                   ) -> tuple[list[dict], list[dict]]:
    """Assign C001... by first named appearance; use BLANK/UNKNOWN buckets.

    Names compare case-insensitively with normalized whitespace. Labels, interval
    edges, depths and match evidence are not changed. Original visual IDs are
    retained in visual_cluster_id, making repeated normalization idempotent.
    Group summaries use a representative's evidence, not invented group confidence.
    """
    records, groups, names = [], {}, {}
    for original in rows:
        row = dict(original)
        visual = _text(row.get('visual_cluster_id')) or _text(row.get('cluster_id'))
        row['visual_cluster_id'] = visual
        label = _text(row.get('lithology'))
        status = _text(row.get('match_status')).casefold()
        if status == 'blank':
            cluster_id = 'BLANK'
        elif not label or label.casefold() == 'unknown':
            cluster_id = 'UNKNOWN'
        else:
            key = label.casefold()
            if key not in names:
                names[key] = f'C{len(names) + 1:03d}'
            cluster_id = names[key]
        row['cluster_id'] = cluster_id
        groups.setdefault(cluster_id, []).append(len(records))
        records.append(row)

    by_index, by_visual = {}, {}
    for summary in source_clusters:
        by_visual[_text(summary.get('cluster_id'))] = summary
        for index in summary.get('member_indices', []):
            if isinstance(index, int) and 0 <= index < len(records):
                by_index[index] = summary

    summaries = []
    for cluster_id, members in groups.items():
        def quality(index):
            row = records[index]
            verified = _text(row.get('match_status')).casefold() == 'matched' and not _true(row.get('lithology_fallback'))
            distance = row.get('fallback_distance') if _true(row.get('lithology_fallback')) else row.get('match_distance')
            return (not verified, -max(0.0, _number(row.get('bottom_y')) - _number(row.get('top_y'))),
                    _number(distance, math.inf), index)
        representatives = sorted(members, key=quality)[:3]
        representative = representatives[0]
        row = records[representative]
        source = by_index.get(representative, by_visual.get(row['visual_cluster_id'], {}))
        summary = dict(source)
        statuses = Counter(_text(records[i].get('match_status')) for i in members)
        fallback_count = sum(_true(records[i].get('lithology_fallback')) for i in members)
        verified_count = sum(_text(records[i].get('match_status')).casefold() == 'matched'
                             and not _true(records[i].get('lithology_fallback')) for i in members)
        summary.update(cluster_id=cluster_id, lithology=row.get('lithology', ''),
                       lithology_labels=list(dict.fromkeys(_text(records[i].get('lithology')) for i in members)),
                       interval_count=len(members), member_indices=members,
                       representative_index=representative, representative_indices=representatives,
                       visual_cluster_ids=list(dict.fromkeys(records[i]['visual_cluster_id'] for i in members)),
                       grouping_status='by_final_lithology',
                       match_status=next(iter(statuses)) if len(statuses) == 1 else 'mixed',
                       match_status_counts=dict(statuses), verified_intervals=verified_count,
                       fallback_intervals=fallback_count, lithology_fallback=fallback_count == len(members),
                       candidates=source.get('candidates', []))
        for field in ('match_distance', 'match_margin', 'runner_up', 'fallback_distance',
                      'match_descriptor_distance', 'match_seam_score', 'match_seam_status'):
            summary[field] = row.get(field)
        summaries.append(summary)
    return records, summaries
