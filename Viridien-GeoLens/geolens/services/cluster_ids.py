"""Use the extraction pipeline's pure label-to-ID policy for legacy CSV views.

Only a standard-library module is loaded; ML/segmentation modules stay out of
Streamlit. Original CSV values remain in DocumentRecord.source_df.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path

POLICY = Path(__file__).resolve().parents[3] / 'info extraction/lithology_groups.py'


@lru_cache(maxsize=1)
def _policy(modification_time: int):
    spec = importlib.util.spec_from_file_location('geolens_lithology_groups', POLICY)
    if spec is None or spec.loader is None:
        raise RuntimeError('The shared lithology cluster policy could not be loaded.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.consolidate_lithology_clusters


def align_pipeline_cluster_ids(frame):
    """Normalize pipeline-style data only; generic six-column CSVs are untouched."""
    if ('cluster_id' not in frame or 'lithology' not in frame or 'match_status' not in frame
            or not ({'runner_up', 'lithology_fallback'} & set(frame.columns))):
        return frame
    records, _ = _policy(POLICY.stat().st_mtime_ns)(frame.to_dict('records'))
    out = frame.copy()
    out['cluster_id'] = [row['cluster_id'] for row in records]
    out['visual_cluster_id'] = [row['visual_cluster_id'] for row in records]
    return out
