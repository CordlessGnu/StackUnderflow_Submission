"""Legacy pipeline CSV analytics follow names without rewriting source values."""
from geolens.services.data_processor import build_document


def test_pipeline_csv_ids_follow_names_and_preserve_original_visual_ids():
    csv = (b'top_y,bottom_y,cluster_id,lithology,top_depth,base_depth,match_status,lithology_fallback\n'
           b'0,10,C100,SHALE,0,10,matched,False\n'
           b'10,20,C400,Shale,10,20,ambiguous,True\n'
           b'20,30,C900,,20,30,blank,False\n')
    record = build_document(csv_bytes=csv, csv_name='old-pipeline.csv')
    assert record.source_df['cluster_id'].tolist() == ['C100', 'C400', 'C900']
    assert record.processed_df['cluster_id'].tolist() == ['C001', 'C001', 'BLANK']
    assert record.processed_df['visual_cluster_id'].tolist() == ['C100', 'C400', 'C900']
    assert bool(record.processed_df.loc[1, 'is_runner_up_fallback'])


def test_generic_csv_keeps_its_source_clustering_semantics():
    csv = (b'top_y,bottom_y,cluster_id,lithology,top_depth,base_depth\n'
           b'0,10,A,Shale,0,10\n10,20,B,Shale,10,20\n')
    record = build_document(csv_bytes=csv, csv_name='external.csv')
    assert record.processed_df['cluster_id'].tolist() == ['A', 'B']
