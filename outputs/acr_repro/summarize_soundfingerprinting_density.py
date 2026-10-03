"""Derive frozen PEX/expanded MinHash density controls with independent gates.

Reference thinning retains native original sequence numbers and physical times;
queries remain dense. Original835 predictions can be extracted from the same
extended run because gallery, query bounds and matching configuration coincide.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from summarize_results import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-protocol',type=Path,required=True)
    parser.add_argument('--extended-protocol',type=Path,required=True)
    parser.add_argument('--extended-results',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--original-output-root',type=Path,required=True)
    args=parser.parse_args()
    original=json.loads(args.original_protocol.read_text());extended=json.loads(args.extended_protocol.read_text())
    original_ids={q['query_id'] for q in original['queries'] if q['evaluate']}
    extended_ids={q['query_id'] for q in extended['queries'] if q['evaluate']}
    gallery=lambda p:[r for r in p['references'] if r['role'] in p['protocol']['gallery_roles']]
    assert gallery(original)==gallery(extended)
    assert len(original_ids)==835 and len(extended_ids)==5835
    original_queries={q['query_id']:q for q in original['queries'] if q['evaluate']}
    extended_queries={q['query_id']:q for q in extended['queries'] if q['evaluate']}
    input_keys=('path','start_s','duration_s','expected_reference_start_s','expected_time_scale',
                'reference_id','source_id','role','evaluate')
    for query_id in original_ids:
        assert all(original_queries[query_id][key]==extended_queries[query_id][key] for key in input_keys)
    summary=[]
    for path in args.extended_results:
        result=json.loads(path.read_text());factor=result['factor']
        assert factor in [2,4] and result['config']['threshold_votes']==4
        assert result['config']['reference_thinning_factor']==factor
        assert result['protocol_sha256']==digest(args.extended_protocol)
        predictions=result['predictions'];assert len(predictions)==5835
        assert {p['query_id'].split('@')[0] for p in predictions}==extended_ids
        assert len({p['query_id'] for p in predictions})==5835
        stats=result['stats'];assert stats['reference_cache_hits']==659
        assert stats['original_dense_reference_fingerprint_count']==1605108
        assert stats['reference_feature_hash_payload_bytes']==stats['reference_fingerprint_count']*100
        assert all(r['original_records_and_physical_timestamps_preserved'] for r in stats['reference_thinning_audit'])
        assert sum(r['retained_count'] for r in stats['reference_thinning_audit'])==stats['reference_fingerprint_count']
        assert all(r['retained_first_sequence']%factor==0 and r['retained_last_sequence']%factor==0 for r in stats['reference_thinning_audit'])
        subset=copy.deepcopy(result)
        subset['predictions']=[p for p in subset['predictions'] if p['query_id'].split('@')[0] in original_ids]
        assert len(subset['predictions'])==835
        subset['protocol_sha256']=digest(args.original_protocol)
        subset['protocol']=original['protocol']
        subset['derived_from_extended_run']={'source_result_path':str(path),'source_result_sha256':digest(path),
            'source_protocol_sha256':digest(args.extended_protocol),'same_ordered_gallery':True,
            'same_query_bounds_and_dense_fingerprint_config':True,'scores_candidates_offsets_unchanged':True,
            'pcm_preparation_hash_retained_from_actual_extended_manifest':True}
        directory=args.original_output_root/f'soundfingerprinting_density_factor{factor}'
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'soundfingerprinting_5s.json').write_text(json.dumps(subset,indent=2)+'\n')
        for population,value,protocol in [('original',subset,original),('expanded',result,extended)]:
            row=summarize(value,protocol)
            row.update(population=population,reference_thinning_factor=factor,
                reference_hash_payload_bytes_per_audio_hour=stats['reference_feature_hash_payload_bytes']/stats['reference_audio_duration_s']*3600,
                queries_remain_native_dense=True,query_fingerprint_count_values=sorted({p['query_fingerprint_count'] for p in predictions}),
                reference_sequence_and_timestamps_preserved=True,density_specific_matcher_configuration_changed=False)
            summary.append(row)
            k=row['test_known']['all'];print(population,'factor',factor,'raw',k['correct_reference']['successes'],
                'accepted',k['accepted_correct']['successes'],'localized',k['accepted_localized']['successes'],
                'unknown',row['unknown_test']['false_accepts'],'threshold',row['threshold'],'payload/h',row['reference_hash_payload_bytes_per_audio_hour'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
