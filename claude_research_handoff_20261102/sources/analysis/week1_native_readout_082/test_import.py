"""Synthetic exporter/importer E2E checks. No real result is generated."""
import collections
import copy
import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import import_native as imp

HERE=Path(__file__).resolve().parent
EXPORT_PACKAGE=imp.DEFAULT_PACKAGE


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def csv_write(path, rows, fields):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:("NA" if v is None else str(v).lower() if isinstance(v,bool) else json.dumps(v) if isinstance(v,dict) else v) for k,v in row.items()})


def digest_record(path):
    path=Path(path)
    return dict(path=str(path),sha256=imp.sha(path),bytes=path.stat().st_size)


def refresh_manifest(export):
    manifest=imp.load(export/"manifest.json")
    manifest["outputs"]={p.name:digest_record(p) for p in export.iterdir() if p.is_file() and p.name not in ("status.json","manifest.json")}
    dump(export/"manifest.json",manifest)


def fixture(root,native_f=True):
    package=root/"package";export=root/"export";package.mkdir();export.mkdir()
    for name in ("schema.json","export_native.py","native_geometry.py","PROTOCOL.md"):
        shutil.copyfile(EXPORT_PACKAGE/name,package/name)
    schema=imp.load(package/"schema.json")
    common=dict(subject_class="person",object_class="cup",fps=10.,coexist_sec=20.,native_F=native_f,
                original_group="C",VrdONE_hit=False,BIG_hit=False,
                slices={d:"S-L-solo" for d in imp.DEFINITIONS},competition_flags={d:False for d in imp.DEFINITIONS})
    metadata=[dict(common,video_name="a",gt_index=0,subject_tid=1,object_tid=2,predicate="hold",begin_fid=10,end_fid=20,relation_sec=1.),
              dict(common,video_name="a",gt_index=1,subject_tid=1,object_tid=2,predicate="hold",begin_fid=30,end_fid=40,relation_sec=1.),
              dict(common,video_name="b",gt_index=0,subject_tid=1,object_tid=3,predicate="hold",begin_fid=10,end_fid=20,relation_sec=1.)]
    metadata_path=root/"per_gt_slices.csv"
    csv_write(metadata_path,metadata,list(metadata[0]))
    exported_metadata=[{k:r[k] for k in imp.METADATA_FIELDS} for r in metadata]
    csv_write(package/"metadata.csv",exported_metadata,list(imp.METADATA_FIELDS))
    inventory=dict(video_names=["a","b"],N_selected_videos=2);dump(package/"inventory.json",inventory)
    receipt=dict(metadata_source_sha256=imp.sha(metadata_path),metadata_csv_sha256=imp.sha(package/"metadata.csv"),
                 inventory_sha256=imp.sha(package/"inventory.json"),N_GT=3,N_GT_bearing_videos=2,N_inventory_videos=2,
                 baseline_all_pass=True,maximum_score_diff=0.)
    dump(package/"input_receipt.json",receipt);dump(export/"input_receipt.json",receipt)
    dump(export/"schema.json",schema)
    # Reuse the exporter temporal-row helper only to construct synthetic inputs;
    # importer independently reconstructs all expected choices from saved ranks.
    spec=importlib.util.spec_from_file_location("geometry_fixture",package/"native_geometry.py")
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    binding_rows=[];pairs=[];candidates=[];temporal=[];receipts=[];sources={}
    for name,hash_value in imp.SOURCE_EXPECTED.items():
        sources[name]=dict(path="synthetic/"+name,sha256=hash_value,bytes=1)
    sources["metadata"]=digest_record(package/"metadata.csv");sources["inventory"]=digest_record(package/"inventory.json")
    for video in ("a","b"):
        # b has both GT roles separately eligible but NEVER the actual (1,3)
        # identity pair. The Cartesian-product shortcut would be wrong.
        actual=[(0,1)] if video=="a" else [(0,1),(1,2)]
        mapped=[1,2] if video=="a" else [1,2,3]
        cats=["person","cup"] if video=="a" else ["person","cup","cup"]
        bindings={}
        for i,tid in enumerate(mapped):
            b=dict(mapped_tid=tid,status="bound",best_tid=tid,mean_iou=.9,coverage=1.,second_mean_iou=.1,margin=.8,
                   N_same_class=2,low_geometry=False,low_coverage=False,ambiguous=False)
            bindings[i]=b
            binding_rows.append(dict(video_name=video,proposal_id=i,input_eligible=True,proposal_begin_fid=0,proposal_end_fid=200,
                                     proposal_category=cats[i],**b))
        pairs.extend(dict(video_name=video,pair_idx=i,sid=s,oid=o) for i,(s,o) in enumerate(actual))
        predictions=[];selected=[]
        if video=="a":
            for rank in range(1,56):
                predicate="hold" if rank==55 else "watch"
                predictions.append(dict(duration=[10,20],score=1-rank/1000,triplet=["person",predicate,"cup"]))
                s=dict(sid=0,oid=1,pair_idx=0,candidate_id=rank-1,query=rank%9)
                selected.append(s)
                candidates.append(dict(video_name=video,saved_rank_1based=rank,pred_score=predictions[-1]["score"],
                    prediction_key=f"{video}:saved_rank:{rank}",pred_begin_fid=10,pred_end_fid=20,subject_class="person",predicate=predicate,object_class="cup",
                    source_sid=0,source_oid=1,source_pair_idx=0,source_candidate_id=rank-1,source_query=rank%9,selected_input_eligible=True,
                    subject_binding_status="bound",object_binding_status="bound",mapped_subject_tid=1,mapped_object_tid=2))
        metas=[r for r in exported_metadata if r["video_name"]==video]
        generated=geometry.temporal_rows(video,metas,predictions,{"selected":selected},{"pairs":actual},bindings)
        temporal.extend(generated)
        vs={}
        for tag in ("cache","predictions","selected","annotation"):
            rec=dict(path=f"synthetic/{video}/{tag}",sha256="a"*64,bytes=1)
            sources[tag+":"+video]=rec;vs[tag]=rec
        receipts.append(dict(video_name=video,empty_input=False,N_GT=len(metas),N_predictions=len(predictions),N_selected_checked=len(predictions),
            N_proposals=len(mapped),N_eligible_proposals=len(mapped),N_actual_input_pairs=len(actual),
            binding_status_counts={"bound":len(mapped)},N_output_rows_by_topk={str(k):len(metas) for k in imp.TOPKS},source_files=vs))
    tables={"proposal_binding.csv":binding_rows,"actual_pairs.csv":pairs,"saved_candidates.csv":candidates,"per_gt_temporal.csv":temporal}
    for name,rows in tables.items():
        csv_write(export/name,rows,[field["name"] for field in schema["csv"][name]])
    with (export/"per_video_receipts.jsonl").open("w") as handle:
        for row in receipts:handle.write(json.dumps(row)+"\n")
    config=dict(topks=list(imp.TOPKS),mapping_rule=dict(mean_min=.5,coverage_min=.8,margin_min=.05),
                loader_attributes=dict(with_clip_feature=False,feat_stride=4,proposal_min_frames=5,random_stride=False,stride_offset=0),viou_threshold='unmodified loader default',
                relation_time_used_for_identity_or_prediction_selection=False,official_recall=False,inference_forward=False,limit_videos=2)
    summaries=[]
    for k in imp.TOPKS:
        for population in ('all_GT','mapped_actual_input_pair'):
            rs=[r for r in temporal if r['topk']==k and (population=='all_GT' or r['mapped_input_pair'])]
            ps=[r for r in rs if r['pred_available']]
            summ=dict(topk=k,population=population,N=len(rs),N_pred=len(ps),no_prediction=len(rs)-len(ps),N_videos=len({r['video_name'] for r in rs}),
                      tiou_sum=sum(r['tiou'] for r in rs),ratio_sum=sum(r['duration_ratio'] for r in ps),
                      mean_tiou=sum(r['tiou'] for r in rs)/len(rs) if rs else None,duration_bias=sum(r['duration_ratio'] for r in ps)/len(ps) if ps else None)
            for tag,t in (('03',.3),('05',.5),('07',.7)):summ['miss_count_tiou'+tag]=sum(r['tiou']<t for r in rs)
            summaries.append(summ)
    dump(export/"config.json",config);dump(export/"summary.json",dict(official_recall=False,rows=summaries))
    final=dict(status="COMPLETE_PARTIAL_SMOKE",complete=False,complete_selected_scope=True,N_videos=2,N_GT=3,
               N_rows_by_topk={str(k):3 for k in imp.TOPKS},videos_completed=["a","b"])
    dump(export/"status.json",final)
    manifest=dict(final,video_names=["a","b"],source_files=sources,outputs={},code={name:imp.sha(package/name) for name in ("export_native.py","native_geometry.py")},
                  protocol=digest_record(package/'PROTOCOL.md'),model_forwards=0,checkpoint_loaded=False,cuda_initialized=False,models_imported=False,N_saved_prediction_checks=55)
    dump(export/"manifest.json",manifest);refresh_manifest(export)
    return export,metadata_path,package,schema


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix="synthetic_test_")
        self.root=Path(self.temp.name)
        self.export,self.metadata,self.package,self.schema=fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        return imp.verify_export(self.export,self.metadata,self.package,allow_partial=True)

    def mutate_csv(self,name,change,rehash=True):
        rows=imp.typed_csv(self.export/name,self.schema["csv"][name]);change(rows)
        csv_write(self.export/name,rows,[f["name"] for f in self.schema["csv"][name]])
        if rehash:refresh_manifest(self.export)

    def test_end_to_end_partial_explicit_and_no_cartesian_pair(self):
        verified=self.verify()
        self.assertFalse(verified["rows_by_k"][50][("a",0)]["pred_available"])
        self.assertEqual(verified["rows_by_k"][100][("a",0)]["saved_rank_1based"],55)
        self.assertFalse(verified["rows_by_k"][100][("b",0)]["mapped_input_pair"])
        out=self.root/"imported"
        result=imp.import_export(self.export,out,metadata_path=self.metadata,package_root=self.package,allow_partial=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["status"],"PARTIAL_DIAGNOSTIC_ONLY")
        self.assertTrue((out/"diag_summary.csv").is_file())
        self.assertTrue((out/"official_historical_top100.csv").is_file())
        with (out/'diag_summary.csv').open(encoding='utf-8-sig') as handle:
            summaries=list(csv.DictReader(handle))
        total=next(r for r in summaries if r['analysis_population']=='all' and r['slice']=='ALL'
                   and r['slice_definition']==imp.DEFINITIONS[0] and 'Top100_' in r['readout'])
        self.assertEqual((total['N'],total['N_pred'],total['no_pred_count']),('3','2','1'))
        self.assertAlmostEqual(float(total['mean_tiou']),1/3)
        with (out/'readout_audit.csv').open(encoding='utf-8-sig') as handle:
            audits=list(csv.DictReader(handle))
        audit=next(r for r in audits if r['topk']=='100' and r['population']=='all' and r['slice_definition']==imp.DEFINITIONS[0])
        self.assertEqual((audit['disjoint_numerator'],audit['disjoint_denominator']),('1','2'))
        self.assertEqual((audit['reused_prediction_GT_numerator'],audit['reused_prediction_GT_denominator']),('2','2'))
        with self.assertRaises(ValueError):
            imp.import_export(self.export,out,metadata_path=self.metadata,package_root=self.package,allow_partial=True)

    def test_partial_default_rejected(self):
        with self.assertRaises(ValueError):
            imp.verify_export(self.export,self.metadata,self.package)

    def test_running_never_accepted_even_with_partial(self):
        status=imp.load(self.export/"status.json");status["status"]="PARTIAL_RUNNING"
        dump(self.export/"status.json",status)
        with self.assertRaises(ValueError):self.verify()

    def test_falsely_declared_fullval_rejected(self):
        for name in ("manifest.json","status.json"):
            m=imp.load(self.export/name);m.update(complete=True,status="COMPLETE_FULLVAL_CPU_EXPORT");dump(self.export/name,m)
        with self.assertRaises(ValueError):self.verify()

    def test_file_tampering_rejected_before_statistics(self):
        self.mutate_csv("per_gt_temporal.csv",lambda rows:rows[0].update(tiou=.4),rehash=False)
        with self.assertRaises(ValueError):self.verify()

    def test_duplicate_and_missing_gt_rejected_after_rehash(self):
        self.mutate_csv("per_gt_temporal.csv",lambda rows:rows.append(copy.deepcopy(rows[0])))
        with self.assertRaises(ValueError):self.verify()

    def test_missing_gt_rejected(self):
        self.mutate_csv("per_gt_temporal.csv",lambda rows:rows.pop())
        with self.assertRaises(ValueError):self.verify()

    def test_metadata_change_rejected(self):
        self.mutate_csv("per_gt_temporal.csv",lambda rows:rows[0].update(predicate="wrong"))
        with self.assertRaises(ValueError):self.verify()

    def test_tiou_and_disjoint_recomputed(self):
        def change(rows):
            chosen=next(r for r in rows if r['pred_available'])
            chosen['tiou']=.5
        self.mutate_csv("per_gt_temporal.csv",change)
        with self.assertRaises(ValueError):self.verify()

    def test_false_prediction_ratio_not_zero(self):
        self.mutate_csv("per_gt_temporal.csv",lambda rows:rows[0].update(duration_ratio=0))
        with self.assertRaises(ValueError):self.verify()

    def test_actual_pair_flag_independently_checked(self):
        def change(rows):
            next(r for r in rows if r['video_name']=='b').update(mapped_input_pair=True,geometry_binding_status='mapped_actual_pair')
        self.mutate_csv("per_gt_temporal.csv",change)
        with self.assertRaises(ValueError):self.verify()

    def test_rank_selection_and_reuse_verified(self):
        def change(rows):
            next(r for r in rows if r['pred_available']).update(saved_rank_1based=54)
        self.mutate_csv("per_gt_temporal.csv",change)
        with self.assertRaises(ValueError):self.verify()

    def test_candidate_completeness_from_receipt(self):
        self.mutate_csv("saved_candidates.csv",lambda rows:rows.pop())
        with self.assertRaises(ValueError):self.verify()

    def test_raw_candidate_pair_not_eligible(self):
        self.mutate_csv("saved_candidates.csv",lambda rows:rows[0].update(selected_input_eligible=False))
        with self.assertRaises(ValueError):self.verify()

    def test_reuse_and_disjoint_flags_cannot_be_invented(self):
        def change(rows):
            next(r for r in rows if r['pred_available']).update(same_prediction_GT_uses=1)
        self.mutate_csv("per_gt_temporal.csv",change)
        with self.assertRaises(ValueError):self.verify()

    def test_binding_threshold_status_is_recomputed(self):
        self.mutate_csv('proposal_binding.csv',lambda rows:rows[0].update(mean_iou=.4,margin=.3))
        with self.assertRaises(ValueError):self.verify()

    def test_zero_native_f_population_still_emitted(self):
        root=self.root/'zero';root.mkdir()
        export,metadata,package,_=fixture(root,native_f=False)
        out=root/'imported'
        imp.import_export(export,out,metadata_path=metadata,package_root=package,allow_partial=True)
        with (out/'diag_summary.csv').open(encoding='utf-8-sig') as handle:
            rows=[r for r in csv.DictReader(handle) if r['analysis_population']=='native_F']
        self.assertEqual(len(rows),3*4*7)
        self.assertTrue(all(r['N']=='0' and r['mean_tiou']=='NA' for r in rows))

    def test_protocol_hash_must_match_trusted_package(self):
        manifest=imp.load(self.export/'manifest.json')
        manifest['protocol']['sha256']='0'*64
        dump(self.export/'manifest.json',manifest)
        with self.assertRaises(ValueError):self.verify()

    def test_empty_input_cannot_have_actual_pairs_or_candidates(self):
        path=self.export/'per_video_receipts.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['empty_input']=True
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        refresh_manifest(self.export)
        with self.assertRaises(ValueError):self.verify()


if __name__=="__main__":
    unittest.main()
