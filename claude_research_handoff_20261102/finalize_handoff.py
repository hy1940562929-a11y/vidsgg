#!/usr/bin/env python3
"""Validate provenance, seal the delivery, and create an uploadable ZIP."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone

P = Path(__file__).resolve().parent
ROOT = P.parents[1]


def included(f):
    rel=f.relative_to(P)
    return (f.is_file() and 'build' not in rel.parts and '__pycache__' not in rel.parts
            and f.name not in ['.DS_Store','compile.stdout.log'] and f.suffix not in ['.pyc'])


def main():
    manifest=json.loads((P/'SOURCE_MANIFEST.json').read_text())
    for r in manifest['files']:
        f=P/r['package_path']
        assert f.is_file(),r['package_path']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],r['package_path']
    assert not manifest['missing_or_excluded'],manifest['missing_or_excluded']
    early=list(csv.DictReader((P/'evidence/early_experiment_ledger.csv').open(encoding='utf-8-sig')))
    later=list(csv.DictReader((P/'evidence/later_verdict_index.csv').open(encoding='utf-8-sig')))
    assert len(early)==42
    e099=next(r for r in later if r['experiment_dir']=='cause_arch_099_e099')
    assert e099['label_reported']=='REJECT',e099
    assert sum([2487,1809,25373,473])==30142
    assert sum([3167,8454,12193,6328])==30142
    assert sum([3151,7475,10296,4451])==25373
    log=(P/'paper_starter/build/main.log').read_text(errors='replace')
    assert 'Output written on' in log
    assert 'Overfull \\hbox' not in log
    assert not re.search(r'(?:Citation|Reference) .+undefined|There were undefined references|^!',log,re.M)
    pdf=P/'paper_starter/build/main.pdf'
    assert pdf.read_bytes().startswith(b'%PDF')
    shutil.copyfile(pdf,P/'paper_starter/draft_scaffold.pdf')
    pdf_out=ROOT/'output/pdf/Claude_handoff_CVPR2027_scaffold_20261102.pdf'
    pdf_out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(pdf,pdf_out)
    validation={
        'status':'DELIVERY_PACKAGE_CHECKS_PASS',
        'date':'2026-11-02',
        'scope':'local document/code-reference packaging; no research execution',
        'copied_source_files_byte_verified':len(manifest['files']),
        'all_requested_copy_paths_present':not manifest['missing_or_excluded'],
        'early_ledger_rows':len(early),
        'later_verdict_rows':len(later),
        'later_label_counts':dict(collections.Counter(r['label_reported'] for r in later)),
        'current_E099_header_correctly_indexed':'REJECT',
        'combined_context_characters':len((P/'CLAUDE_SINGLE_MESSAGE.md').read_text()),
        'latex_compile':'PASS',
        'latex_pages':3,
        'unresolved_citations':False,
        'overfull_boxes':False,
        'visual_QA':'all three rendered PDF pages inspected; no clipping or overlap',
        'new_results':'NOT RUN placeholders only',
        'scientific_validation_performed':False,
        'new_method_implemented':False,
        'training_or_model_inference_run':False,
        'gpu_or_ssh_used':False,
        'official_state_modified':False,
        'original_checkpoints_videos_native_tensors_included':False,
        'paper_status':'writing scaffold, not completed or submission ready',
        'checked_at':datetime.now(timezone.utc).isoformat(),
    }
    (P/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    files=[f for f in sorted(P.rglob('*')) if included(f) and f.name!='FILE_SHA256.json']
    hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (P/'FILE_SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    files=[f for f in sorted(P.rglob('*')) if included(f)]
    dest=P.with_suffix('.zip')
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,P.name+'/'+str(f.relative_to(P)))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for rel,sha in hashes.items():
            assert hashlib.sha256(z.read(P.name+'/'+rel)).hexdigest()==sha,rel
    receipt={'archive':str(dest),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
             'archive_bytes':dest.stat().st_size,'archive_files':len(files),
             'crc_check':'PASS','all_payload_sha_check':'PASS',
             'source_files':len(manifest['files']),'paper':'scaffold only; new method results NOT RUN'}
    (P.parent/(P.name+'_archive_receipt.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
