#!/usr/bin/env python3
"""Read-only software/data preflight. No model forward, no installs, no SSH."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def preflight(config_path, check_data=False, require_cuda=False):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    resolve = lambda v: Path(v) if Path(v).is_absolute() else base / v
    checks = []
    def check(name, value, detail=''):
        checks.append(dict(check=name, passed=bool(value), detail=str(detail)))
    for name in ('sampling_probe.py', 'sampling_math.py', 'layer_probe.py', 'loss_probe.py', 'loss_math.py'):
        p = HERE / name
        try:
            ast.parse(p.read_text(), filename=str(p)); check('syntax:' + name, True)
        except Exception as e: check('syntax:' + name, False, e)
    manifest = HERE / 'MANIFEST.json'
    if manifest.exists():
        locked=json.loads(manifest.read_text())
        for rel, expected in locked['files'].items():
            p = HERE / rel
            check('package:' + rel, p.is_file() and sha(p) == expected)
        current={k:v for k,v in cfg.items() if k not in ('paths','path_remap','device')}
        check('protocol_parameters_unchanged',current==locked['protocol_parameters'],
              'Only paths, path_remap and device may differ from the packaged protocol.')
    check('no_full_benchmark', cfg['scope']['allow_full_benchmark'] is False)
    check('no_checkpoint_overwrite', cfg['scope']['allow_checkpoint_overwrite'] is False)
    contract_path = resolve(cfg['files']['source_contract043'])
    contract = json.loads(contract_path.read_text())
    vendor = resolve(cfg['paths']['vendor078'])
    frozen = json.loads((vendor / 'runtime_manifest.json').read_text())['files']
    for rel, expected in frozen.items():
        p = vendor / rel
        check('frozen078:' + rel, p.is_file() and sha(p) == expected)
    if check_data:
        paths = {k:resolve(v) for k,v in cfg['paths'].items()}
        remaps = {'/root/rebinding_project/VrdONE': str(paths['repo_root']),
                  '/root/autodl-tmp/VrdONE-data': str(paths['data_root']), **cfg.get('path_remap',{})}
        def remap(value):
            p=Path(value)
            for old,new in sorted(remaps.items(),key=lambda v:-len(v[0])):
                try:return Path(new)/p.relative_to(old)
                except ValueError:pass
            return p if p.is_absolute() else base/p
        for key in ('repo_root','checkpoint','gt_proposals','gt_features','annotations','cache_root','train_export_root'):
            check('exists:' + key, paths[key].exists(), paths[key])
        for rel, expected in contract['repo_sources'].items():
            p = paths['repo_root'] / rel
            check('repo:' + rel, p.is_file() and sha(p) == expected)
        cp = paths['checkpoint']
        check('checkpoint_sha256', cp.is_file() and sha(cp)==contract['checkpoint_sha256'])
        inventory = json.loads(resolve(cfg['files']['cache_inventory']).read_text())
        for role in ('train', 'adapter_dev'):
            p = paths['cache_root'] / (role + '_cache.pt')
            check('cache:' + role, p.is_file() and sha(p)==inventory[role]['sha256'])
            mp=paths['train_export_root']/role/'manifest.json'
            expected={'train':'39c0555bda3d99e928aef89a9f346a9b2c418a4e051205729a3838927a2306c5',
                      'adapter_dev':'4a9dc55807723c69b2f8d10d73ad827ca3b9c9f2007f2e6607603f8a16db7edc'}[role]
            ok=mp.is_file() and sha(mp)==expected
            check('source_manifest:' + role,ok,mp)
            if ok:
                old=json.loads(mp.read_text())
                for ri,ref in enumerate(old['records']):
                    p=remap(ref['path'])
                    check(f'export_record:{role}:{ri}',p.is_file() and sha(p)==ref['sha256'],p)
                for video,ref in old['inputs'].items():
                    p=remap(ref['annotation'])
                    check(f'train_annotation:{role}:{video}',p.is_file() and sha(p)==ref['annotation_sha256'],p)
                dc=old['effective_dataset_config']
                loader=remap(dc['cache_dir'])/(dc['cache_tag']+'_VidOR_training')
                for video in old['sampled_video_ids']:
                    check(f'loader_cache:{role}:{video}',(loader/(video+'.pkl')).is_file())
        cohorts=json.loads(resolve(cfg['files']['cohorts']).read_text())
        for name,ref in cohorts.items():
            ex=remap(ref['export']);mp=ex/'manifest.json'
            ok=mp.is_file() and sha(mp)==ref['manifest_sha256']
            check('exposed_cohort:'+name,ok,mp)
            if ok and name=='047':
                manifest=json.loads(mp.read_text());p=ex/'effective_config.json'
                check('C_effective_config',p.is_file() and sha(p)==manifest['outputs']['effective_config.json']['sha256'])
        proposals = sorted(paths['gt_proposals'].glob('*.pkl'))
        check('gt27_count', len(proposals)==27, len(proposals))
        for p in proposals:
            group, vid = p.stem.split('_', 1)
            check('gt_features:' + p.stem, (paths['gt_features']/p.name).is_file())
            check('annotation:' + p.stem, (paths['annotations']/group/(vid+'.json')).is_file())
        for mod in ('numpy','torch','yaml','scipy','einops'):
            check('dependency:' + mod, importlib.util.find_spec(mod) is not None)
    if require_cuda:
        try:
            import torch
            check('cuda_available', torch.cuda.is_available())
            if torch.cuda.is_available():
                device = cfg.get('device','cuda:0')
                check('cuda_device', True, torch.cuda.get_device_name(torch.device(device)))
        except Exception as e: check('cuda_available', False, e)
    result = dict(complete=all(x['passed'] for x in checks),
                  stage='SOFTWARE_AND_DATA_CHECK' if check_data else 'LOCAL_SOFTWARE_CHECK',
                  experiment_executed=False, config_sha256=sha(config_path), checks=checks)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE/'config.json'))
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--require-cuda',action='store_true')
    p.add_argument('--out',help='New JSON receipt; refuses overwrite')
    args=p.parse_args()
    try: result=preflight(args.config,args.check_data,args.require_cuda)
    except Exception as e: result=dict(complete=False,experiment_executed=False,error=repr(e))
    if args.out:
        with Path(args.out).open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['complete'] else 2

if __name__=='__main__': sys.exit(main())
