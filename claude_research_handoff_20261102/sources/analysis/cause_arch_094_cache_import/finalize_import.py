"""094 local integrity verification and acceptance receipt; standard library only."""
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
STAGING=Path('/private/tmp/cvpr-cache094-v4')
DEST=PROJECT/'handoff/incoming/VIDOR-E091-GEOMETRY-CONTRACT-092/cache_v1'
REMOTE='/root/autodl-tmp/VrdONE-data/commander_cache_import_094/cache_v4'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()
def require(ok,message):
    if not ok: raise RuntimeError(message)
def js(p):
    return json.loads(Path(p).read_text())
def write(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

m=js(STAGING/'manifest.json')
require(m['status']=='REMOTE_ASSEMBLY_PASS_PENDING_LOCAL_IMPORT','assembly status')
require(m['training'] is False and m['E091_evaluation'] is False,'task scope')
require(len(m['files'])==len({f['path'] for f in m['files']}),'duplicate paths')
for f in m['files']:
    p=STAGING/f['path']
    require(p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'import integrity '+f['path'])
require(m['bytes']==sum(f['bytes'] for f in m['files']),'byte sum')
audit=js(STAGING/'support_audit.json'); consistency=js(STAGING/'forward_consistency.json')
require(audit['status']==consistency['status']=='PASS','remote audit')
require(len(consistency['checks'])==1725,'native check count')
require(all(all(v==0 for v in c['max_abs'].values()) for c in consistency['checks']),'nonzero reconstruction')
require(consistency['encoder_forwards']==consistency['optimizer_steps']==0,'execution scope')
counts=m['counts']
require((counts['train']['pairs'],counts['train']['targets'])==(350,563),'train counts')
require((counts['adapter_dev']['pairs'],counts['adapter_dev']['targets'])==(89,169),'dev counts')
require((counts['eval']['main400'],counts['eval']['subsecond163'],counts['eval']['main_H_eligible'])==(400,163,381),'fixed members')
require((counts['eval']['native_pairs'],counts['eval']['selected_pairs'],counts['eval']['long_protection_GT'],counts['eval']['locked_GT'],counts['eval']['no_pair_GT'])==(1614,890,1101,2573,0),'eval support counts')
for role,expected in [('train',(88,350,563,561,2)),('adapter_dev',(23,89,169,168,1))]:
    s=audit['splits'][role]
    require(tuple(s.get(k,0) for k in ['batches','pairs','targets','at_least_two_positive','single_positive'])==expected and s.get('zero_positive',0)==0,'audited support '+role)
require(audit['splits']['eval']==dict(pairs=1614,batches=1614),'eval shards')
sets=[set(m['split_videos']['train']),set(m['split_videos']['adapter_dev']),set(m['eval_videos'])]
require(list(map(len,sets))==[32,8,27] and all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),'split overlap')
prior=js(PROJECT/'analysis/mech_query_score_089/input_manifest_initial.json')
for name,filename in [('members','members.json'),('eval_locked','eval_locked.csv'),('coordinates','eval_coordinates.json')]:
    require(sha(STAGING/'sources/087'/filename)==prior['sources'][name]['sha256'],'historical lock '+name)
events=list(csv.DictReader((STAGING/'event_support.csv').open()))
require(len(events)==732 and len({r['event_id'] for r in events})==732,'event ledger')
require(all(int(r['positive_samples_checked'])>=1 for r in events),'unexpected zero target')
for role,total,single in [('train',563,2),('adapter_dev',169,1)]:
    rows=[r for r in events if r['split']==role]
    require(len(rows)==total and sum(int(r['positive_samples_checked'])==1 for r in rows)==single,'event support ledger '+role)
locked=list(csv.DictReader((STAGING/'sources/087/eval_locked.csv').open()))
eval_support=js(STAGING/'eval_support.json')
require(len(eval_support)==len(locked)==2573,'eval ledger count')
locked_keys={(r['video_name'],int(r['gt_index']),int(r['pair_idx']),int(r['query'])) for r in locked}
support_keys={(r['video_name'],r['gt_index'],r['pair_idx'],r['legacy_query']) for r in eval_support}
require(locked_keys==support_keys and len(locked_keys)==2573 and all(r['status']=='present' and r['missing_reason'] is None for r in eval_support),'eval support identities')
matching=js(STAGING/'frozen_matching.json')
require(len(matching['entries'])==439 and sum(len(e['event_ids']) for e in matching['entries'])==732,'matching totals')
query_audit=js(STAGING/'fixed_query_audit.json')
require(query_audit['all_legacy_queries_preserved'] and query_audit['main400_probability_consistent'],'query preservation')
require(all(not r['in_main400'] for r in query_audit['drifts_outside_main']),'main rank drift')
require(len(query_audit['drifts_outside_main'])==1,'outside drift count')
drift=query_audit['drifts_outside_main'][0]
require((drift['video'],drift['gt_index'],drift['pair_idx'],drift['legacy_query'],drift['current_probability_argmax'])==('1161_8966147395',3,2,0,1),'outside drift identity')
require(drift['legacy_probability']==0.3675785877732299 and drift['current_probability']==0.3675795173808527,'outside drift values')

# Preserve local-only protocol snapshots. No documents are uploaded.
snapshot=HERE/'protocol_snapshot';snapshot.mkdir(exist_ok=True)
protocol_sources=[
    'docs/deliverables/CACHE_IMPORT_CHECKLIST.md',
    'analysis/cause_arch_092/STATUS.md',
    'analysis/cause_arch_091/CAUSE_STATEMENT.md',
    'analysis/cause_arch_091/ARCHITECTURE.md',
    'analysis/cause_arch_091/REVIEW.md',
    'analysis/short_duration_mechanisms_079/vendor078/source_contract043.json',
]
protocol_files=[]
for rel in protocol_sources:
    src=PROJECT/rel;dst=snapshot/rel.replace('/','__');shutil.copyfile(src,dst)
    protocol_files.append(dict(source=rel,snapshot=str(dst.relative_to(PROJECT)),sha256=sha(dst)))
contract_sha='dd4be25bd4f35a394c387d0905ef747d0a3c051f7f62e3d0931244caa6525bd9'
require(protocol_files[-1]['sha256']==contract_sha,'remote/local source contract mismatch')
write(HERE/'protocol_sources.json',dict(files=protocol_files,remote_source_contract_sha256=contract_sha,remote_sha_observed_by_readonly_command=True,local_documents_uploaded=False))

# Move verified payloads into the requested project path; replace only this task's partial import.
DEST.mkdir(parents=True,exist_ok=True)
for f in m['files']+[dict(path='manifest.json')]:
    src=STAGING/f['path'];dst=DEST/f['path'];dst.parent.mkdir(parents=True,exist_ok=True)
    os.replace(src,dst)
    expected=f.get('sha256')
    if expected:require(sha(dst)==expected,'destination SHA '+f['path'])
print('DESTINATION_SHA_PASS',len(m['files']),flush=True)

groups={}
for role in ['train','adapter_dev','eval']:
    files=[f for f in m['files'] if f['path'].startswith(role+'/')]
    body=dict(role=role,remote_root=REMOTE,local_root=str(DEST),files=files)
    p=HERE/(role+'_file_index.json');write(p,body)
    groups[role]=dict(index=str(p.relative_to(PROJECT)),sha256=sha(p),files=len(files),fields_complete=True)
items={}
for name in ['manifest.json','frozen_matching.json','initial_mask_embed.pt','initial_class_embed.pt','forward_consistency.json','tensor_schema.json','support_audit.json','event_support.csv','eval_support.json','fixed_query_audit.json']:
    items[name]=dict(path=str(DEST/name),sha256=sha(DEST/name),fields_complete=True)
receipt=dict(status='CACHE_GATE_PASS',scope='material acceptance only; E091 not run',remote_root=REMOTE,local_root=str(DEST),verified_files=len(m['files']),verified_payload_bytes=m['bytes'],manifest_sha256=items['manifest.json']['sha256'],groups=groups,required_items=items,counts=counts,support=audit['splits'],projection_max_abs=0.0,checked_native_batches=1725,encoder_forwards=0,optimizer_steps=0,training_started=False,E091_evaluation=False,query_drift_outside_main=query_audit['drifts_outside_main'],protocol_snapshot_index_sha256=sha(HERE/'protocol_sources.json'),local_verification='all manifest payload SHA checked before and after move; remote field audits bound by those hashes; no local torch dependency',completed_at=datetime.now().astimezone().isoformat())
write(HERE/'material_audit.json',receipt)

rows=[]
labels={'train':'训练native cache','adapter_dev':'开发native cache','eval':'评价native cache'}
for role,info in groups.items():
    rows.append(f"| {labels[role]} | `cache_v1/{role}/`，{info['files']}个分片；[{role}索引]({role}_file_index.json) | `{info['sha256']}`（逐文件SHA索引自身） | 齐全 |")
for name in ['frozen_matching.json','initial_mask_embed.pt','manifest.json','forward_consistency.json','tensor_schema.json','event_support.csv']:
    rows.append(f"| {name} | `cache_v1/{name}` | `{items[name]['sha256']}` | 齐全 |")
status=f"""# 094 E091材料导入与验收

**CACHE_GATE_PASS** — 2026-10-20。仅材料通过；A/B/S训练步数为0，E091评价未运行。

## 材料范围

- 本地根目录：`{DEST}`。
- 远程组装目录：`{REMOTE}`；原087 raw和预训练checkpoint未覆盖。
- 已逐文件核验 {len(m['files'])} 个manifest负载文件，共 {m['bytes']:,} 字节。另封存manifest自身SHA与本地原协议快照。
- 分片保留原生`f/hs/valid/cls/masks`的数值和dtype。训练/开发附独立事件`preds/targets`；评价时间target由原GT27标注、锁定清单与坐标连接，完整标注存于`sources/annotations/`，未用50类union probe替代。

| 必备项 | 路径／映射 | SHA256 | 字段 |
|---|---|---|---|
{chr(10).join(rows)}

分片的每个SHA、源路径、字节数见三个索引及原manifest；不是仅校验压缩包。
逐项机器回执：[material_audit.json](material_audit.json)。

## 支持账与固定协议

| 分组 | 视频 | 实体对 | 独立事件target | 至少2个正点 | 仅1个正点 | 零正点 |
|---|---:|---:|---:|---:|---:|---:|
| train | 32 | 350 | 563 | 561 | 2 | 0 |
| adapter_dev | 8 | 89 | 169 | 168 | 1 | 0 |

上述是本缓存实际支持，不能推广为整个数据集无采样缺失。裁剪、网格起止与GT边界差异、采样种子、offset、随机抽样记录和`censored_positive_cells`逐pair/event保留在`pair_sampling_support.json`与`event_support.csv`。网格端点不等于GT端点不自动解释为裁剪。

- train/dev/eval视频互不重叠；评价保留27视频、2573条原锁定GT，全部有固定pair缓存。
- 评价native cache共1614个pair，原锁定GT使用其中890个pair；保留全部1614，不依效果选pair。
- 原400/163成员、固定pair_idx和F query保持；原H准入381/400，其余19/400仍留主分母。
- 关系≥3秒且共存≥10秒、可复用固定读出的保护组为1101条；仅封存支持，不报告新性能。
- 439个训练/开发pair匹配记录覆盖732/732事件，每target恰好一次，同pair query唯一，目标数不超过9。匹配绑定初始化及每个缓存SHA，未来A/B/S共用。
- 原共享mask embedding初始化来自预训练EMA checkpoint，非继续训练权重；未来若执行E091只将其更新作用于decoder层3，早期层输出保持缓存。

## 数值验收

1725/1725批缓存（88 train + 23 dev + 1614 eval）重算mask/class分支与同源原生输出的最大绝对误差均为0；既定容差保持4e-6，mask二值、分类top1/top6一致。匹配使用的重算与缓存张量完全相同，故Hungarian输入一致；匹配覆盖另经CPU核验。

本轮用GPU做冻结mask/class分支重算验收；未重跑encoder，未新导出视频前向，未执行optimizer，未训练任何一臂。不能把缓存PASS写成机制或方法有效。

## 异常与脚本修正均保留

1. v1误用raw logit跨query排名；改为089既定的全51类softmax概率，旧query不变。
2. v2额外要求全部2573条旧F等于当前argmax；原091只要求保留F，089排序一致性范围为400。v4保留全GT旧F，仅对原400执行排序检查。组外1条发生微小排名漂移：`1161_8966147395/GT3/pair2`，旧q0概率0.3675785877732299，q1概率0.3675795173808527；仍使用q0，未删样本。
3. v3将成员表字符串GT ID与整数比较，造成主组连接为空。v4按历史089做规范化，恢复原400/163；没有改成员或放宽数值门槛。

旧脚本、失败日志、修正说明保存在本目录，远程v1/v2/v3输出未删除。上述为验收实现错误，不是模型效果负结果，也不掩盖组外排序漂移。

## 来源与执行边界

原协议快照与SHA见[protocol_sources.json](protocol_sources.json)。远程实际使用043来源合同SHA与本地副本一致：`{contract_sha}`。本地文档批量上传被自动审批拒绝（未单独授权上传）；改在本地封存，无相关文档外传。

状态：仅`CACHE_GATE_PASS`；092当时的`NOT_RUN_MISSING_CACHE`历史记录不改写。RQ不变，M091未采纳，E091未启动，不自动排队下一项。**094到此停止。**
"""
(HERE/'STATUS.md').write_text(status)
print(json.dumps(dict(status='CACHE_GATE_PASS',files=len(m['files']),bytes=m['bytes'],status_path=str(HERE/'STATUS.md'))),flush=True)
