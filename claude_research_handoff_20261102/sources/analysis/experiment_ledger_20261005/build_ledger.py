"""Assemble a source-linked Phase-1 ledger, never alter historical experiments."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
records=[]
sources={}
def js(p):
    f=ROOT/p;sources[p]=hashlib.sha256(f.read_bytes()).hexdigest()
    return json.loads(f.read_text())
def rows(p):
    f=ROOT/p;sources[p]=hashlib.sha256(f.read_bytes()).hexdigest()
    with f.open(encoding='utf-8-sig',newline='') as s:return list(csv.DictReader(s))
def key(r):return r['video_name'],int(r['gt_index'])
def yes(v):return str(v).lower() in ('true','1')
def card(id,name,kind,hypothesis,change,fixed,scope,baseline,metrics,result,support,cannot,evidence,implementation=None,**kwargs):
    r=dict(id=id,name=name,evidence_type=kind,hypothesis=hypothesis,changed=change,fixed=fixed,
      data_scope=scope,baseline=baseline,main_metrics=metrics,
      gained_correct=kwargs.pop('gained_correct','不适用：未产生新的自动检测结果'),
      lost_correct=kwargs.pop('lost_correct','不适用：未产生新的自动检测结果'),
      C_recovered=kwargs.pop('C_recovered','不适用：未产生新的自动检测结果'),
      target_feature_results=kwargs.pop('target_feature_results',{}),
      whole_dataset_results=kwargs.pop('whole_dataset_results','无新的全量检测指标'),
      result_assessment=result,supported_conclusions=support,unsupported_conclusions=cannot,
      evidence_paths=evidence,implementation_paths=implementation or [],**kwargs)
    for p in evidence+(implementation or []):
        assert (ROOT/p).is_file(),p
        if p not in sources:sources[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    records.append(r);return r

P005='handoff/incoming/VIDOR-PAIR-AUDIT-005/'
P008='handoff/incoming/VIDOR-VRDONE-FAILURE-008/'
P009='handoff/incoming/VIDOR-COMMON-FAILURE-009/'
P010='analysis/vrdone_architecture_010/new_container_preflight/'
P011='analysis/vrdone_architecture_011/remote/'
P013='analysis/vrdone_common_failure_013/'
PGT='handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/'
inst=rows(P005+'raw-20260928/instance_comparison.csv');imap={key(r):r for r in inst}
features={key(r):r for r in rows(P008+'input_audit_v2/per_gt_input_features.csv')}
repeats=Counter((r['video_name'],r['subject_tid'],r['object_tid'],r['predicate']) for r in inst)
VAL='VidOR validation；835视频清单/832有GT视频/30142关系；非训练'
SMALL='VidOR validation固定32视频/1209 GT/C1047；曾用于008及009筛查，非独立测试'
FIX='同一VrdONE EMA权重、输入轨迹/特征、stride4、9query、谓词Top6、至少5帧、最多200保存预测；评价Top100'
TRAIN='官方training按视频划分：train32原标注1237 GT/缓存350 pair与563 GT，adapter_dev8原标注827 GT/缓存89 pair与169 GT；M010单seed10、M011单seed11；评价为已暴露validation32/1209GT/C1047'

def feature_stats(outcomes):
    groups=defaultdict(list)
    for r in outcomes:
        i=imap[key(r)];f=features[key(r)];d=float(i['duration_sec'])
        labels=['all','short<3s' if d<3 else '>=3s','<1s' if d<1 else ('1-3s' if d<3 else ('3-10s' if d<10 else '>=10s'))]
        size=float(f['min_entity_area_fraction'] or 'nan');speed=float(f['relative_speed_per_second'] or 'nan')
        if math.isfinite(size) and size<.01:labels.append('small<1%')
        if math.isfinite(speed) and speed>=.1:labels.append('fast>=0.1/s')
        if not math.isfinite(speed):labels.append('motion_missing')
        nrep=repeats[(i['video_name'],i['subject_tid'],i['object_tid'],i['predicate'])]
        if nrep>=2:labels.append('repeat>=2')
        if nrep>=3:labels.append('repeat>=3')
        if f['input_pair_state']=='feasible_input_pair':labels.append('input_feasible')
        b=yes(i['VrdONE_hit_at100']);n=yes(r['hit100'])
        for label in labels:groups[label].append((b,n,i['group_at100']=='C'))
    result={}
    for label,a in groups.items():
        result[label]=dict(N=len(a),baseline_hits=sum(x for x,y,c in a),hits=sum(y for x,y,c in a),
          gains=sum(not x and y for x,y,c in a),losses=sum(x and not y for x,y,c in a),
          C_N=sum(c for x,y,c in a),C_recovered=sum(c and y for x,y,c in a))
    return result

d005=js(P005+'analysis/results.json')
card('005-COMPARE','BIG与VrdONE配对及四象限','Observation','VrdONE新增与损失哪些GT；配对是否一致','不改模型；比较已有预测命中','GT实例、TopK和原匹配结果',VAL,'发布BIG命中缓存 vs VrdONE源码评估导出',
 {'protocols':d005['protocols'],'missing_prediction_GT':d005['missing_prediction_GT']},
 '已核实描述；不是共享输入的架构受控对照','Top100 A2487/B1809/C25373/D473；VrdONE4296、BIG2960命中','B不能等同VrdONE架构独有解决的语义类型；未恢复BIG原预测/导出器',
 [P005+'raw-20260928/summary.json',P005+'raw-20260928/run.log',P005+'raw-20260928/instance_comparison.csv',P005+'analysis/results.json'],[P005+'analysis/analyze_import.py'],
 gained_correct=1809,lost_correct=473,C_recovered='不适用：C由两模型共同失败定义',whole_dataset_results=d005['protocols'],target_feature_results=d005['duration'])
card('005-PROFILE','时长、谓词、实体类别与保存排名画像','Observation / Correlation','短关系及某些谓词是否集中漏检','只重分组已有GT与命中','同一30142条GT和原预测',VAL,'各特征组与互补组',
 {'duration':d005['duration'],'miss100_disjoint_partition':d005['miss100_disjoint_partition']},'描述性结果','<1s共同漏检3151/3167；1-3s7475/8454；3-10s10296/12193；>=10s4451/6328','时长因果效应；检测/跟踪/边界的独立机制',
 [P005+'analysis/results.json',P005+'analysis/duration_summary.csv'],[P005+'analysis/analyze_import.py'],whole_dataset_results='全量已有预测画像；无新方法')
card('006-DURATION-RESOLUTION','已保存匹配预测的时间分辨率审计','Observation / Diagnosis','亚秒关系是否与采样/预测时长有关','仅分析已匹配保存区间','旧预测与GT',VAL,'短/长关系及匹配预测',js(P005+'analysis/duration_resolution_results.json'),
 '仅描述；远端006包未返回','可以统计已命中候选的区间长度/分辨率','幸存者样本不能证明漏检由stride或最短长度造成',
 [P005+'analysis/duration_resolution_results.json'],[P005+'analysis/duration_resolution_audit.py'])
card('007-PROFILE','共同失败画像与短关系阈值敏感性','Correlation','共同失败中哪些结构特征更多','只统计/抽样，不改模型','旧GT/四象限',VAL,'共同失败C和其他GT',
 js(P005+'analysis/shared_failure_profile/results.json'),'描述性完成；视觉审核未完成','记录时长、谓词、关系段数、视频负载的分布和分母','表中inspection候选不等于已看视频；交互/运动语义未核验',
 [P005+'analysis/shared_failure_profile/results.json',P005+'analysis/shared_failure_profile/inspection_sample.csv'],[P005+'analysis/profile_shared_failures.py'])
card('007-R2','时长分层控制与成对抽样','Correlation','短长组差距能否由谓词/类别/重复性构成解释','在共同支持分层内标准化；生成盲审样本','相同旧命中结果',VAL,'同层短/长关系',
 js(P005+'analysis/duration_controls/results.json'),'调整后的统计关联','可比较指定分层内的短长组失败率','共同支持群体不同；无运动/视觉人工审核；不是时长因果干预',
 [P005+'analysis/duration_controls/results.json',P005+'analysis/duration_controls/standardized_comparisons.csv'],[P005+'analysis/compare_duration_within_groups.py'])
cpu=js(P008+'cpu_audit/summary.json')
card('008-SAVED-ORACLE','保存预测的谓词/裁时间oracle诊断','Diagnosis','失败在已有预测中是否可用改谓词/裁时间解释','GT辅助匹配/裁时间，不产生可部署预测','原保存轨迹与候选',VAL,'原VrdONE保存结果',cpu,
 '诊断完成','识别time_crop3707、predicate_only770、两者1967等诊断桶','oracle可行不等于自动恢复；互斥桶不是永久单因故障',
 [P008+'cpu_audit/summary.json',P008+'cpu_audit/per_gt_diagnosis.csv'],[P008+'code/audit_saved.py',P008+'code/temporal_oracle_reference.py'])
inp=js(P008+'input_audit_v2/summary.json')
card('008-INPUT-V1','旧版GT交集裁剪输入诊断','Diagnosis','当前轨迹裁到GT交集是否达标','仅测试GT交集裁剪；后由v2任意连续区间搜索替代','旧输入轨迹和类别',VAL,'当前输入pair',js(P008+'input_audit/summary.json'),
 '旧定义，保留但不与v2混用','可裁剪达标13143；v2新增151条可行，属于oracle定义放宽','151不是模型恢复；v1与v2运动lag支持规则也不同',
 [P008+'input_audit/summary.json',P008+'input_audit/per_gt_input_features.csv'],[P008+'code/audit_input_pairs.py'])
card('008-INPUT-V2','进入模型的合格实体对输入可行性','Diagnosis','固定轨迹上是否存在能匹配GT的任意连续区间','穷举oracle时间；v2复现loader合格pair筛选','固定当前输入轨迹、类别和空间框',VAL,'原VrdONE输入候选',inp,
 '诊断完成','全体F13294；C中F8545、不可行16828','不是同时可达召回；不能分解为检测、分类、跟踪各自损失；不是BIG输入诊断',
 [P008+'input_audit_v2/summary.json',P008+'input_audit_v2/per_gt_input_features.csv'],[P008+'code/audit_input_pairs_v2.py',P008+'code/temporal_oracle_reference.py'],whole_dataset_results=inp['counts'])
q=js(P008+'query_audit/summary.json')
card('008-QUERY32','原始query、谓词Top6与排名诊断','Diagnosis','输入可行后是无匹配mask、无正确谓词还是候选排序损失','读取9query原始mask与分类，不训练','固定输入/权重/原decode',SMALL,'原模型139命中',q,
 '诊断完成','1070漏检=722输入不可行+182无匹配hull+28谓词不在同query Top6+138排序/分配；C1047中721+180+26+120','诊断路径不等于互斥物理原因；不证明旧关系记忆或上下文压制',
 [P008+'query_audit/summary.json',P008+'query_audit/per_gt_query_diagnosis.csv',P008+'pilot_joined_diagnosis.csv'],[P008+'code/audit_pilot_queries.py'],target_feature_results=q['duration'])

def add_inference(id,name,path,mode,hyp,change,scope,interpretation,cannot):
    summary=js(path+'summary.json');m=summary['metrics'][mode];b=summary['metrics']['baseline']
    allrows=rows(path+'per_gt_outcomes.csv')
    rr=[r for r in allrows if r['mode']==mode]
    br=[r for r in allrows if r['mode']=='baseline']
    m=dict(m);b=dict(b)
    for metrics,subset in ((m,rr),(b,br)):
        for k in (50,100,200):
            metrics['hit'+str(k)]=sum(0<=int(r['rank'])<k for r in subset)
            metrics['R@'+str(k)+'_fraction']={'numerator':metrics['hit'+str(k)],'denominator':len(subset)}
    f=feature_stats(rr);a=f['all'];assert a['hits']==m['hit100']
    return card(id,name,'Intervention',hyp,change,FIX,scope,b,m,
      interpretation,'仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表',cannot,
      [path+'summary.json',path+'per_gt_outcomes.csv'],
      [P008+'code/pilot_decode.py'] if id.startswith('008') else ['analysis/vrdone_common_failure_009/experiment/run_inference.py','analysis/vrdone_common_failure_009/experiment/evaluate_trials.py'],
      gained_correct=a['gains'],lost_correct=a['losses'],C_recovered=a['C_recovered'],target_feature_results=f,
      whole_dataset_results=m if '835' in id else '未做该变体完整835视频评估')

add_inference('008-SPLIT','原mask按连通分量拆段',P008+'pilot_retry/','split_components','拆开跨gap区间是否恢复关系','同一query的正mask分成多个连续输出',SMALL,'不确定：命中+1但mAP下降','拆段同时改变候选数/排序；不能单独归因于gap建模')
add_inference('008-RIGHT','末端补齐一个采样单元',P008+'pilot_retry/','right_cell','末端+1改为补满stride是否减少损失','解码末端补至下个采样单元并截于pair结束',SMALL,'负/无净增：新增2损失2且mAP下降','不能证明采样无影响；只检验该后处理')

prof=js('analysis/vrdone_common_failure_009/features/results.json')
card('009-FEATURE-CONTROLS','大小/速度/时长/重复与输入可行分层','Correlation','原始高失败率是否在控制群体构成后仍存在','GT几何代理量分箱与共同支持标准化','30142 GT、旧C/F定义',VAL,'特征组对互补组；全体和F分开',
 {'C_input_states':prof['C_input_states'],'standardized':prof['standardized'],'tables':prof['tables']},'关联有正有负','高速全体+5.816pp，F中-0.788pp；小实体全体+11.279pp，F中+3.595pp；重复同视频全体-0.660pp','不能把GT速度当轨迹误差；F条件化是oracle筛选且可能有选择偏差；非因果效应',
 ['analysis/vrdone_common_failure_009/features/results.json',P008+'input_audit_v2/per_gt_input_features.csv'],['analysis/vrdone_common_failure_009/features/profile_common.py'],whole_dataset_results='全量描述/分层，非新方法')
specs=[('R1','几何平均评分','关系与两实体分数算术平均→几何平均','评分是否影响候选选择','不确定：局部正向点估计；未扩大'),('R2','mask质量重排','原分数乘首末正点跨度内mask概率均值（含内部负点）','mask置信度是否代表匹配质量','混合：mAP升但Top100净-4'),('R3','同pair谓词时间NMS','tIoU>0.7的同pair同谓词候选抑制','去重复能否释放正确候选名额','Top100无恢复；mAP及R200略升'),('R4','提高mask阈值','sigmoid阈值0.5→0.75','更紧时间区间是否改善检测','不确定：局部正向点估计；候选由此筛选')]
for mode,name,change,hyp,result in specs:
    add_inference('009-'+mode+'-32',name,P009+'remote/old32_v1/evaluation/',mode,hyp,change,SMALL,result,'不能由已暴露小样本外推全量；与其他变体筛选存在选择效应')
add_inference('009-R4-64','R4扩展64视频',P009+'remote/expanded64_v2/evaluation/','R4','冻结R4是否在另一批视频净增','阈值0.5→0.75','validation64/2601 GT；与探索32不重叠，但此前全验证诊断已暴露','不确定：新增11损失6的正点估计','不是未见测试集；不能替代完整benchmark')
add_inference('009-R4-835','R4全835视频',P009+'remote/full835/evaluation/','R4','冻结R4能否全量净提升','阈值0.5→0.75',VAL,'负结果：mAP下降；新增78、损失108、净-30','局部短组收益不等于全体有效；不能据此声称学习到边界机制')
card('009-WITNESS','R4全量变化的同候选见证','Diagnosis of Intervention','恢复/损失来自几何还是排名变化','事后跟踪同pair/query/predicate','已保存的baseline/R4推理',VAL+'；仅186条Top100状态变化GT','baseline和R4同一候选',
 {'changed_GT':186,'gains':78,'losses':108,'C_gains':64,'C_same_candidate_geometry':58,'C_rank_assignment':6,'loss_retained_bad_geometry':103,'loss_candidate_removed':5},
 '支持局部操作效应；不支持总体方法收益','阈值改变确实使部分同候选vIoU跨阈值；103条保留候选损失中mask均值升而匹配质量降','这103条是按损失筛出的条件样本，不能当全体质量负相关；不是历史上下文因果证据',
 [P009+'remote/full835/evaluation/r4_same_query_witnesses.json',P009+'remote/full835/evaluation/boundary_extent_audit.json'],['analysis/vrdone_common_failure_009/experiment/audit_r4_witnesses.py'])
card('009-ANNOTATION','时长切点与重复标注结构审计','Observation / Correlation','短关系分箱/重复多段是否是标注或fps边界假象','按整数帧与真实fps核分箱；同键合并重叠/相邻段','旧GT与C/F',VAL,'<2.5/<3/<3.5秒；单段/多段',js('analysis/vrdone_common_failure_009/annotation_structure/results.json'),
 '描述性完成','6097多记录键、18063关系合并后仍分离；90帧记录1906条均>3秒','分离标注不等于已核验重复动作；没跑合并GT后的新评价',
 ['analysis/vrdone_common_failure_009/annotation_structure/results.json'],['analysis/vrdone_common_failure_009/annotation_structure/audit.py'])

def add_training(id,mode,steps,base_dir,train_dir,family):
    metricfile=base_dir+'/official/summary.json';outcomefile=base_dir+'/outcomes/summary.json'
    m=js(metricfile);o=js(outcomefile);t=js(train_dir+'/summary.json')
    if family=='010':
        change={'class_only':'仅训练分类残差分支','mask_only':'仅训练时间mask残差分支','full':'训练分类与谓词时间证据联合残差分支'}[mode]
        hyp='谓词时间证据耦合是否超过单独分类/时间分支'
        imp=['analysis/vrdone_architecture_010/integration/train_adapter.py','analysis/vrdone_architecture_010/integration/adapter_model.py','analysis/vrdone_architecture_010/prototype/predicate_evidence.py','analysis/vrdone_architecture_010/integration/infer_head_v3.py']
        result='无召回变化' if steps==200 or mode=='mask_only' else '不确定：局部+1；仅排序变化，时间分支未提供独立收益'
        support='200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变'
    else:
        change='训练query条件化联合起止点头；'+('局部时间视觉特征' if mode=='local' else '以时间均值替换局部视觉，保留mask/位置等信息')
        hyp='局部视觉驱动的区间解码是否优于同参数pooled控制'
        imp=['analysis/vrdone_architecture_011/prototype/train_interval.py','analysis/vrdone_architecture_011/prototype/structured_interval.py','analysis/vrdone_architecture_011/prototype/training.py','analysis/vrdone_architecture_011/prototype/inference.py']
        result='负结果：新增11损失29，且弱于pooled' if mode=='local' else '不确定：单seed局部+1，R200下降2'
        support='local确有9条C几何恢复，但总体下降；pooled仅正点估计，paired bootstrap包含0'
    return card(id,f'M{family} {steps}步 {mode}','Intervention',hyp,change,FIX+'；冻结骨干；固定相同训练顺序、query/谓词输出预算；最终步权重',TRAIN,
       m['metrics']['baseline'],m['metrics']['head'],result,support,'不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益',
       [metricfile,outcomefile,base_dir+'/outcomes/per_gt_outcomes.csv',train_dir+'/summary.json',train_dir+('/adapter_last.pt' if family=='010' else '/interval_last.pt')],imp,
       gained_correct=o['gains'],lost_correct=o['losses'],C_recovered=o['C_recovered'],target_feature_results=o['strata'],
       whole_dataset_results='未做M'+family+'完整835验证/独立测试',training_summary=t)

for steps in (200,2000):
 for mode in ('class_only','mask_only','full'):
    ev=P010+f'evaluation_results_v1/steps{steps}/{mode}'
    tr=P010+('training_results_v1/training_v1/' if steps==200 else 'training_stage2_results_v1/training_stage2_v1/')+mode
    add_training(f'010-{steps}-{mode}',mode,steps,ev,tr,'010')
for mode in ('local','pooled'):
    add_training('011-'+mode,mode,2000,P011+'final_results_v2/evaluation_v2/'+mode,P011+'training_results_v2/training_v2/'+mode,'011')
local_before=js(P011+'training_results_v2/training_v2/local/adapter_dev_before.json')['statistics']
local_after=js(P011+'training_results_v2/training_v2/local/adapter_dev_after.json')['statistics']
card('011-TRAIN-DIAG','M011训练目标/解码及候选有效性诊断','Diagnosis of Intervention','训练似然与区间质量是否一致；解码为何丢候选','对已完成训练权重及原始解码做诊断','固定train/dev缓存与同一32验证输出',TRAIN,
 '初始化区间头/原合法候选',{'local_dev_NLL':[local_before['endpoint_nll'],local_after['endpoint_nll']],'local_dev_MAP_tIoU':[local_before['visible_gt_tiou'],local_after['visible_gt_tiou']],
 'NLL_denominator':119,'tIoU_denominator':169,
 'raw_candidates':186948,'baseline_valid':186690,'local_valid':141110,'local_valid_to_invalid':45581,'pooled_valid':182207,'pooled_valid_to_invalid':4483},
 '观察到目标/决策结果背离及退化候选','局部头NLL降低而dev MAP tIoU降低；大量单点/短区间被min5规则过滤','候选45581不是GT损失；不能把整个负结果只归因输入域差；没有对损失函数做独立因果消融',
 [P011+'training_results_v2/training_v2/local/adapter_dev_before.json',P011+'training_results_v2/training_v2/local/adapter_dev_after.json',P011+'final_results_v2/inference_v2/local/receipts.json',P011+'final_results_v2/inference_v2/pooled/receipts.json','handoff/incoming/VIDOR-ARCHITECTURE-011/REPORT.md'],
 ['analysis/vrdone_architecture_011/prototype/training.py','analysis/vrdone_architecture_011/prototype/structured_interval.py','analysis/vrdone_architecture_011/prototype/inference.py'])
card('010-EVIDENCE-DIAG','M010已训练时间证据可分性诊断（存于011目录）','Diagnosis','M010时间证据是否可分、是否改动原mask','读取M010 full既有权重在冻结train/dev缓存的逐时刻证据；时间打乱控制','既有缓存、M010权重，无新训练',TRAIN,'原mask与M010证据；时间打乱',
 js(P011+'diagnostic_results_v2/evidence_diagnostic_v2/summary.json'),'诊断完成，不是M011新训练','可核查证据可分性、阈值翻转位置与训练/开发差距','目录名含011不代表M011模型结果；AUROC或dense BCE不能替代检测效果',
 [P011+'diagnostic_results_v2/evidence_diagnostic_v2/summary.json'],['analysis/vrdone_architecture_011/diagnose_evidence.py'])
d013=js(P013+'results.json')
card('013-STRUCTURE','十项结构特征及敏感性/控制变量','Correlation','非显眼特征是否具有稳定条件差异','构建接续、并存、相对跨度、后续段等代理量；标准化','旧C、v2输入可行F；无新推理',VAL,
 '各曝光组和互补组；全体/F/同视频控制',{'overall':d013['overall'],'features':d013['features'],'sensitivity':d013['sensitivity'],'budget_bound':d013['budget_bound']},
 '混合关联，若干方向反转','后续段加重复控制后仍小正差；不同客体接续同视频差值反转；视频输出预算形成容量约束','接续候选不是真实切换；相对标注跨度不是物体共存时长；未证实记忆/冲突机制',
 [P013+'results.json',P013+'per_gt_features.csv'],[P013+'analyze.py'])
card('013-WITHIN-KEY','同视频同实体对同谓词的首段/后续段对照','Correlation','后续段较难是否只是跨视频/谓词构成差异','同键比较；增加相同时长档和双侧F限制','同一关系键、原预测、每键等权',VAL+'；严格子组1189键/322视频/1189首段+1675后续段',
 '同键真正首段',d013['same_key_followup'],'描述性小正差','严格子组首段漏852/1189、后段1304/1675；键等权差为24.4738095/1189×100个百分点','不能用两组合并率差替代键等权差；同一时长档不等于相同时长；无不确定性区间/机制干预',
 [P013+'results.json',P013+'per_gt_features.csv'],[P013+'analyze.py'])
re=js('analysis/experiment_ledger_20261005/recomputation.json')
card('DURATION-ENTITY','实体时长与关系时长交叉','Observation / Correlation','关系短是否也意味着实体短','官方GT累计/当前连续/共同连续时长分别统计','30142条原GT/旧命中与v2输入可行',VAL,
 '关系<3秒；实体<3秒 vs >=3秒',re['entity_duration'],'描述完成','短关系11621；两实体累计均长10230，但当前连续均长5870，共同连续>=3为5311','GT支持含自动框，不是人工核验物理可见性；不能反推出预测轨迹错误类型',
 ['analysis/entity_relation_duration_20260929/per_gt_durations.csv','analysis/entity_relation_duration_20260929/results.json','analysis/entity_relation_duration_20260929/acquisition.json'],['analysis/entity_relation_duration_20260929/analyze.py'])
gt=js(PGT+'evaluation/summary.json')
card('GT-PAIRED27','GT实体/轨迹/正确配对输入双臂诊断','Diagnosis / Oracle-input intervention','纠正上游输入后能多产生多少可匹配raw mask','native输入→GT框/类/身份pair/重新ROI特征；GT关闭轨迹去重','原EMA、stride4、9query、阈值0.5；两臂同进程fresh forward；无训练',
 'validation strict切换筛选27视频/2573GT/C2270，非随机；全部关系纳入分母','native raw-mask vIoU候选覆盖683/2573',re['gt_probe'],
 '诊断候选覆盖正变化，不是新方法检测提升','GT输入覆盖1678，新增1059、损失64；短<3组144→452、+328/-20；C覆盖385→1385、+1054/-54',
 '不能把oracle query挑选当自动Recall；不能分离框/特征/配对/去重的独立贡献；不能外推835',
 [PGT+'evaluation/summary.json',PGT+'evaluation/per_gt.csv',PGT+'run/completion.json',PGT+'run/preflight.json',PGT+'run/effective_config.json'],
 ['analysis/gt_input_mask_probe_20260929/run_probe.py','analysis/gt_input_mask_probe_20260929/evaluate_masks.py'],
 gained_correct='不适用；raw候选覆盖新增1059',lost_correct='不适用；raw候选覆盖损失64',C_recovered='不适用；C内raw覆盖新增1054、损失54',target_feature_results=re['gt_probe'],whole_dataset_results='未做全835 GT-input诊断')
card('GT-TIME27','GT输入纯时间、二值mask与采样网格上界','Diagnosis','GT输入后是否仍缺少短时间mask；网格是否可表达','GT辅助选最优query/穷举合法网格区间','同一GT臂输出，min5，stride4','与GT-PAIRED27同27视频/2573GT','实际GT时间候选 vs 网格oracle',
 {'gt_binary50':1717,'gt_hull_tiou50':1731,'grid_tiou50':2563,'N':2573,'short_joint_ge3':re['gt_probe']['short_joint_ge3'],'short_joint_ge10':re['gt_probe']['short_joint_ge10']},
 '诊断完成','亚秒二值55/333，区间75/333；短关系且共同>=3秒二值178/626；网格可行但实际区间失败832','二值mask与hull tIoU不可互换；网格可表达不证明采样保留足够视觉信息；尚未证明长上下文稀释',
 [PGT+'evaluation/per_gt.csv',PGT+'evaluation/summary.json','analysis/gt_input_mask_probe_20260929/derived_summary.json'],['analysis/gt_input_mask_probe_20260929/evaluate_masks.py','analysis/gt_input_mask_probe_20260929/summarize_local.py'])
card('GT-ONE-TO-ONE','独立时间事件的一对一query容量','Diagnosis','GT独立挑query是否重复使用同一掩码','合并只谓词不同而身份/时间相同的GT；最大匹配','固定GT臂query集合','同27视频/2512独立时间事件（来自2573关系）','逐事件独立存在性',re['gt_event_matching'],
 '诊断完成','二值一对一1644/2512；hull时间1658/2512','GT最大匹配不是模型实际选择；没有native与GT一对一增益',
 [PGT+'evaluation/unique_time_events.csv',PGT+'evaluation/summary.json'],['analysis/gt_input_mask_probe_20260929/evaluate_masks.py'])

pending=[
 ('012-UNRUN','M012有界相对收益修正','Hypothesis','保留原区间并学习修正收益','设计/代码已准备，未训练/未评估',['methods/vidor-architecture-012-mathematical-design.md','tasks/VIDOR-ARCHITECTURE-012/EXECUTION_PROGRESS.md']),
 ('003-NO-RETURN','PVSG参与者替换普查任务包','Hypothesis / planned diagnosis','统计真实参与者替换','READY_FOR_REMOTE；无incoming结果',['handoff/outgoing/VIDSGG-REBIND-003/task.yaml']),
 ('006-NO-RETURN','006短关系远端诊断包','Hypothesis / planned diagnosis','定位短关系失败阶段','NOT_STARTED/NOT_HANDED_OFF；部分问题后来由008处理',['handoff/outgoing/VIDOR-SHORT-AUDIT-006/task.yaml']),
 ('007-VISUAL-NO-RETURN','007/007-r2视觉盲审','Planned observation','核验实体交互/运动语义','本地抽样完成；无人工填写返回',['handoff/outgoing/VIDOR-FAILURE-PROFILE-007/task.yaml','handoff/outgoing/VIDOR-FAILURE-PROFILE-007-r2/task.yaml']),
 ('STRICT44-NO-RESULT','strict44切换raw指标与单例脚本','UNVERIFIED','旧边残留/新边建立延迟','本地有脚本；未找到strict_candidates.csv、strict44_raw_metrics.csv或对应结果日志',['analysis/gt_input_mask_probe_20260929/source_snapshot/batch_raw_strict44.py','analysis/gt_input_mask_probe_20260929/source_snapshot/run_gt_s00001.py']),
 ('BIG-BINDING-NO-RESULT','BIG候选oracle及role_top2/top4探针','UNVERIFIED','实体绑定候选是否被过早丢弃','当前项目未发现对应结果/脚本；不使用聊天数字作为实验证据',[]),
]
for id,name,kind,hyp,status,paths in pending:
    card(id,name,kind,hyp,'无可纳入本总账的完成干预','不适用','计划或材料缺失；未作为已完成实验计数','未核实',{},'未验证/缺材料',status,'任何恢复数量、性能提升或机制结论均不能由这些材料单独支持',paths)

ledger=dict(audit_date='2026-10-05',phase=1,scope='仅本地项目真实文件；不读聊天作为实验事实；不查外网；不跑新模型/训练；不进入阶段2',
 definitions={'C':'固定原BIG与VrdONE Top100均漏检，不随新方法重新定义','F':'008 v2中VrdONE合格输入对存在几何可行的任意连续区间；oracle条件',
 'hit100':'官方triplet+min(主体vIoU,客体vIoU)>=0.5，置信度排序贪心一对一；每视频前100预测',
 'mAP':'按有GT视频AP平均，保存0-1小数；完整835清单实际平均832视频；不能用TP/GT冒充分子分母',
 'percentages':'所有比例保留原计数；标准化率的分子可为加权非整数，不是实际样本数量',
 'causal_evidence':'固定候选/权重的单操作对预测的直接影响可核验；历史记忆、长上下文压制、重复query竞争均未形成机制因果证据'},
 records=records,source_sha256=sources)
(OUT/'ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
print('Ledger records:',len(records))
print('Intervention variants:',sum(r['evidence_type']=='Intervention' for r in records))
print('Source files:',len(sources))
