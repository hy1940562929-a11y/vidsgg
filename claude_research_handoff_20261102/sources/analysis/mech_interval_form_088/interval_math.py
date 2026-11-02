"""088 fixed CPU interval construction. Candidate functions accept no GT data."""
import hashlib
import numpy as np
N_QUERY=9
N_NULL=100

def require(ok,message):
    if not ok:raise ValueError(message)

def probability_audit(logits):
    x=np.asarray(logits)
    require(x.dtype==np.float32 and x.ndim==2 and x.shape[0]==N_QUERY and x.shape[1]>0,'expected finite float32 [9,N] native logits')
    require(np.isfinite(x).all(),'nonfinite native logit')
    with np.errstate(over='ignore',under='ignore'):
        p32=np.float32(1.)/(np.float32(1.)+np.exp(-x))
        p64=1./(1.+np.exp(-x.astype(np.float64)))
    b32=p32>np.float32(.5);b64=p64>.5;braw=x>0
    audit=dict(float32_float64_bit_differences=int(np.count_nonzero(b32!=b64)),float32_logit_zero_bit_differences=int(np.count_nonzero(b32!=braw)),N_values=int(x.size),cpu_boundary_audit_not_GPU_reproduction=True)
    require(audit['float32_float64_bit_differences']==0 and audit['float32_logit_zero_bit_differences']==0,'CPU sigmoid threshold boundary mismatch')
    return p32,b32,audit

def components(bits):
    edges=np.diff(np.r_[False,np.asarray(bits,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1).tolist(),(np.flatnonzero(edges==-1)-1).tolist()))

def interval(frames,query,first,last):
    first,last=int(first),int(last);a,b=int(frames[first]),int(frames[last])+1
    if b-a<5:return None
    return dict(query=int(query),first_index=first,last_index=last,start=a,end=b,sample_length=last-first+1,frame_length=b-a)

def shared_hash64(video,pair,query,repetition):
    require(0<=int(query)<9 and 0<=int(repetition)<100,'invalid null index')
    key=f'88/{video}/{int(pair)}/{int(query)}/{int(repetition)}'
    digest=hashlib.sha256(key.encode('utf-8')).hexdigest()
    return int(digest[:16],16),digest

def random_interval(real,frames,numerator):
    if real is None:return None
    n=len(frames);length=real['sample_length'];available=n-length+1
    require(available>0 and 0<=numerator<2**64,'invalid null support')
    first=(int(numerator)*available)//(2**64)
    result=interval(frames,real['query'],first,first+length-1)
    require(result is not None and result['sample_length']==length and result['frame_length']==real['frame_length'],'null length mismatch')
    return result

def generate_pair(video,pair_idx,frames,native_masks):
    """Construct and audit all real/null slots once. No GT time/class accepted."""
    frames=np.asarray(frames)
    require(frames.ndim==1 and np.issubdtype(frames.dtype,np.integer) and len(frames)>0,'integer sampled frames required')
    require(len(frames)==native_masks.shape[-1] and np.all(np.diff(frames)==4),'frozen stride4 grid mismatch')
    p,bits,numeric=probability_audit(native_masks);H=[];C=[];selection=[]
    for q in range(9):
        ids=np.flatnonzero(bits[q]);h=interval(frames,q,ids[0],ids[-1]) if len(ids) else None
        raw=components(bits[q]);legal=[]
        for a,b in raw:
            z=interval(frames,q,a,b)
            if z is not None:
                z.update(mean_probability=float(np.mean(p[q,a:b+1],dtype=np.float64)),peak_probability=float(np.max(p[q,a:b+1])))
                legal.append(z)
        chosen=min(legal,key=lambda z:(-z['mean_probability'],-z['peak_probability'],z['first_index'])) if legal else None
        require(chosen is None or h is not None,'C output without H capacity')
        H.append(h);C.append(chosen);selection.append(dict(query=q,positive_points=len(ids),components_before_min5=len(raw),selectable_components=len(legal),legal_components=legal))
    real=dict(video_name=str(video),pair_idx=int(pair_idx),N_valid=len(frames),origin=int(frames[0]),frames=frames.tolist(),H=H,C=C,selection=selection)
    null={'null_H':[],'null_C':[],'shared_hash64_numerators':[]}
    for rep in range(100):
        draws=[shared_hash64(video,pair_idx,q,rep)[0] for q in range(9)]
        null['shared_hash64_numerators'].append([str(v) for v in draws])
        for name,slots in (('null_H',H),('null_C',C)):
            moved=[random_interval(z,frames,draws[q]) for q,z in enumerate(slots)]
            require(sum(z is not None for z in moved)==sum(z is not None for z in slots),'null actual capacity mismatch')
            null[name].append(moved)
    hn=sum(z is not None for z in H);cn=sum(z is not None for z in C)
    require(cn<=hn<=9,'slot budget violated')
    audit=dict(video_name=str(video),pair_idx=int(pair_idx),H_candidates=hn,C_candidates=cn,H_empty_slots=9-hn,C_empty_slots=9-cn,C_selectable_components=sum(r['selectable_components'] for r in selection),N_query_slots=9,N_null_repetitions=100,matched_null_count_and_lengths=True,**numeric)
    return real,null,audit

def tiou(candidate,start,end):
    if candidate is None:return 0.
    intersection=max(0,min(candidate['end'],end)-max(candidate['start'],start))
    union=(candidate['end']-candidate['start'])+(end-start)-intersection
    return intersection/union

def score_slots(slots,start,end):
    scores=[tiou(z,start,end) for z in slots];best=max(scores,default=0.)
    return dict(hit=bool(best>=.5),max_tiou=float(best),matched_queries=[i for i,v in enumerate(scores) if slots[i] is not None and v>=.5])

def evaluate_event(event,real,null):
    """GT enters only after the complete real/null candidate library is sealed."""
    start,end=int(event['begin_fid']),int(event['end_fid']);require(start<end,'invalid GT interval')
    row=dict(event);row['H']=score_slots(real['H'],start,end);row['C']=score_slots(real['C'],start,end)
    for name in ('null_H','null_C'):
        require(len(null[name])==100,'each null must have exactly100 separate libraries')
        hits=[int(score_slots(slots,start,end)['hit']) for slots in null[name]]
        row[name]=dict(hits_by_repetition=hits,hit_count=sum(hits),repetitions=100,expected_hit=sum(hits)/100.)
    return row

def summarize(rows):
    require(bool(rows),'empty frozen cohort');videos=sorted({r['video_name'] for r in rows});N=len(rows)
    fields=('H','C','null_H','null_C')
    rates=lambda r:np.array([int(r['H']['hit']),int(r['C']['hit']),r['null_H']['expected_hit'],r['null_C']['expected_hit']],dtype=float)
    pervideo={v:dict(zip(fields,np.mean([rates(r) for r in rows if r['video_name']==v],axis=0).tolist())) for v in videos}
    matrix=np.array([[pervideo[v][f] for f in fields] for v in videos]);means=matrix.mean(0)
    indices=np.random.default_rng(88).integers(0,len(videos),size=(10000,len(videos)))
    boot=matrix[indices].mean(1);comparisons={}
    for name,a,b in (('C_minus_H',1,0),('C_minus_null_C',1,3),('H_minus_null_H',0,2)):
        effect=float(means[a]-means[b]);ci=np.quantile(boot[:,a]-boot[:,b],[.025,.975]).tolist()
        comparisons[name]=dict(video_equal_difference=effect,ci95=ci,passes_minimum_effect=bool(effect>=.05 and ci[0]>0),minimum_effect=.05)
    both=sum(r['H']['hit'] and r['C']['hit'] for r in rows);gains=sum(not r['H']['hit'] and r['C']['hit'] for r in rows);losses=sum(r['H']['hit'] and not r['C']['hit'] for r in rows);neither=N-both-gains-losses
    integers=dict(H_hits=both+losses,C_hits=both+gains,both_hit=both,gain=gains,loss=losses,neither_hit=neither,net_gain=gains-losses,denominator=N)
    integers['fractions']={k:v/N for k,v in integers.items() if k!='denominator'}
    nulls={name:dict(hit_indicator_sum=sum(r[name]['hit_count'] for r in rows),expectation_denominator=100,expected_hit_count=sum(r[name]['hit_count'] for r in rows)/100.,GT_denominator=N,event_equal_expected_rate=sum(r[name]['hit_count'] for r in rows)/(100.*N),per_repetition_hit_counts=[sum(r[name]['hits_by_repetition'][i] for r in rows) for i in range(100)]) for name in ('null_H','null_C')}
    out=dict(N_GT=N,N_videos=len(videos),video_order=videos,integer_counts=integers,null_expected_counts=nulls,video_equal_rates=dict(zip(fields,means.tolist())),comparisons=comparisons,per_video=pervideo,bootstrap=dict(seed=88,repetitions=10000,shared_indices_across_arms=True,indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),uncertainty='video bootstrap conditional on the100 fixed random libraries'))
    return out,indices

def decide(summary):
    c=summary['comparisons'];rates=summary['video_equal_rates']
    if c['C_minus_H']['passes_minimum_effect'] and c['C_minus_null_C']['passes_minimum_effect']:
        return dict(status='SUPPORTED',exit=None,scope='fixed selector time-shape coverage improvement only')
    if c['C_minus_H']['ci95'][1]<.05:
        return dict(status='NOT_SUPPORTED',exit='A',scope='no prespecified material improvement from this fixed recipe')
    if rates['H']<.20 and rates['C']<.20 and c['H_minus_null_H']['ci95'][1]<.05 and c['C_minus_null_C']['ci95'][1]<.05:
        return dict(status='NOT_SUPPORTED',exit='B',scope='both rules have low coverage without prespecified positional value')
    return dict(status='INCONCLUSIVE',exit=None,scope='no selector, threshold, candidate or training expansion')
