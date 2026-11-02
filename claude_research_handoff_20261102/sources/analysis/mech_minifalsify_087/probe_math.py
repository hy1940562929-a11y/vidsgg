"""087-only evaluation math; 079 training mathematics remains imported unchanged."""
import numpy as np

def position_full(n):
    if n < 1:raise ValueError('positive length required')
    z=np.zeros((256,n),np.float32);u=np.linspace(-1,1,n,dtype=np.float32)
    z[0],z[1],z[2]=1,u,u*u
    for k in range(1,4):z[2*k+1],z[2*k+2]=np.sin(k*np.pi*u),np.cos(k*np.pi*u)
    return z

def local_points(frames, start, end, same_predicate_intervals):
    frames=np.asarray(frames);length=end-start
    if length<=0:raise ValueError('invalid interval')
    pos=(frames>=start)&(frames<end)
    neg=((frames>=start-length)&(frames<start))|((frames>=end)&(frames<end+length))
    for a,b in same_predicate_intervals:neg &= ~((frames>=a)&(frames<b))
    return pos,neg

def auroc(scores,pos,neg):
    scores=np.asarray(scores)
    if not np.isfinite(scores).all():raise ValueError('nonfinite score')
    if not pos.any() or not neg.any():return None
    # Equivalent pairwise Mann-Whitney definition without an O(P*N) matrix.
    n=np.sort(scores[neg]);p=scores[pos]
    return float((np.searchsorted(n,p,'left')+.5*(np.searchsorted(n,p,'right')-np.searchsorted(n,p,'left'))).mean()/len(n))

def aggregate(rows,arms=('fine','fused','position'),reps=10000,seed=87):
    selected=[r for r in rows if r.get('eligible')];videos=sorted({r['video_name'] for r in selected})
    pervideo={v:{a:float(np.mean([r['auroc'][a] for r in selected if r['video_name']==v])) for a in arms} for v in videos}
    comparisons={};rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(videos),size=(reps,len(videos))) if videos else None
    for a,b in (('fine','fused'),('fine','position'),('fused','position')):
        diffs=np.asarray([pervideo[v][a]-pervideo[v][b] for v in videos])
        ci=np.quantile(diffs[sample].mean(1),[.025,.975]).tolist() if videos else None
        effect=float(diffs.mean()) if videos else None
        comparisons[a+'_minus_'+b]=dict(mean=effect,ci95=ci,passes=bool(len(videos)>=20 and len(selected)>=100 and effect>=.05 and ci[0]>0))
    return dict(N_total=len(rows),N_evaluable=len(selected),N_videos=len(videos),N_excluded=len(rows)-len(selected),per_video=pervideo,
                video_equal_mean={a:float(np.mean([pervideo[v][a] for v in videos])) if videos else None for a in arms},comparisons=comparisons)

def decision(group):
    if group['N_videos']<20 or group['N_evaluable']<100:return 'INCONCLUSIVE_INSUFFICIENT_SUPPORT'
    a,b,c=[group['comparisons'][k] for k in ('fine_minus_fused','fine_minus_position','fused_minus_position')]
    if a['passes'] and b['passes']:return 'SUPPORTED_FIXED_RECIPE_READABILITY_DIFFERENCE'
    if c['passes'] and a['ci95'][1]<.05:return 'STOP_FIXED_RECIPE_FUSION_DAMAGE_EXPLANATION'
    if b['ci95'][1]<.05 and c['ci95'][1]<.05:return 'STOP_THIS_LINEAR_READOUT_RECIPE'
    return 'INCONCLUSIVE_NO_EXPANSION'
