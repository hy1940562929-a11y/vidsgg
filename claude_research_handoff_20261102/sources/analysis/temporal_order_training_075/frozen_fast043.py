"""Materialize trajectories after unchanged CUDA candidate sorting, no GT."""
import numpy as np
import torch

def fast_decode(model,inp,cls,masks):
    """Exact source score arithmetic/order; boxes copied for selected200 only."""
    probs=cls.softmax(-1)
    score,pred=probs[...,1:].topk(model.topk,dim=-1);pred=pred+1
    sids=inp['sids'].tolist();oids=inp['oids'].tolist();durs=inp['traj_durations'].tolist();offsets=inp['so_offset'].tolist()
    cats=inp['cat_ids'].tolist();triplets=[];durations=[];pairs=[];pis=[];qs=[];ks=[]
    for pi,(s,o) in enumerate(zip(sids,oids)):
        origin=max(durs[s][0],durs[o][0]);end=min(durs[s][1],durs[o][1]);n=inp['so_features_list'][pi].shape[-1]
        bits=(masks[pi][:,:n].sigmoid()>.5).cpu().numpy()
        for q in range(9):
            idx=np.flatnonzero(bits[q])
            if not len(idx):continue
            a=int(idx[0])*model.feat_stride+int(offsets[pi]);b=int(idx[-1])*model.feat_stride+int(offsets[pi])+1
            assert 0<=a<b<=end-origin
            if b-a<model.pred_min_frames:continue
            for k in range(model.topk):
                pis.append(pi);qs.append(q);ks.append(k);pairs.append([s,o]);durations.append([origin+a,origin+b])
    if not pis:return None
    device=cls.device;ip=torch.tensor(pis,device=device);iq=torch.tensor(qs,device=device);ik=torch.tensor(ks,device=device)
    ss=torch.as_tensor(inp['sids'],device=device)[ip];oo=torch.as_tensor(inp['oids'],device=device)[ip]
    cs=inp['cat_scores'].to(device)
    triple_scores=torch.stack([cs[ss],score[ip,iq,ik],cs[oo]],-1)
    avg=triple_scores.mean(-1);index=torch.argsort(avg,descending=True)[:model.n_max_pair]
    keep=index.cpu().tolist();pp=pred[ip,iq,ik][index].cpu().tolist();selected_pairs=[pairs[k] for k in keep];selected_durs=[durations[k] for k in keep]
    boxes=[b.detach().cpu() for b in inp['bboxes_list']];trajectories=[]
    for j,(s,o) in enumerate(selected_pairs):
        a,b=selected_durs[j];sb=boxes[s][a-durs[s][0]:b-durs[s][0]];ob=boxes[o][a-durs[o][0]:b-durs[o][0]]
        assert len(sb)==len(ob)==b-a
        trajectories.append(torch.stack([sb,ob]).tolist());triplets.append([cats[s],pp[j],cats[o]])
    return {'triplets':triplets,'triple_scores':triple_scores[index].cpu().tolist(),'triple_scores_avg':avg[index].cpu().tolist(),'so_trajs':trajectories,'pred_durations':selected_durs,'so_tids':selected_pairs}

def collect(model,inp,Capture):
    """Same original preprocessing groups, all layer forward once per batch."""
    n=len(inp['sids']);records=[];mapping=[]
    with torch.no_grad(),Capture(model) as capture:
        for offset in range(0,n,model.max_so_pair):
            xs,ms,ids=model.preprocessing(inp['so_features_list'][offset:offset+model.max_so_pair])
            for x,m,ii in zip(xs,ms,ids):
                if x is None:assert not ii;continue
                model._mask_vrd(x,m);mapping.append([offset+i for i in ii])
        records=capture.records
    assert len(records)==len(mapping) and sorted(i for ii in mapping for i in ii)==list(range(n))
    return records,mapping

def decode_head(model,head,inp,records,mapping,project):
    n=len(inp['sids']);cls=[None]*n;masks=[None]*n
    with torch.no_grad():
        for r,ids in zip(records,mapping):
            mm=project(head,r)[-1]
            for row,pi in enumerate(ids):cls[pi]=r['cls'][-1,row];masks[pi]=mm[row]
        if not n:return None
        return fast_decode(model,inp,torch.stack(cls),masks)
