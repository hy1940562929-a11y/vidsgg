"""CPU-only E099 joint capacity acceptance; no training or model forward.

The MILP searches a sufficient pool, NOT the entire legal oracle universe.
Failure/infeasibility/timeout without a verified witness is always
JOINT_CAPACITY_UNRESOLVED, never a scientific REJECT.
"""
import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

MANIFEST_SHA = "2613d524a528528212cb28c1d2a543431d7fc9b08cfc9d61f324a6aebd2f2269"
EVENT_SHA = "0c412de71d7ba10dbd1380798741c0069fb65557ee6659ff21e7992c2dc781fa"
GROUPS = ("main400", "sub163", "residual334", "long_protection", "all")
EXPECTED = {"main400": (400, 46, 396), "residual334": (334, 29, 331),
            "sub163": (163, 0, 163), "long_protection": (1101, 540, 815),
            "all": (2573, 989, 1591)}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def ratio(candidate, event):
    inter = max(0, min(candidate["end"], event["end_fid"]) -
                max(candidate["start"], event["begin_fid"]))
    union = candidate["end"]-candidate["start"] + event["end_fid"]-event["begin_fid"]-inter
    return inter, union


def match(candidates, rows, class_consistent=False):
    """Exact count, exact rational total tIoU, GT-ID/query lexicographic order.

    At most twelve candidates (E111 expanded query bank). Dynamic programming over 2^9 query subsets avoids
    floating-point tie perturbations. Integer LCM scaling represents all tIoUs
    exactly. Distinct duplicate slots remain distinct candidates.
    """
    require(len(candidates) <= 12, "CAPACITY_EXCEEDED")
    rows = sorted(rows, key=lambda e: (e["video_name"], int(e["gt_index"])))
    edges, scale = [], 1
    for gi, e in enumerate(rows):
        for qi, c in enumerate(candidates):
            num, den = ratio(c, e)
            if 2*num >= den and (not class_consistent or
                                  c["predicate_id"] > 0 and c["predicate_id"] == e["predicate_id"]):
                scale = math.lcm(scale, den)
                edges.append((gi, qi, num, den))
    bygt = defaultdict(list)
    for gi, qi, num, den in edges:
        bygt[gi].append((qi, num*(scale//den)))
    # mask -> (scaled sum tIoU, tuple of (GT index, query index)).
    dp = {0: (0, ())}
    for gi, e in enumerate(rows):
        nxt = dict(dp)
        gid = int(e["gt_index"])
        for mask, (score, pairs) in dp.items():
            for qi, weight in bygt[gi]:
                if not (mask >> qi) & 1:
                    new = (score+weight, pairs+((gid, qi),))
                    key = mask | (1 << qi)
                    old = nxt.get(key)
                    if old is None or new[0] > old[0] or new[0] == old[0] and new[1] < old[1]:
                        nxt[key] = new
        dp = nxt
    mask, (score, pairs) = min(dp.items(),
                               key=lambda item: (-item[0].bit_count(), -item[1][0], item[1][1]))
    return {"hits": mask.bit_count(), "sum_tIoU_fraction": [score, scale],
            "pairs": [{"gt_index": g, "query_index": q,
                       "candidate_id": candidates[q]["candidate_id"]} for g, q in pairs]}


def evaluate(candidate_sets, events):
    out = {}
    for group in GROUPS:
        selected = [e for e in events if group == "all" or e[group]]
        bp = defaultdict(list)
        for e in selected:
            bp[(e["video_name"], e["pair_idx"])].append(e)
        counts = defaultdict(lambda: [0, 0, 0])
        details = []
        for pair, rs in sorted(bp.items()):
            cs = candidate_sets.get(pair, [])
            t, c = match(cs, rs), match(cs, rs, True)
            counts[pair[0]][0] += len(rs)
            counts[pair[0]][1] += t["hits"]
            counts[pair[0]][2] += c["hits"]
            details.append({"video_name": pair[0], "pair_idx": pair[1],
                            "time": t, "class": c})
        denominator = sum(c["predicate_id"] > 0 for cs in candidate_sets.values() for c in cs)
        out[group] = {"N": len(selected), "videos": len(counts),
                      "time_hits": sum(v[1] for v in counts.values()),
                      "class_hits": sum(v[2] for v in counts.values()),
                      "time_macro": float(sum((Fraction(v[1], v[0]) for v in counts.values()), Fraction())/len(counts)),
                      "class_macro": float(sum((Fraction(v[2], v[0]) for v in counts.values()), Fraction())/len(counts)),
                      "video_N_time_class": dict(counts), "matching": details}
        if group == "all":
            out[group]["foreground_candidates"] = denominator
            out[group]["class_precision"] = out[group]["class_hits"]/denominator if denominator else 0.
    return out


def macro_delta(candidate, baseline, kind):
    index = 1 if kind == "time" else 2
    a, b = candidate["video_N_time_class"], baseline["video_N_time_class"]
    require(set(a) == set(b), "MACRO_VIDEO_SUPPORT")
    return sum((Fraction(a[v][index]-b[v][index], b[v][0]) for v in b), Fraction())/len(b)


def witness_checks(value, base):
    tests = {"residual_gain_at_least61": value["residual334"]["time_hits"]-base["residual334"]["time_hits"] >= 61}
    fractions = {}
    for mode in ("time", "class"):
        d = macro_delta(value["main400"], base["main400"], mode)
        tests["main400_"+mode+"_strict_increase"] = d > 0
        fractions["main400_"+mode] = str(d)
        d = macro_delta(value["long_protection"], base["long_protection"], mode)
        tests["long_"+mode+"_protection"] = d >= Fraction(-1, 100)
        fractions["long_"+mode] = str(d)
        tests["all_"+mode+"_count_protection"] = value["all"][mode+"_hits"] >= base["all"][mode+"_hits"]
    a, b = value["all"], base["all"]
    # Zero-candidate convention is precisely zero precision.
    pa = Fraction(a["class_hits"], a["foreground_candidates"]) if a["foreground_candidates"] else Fraction()
    pb = Fraction(b["class_hits"], b["foreground_candidates"]) if b["foreground_candidates"] else Fraction()
    tests["precision_protection"] = pa >= pb
    return {"pass": all(tests.values()), "tests": tests, "exact_macro_differences": fractions,
            "precision_fractions": [str(pa), str(pb)]}


def best_legal(event, n, origin):
    a, b = np.triu_indices(n, k=1)
    if not len(a):
        return None, False
    left, right = origin+4*a, origin+4*b+1
    inter = np.maximum(0, np.minimum(right, event["end_fid"])-np.maximum(left, event["begin_fid"]))
    union = right-left+event["end_fid"]-event["begin_fid"]-inter
    # Ratio maximization with exact integer cross multiplication; loop only tied
    # float maxima, then compare exact fractions. Integers are small frame IDs.
    scores = inter/union
    idx = int(np.argmax(scores))
    tied = np.flatnonzero(scores == scores[idx])
    for j in tied:
        if int(inter[j])*int(union[idx]) > int(inter[idx])*int(union[j]):
            idx = int(j)
    result = {"start": int(left[idx]), "end": int(right[idx]),
              "predicate_id": int(event["predicate_id"]),
              "candidate_id": f"GT_ORACLE/{event['video_name']}/{event['gt_index']}"}
    return result, bool(2*inter[idx] >= union[idx])


def predicate_map(path):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "vidor_pred_categories" for t in node.targets):
            values = ast.literal_eval(node.value)
            require(len(values) == 51 and values[0] == "__background__", "PREDICATE_MAP")
            return {name: i for i, name in enumerate(values)}
    raise RuntimeError("PREDICATE_MAP_MISSING")


def material(args):
    import torch
    torch.set_num_threads(2)
    cache, events_path, preflight = Path(args.cache), Path(args.events), Path(args.preflight)
    require(sha(cache/"manifest.json") == MANIFEST_SHA, "MANIFEST_SHA")
    require(sha(events_path) == EVENT_SHA, "EVENT_SHA")
    manifest = json.loads((cache/"manifest.json").read_text())
    for rec in manifest["files"]:
        require(sha(cache/rec["path"]) == rec["sha256"], "CACHE_SHA "+rec["path"])
    status = json.loads((preflight/"status.json").read_text())
    require(status["status"] == "NATIVE_REBUILD_PASS" and status["checked"] == 1725 and status["max_abs"] <= 4e-6, "NATIVE_GATE")
    for path, digest in json.loads((preflight/"material_receipt.json").read_text())["models"].items():
        require(sha(path) == digest, "MODEL_SOURCE_CHANGED "+path)
    split_counts = {}
    for split, nb, npair, ne in (("train",88,350,563), ("adapter_dev",23,89,169)):
        paths = sorted((cache/split).glob("*.pt"))
        require(len(paths) == nb, "TRAIN_DEV_BATCH")
        pairs, videos, total = [], set(), 0
        for path in paths:
            c = torch.load(path, map_location="cpu", weights_only=False)
            require(len(c["metadata"]) == len(c["preds"]) == len(c["targets"]) == c["f"].shape[0], "TARGET_BATCH")
            for j,(meta,pred,target) in enumerate(zip(c["metadata"],c["preds"],c["targets"])):
                n=len(pred)
                require(target.shape==(n,c["f"].shape[-1]) and n==len(meta["events"]) and n<=9, "TARGET_COMPLETE")
                require([e["target_index"] for e in meta["events"]]==list(range(n)), "TARGET_INDEX")
                require([e["predicate_id"] for e in meta["events"]]==pred.tolist(), "TARGET_LABEL")
                require(bool((pred>0).all()) and bool((pred<=50).all()), "TARGET_CLASS_RANGE")
                require(len(set(e["annotation_gt_index"] for e in meta["events"]))==n, "TARGET_DUPLICATE")
                require(bool(((target==0)|(target==1)).all()) and bool((target.sum(-1)>0).all()), "TARGET_MASK")
                require(bool((target[:,~c["valid"][j,0]]==0).all()), "TARGET_PADDING")
                pairs.append((meta["video_id"],tuple(meta["pair_key"])))
                videos.add(meta["video_id"]);total+=n
        require(len(pairs)==npair and total==ne and videos==set(manifest["split_videos"][split]),"SPLIT_IDENTITY")
        split_counts[split]={"batches":nb,"pairs":npair,"events":ne,"videos":len(videos)}
    require(set(manifest["split_videos"]["train"]).isdisjoint(manifest["eval_videos"]) and
            set(manifest["split_videos"]["adapter_dev"]).isdisjoint(manifest["eval_videos"]), "SPLIT_OVERLAP")
    events=json.loads(events_path.read_text())
    require(len(events)==2573 and {g:sum(bool(e[g]) for e in events) for g in GROUPS[:-1]} ==
            {"main400":400,"sub163":163,"residual334":334,"long_protection":1101}, "GROUP_COUNTS")
    pmap=predicate_map(cache/"sources/repo/dataloaders/category.py")
    annotations={}
    for e in events:
        video=e["video_name"]
        if video not in annotations:
            annotations[video]=json.loads((cache/"sources/annotations"/(video+".json")).read_text())
        gt=annotations[video]["relation_instances"][e["gt_index"]]
        require((gt["begin_fid"],gt["end_fid"])==(e["begin_fid"],e["end_fid"]), "GT_TIME_IDENTITY")
        e["predicate_id"]=pmap[gt["predicate"]]
        require(e["predicate_id"]>0,"GT_BACKGROUND")
    require(len({(e["video_name"],e["gt_index"]) for e in events})==2573,"EVENT_DUPLICATE")
    bp=defaultdict(list)
    for e in events:bp[(e["video_name"],e["pair_idx"])].append(e)
    baseline,pool,feasible={},{},{}
    for (video,pi),rows in sorted(bp.items()):
        c=torch.load(cache/"eval"/f"gt27_{video}_{pi:04d}.pt",map_location="cpu",weights_only=False)
        meta=c["metadata"];n=int(meta["N_valid"]);origin=int(meta["origin"])
        require(all(e["origin"]==origin and e["N_valid"]==n and e["stride"]==4 for e in rows),"FIXED_PAIR")
        require(c["masks"].shape[-2]==9 and c["cls"].shape[-1]==51,"NATIVE_NINE_CLASS")
        cls=c["cls"][-1,0].softmax(-1).argmax(-1).tolist()
        masks=c["masks"][-1,0,:,:n].sigmoid()
        slots=[]
        for q in range(9):
            ix=torch.where(masks[q]>.5)[0]
            if len(ix) and ix[-1]>ix[0]:
                slots.append({"start":origin+4*int(ix[0]),"end":origin+4*int(ix[-1])+1,
                              "predicate_id":int(cls[q]),"candidate_id":f"C0/{video}/{pi}/{q}","native_query":q})
        baseline[(video,pi)]=slots
        pool[(video,pi)]=[dict(x) for x in slots]
        for e in rows:
            candidate,ok=best_legal(e,n,origin)
            feasible[(video,e["gt_index"])]=ok
            if ok:pool[(video,pi)].append(candidate)
    return events,baseline,pool,feasible,{"manifest_sha256":MANIFEST_SHA,"events_sha256":EVENT_SHA,
            "verified_files":len(manifest["files"]),"splits":split_counts,
            "predicate_source":"094 annotation relation_instances[gt_index] + hashed category.py; never query class"}


def solve(pool, events, base, seconds):
    """MILP sufficient witness, one set reused for three seeds."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    pairs=sorted(pool);candidates=[];by_pair={}
    for pair in pairs:
        by_pair[pair]=list(range(len(candidates),len(candidates)+len(pool[pair])))
        candidates.extend(pool[pair])
    nx=len(candidates);objective=[0.]*nx
    ri=[];ci=[];vals=[];lower=[];upper=[];named=[]
    def row(coeff,lo=-np.inf,hi=np.inf,name=""):
        k=len(lower)
        for j,v in coeff.items():
            if v:ri.append(k);ci.append(j);vals.append(v)
        lower.append(lo);upper.append(hi);named.append(name)
    for pair,ids in by_pair.items():row(dict.fromkeys(ids,1),hi=9,name="capacity/"+str(pair))
    group_variables={}
    for group in GROUPS:
        rs=[e for e in events if group=="all" or e[group]]
        vc=Counter(e["video_name"] for e in rs)
        for mode in ("time","class"):
            yall=[];macro={};by_gt=defaultdict(list);by_candidate=defaultdict(list)
            for e in rs:
                pair=(e["video_name"],e["pair_idx"])
                for xi in by_pair[pair]:
                    c=candidates[xi];a,b=ratio(c,e)
                    if 2*a<b or mode=="class" and (c["predicate_id"]==0 or c["predicate_id"]!=e["predicate_id"]):continue
                    yi=len(objective)
                    objective.append(-100. if group=="residual334" and mode=="time" else
                                     -1./(len(vc)*vc[e["video_name"]]) if group=="main400" else -.00001)
                    yall.append(yi);macro[yi]=1./(len(vc)*vc[e["video_name"]])
                    by_gt[(e["video_name"],e["gt_index"])].append(yi)
                    by_candidate[xi].append(yi)
            for key,ys in by_gt.items():row(dict.fromkeys(ys,1),hi=1,name=f"GT/{group}/{mode}/{key}")
            for xi,ys in by_candidate.items():
                coeff=dict.fromkeys(ys,1);coeff[xi]=-1
                row(coeff,hi=0,name=f"slot/{group}/{mode}/{xi}")
            group_variables[(group,mode)]=(yall,macro)
            if group=="residual334" and mode=="time":
                row(dict.fromkeys(yall,1),lo=base[group]["time_hits"]+61,name="residual334_plus61")
            if group=="main400":
                # Numerical MILP bound is baseline; STRICT > is subsequently
                # verified with exact Fractions. Objective rewards main gain.
                row(macro,lo=base[group][mode+"_macro"],name="main400_"+mode+"_nondecrease_relaxation")
            if group=="long_protection":
                row(macro,lo=base[group][mode+"_macro"]-.01,name="long_"+mode+"_minus_one_pp")
            if group=="all":
                row(dict.fromkeys(yall,1),lo=base[group][mode+"_hits"],name="all_"+mode+"_count")
                if mode=="class":
                    coeff=dict.fromkeys(yall,1.)
                    p0=base[group]["class_precision"]
                    for xi,c in enumerate(candidates):
                        if c["predicate_id"]>0:coeff[xi]=-p0
                    row(coeff,lo=0,name="all_precision")
    matrix=coo_matrix((vals,(ri,ci)),shape=(len(lower),len(objective))).tocsc()
    result=milp(np.asarray(objective),integrality=np.ones(len(objective)),
                bounds=Bounds(np.zeros(len(objective)),np.ones(len(objective))),
                constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),
                options={"time_limit":seconds,"mip_rel_gap":0.001})
    chosen=None
    if result.x is not None:
        chosen={pair:[dict(candidates[i]) for i in ids if result.x[i]>.5] for pair,ids in by_pair.items()}
    receipt={"solver":"scipy.optimize.milp/HiGHS","status":int(result.status),"message":str(result.message),
             "objective":float(result.fun) if result.fun is not None else None,
             "variables":len(objective),"candidate_variables":nx,"constraints":len(lower),
             "time_limit_seconds":seconds,"pool_complete":False,
             "main_strict_constraint":"MILP relaxed to >=baseline; exact postcheck requires strictly >0",
             "negative_result_exit":"JOINT_CAPACITY_UNRESOLVED; not full-universe or three-set infeasibility",
             "constraint_names":named,"candidate_selection":result.x[:nx].tolist() if result.x is not None else None}
    return chosen,receipt,{"row":ri,"column":ci,"value":vals,"lower":lower,"upper":upper,
                           "objective":objective,"shape":matrix.shape}


def main():
    parser=argparse.ArgumentParser()
    root=Path("/root/autodl-tmp/VrdONE-data")
    parser.add_argument("--cache",default=str(root/"commander_cache_import_094/cache_v4"))
    parser.add_argument("--events",default=str(root/"commander_cause_arch_095/results_v2/events.json"))
    parser.add_argument("--preflight",default=str(root/"commander_cause_arch_099/preflight_v1"))
    parser.add_argument("--out",default=str(root/"commander_cause_arch_099/capacity_v1"))
    parser.add_argument("--seconds",type=float,default=180)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();phase="MATERIAL"
    try:
        events,c0,pool,feasible,identity=material(args)
        write(out/"material_identity.json",identity)
        phase="C0_AND_HISTORICAL_U9"
        base=evaluate(c0,events);audit={}
        for group in GROUPS:
            rs=[e for e in events if group=="all" or e[group]]
            bypair=Counter((e["video_name"],e["pair_idx"]) for e in rs if feasible[(e["video_name"],e["gt_index"])])
            u=sum(min(9,n) for n in bypair.values())
            audit[group]={"N":len(rs),"C0":base[group]["time_hits"],"U9":u}
            if group in EXPECTED:require((len(rs),base[group]["time_hits"],u)==EXPECTED[group],"HISTORICAL_C0_U9_"+group)
        write(out/"C0_metrics.json",base);write(out/"historical_capacity.json",audit)
        write(out/"candidate_pool.json",{"uses_GT":True,"purpose":"capacity audit only","complete_universe":False,
              "pairs":[{"video_name":p[0],"pair_idx":p[1],"candidates":v} for p,v in pool.items()]})
        phase="SUFFICIENT_JOINT_WITNESS"
        chosen,solver,constraints=solve(pool,events,base,args.seconds)
        write(out/"solver.json",solver)
        # Preserve exact linear optimization representation without JSON infinities.
        np.savez_compressed(out/"milp_constraints.npz",**{k:np.asarray(v) for k,v in constraints.items()})
        checks={"pass":False,"reason":"no incumbent"}
        if chosen is not None:
            require(all(len(cs)<=9 for cs in chosen.values()),"INCUMBENT_CAPACITY")
            value=evaluate(chosen,events);checks=witness_checks(value,base)
            write(out/"witness_metrics.json",value)
            write(out/"witness_candidates.json",{"uses_GT":True,"never_use_as_model_predictions":True,
                  "three_seed_sets":"identical verified set repeated three times; never 27 slots",
                  "pairs":[{"video_name":p[0],"pair_idx":p[1],"candidates":v} for p,v in chosen.items()]})
        write(out/"witness_checks.json",checks)
        passed=checks["pass"]
        write(out/"status.json",{"status":"JOINT_CAPACITY_PASS" if passed else "TECHNICAL_FAILURE",
             "subreason":None if passed else "JOINT_CAPACITY_UNRESOLVED",
             "scientific_REJECT":False,"training_runs":0,"elapsed_seconds":time.monotonic()-start,
             "witness_sets":3 if passed else 0,
             "scope":"capacity feasibility only; not network-achievability or intervention evidence"})
        print((out/"status.json").read_text(),flush=True)
        return 0 if passed else 2
    except Exception as exc:
        write(out/"status.json",{"status":"TECHNICAL_FAILURE","phase":phase,"error":str(exc),
             "traceback":traceback.format_exc(),"training_runs":0,"elapsed_seconds":time.monotonic()-start})
        print((out/"status.json").read_text(),flush=True);return 2


if __name__=="__main__":
    raise SystemExit(main())
