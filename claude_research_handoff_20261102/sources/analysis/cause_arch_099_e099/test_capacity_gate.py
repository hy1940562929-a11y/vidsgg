"""Small synthetic checks only; no native material or real experiment."""
import unittest
import importlib.util
from capacity_gate import match, best_legal, witness_checks, evaluate, solve


def event(i,a=0,b=9,p=1):
    return dict(video_name="v",pair_idx=0,gt_index=i,begin_fid=a,end_fid=b,predicate_id=p,
                main400=True,sub163=False,residual334=True,long_protection=False)


def candidate(i,a=0,b=9,p=1):
    return dict(candidate_id=str(i),start=a,end=b,predicate_id=p)


class CapacityTests(unittest.TestCase):
    def test_duplicate_slots_and_lexicographic_ties(self):
        result=match([candidate(0),candidate(1)],[event(2),event(1)])
        self.assertEqual(result["hits"],2)
        self.assertEqual([(x["gt_index"],x["query_index"]) for x in result["pairs"]],[(1,0),(2,1)])

    def test_count_precedes_iou_and_classes(self):
        cs=[candidate(0,0,9),candidate(1,4,13,p=2)]
        es=[event(0,0,9),event(1,4,13,p=1)]
        self.assertEqual(match(cs,es)["hits"],2)
        self.assertEqual(match(cs,es,True)["hits"],1)
        self.assertEqual(match([candidate(0,p=0)],es,True)["hits"],0)

    def test_grid_minimum(self):
        c,ok=best_legal(event(0,4,9),4,0)
        self.assertTrue(ok);self.assertEqual((c["start"],c["end"]),(4,9))
        self.assertEqual(best_legal(event(0),1,0),(None,False))
        self.assertFalse(best_legal(event(0,500,510),4,0)[1])

    def test_global_budget(self):
        self.assertEqual(match([candidate(i) for i in range(9)],[event(i) for i in range(12)])["hits"],9)

    def test_joint_exact_checks(self):
        es=[];pool={};base={}
        for i in range(120):
            e=event(i,10*(i%6),10*(i%6)+9)
            e.update(pair_idx=i//6,main400=i<100,residual334=i<100,
                     sub163=i<20,long_protection=i>=100)
            es.append(e);key=("v",i//6)
            pool.setdefault(key,[]).append(candidate(i,e["begin_fid"],e["end_fid"]))
            base[key]=[]
        before=evaluate(base,es);after=evaluate(pool,es)
        self.assertTrue(witness_checks(after,before)["pass"])
        self.assertFalse(witness_checks(before,before)["pass"])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "existing SciPy unavailable")
    def test_milp_sufficient_witness(self):
        es=[];pool={};base={}
        for i in range(120):
            e=event(i,10*(i%6),10*(i%6)+9)
            e.update(pair_idx=i//6,main400=i<100,residual334=i<100,
                     sub163=i<20,long_protection=i>=100)
            es.append(e);key=("v",i//6)
            pool.setdefault(key,[]).append(candidate(i,e["begin_fid"],e["end_fid"]))
            base[key]=[]
        before=evaluate(base,es)
        selected,receipt,_=solve(pool,es,before,10)
        self.assertIsNotNone(selected,receipt)
        self.assertTrue(witness_checks(evaluate(selected,es),before)["pass"])


if __name__=="__main__":
    unittest.main()
