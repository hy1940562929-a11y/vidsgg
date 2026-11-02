#!/usr/bin/env python3
"""List actual result/status files without manufacturing missing metrics."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    root=Path(a.run)
    print('079 实测结果入口；没有运行或未完成的阶段不能解释为负结果。')
    for name in ('A_sampling_architecture','B_layer_readout','C_objective'):
        stage=root/name
        print('\n'+name)
        files=sorted(set(stage.glob('*summary*.json'))|set(stage.glob('*status*.json'))|set(stage.glob('*decision*.json')))
        if not files: print('未找到结果：NOT_EXECUTED_OR_INCOMPLETE');continue
        for f in files:
            print(str(f))
            d=json.loads(f.read_text())
            for k in ('complete','status','decision','stage','scope','scientific_decision','elapsed_seconds'):
                if k in d:print(k+': '+str(d[k]))
    print('\n先查看新增/损失、总分母、视频支持数与时间输出，再决定是否扩展。')
    print('这些是诊断/小训练结果，不是完整VidOR mAP或正式投稿结果。')

if __name__=='__main__': main()
