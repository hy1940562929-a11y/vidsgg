# 083 复现入口

只读已导入的015原始输出诊断和077/080逐GT表；无需GPU、PyTorch或新增依赖。

```bash
python3 -m unittest discover -s analysis/week1_query_feasibility_083 -p 'test_*.py'
python3 analysis/week1_query_feasibility_083/run.py --out /absolute/path/to/new_output_directory
```

脚本拒绝覆盖结果，校验来源SHA、2573唯一GT、默认oracle逐条一致及所有上界单调性。8项合成测试通过，Reviewer独立重算全部24行。

结果见 [简短报告](results_v1/REPORT.md)、[分解CSV](results_v1/summary.csv)、[逐GT](results_v1/per_gt.csv)、[来源清单](results_v1/manifest.json)。

覆盖GT亚秒组187条：默认形状oracle命中11，失败176；176中1条网格不可行、25条阈值oracle可达、150条阈值hull族不可达。固定分类读出命中10条，不能混用11。所有oracle跨同identity全部连续段和query，不要求正确谓词，均不是实际恢复数量。
