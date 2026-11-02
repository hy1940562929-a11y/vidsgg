# 084复现

仅Python标准库：

```bash
python3 -m unittest discover -s analysis/week1_component_readout_084 -p 'test_*.py'
python3 analysis/week1_component_readout_084/run.py --out /absolute/path/to/new_output_directory
```

[最终解释与既有负结果](REPORT.md)优先于生成结果里的初版解释；它增加独立审查要求的query库范围与015D负结果，没有修改任何数字或冻结协议。4项测试通过，Reviewer独立复核2573逐GT、54分组和时间见证。
