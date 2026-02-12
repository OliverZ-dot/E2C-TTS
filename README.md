# E2C Test-Time Scaling 实验

E2C 的 Test-Time Scaling 实验（AIME 2024）。

## 结构

详见 [结构.md](结构.md)。

## 环境

```bash
pip install -r requirements.txt
```

## 数据准备

将 AIME 2024 数据放到 `data/aime2024/aime2024.parquet`。

下载：

```bash
python scripts/download_aime2024.py
```

或按 [data/aime2024/README.md](data/aime2024/README.md) 手动下载。

## 运行

```bash
python run_tts.py
```

可选参数：

```bash
python run_tts.py --config config/tts.yaml --methods e2c_select_lm_judge e2c_sc --budgets 4 8 --limit 5
```

- `--config`：配置文件路径  
- `--methods`：指定方法列表  
- `--budgets`：指定 budget 列表（K/N）  
- `--limit`：限制样本数（调试用）

## 方法说明

| 方法 | 说明 |
|------|------|
| greedy_cot | Greedy CoT，N=1 |
| self_consistency | Self-Consistency，N 条链，多数投票 |
| e2c_select_lm_judge | E2C-Select (Self LM-Judge) |
| e2c_select_semantic_cluster | E2C-Select (Semantic Cluster) |
| e2c_sc | E2C-SC，执行全部 K 个 plan |
| e2c_rp | E2C-RP，随机选 1 个 plan |

## 句子嵌入（e2c_select_semantic_cluster）

需要句子向量。配置在 `config/tts.yaml` 的 `embedding`：`backend` 可选 `modelscope` / `huggingface` / `auto`，`modelscope_model` 默认 GTE 英文。需安装 `modelscope`。

## 输出

不同 budget 下的 Acc (%) 和 Tokens (k)。

## 参考

Explore–Execute Chain 论文及 Appendix A.4 实验设置。

