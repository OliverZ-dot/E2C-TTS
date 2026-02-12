# AIME 2024 Benchmark

请将 AIME 2024 数据集下载到本目录。

## 数据格式要求

文件需为 **parquet** 格式，且包含以下字段（或兼容别名）：

| 字段 | 别名 | 说明 |
|------|------|------|
| question | Question, problem, prompt | 数学题目 |
| answer | Answer, response | 正确答案 |

## 下载方式

### 方式 1：Hugging Face Datasets

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/aime_2024")
ds["test"].to_parquet("aime2024.parquet")
```

或使用命令行：

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/aime_2024')
ds['test'].to_parquet('aime2024.parquet')
"
```

### 方式 2：Maxwell-Jia/AIME_2024

```python
from datasets import load_dataset 
ds = load_dataset("Maxwell-Jia/AIME_2024")
# 根据实际 split 名称调整，如 "test" 或 "validation"
split = list(ds.keys())[0]
df = ds[split].to_pandas()
# 确保列名为 question, answer
df = df.rename(columns={
    "problem": "question", "Problem": "question",
    "answer": "answer", "Answer": "answer",
})[["question", "answer"]]
df.to_parquet("aime2024.parquet", index=False)
```

### 方式 3：直接下载 parquet

部分数据集提供 parquet 直接下载，例如：

- BytedTsinghua-SIA/AIME-2024:  
  https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024

下载后将文件重命名为 `aime2024.parquet` 并放到本目录。

## 放置位置

```
data/aime2024/
├── README.md         # 本说明
└── aime2024.parquet  # 数据文件（需自行下载）
```

数据准备好后，在项目根目录运行：

```bash
python run_tts.py
```
