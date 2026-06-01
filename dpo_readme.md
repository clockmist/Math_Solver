## DPO 训练方案调研与实践（v2）

任务：小学 1–6 年级数学应用题，基于 SFT 后的 Qwen2.5-0.5B-Instruct 做 DPO，提升测试集准确率。

**结论：竞赛实测 v2「最小改动」方案效果最好，提交文件为 `submit.csv`（模型 `qwen_dpo_merged_final/`）。**

---

### 1. 问题背景

SFT 合并模型（`qwen_sft_full/`）在 `small_val.json`（84 条）上约 **59.5%**。尝试用 DPO 进一步优化时，发现**数据构造方式**对效果影响远大于超参微调。

原始 v1 方案（`generate_DPO_dataset.py`，已删除）：
- chosen = `train_cot.json` 中的正确 CoT
- rejected = SFT 模型自由生成的错误推理

训练后 small_val **下降 10+ 个百分点**，竞赛效果亦不佳。

---

### 2. 调研结论：小学数学 DPO 数据该怎么建

参考 Step-DPO、DPO-ST 等数学推理 DPO 工作，结合本任务特点：

1. **Minimal-Edit 优于 Free-Generation**  
   rejected 应从 chosen 派生，而非让模型自由生成。改动局限在**一个算式**（算错结果、改运算符、改操作数）。

2. **保持推理结构一致**  
   「已知条件 / 求解目标 / 计算过程」段落结构应相同，避免 chosen 用乘倍数法、rejected 用比例法这类路径分歧。

3. **错误类型应可定位**  
   优先末步算术错误（wrong_result）；模型只需学会「315 对、316 错」，而非换一套解题思路。

4. **少而精**  
   DPO 通常 300–1000 条高质量偏好对即可；本方案用 **500 条**（`dpo_data_v2_500.json`）。

5. **DPO 擅长细粒度偏好，不擅长风格对齐**  
   chosen/rejected 差异过大时，模型只学到「偏好某种写法」，泛化反而变差。

---

### 3. 方案对比与实验结果

#### 3.1 v1 vs v2 数据质量（500 条抽样）

| 指标 | v1（SFT 自由生成 rejected） | v2（最小改动） |
|------|------------------------------|----------------|
| chosen/rejected 文本相似度 | **0.431** | **0.985** |
| 相似度 > 0.8 的样本占比 | 6/500 | **500/500** |
| 计算行重叠率 | 0.001 | **0.382** |
| 答案不同比例 | 1.000 | 1.000 |

v1 的 rejected 与 chosen 几乎是两套解题思路；v2 仅在最后一个算式和最终答案处有差异，符合 DPO 对「计算纠错」的学习需求。

#### 3.2 各版本 small_val（84 条）与竞赛

| 版本 | 数据策略 | small_val | 竞赛表现 |
|------|----------|-----------|----------|
| SFT | — | 59.52% | — |
| **DPO v2** | 最小改动，500 条 | 57.14% | **最好** |
| DPO v3 | SFT 扫描 hard negative + fallback | 53.57% | 差 |
| DPO v3b | v2 主体 + 少量 hard negative | 59.52% | 不如 v2 |

说明：
- **small_val 不能代表 8000 条测试集**（v3b 本地与 SFT 持平，竞赛仍不如 v2）。
- v3/v3b 引入 hard negative（相似度 ~0.79）或改变样本分布，易过拟合偏好对、损伤泛化。
- v2 虽在 small_val 略低于 SFT，但**测试集/竞赛上最稳**，是当前保留方案。

v3/v3b 相关代码与数据已清理，仓库仅保留 v2 流水线。

---

### 4. 当前方案（v2）说明

#### 4.1 数据构造（`generate_DPO_dataset_v2.py`）

- 输入：`train_cot.json`（DeepSeek 生成的正确 CoT，生成时已与 `train.json` 标签校验）
- 输出：
  - `dpo_data_v2.json` — 全量约 11250 条
  - `dpo_data_v2_500.json` — 随机抽样 500 条，供训练
  - `dpo_data_v2_stats.json` — 统计信息
- 构造方式：从 chosen 正确 CoT 出发，对**最后一个可改算式**做 minimal-edit（改结果 / 运算符 / 操作数），保证 rejected 答案错误且结构高度一致。

#### 4.2 训练流程

```
checkpoint-7325 (SFT LoRA)
        ↓ merge_model.py
   qwen_sft_full/
        ↓ train_DPO.py + dpo_data_v2_500.json
   qwen_dpo_output_enhanced/  (LoRA)
        ↓ 训练内 merge
   qwen_dpo_merged_final/
        ↓ infer_dpo.py + test.json
      submit.csv
```

#### 4.3 超参（`train_DPO.py`）

| 参数 | 值 |
|------|-----|
| 数据 | `dpo_data_v2_500.json` |
| beta | 0.3 |
| learning_rate | 5e-6 |
| num_epochs | 1 |
| batch_size | 1 × grad_accum 4 |
| max_length | 768 |
| LoRA | r=16, alpha=32 |

相对原始 `origin/dpo` 分支的改动：
- `merge_model.py`：SFT LoRA 路径改为 `./checkpoint-7325`，输出 `qwen_sft_full/`
- `train_DPO.py`：数据改为 v2；epoch 2→1；移除 trl 不支持的 `max_prompt_length`
- 新增 `generate_DPO_dataset_v2.py`、`infer_dpo.py`
- 删除 v1 的 `generate_DPO_dataset.py` 与 `dpo_data.json`

---

### 5. 相关代码/文件

| 文件 | 作用 |
|------|------|
| `merge_model.py` | 合并 SFT LoRA 与基座 → `qwen_sft_full/` |
| `generate_DPO_dataset_v2.py` | 构造 v2 偏好数据 |
| `train_DPO.py` | DPO 训练 → `qwen_dpo_output_enhanced/`、`qwen_dpo_merged_final/` |
| `evaluate_dpo.py` | 在 `small_val.json` 上评估 |
| `infer_dpo.py` | 在 `test.json` 上推理 → **`submit.csv`** |

模型与数据（体积大，通常不入 git）：
- `qwen_sft_full/` — SFT 合并全量模型
- `qwen_dpo_merged_final/` — **竞赛用 DPO 合并模型**
- `submit.csv` — **竞赛提交结果（v2）**

---

### 6. 使用步骤

```bash
conda activate StarWM
cd /data3/xuyuhang/Math_Solver

# 1. 生成 v2 数据（无需 GPU）
python generate_DPO_dataset_v2.py

# 2. 合并 SFT 模型（需 checkpoint-7325/ 与基座 Qwen/）
python merge_model.py

# 3. DPO 训练（需 GPU）
python train_DPO.py

# 4. 验证集评估
python evaluate_dpo.py

# 5. 测试集推理（竞赛提交）
python infer_dpo.py
# 或指定 GPU：INFER_GPU=1 python infer_dpo.py
```

---

### 7. 后续可尝试方向（未验证）

若继续优化，建议**在 v2 框架内**小步迭代，避免再引入 hard negative：
- 在 `dpo_data_v2_500.json` 基础上微调 beta（0.1–0.3）、lr（1e-6–5e-6）
- 构建与测试分布更接近的 holdout 集做验证，**不要仅依赖 small_val**
- 全量 `dpo_data_v2.json` 子采样 vs 500 条固定集对比
