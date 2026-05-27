#!/bin/bash
# run_pipeline.sh - 完整流水线：SFT训练 → DPO数据构建

set -e  # 任何命令失败就退出，方便排查

echo "=========================================="
echo "流水线启动时间: $(date)"
echo "=========================================="

# ==================== 阶段1: SFT训练 ====================
echo ""
echo "[阶段1/2] 开始 SFT 训练..."
echo ""


# 启动训练（后台子进程，但脚本会等待它完成）
python qwen_cot_ft.py

# 检查训练是否正常完成
if [ ! -d "./output/Qwen_CoT_v2" ]; then
    echo "错误: SFT训练未完成或输出目录不存在"
    exit 1
fi

echo ""
echo "SFT训练完成！输出目录: ./output/Qwen_CoT_v2"
echo ""

# ==================== 阶段2: DPO数据构建 ====================
echo "[阶段2/2] 开始 DPO 数据集构建..."
echo ""


python generate_DPO_dataset.py  # ← 你的DPO构建脚本名

echo ""
echo "=========================================="
echo "流水线完成时间: $(date)"
echo "DPO数据输出: dpo_data.json"
echo "=========================================="
