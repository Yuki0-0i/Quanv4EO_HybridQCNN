#!/bin/bash
# ============================================================
# 一键复现脚本 - 4 配置 × 4 实验 × 5-seed 完整评估
# ============================================================
# 用法:
#   bash run_all.sh                  # 完整 (含 QConv 提取, ~10h)
#   bash run_all.sh --skip-qconv     # 跳过 QConv (用已有 npz, ~1h)
#
# 假设:
#   1. conda env plenv_gpu 已创建 + 依赖已装
#   2. datasets/EuroSAT/ 已准备 27000 张
#   3. 当前在 Quanv4EO_HybridQCNN/ 目录
# ============================================================

set -e
SKIP_QCONV=0
for arg in "$@"; do
    case $arg in
        --skip-qconv) SKIP_QCONV=1 ;;
    esac
done

# ============== 0. 环境 ==============
source /home/admin001/miniconda3/etc/profile.d/conda.sh
conda activate plenv_gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMP=/tmp
export TEMP=/tmp
export TMPDIR=/tmp

cd "$(dirname "$0")"
EXP=experiments

echo "============================================================"
echo "Quanv4EO 一键复现"
echo "  SKIP_QCONV=$SKIP_QCONV"
echo "  路径: $(pwd)"
echo "============================================================"

# ============== 1. 烟测 ==============
echo ""
echo "[1/8] 烟测..."
python $EXP/smoke_test.py 2>&1 | tail -20

# ============== 2. QConv 提取 (可选跳过) ==============
if [ $SKIP_QCONV -eq 0 ]; then
    echo ""
    echo "[2/8] QConv 提取 (4 配置 × 5000 张, ~10h)..."

    # 4 配置各跑一次 (后台)
    for cfg in "4 2 ry baseline" "6 2 ry qubit6" "4 4 ry depth4" "4 2 rxry enc_rxry"; do
        read q l e tag <<< "$cfg"
        if [ ! -f "experiments/features_${tag}_5000_${q}q${l}l_${e}.npz" ]; then
            echo "  → 跑 ${tag}..."
            python $EXP/run_pipeline_5000_chunked.py \
                --max_per_class 500 --n_jobs 8 --qubits $q --n_layers $l --encoding $e \
                --chunk_size 1000 --tag "${tag}_5000_${q}q${l}l_${e}"
        else
            echo "  ✓ ${tag} npz 已存在, 跳过"
        fi
    done
else
    echo "[2/8] SKIP QConv 提取"
fi

# ============== 3. 经典 CNN baseline ==============
echo ""
echo "[3/8] 经典 CNN baseline (1000/2000/5000 张)..."
python $EXP/classic_cnn_baseline.py 2>&1 | tail -15

# ============== 4. 4 配置 × 4 实验 × 5-seed ==============
echo ""
echo "[4/8] 4 配置 × 4 实验 × 5-seed (60 runs, ~40 min)..."
python $EXP/exp_matrix.py 2>&1 | tail -50

# ============== 5. K-fold CV ==============
echo ""
echo "[5/8] K-fold CV (75 runs, ~20 min)..."
python $EXP/kfold_3configs.py 2>&1 | tail -30

# ============== 6. 画图 ==============
echo ""
echo "[6/8] 画 4 张关键实验图..."
python $EXP/plot_figures.py 2>&1 | tail -5

# ============== 7. 综合报告 ==============
echo ""
echo "[7/8] 生成综合报告..."
python $EXP/all_experiments_summary.py 2>&1 | tail -5

# ============== 8. 总结 ==============
echo ""
echo "============================================================"
echo "全部完成!"
echo "  实验数据: $EXP/features_*.npz (gitignore)"
echo "  报告:     $EXP/../26项目报告1.docx (上级目录)"
echo "  Markdown: reports/*.md"
echo "  图:       experiments/fig*.png"
echo "============================================================"
