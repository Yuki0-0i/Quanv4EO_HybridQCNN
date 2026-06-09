#!/bin/bash
# QConv 排队: 等待 Q6 完成后自动跑 D4, 然后 RXRY
# 每个 task 完成时间 ~4.5h

set -e
SRC=/hdd/Dengxuanyu/dxy1/Quanv4EO_0604
PYTHON="source /home/admin001/miniconda3/etc/profile.d/conda.sh && conda activate plenv_gpu && cd $SRC"

# 当前在跑的 Q6 PID
Q6_PID=2480210

echo "=== 监控 Q6 完成 (PID=$Q6_PID) ==="
while ps -p $Q6_PID > /dev/null 2>&1; do
    sleep 60
    echo "$(date '+%H:%M:%S') - Q6 still running, elapsed $(ps -p $Q6_PID -o etime= 2>/dev/null | tr -d ' ')"
done
echo "$(date '+%H:%M:%S') - Q6 done, waiting 30s for cleanup..."

# 等 workers 退出
sleep 30
pkill -9 -f "LokyProcess" 2>/dev/null || true
pkill -9 -f "loky" 2>/dev/null || true
sleep 10

echo ""
echo "=== 启动 D4 (Depth=4, 4q 4l RY, 5000 张) ==="
rm -f /tmp/qconv_d4.log
bash -c "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup setsid python -u $SRC/experiments/run_pipeline_5000_chunked.py --max_per_class 500 --n_jobs 8 --qubits 4 --n_layers 4 --encoding ry --chunk_size 1000 --tag depth4_5000_4q4l_ry > /tmp/qconv_d4.log 2>&1 < /dev/null" &
D4_PID=$!
disown
echo "D4 started PID: $D4_PID at $(date '+%H:%M:%S')"

echo "=== 监控 D4 完成 ==="
while ps -p $D4_PID > /dev/null 2>&1; do
    sleep 60
    echo "$(date '+%H:%M:%S') - D4 still running, elapsed $(ps -p $D4_PID -o etime= 2>/dev/null | tr -d ' ')"
done
echo "$(date '+%H:%M:%S') - D4 done, waiting 30s..."
sleep 30
pkill -9 -f "LokyProcess" 2>/dev/null || true
pkill -9 -f "loky" 2>/dev/null || true
sleep 10

echo ""
echo "=== 启动 RXRY (Encoding RX+RY, 4q 2l, 5000 张) ==="
rm -f /tmp/qconv_rxry.log
bash -c "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup setsid python -u $SRC/experiments/run_pipeline_5000_chunked.py --max_per_class 500 --n_jobs 8 --qubits 4 --n_layers 2 --encoding rxry --chunk_size 1000 --tag enc_rxry_5000_4q2l_rxry > /tmp/qconv_rxry.log 2>&1 < /dev/null" &
RXRY_PID=$!
disown
echo "RXRY started PID: $RXRY_PID at $(date '+%H:%M:%S')"

echo "=== 监控 RXRY 完成 ==="
while ps -p $RXRY_PID > /dev/null 2>&1; do
    sleep 60
    echo "$(date '+%H:%M:%S') - RXRY still running, elapsed $(ps -p $RXRY_PID -o etime= 2>/dev/null | tr -d ' ')"
done
echo "$(date '+%H:%M:%S') - RXRY done!"
sleep 30
pkill -9 -f "LokyProcess" 2>/dev/null || true
pkill -9 -f "loky" 2>/dev/null || true
echo ""
echo "=== 全部 3 个 QConv 任务完成 ==="
ls -la /hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_*_5000_*.npz
