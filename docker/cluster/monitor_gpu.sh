#!/bin/bash
# GPU monitoring script for cluster jobs
# Logs GPU utilization during training

LOGFILE="${1:-gpu_utilization.log}"
INTERVAL="${2:-5}"  # seconds

echo "======================================"
echo "GPU Monitoring Started"
echo "======================================"
echo "Logfile: ${LOGFILE}"
echo "Interval: ${INTERVAL}s"
echo ""

# Header
echo "Timestamp,GPU_ID,GPU_Util_%,Memory_Used_MB,Memory_Total_MB,Temp_C,Power_W" > "${LOGFILE}"

# Monitor loop
while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
               --format=csv,noheader,nounits >> "${LOGFILE}" 2>/dev/null
    sleep "${INTERVAL}"
done
