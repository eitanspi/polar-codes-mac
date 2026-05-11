#!/bin/bash
# Monitor running experiments every 10 minutes
# Usage: nohup bash class_c_npd/monitor.sh > class_c_npd/results/monitor.log 2>&1 &

while true; do
    echo ""
    echo "========== $(date) =========="

    # Check if comparison is running
    if pgrep -f "cg_vs_npd_comparison" > /dev/null 2>&1; then
        echo ">>> CG vs NPD comparison: RUNNING"
        tail -5 class_c_npd/results/cg_vs_npd.log 2>/dev/null
    else
        echo ">>> CG vs NPD comparison: FINISHED"
        tail -3 class_c_npd/results/cg_vs_npd.log 2>/dev/null
    fi

    echo ""
    echo "--- Process status ---"
    ps aux | grep -E "cg_vs_npd|npd_design" | grep -v grep | awk '{printf "  PID=%s CPU=%s CMD=%s %s %s\n", $2, $3, $11, $12, $13}'

    # Check for errors
    if grep -qi "error\|traceback\|exception" class_c_npd/results/cg_vs_npd.log 2>/dev/null; then
        echo "!!! ERRORS DETECTED in cg_vs_npd.log !!!"
        grep -i "error\|traceback" class_c_npd/results/cg_vs_npd.log | tail -5
    fi

    sleep 600  # 10 minutes
done
