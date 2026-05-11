#!/bin/bash
# Overnight automation: runs after N=128 sweep completes
# Launch with: nohup bash class_c_npd/run_overnight.sh > class_c_npd/results/overnight.log 2>&1 &

set -e
cd /Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2

echo "=== Overnight automation started: $(date) ==="

# Wait for the N=128 sweep to finish (check every 5 min)
echo "Waiting for npd_design_sweep to finish..."
while pgrep -f "npd_design_sweep" > /dev/null 2>&1; do
    sleep 300
    echo "  Still running... $(date)"
done
echo "npd_design_sweep finished: $(date)"

# Step 1: Extend to N=256
echo ""
echo "=== Step 1: NPD design sweep N=256 ==="
python -u class_c_npd/training/npd_design_sweep.py --N_list 256 2>&1 | tee class_c_npd/results/npd_design_n256.log

# Step 2: Multi-channel (BEMAC) with NPD design
echo ""
echo "=== Step 2: BEMAC NPD design sweep ==="
# (would need a separate script for BEMAC NPD design — skip for now)

# Step 3: Generate final report
echo ""
echo "=== Step 3: Generate plots and report ==="
python -u class_c_npd/eval/generate_report.py 2>&1
python -u class_c_npd/eval/plot_sweep.py 2>&1

echo ""
echo "=== Overnight automation finished: $(date) ==="
