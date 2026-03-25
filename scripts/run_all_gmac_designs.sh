#!/bin/bash
# Generate reliable MC designs for Gaussian MAC at SNR=6dB
# Uses 4 hours total budget — enough for ~50K+ trials at small N, ~1K+ at N=1024
cd /Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git

for cls in C A B; do
    echo "=== Class $cls ==="
    python scripts/run_design_gmac.py \
        --class $cls \
        --N 8 16 32 64 128 256 512 1024 \
        --snr-db 6 \
        --hours 4 \
        --force \
        2>&1 | tee designs/gmac_design_${cls}_log.txt
    echo
done

echo "All designs complete."
