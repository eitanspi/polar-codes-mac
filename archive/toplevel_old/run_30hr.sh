#!/bin/bash
# 30-hour training campaign
# Run from: /Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2
#
# Launches:
# 1. Main campaign (scheduled sampling N=256, then N=512) — uses ~80% CPU
# 2. Regular N=512 training in background — uses ~20% CPU (lower priority)

cd /Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2

echo "=== 30-Hour Training Campaign ==="
echo "Started: $(date)"
echo ""

# Kill any existing training
pkill -f "train_n256\|train_n512\|train_30hr" 2>/dev/null
sleep 2

# Launch main campaign (scheduled sampling)
echo "Launching main campaign (scheduled sampling N=256 → N=512)..."
nohup python3 -u neural/train_30hr_campaign.py > neural/train_30hr_campaign_stdout.log 2>&1 &
MAIN_PID=$!
echo "  PID: $MAIN_PID"

# Launch regular N=512 training at lower priority
echo "Launching regular N=512 training (low priority)..."
nohup nice -n 10 python3 -u neural/train_n512_long.py > neural/train_n512_long_stdout.log 2>&1 &
N512_PID=$!
echo "  PID: $N512_PID"

echo ""
echo "Both processes launched."
echo ""
echo "To monitor:"
echo "  tail -f neural/train_30hr_campaign.log        # Main campaign"
echo "  tail -f neural/train_n512_long.log             # N=512 regular"
echo "  tail -f neural/train_30hr_campaign_stdout.log  # Main stdout"
echo ""
echo "To check status:"
echo "  ps aux | grep 'train_30hr\|train_n512' | grep -v grep"
echo ""
echo "PIDs saved to neural/campaign_pids.txt"
echo "$MAIN_PID" > neural/campaign_pids.txt
echo "$N512_PID" >> neural/campaign_pids.txt
