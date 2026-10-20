#!/bin/bash
# 10.10.10.104, after the ESC-RANK fill: a second judge from another family (Llama-3.1-8B, models.yaml judge_fallback)
# rescores IP, PRI and Success on the key arms, into <run>/scores_judge2/<arm>; scripts/judge_agreement.py compares.
#   (setsid nohup ./scripts/second_judge_104.sh > runs/second_judge_104.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for the ESC-RANK fill"
until [ -n "$NO_WAIT" ] || grep -q "escrank fill complete" runs/escrank_fill.log 2>/dev/null; do sleep 120; done
echo "=== pull llama3.1:8b-instruct-q4_K_M $(date)"
curl -s localhost:11434/api/pull -d '{"model": "llama3.1:8b-instruct-q4_K_M", "stream": false}'; echo
free_gpu
for a in runs/baselines_1000:cellA_mono_ungated runs/baselines_1000:cellB_mono_gated runs/baselines_1000:cellC_dec_ungated \
         runs/baselines_1000:cellD_dec_gated runs/memory_v2:mem_none runs/memory_v2:mem_needstate \
         runs/steering:steer_mono_L12_c0 runs/steering:steer_mono_L12_c0.2 runs/sft_agents:sft_dec_gated_mem; do
    r=${a%%:*}; arm=${a#*:}
    [ -s $r/scores_judge2/$arm/success.jsonl ] || run python src/metrics.py --dialogues $r/dialogues/$arm.jsonl \
        --scale ip --scale pri --scale success --judge-role judge_fallback --backend ollama --out $r/scores_judge2/$arm
done
echo "=== second judge complete $(date)"
