# Study guide: the multi-agent architecture, the single-session baselines, and the model choices

Status as of 7 Oct 2026. Numbers are copied from `results/v1|v2|v3_full1000/README.md`; nothing here is new.

---

## Part 1. The multi-agent architecture

### 1.1 What it does

One supporter reply is produced by four LLM calls instead of one, each with its own prompt and job, plus a
gate that decides whether the draft is sent.

```
conversation so far + memory brief (<= 320 tokens)
        |
   Analyzer    -> surface feeling + candidate needs, each with a verbatim seeker quote, a depth (1-3),
        |         a parent need and a status hint (hypothesis / confirmed / disconfirmed / resolved)
   [pacing]    -> highest allowed rung this turn: L1 if Analyzer confidence < 0.45; L3 only if memory
        |         holds a confirmed need (src/pacing.py)
   Strategist  -> the move and the rung it wants, within the allowed rungs
        |
   Responder   -> the draft reply, told "Rung you must not exceed: Lx"
        |
   Critic      -> grounding checks (unsupported / denied inference, stale reference, overreaching depth)
        |         + a predicted intrusiveness score IP (1-7 items)
   Gate        -> score s = 0.7 * predicted IP + 0.3 * violation mass;
        |         release if s <= lambda-hat, else revise (up to 2), else safe reflective fallback
   Memory      -> the Analyzer's grounded claims update the need-state graph (propose / confirm / deny / resolve)
```

The point of decomposing: each step can be inspected and constrained (evidence quotes, depth cap, gate),
which a single prompt cannot guarantee. The monolithic baseline writes analysis, strategy and reply in one
call, so comparing the two isolates the value of decomposition.

### 1.2 The experiment it is tested in: the 2 x 2

| Cell | Architecture | Gate | Memory |
|---|---|---|---|
| A `cellA_mono_ungated` | monolithic | off | need-state (but see note) |
| B `cellB_mono_gated` | monolithic | conformal | need-state (but see note) |
| C `cellC_dec_ungated` | multi-agent | off | need-state, written by the Analyzer |
| D `cellD_dec_gated` | multi-agent | conformal | need-state, written by the Analyzer |

All four run the untuned Qwen2.5-7B (no adapter mapped on the lab), 2-4 linked sessions per test person.
Note: cells A and B have no Analyzer, so their memory is never written (v2 README says so; fixed for future
monolithic runs by the `memory_writer` switch added 7 Oct).
Main effects: decomposition = (C + D) - (A + B), gate = (B + D) - (A + C), both halved; interaction = the rest.

### 1.3 Results so far

**v1 (2 Oct, 30 profiles) - not usable as evidence.** It was the bug-finding run. Its own README lists why:
Success failed to parse for 14 of 30 dialogues (B11); the Critic's predicted IP was always 0 because it copied
the prompt's example (B13); the Critic invented "denied inference" flags, so the gate rewrote 72 % of cell B
(B14); single-model replies were ~85 words vs ~21 for the multi-agent one (length confound, B19); 7-13 % of
single-model turns leaked the private analysis to the seeker (B28). All were fixed before v2.

**v2 (3 Oct, 40 test profiles, 121 sessions per memory arm) - the first clean 2 x 2.**

| Arm | Success | AELS | Basic | IP (lower better) | PRI obs. | PRI counterfactual |
|---|---|---|---|---|---|---|
| A mono, ungated | 0.381 | 0.855 | 0.801 | 0.078 | 0.215 | +0.003 [-0.006, 0.011] |
| B mono, gated | 0.356 | 0.834 | 0.801 | **0.059** | 0.217 | -0.004 [-0.011, 0.004] |
| C multi-agent, ungated | 0.379 | 0.838 | 0.775 | 0.095 | 0.232 | +0.008 [-0.001, 0.017] |
| D multi-agent, gated | 0.362 | 0.829 | 0.757 | **0.062** | **0.200** | -0.005 [-0.013, 0.002] |

- Success: decomposition +0.002, gate -0.021, interaction +0.009; D - A = -0.019 [-0.039, 0.002], not
  significant.
- The gate cuts IP by about 25-35 % (0.078 to 0.059 monolithic, 0.095 to 0.062 multi-agent).
- Gate paths: B released 49 % / revised 39 % / fallback 13 %; D 45 % / 30 % / 25 %.
- Gate calibration (v2): alpha 0.10, tau 0.20, lambda-hat 0.125, release 54 %.

**v3 (1,000-profile corpus, 100 test people) - in progress.**

| Arm | Sessions | Success | AELS | IP | PRI obs. | PRI counterfactual |
|---|---|---|---|---|---|---|
| A mono, ungated | 310 | 0.382 | 0.859 | 0.068 | 0.184 | -0.003 [-0.007, 0.002] |
| B mono, gated | 310 | 0.375 | 0.833 | **0.061** | 0.187 | still being scored |
| C multi-agent, ungated | 310 | conversations done; judging paused for the fine-tuned 7B runs | | | | |
| D multi-agent, gated | - | not run yet (resumes after the fine-tuned 7B runs) | | | | |

- Gate calibration (v3, on cell C rollouts of the calibration people): alpha 0.05 / 0.10 / 0.20 give
  lambda-hat 0 / 0.2 / 1.0, release 13 / 73 / 100 %.
- Share of turns with IP > 0.20: cell A 4.0 % vs cell B 2.3 % (the bound is 10 %).
- Cell B paths: released 53.5 %, revised 36.8 %, fallback 9.6 %.

**What can be claimed today:** the gate reliably lowers intrusiveness at a small Success cost (v2 and v3
agree); decomposition has not shown a Success gain (v2); the v3 multi-agent result (cells C, D) is pending.

### 1.4 How to study it (reading order)

1. `configs/arms.yaml` - every arm is a set of switches (supporter, architecture, gate, memory, simulator).
2. `src/evaluate.py` - `roll_out` (line ~150: sessions, decay between them) and `run_session` (line 63: the
   turn loop calling `pipeline.step`).
3. `src/agents.py` - read in this order:
   `AgentPipeline.step` (line 234, the whole turn), `analyze` / `strategize` / `generate` (204-232),
   `Critic.review` and `_decide` (99-161, release / revise / fallback), `update_memory` (280).
4. The four prompts: `prompts/agent_analyzer.md`, `agent_strategist.md`, `agent_generator.md`,
   `agent_critic.md` - what each role is asked for and the JSON it must return.
5. `src/pacing.py` (the rung cap) and `src/grounding.py` (the typed violations the Critic checks).
6. `src/conformal.py` (Hoeffding-Bentkus bound, lambda-hat) and `src/calibrate.py` (how the gate is fitted).
7. `src/baselines/listener_mono.py` - the monolithic contrast (one call, same output format).
8. Scoring: `src/metrics.py`, `src/judge.py`, `prompts/metric_ip.md`, `metric_pri.md`, `scale_success.md`;
   `src/counterfactual.py` for counterfactual PRI; `src/report.py` for the 2 x 2 effects.
9. Results in order: `results/v1/README.md` (what broke and why), `results/v2/README.md` section 3,
   `results/v3_full1000/README.md` section 7, and the plots in
   `results/v3_full1000/baselines_1000/extra/baseline_plots/`.
10. Real turns: every multi-agent turn stores the Analyzer state, Strategist plan, Critic verdict and gate
    path. To read one:
    ```
    python3 -c "import json; d=json.loads(open('results/v3_full1000/baselines_1000/dialogues/cellB_mono_gated.jsonl').readline()); a=d['agent_artifacts'][1]; print(json.dumps({k:a[k] for k in ('text','permitted_rung','strategist','critic','path')}, indent=1)[:2000])"
    ```

---

## Part 2. The single-session baselines

### 2.1 What they are

| Arm | Supporter | Architecture | Simulator | Purpose |
|---|---|---|---|---|
| `base_instruct` | untuned Qwen2.5-7B | monolithic | proactive | the starting point: what the base model does |
| `reactive_baseline` | untuned Qwen2.5-7B | monolithic | **reactive** (help-seeking seeker, as in the baseline paper) | the paper's setting |
| `sft_with_thoughts` | fine-tuned (analysis + strategy + reply) | monolithic | proactive | effect of fine-tuning with rationales |
| `sft_wo_thoughts` | fine-tuned (reply only) | monolithic | proactive | isolates the thoughts |

One session per test person, no memory, no gate.

### 2.2 Results

**v2 (40 test people, 7B fine-tuned on the 50-profile corpus):**

| Arm | Success | AELS | Basic | IP | PRI obs. | PRI counterfactual |
|---|---|---|---|---|---|---|
| base_instruct | 0.383 | 0.838 | 0.804 | 0.082 | 0.212 | -0.001 [-0.016, 0.015] |
| reactive_baseline | 0.388 | 0.872 | 0.808 | 0.085 | 0.212 | -0.002 [-0.016, 0.013] |
| sft_with_thoughts (7B) | **0.392** | 0.840 | 0.805 | 0.103 | 0.272 | +0.009 [-0.009, 0.027] |

**v3 (100 test people):**

| Arm | Success | AELS | IP | PRI obs. | PRI counterfactual |
|---|---|---|---|---|---|
| base_instruct | 0.383 | 0.836 | 0.065 | 0.192 | -0.001 [-0.009, 0.007] |
| reactive_baseline | **0.415** | **0.868** | 0.089 | 0.190 | 0.008 [0.000, 0.016] |
| 0.5B with thoughts | 0.397 | | 0.103 | 0.246 | 0.010 [-0.000, 0.021] |
| 0.5B without thoughts | 0.395 | | 0.081 | 0.233 | 0.006 [-0.004, 0.016] |
| 7B with thoughts | **0.415** | | 0.091 | 0.243 | **0.018 [0.008, 0.029]** |
| 7B without thoughts | 0.392 | | 0.083 | 0.225 | 0.010 [0.001, 0.020] |
| 3B, Qwen3-4B | queued | | | | |

Paper-scale checks for the untuned arms (v3): Basic 80.1 (base) vs 80.9 (reactive); Aff 5.71 / 5.86; Neg 1.00 /
1.00; Sup 6.26 / 6.51; Man 5.59 / 5.84.

How to read them: Success sits in a narrow band (0.38-0.42) because the judge gives most dialogues a 3 of 7;
the 7B with thoughts is ahead of the 7B without by 0.023 [0.000, 0.047]; the reactive seeker makes every
supporter look better because that seeker volunteers more; fine-tuning raises IP and reactance a little.

### 2.3 Code for the baselines

1. `configs/arms.yaml` lines 13-16 (the four arm definitions).
2. `src/evaluate.py` - `n_sessions = 1` unless the arm has memory or `multi_session` (inside `roll_out`).
3. `src/baselines/listener_mono.py` - the single-call supporter used by all four.
4. `src/build_sft.py` - `SUPPORTER_SYSTEM` (the one output spec for every supporter) and `parse_target`
   (keeps only the `<response>` part for the seeker).
5. `src/dialogue.py` - `simulator_system`; prompts `user_proactive.md` vs `user_reactive.md` (the only
   difference between base_instruct and reactive_baseline).
6. `src/llm.py` - how the adapter is switched on for the fine-tuned supporter (`use_adapter`).
7. `scripts/scaling_eval.sh` - how the fine-tuned test runs were launched (transformers backend, `--adapter`).
8. Scoring: `src/metrics.py`, `src/judge.py`, `src/counterfactual.py`; plots `extra/baseline_plots.py`,
   cross-size table `extra/scaling.py`.

---

## Part 3. Defending the model choices

### 3.1 Supporter, simulator and agents: Qwen2.5-Instruct (0.5B / 3B / 7B)

1. **A clean size ladder from one recipe.** Qwen2.5 ships 0.5B, 1.5B, 3B, 7B, 14B, 32B and 72B with the same
   tokenizer, chat template and post-training pipeline. That is what makes the scaling study (power-law fit,
   PVI and LoRA geometry across sizes) interpretable: size is the only thing that changes.
2. **It fits the hardware.** The project runs on a 12 GB RTX 3060 and an 8 GB RTX 4060 laptop. A 7B trains
   under QLoRA on 12 GB and serves in about 5 GB in 4-bit, leaving room to swap the 12B judge in and out;
   0.5B and 3B train on the laptop.
3. **Structured output.** The agents, the annotator and the memory writes all need JSON. Qwen2.5-Instruct is
   trained for structured output, and the code tolerates the malformed fields it still produces (coerced, or
   marked `parse_failed`, never a crash; v1 bug B27).
4. **Open weights, no gating, mature tooling.** Downloadable without an access request, supported by Ollama,
   bitsandbytes 4-bit, PEFT and transformers on both machines. (Licences: Apache-2.0 for 0.5B and 7B; the 3B
   is under the Qwen Research licence, which allows research use.)
5. **One family for generator, simulator and agents** keeps the comparison between arms about the component,
   not about different models talking to each other. The cost of this choice is known and measured: the
   corpus replies are written by Qwen2.5-7B, so the 7B fine-tune partly learns its own style (the PVI drop at
   7B). That is why a second family is being added (Qwen3-4B now, Llama next).

### 3.2 Why not Qwen3 as the main family

1. **Built-in thinking mode clashes with our design.** Qwen3 is a hybrid reasoning model that emits its own
   `<think>` block by default. Our method teaches the model its own `<analysis>` and `<strategy>` from
   annotations; a second, untrained reasoning channel would confound the with-vs-without-thoughts comparison.
   It must be switched off (`enable_thinking=False`) in training and inference, which we do for Qwen3-4B.
2. **No 7B.** Qwen3 comes as 0.6B, 1.7B, 4B, 8B, 14B, 32B. The 8B (8.2 B parameters) is heavier than Qwen2.5-7B
   (7.6 B) on a 12 GB card under QLoRA, and the ladder does not line up with the corpus generator's size.
3. **Continuity.** The corpus, the simulator and every v1/v2 result were built on Qwen2.5-7B; switching the main
   family mid-project would make the earlier results incomparable.
4. **It is not ignored:** Qwen3-4B is fine-tuned on the same data as a second family (validation loss 0.698
   after epoch 1 and 0.669 after epoch 2, with thoughts), and Qwen3 data generation is planned after 9 Oct.
   (Qwen3's advantage, fair to state: Apache-2.0 for every size.)

### 3.3 Why not Llama (the baseline paper's family)

1. **Size ladder.** Llama 3.x offers 1B, 3B, 8B, 70B across different releases (3.2 small models, 3.1 8B), not
   one recipe across sizes, which weakens a scaling study.
2. **Access and licence.** Llama weights are gated behind an access request and a custom community licence.
3. **It is planned, for a direct comparison:** fine-tuning Llama-3.1-8B on our corpus after 9 Oct gives a
   like-for-like comparison with the paper's base model and a second family for the self-generated-data check.

### 3.4 The judge: Mistral-Nemo-Instruct-12B

1. **A different family from everything it judges** (Qwen), enforced in code (`judge.assert_judge_separate`),
   to limit self-preference bias - an LLM tends to rate text like its own higher.
2. **Strongest open judge that fits:** 12B in 4-bit is about 7 GB, swapped with the 7B on the 12 GB card;
   long context (128k) for whole multi-session dialogues; Apache-2.0.
3. **Open instead of GPT-4o (the paper's judge):** reproducible (fixed weights, no silent API updates), no
   per-call cost for tens of thousands of ratings, and no emotional-support dialogue sent to a third-party
   API. Known limitation to state: the judge has not been checked against human ratings yet.
4. Mistral-Nemo also wrote the profiles, so profiles and dialogues come from different families, which keeps
   the seeker personas less homogeneous than if Qwen had written both.

### 3.5 Likely questions and short answers

- *Why not a bigger model?* The 12 GB GPU caps us at about 7-8B for QLoRA; the scaling trend from 0.5B to 7B
  is reported so the effect of size is visible.
- *Isn't one family for data and supporter circular?* Partly, and it is measured (PVI at 7B); the Qwen3-4B and
  planned Llama-3.1-8B fine-tunes are the check.
- *Why trust an LLM judge?* Separate family, verbatim published scales, redaction of hidden information,
  per-item parsing; human agreement is the open item.
- *Why QLoRA instead of full fine-tuning?* Full fine-tuning a 7B needs far more than 12 GB; QLoRA is the
  standard way to adapt it on one consumer GPU, and the LoRA-geometry analysis shows rank 32 is more than
  enough (rank 4-16 recovers 95 % of the gain).
