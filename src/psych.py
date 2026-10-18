"""LLM psychology probes: what the ways of making the supporter less intrusive do to its other behaviour.

Systems (SYSTEMS below): the untuned one-call listener (cell A), the same with activation steering, the fine-tuned
one, the DPO one, and the four-agent gated pipeline with the untuned (cell D) and the fine-tuned (step 12) Generator.

  sycophancy  48 hand-written seeker turns (data/psych/sycophancy_probes.jsonl), each a thinking trap or a risky plan
              in 8 categories. Every system replies once (no memory, listening phase); the judge (Mistral-Nemo, as for
              every other metric) scores the reply on prompts/metric_syco.md - sycophancy (items 1-3: agrees with the
              distortion), balance (4-5: validates the feeling, not the belief), harshness (6-7) - and on IP. The
              question: does a less intrusive supporter become a more sycophantic one?

    python src/psych.py respond --system base --backend transformers
    python src/psych.py score   --system base --backend ollama
    python src/psych.py report
"""
from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

from common import DATA, ROOT, load_config, read_jsonl, render_transcript, write_jsonl

OUT = ROOT / "runs/psych"
SFT = "runs/full_1000/adapter_with_thoughts"
SYSTEMS = {
    "base": {"arch": "monolithic"},
    "steered": {"arch": "monolithic", "steer": ("runs/steering/steering_vectors.pt", 12, 0.2)},
    "sft": {"arch": "monolithic", "adapter": SFT},
    "dpo": {"arch": "monolithic", "adapter": "runs/preference/adapter_dpo"},
    "dec_gated": {"arch": "decomposed", "calibration": "runs/baselines_1000/conformal/calibration.json"},
    "sft_dec_gated": {"arch": "decomposed", "adapter": SFT, "calibration": "runs/sft_agents/conformal/calibration.json"},
}
SYCO_SUBSCALES = {"sycophancy": (1, 2, 3), "balance": (4, 5), "harshness": (6, 7)}


def load_system(name: str, backend: str):
    """The resident LLM and a reply function turns -> text for one system, built the way evaluate.py builds it."""
    from agents import AgentPipeline
    from baselines.listener_mono import MonolithicListener
    from baselines.memory_none import NoMemory
    from conformal import Calibration
    from llm import LLM

    spec = SYSTEMS[name]
    if spec.get("adapter"):
        load_config("models")["roles"]["supporter"]["adapter"] = spec["adapter"]
    llm = LLM("supporter", backend=backend)
    cal = Calibration.load(Path(spec["calibration"])) if spec.get("calibration") else None
    steer = ({"steer_vector": spec["steer"][0], "steer_layer": spec["steer"][1], "steer_coef": spec["steer"][2]}
             if spec.get("steer") else None)
    views = {r: llm.view(r if r != "generator" else "supporter") for r in ("analyzer", "strategist", "critic", "generator")}

    def reply(turns: list[dict], key: str) -> dict:
        memory = NoMemory(key)
        if spec["arch"] == "monolithic":
            pipe = MonolithicListener(llm, memory=memory, steer=steer)
        else:
            pipe = AgentPipeline(views["generator"], views=views, memory=memory, calibration=cal, gate=True)
        r = pipe.step(turns, session_id=key, phase="listening")
        return {"text": r.text, "path": r.path}

    return llm, reply


def respond(system: str, backend: str, limit: int | None = None) -> None:
    probes = read_jsonl(DATA / "psych" / "sycophancy_probes.jsonl")[:limit]
    llm, reply = load_system(system, backend)
    try:
        rows = [{**p, "system": system, **reply(p["turns"], p["probe_id"])} for p in probes]
    finally:
        llm.release()
    write_jsonl(OUT / "sycophancy" / f"replies_{system}.jsonl", rows)
    print(f"{system}: {len(rows)} replies")


def score(system: str, backend: str | None) -> None:
    import judge as J
    from llm import pmap

    J.SCALE_PROMPTS["syco"] = "metric_syco.md"
    rows = read_jsonl(OUT / "sycophancy" / f"replies_{system}.jsonl")
    jd = J.Judge(backend=backend)

    def one(r: dict) -> dict:
        turns = r["turns"] + [{"turn_index": len(r["turns"]), "role": "supporter", "text": r["text"]}]
        out = dict(r)
        for scale in ("syco", "ip"):
            res = jd.score(scale, r["probe_id"], render_transcript(turns), {}, target_text=r["text"])
            out[scale] = res.items if not res.parse_failed else None
            if scale == "ip":
                out["ip_norm"] = res.normalized
        return out

    try:
        scored = list(pmap(one, rows, jd.llm))
    finally:
        jd.llm.release()
    write_jsonl(OUT / "sycophancy" / f"scores_{system}.jsonl", scored)


def syco_table() -> list[str]:
    lines = ["== sycophancy probes (48 seeker turns, 8 categories; 0-1, higher = more of it; 95% CI over probes)",
             "system  n  sycophancy  balance  harshness  IP  fallback"]
    cats = defaultdict(dict)
    for name in SYSTEMS:
        path = OUT / "sycophancy" / f"scores_{name}.jsonl"
        if not path.exists():
            continue
        rows = [r for r in read_jsonl(path) if r.get("syco")]
        sub = {k: [st.mean((r["syco"][str(i)] - 1) / 6 for i in items) for r in rows]
               for k, items in SYCO_SUBSCALES.items()}
        ips = [r["ip_norm"] for r in rows if r.get("ip_norm") is not None]
        lines.append(f"{name}  {len(rows)}  " + "  ".join(f"{fmt_ci(v)}" for v in sub.values()) +
                     f"  {fmt_ci(ips)}  {sum(r['path'] == 'fallback' for r in rows) / max(1, len(rows)):.3f}")
        for r, s in zip(rows, sub["sycophancy"]):
            cats[r["category"]].setdefault(name, []).append(s)
    if cats:
        names = [n for n in SYSTEMS if any(n in c for c in cats.values())]
        lines += ["", "sycophancy by category", "category  " + "  ".join(names)]
        lines += [f"{c}  " + "  ".join(f"{st.mean(v[n]):.3f}" if n in v else "n/a" for n in names)
                  for c, v in cats.items()]
    return lines


def fmt_ci(vals: list[float], reps: int = 2000) -> str:
    import random

    if not vals:
        return "n/a"
    rng = random.Random(0)
    boots = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(reps))
    return f"{st.mean(vals):.3f} [{boots[int(.025 * reps)]:.3f}, {boots[int(.975 * reps)]:.3f}]"


def report(dest: Path) -> None:
    lines = [__doc__.split("\n\n")[0], ""] + syco_table()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "llm_psychology.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM psychology probes of the supporter systems.")
    ap.add_argument("cmd", choices=["respond", "score", "report"])
    ap.add_argument("--system", choices=list(SYSTEMS))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "temp_results/19_llm_psychology"))
    args = ap.parse_args()
    if args.cmd == "respond":
        respond(args.system, args.backend or "transformers", args.limit)
    elif args.cmd == "score":
        score(args.system, args.backend)
    else:
        report(Path(args.out))


if __name__ == "__main__":
    main()
