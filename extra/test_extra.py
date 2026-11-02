"""Checks for the extra analyses. Plain asserts, same style as tests/test_all.py; echo backend, no GPU.

Each estimator is checked against a case with a known answer (hand-computed, or simulated from known
parameters), and each model pass is run once against the echo backend to prove the wiring.

  python extra/test_extra.py            (all checks)
  python extra/test_extra.py pvi        (only checks whose name contains 'pvi')
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _shared                                   # noqa: E402  (puts src/ on the path)
import lora_geometry as lg                       # noqa: E402
import memory_calibration as memcal              # noqa: E402
import memory_eval as memeval                    # noqa: E402
import pvi                                       # noqa: E402
import scaling                                   # noqa: E402
import llm as llm_mod                            # noqa: E402
from judge import Judge                          # noqa: E402

CHECKS: list = []
TMP = Path(tempfile.mkdtemp(prefix="coccon_extra_tests_"))


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def echo(role: str):
    return llm_mod.LLM(role, backend="echo", cache_dir=TMP / "cache")


def toy_session(sid: str = "p000001-armX-s1", pid: str = "p000001", need: str = "I need to feel valued.",
                ptype: str = "job crisis", n_pairs: int = 4) -> dict:
    turns = [{"turn_index": 0, "role": "supporter", "text": "Hello, I have time if you want to talk.",
              "meta": {"source": "opener_pool"}}]
    for k in range(n_pairs):
        turns.append({"turn_index": 2 * k + 1, "role": "user", "text": f"Things are fine, I guess ({k})."})
        turns.append({"turn_index": 2 * k + 2, "role": "supporter",
                      "text": "It sounds like you might feel unseen?" if k % 2 else "Tell me more?"})
    turns.append({"turn_index": 2 * n_pairs + 1, "role": "user", "text": "Maybe."})
    return {"session_id": sid, "profile_id": pid, "turns": turns,
            "profile_snapshot": {"terminal_need": need, "feeling": "tired", "problem_type": ptype,
                                 "resistance_level": "medium",
                                 "need_chain": [{"depth": 0, "text": "work is a lot"},
                                                {"depth": 1, "text": "I need rest"},
                                                {"depth": 2, "text": need}]}}


# --------------------------------------------------------------------------- shared


@check("shared.chi2_sf_matches_tables")
def _():
    for x, df in ((3.841459, 1), (5.991465, 2), (18.307038, 10), (124.342, 100)):
        assert abs(_shared.chi2_sf(x, df) - 0.05) < 1e-4, (x, df, _shared.chi2_sf(x, df))
    assert abs(_shared.chi2_sf(0.0158, 1) - 0.9) < 1e-3
    assert abs(_shared.norm_sf(1.959964) - 0.025) < 1e-6


# --------------------------------------------------------------------------- memory calibration


@check("memory_calibration.scoring_rules")
def _():
    p = np.array([0.1] * 50 + [0.9] * 50)
    y = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5, dtype=float)
    d = memcal.brier_decomposition(p, y)
    assert abs(d["reliability"] - d["resolution"] + d["uncertainty"] - d["brier"]) < 1e-12
    assert abs(d["within_bin"]) < 1e-12 and d["ece"] < 1e-12
    assert memcal.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0


@check("memory_calibration.irls_recovers_logistic_coefficients")
def _():
    rng = np.random.default_rng(12)
    X = np.column_stack([np.ones(5000), rng.standard_normal((5000, 2))])
    w_true = np.array([-0.5, 1.2, -0.8])
    y = (rng.random(5000) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    assert np.abs(memcal.logistic_irls(X, y, l2=0.0) - w_true).max() < 0.12


@check("memory_calibration.echo_pass_labels_nodes")
def _():
    nodes = [{"text": "needs rest", "reference": ["I need rest", "I need to feel valued."]},
             {"text": "wants a car", "reference": ["I need rest"]}]
    memcal.label_nodes(echo("judge"), nodes)
    assert [n["real"] for n in nodes] == [0, 0], "echo rates 2/7: no match"


# --------------------------------------------------------------------------- scaling study (CPU parts)


@check("lora_geometry.lowrank_svd_matches_dense")
def _():
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((8, 50)), rng.standard_normal((40, 8))
    u, sig, v = lg.svd_lowrank(A, B, 2.0)
    dense = np.linalg.svd(2.0 * B @ A, compute_uv=False)[:8]
    assert np.allclose(sig, dense), (sig, dense)
    assert np.allclose(u @ np.diag(sig) @ v.T, 2.0 * B @ A)


@check("lora_geometry.spectrum_and_similarity_known_cases")
def _():
    st = lg.spectrum_stats(np.array([1.0, 1.0, 1.0, 1.0, 0, 0]))
    assert abs(st["effective_rank"] - 4) < 1e-9 and abs(st["stable_rank"] - 4) < 1e-9
    assert abs(lg.spectrum_stats(np.array([5.0, 0, 0]))["effective_rank"] - 1) < 1e-9
    q = np.linalg.qr(np.random.default_rng(0).standard_normal((30, 6)))[0]
    assert abs(lg.subspace_similarity(q, q, 4) - 1) < 1e-9
    assert lg.subspace_similarity(q[:, :3], q[:, 3:], 3) < 1e-9          # orthogonal subspaces
    curve = {0: 2.0, 1: 1.6, 2: 1.45, 4: 1.41, 8: 1.4, 32: 1.4}
    assert lg.rank_needed(curve) == 4                                     # 0.59 of 0.60 drop at k=4


@check("scaling.power_law_recovers_exponent")
def _():
    n = [0.36e9, 2.77e9, 6.53e9]
    fit = scaling.power_law(n, [5.0 * x ** -0.07 for x in n])
    assert abs(fit["b"] - 0.07) < 1e-9 and abs(fit["a"] - 5.0) < 1e-6 and fit["residual_df"] == 1


@check("pvi.summary_on_synthetic_turns")
def _():
    rows = [{"profile_id": f"p{i % 5}", "skipped": False, "n_tokens": 10, "thought_chars": 100 + i,
             "ladder_rung": "L1", "logp_with": -10.0, "logp_wo": -10.0 - math.log(2) * (1 if i % 4 else -1),
             "pvi_bits": 1.0 if i % 4 else -1.0} for i in range(40)]
    rows.append({"profile_id": "p0", "skipped": True})
    s = pvi.summarise(rows, 200, "test")
    assert s["n"] == 40 and s["skipped"] == 1
    assert abs(s["v_information"] - 0.5) < 1e-9 and abs(s["share_negative"] - 0.25) < 1e-9
    assert abs(s["reply_nll_with"] - 1.0) < 1e-9
    lo, hi = s["v_information_ci"]
    assert lo <= 0.5 <= hi


@check("memory_eval.gap_buckets_and_denial_reproposal")
def _():
    assert memeval.bucket({"session_index": 1}) == "session 1"
    assert [memeval.bucket({"session_index": 2, "gap_days_from_prev": g}) for g in (3, 7, 7.5, 21, 30)] == \
        ["<= 7 d", "<= 7 d", "7-21 d", "7-21 d", "> 21 d"]
    assert memeval.is_denial("Oh, that's not it at all.") and memeval.is_denial("No, it's more about my dad.")
    assert not memeval.is_denial("It's not really my fault though.")       # cue not at the start
    claim = "Maybe you need to feel that your work is valued by your team."
    assert memeval.inference_sentence("I hear you. " + claim) == claim
    assert memeval.inference_sentence("That sounds like a long week.") is None
    sup = lambda i, text: {"turn_index": i, "role": "supporter", "text": text}
    usr = lambda i, text: {"turn_index": i, "role": "user", "text": text}
    s1 = {"profile_id": "p1", "session_index": 1, "session_id": "p1-a-s1", "turns": [
        sup(0, "Hello."), usr(1, "Work is a lot."), sup(2, claim), usr(3, "No, it's not that."),
        sup(4, "Okay, tell me more?"), usr(5, "Just tired.")]}
    s2 = {"profile_id": "p1", "session_index": 2, "session_id": "p1-a-s2", "turns": [
        sup(0, "Hi again."), usr(1, "Same."), sup(2, "Could it be you need your team to see your work is valued?")]}
    ev = memeval.denials_and_reproposals([s1, s2])
    assert len(ev) == 1 and ev[0]["claim"] == claim
    assert [(r["session_index"], r["turn_index"]) for r in ev[0]["reproposals"]] == [(2, 2)]


def main(argv: list[str]) -> int:
    wanted = [a for a in argv[1:] if not a.startswith("-")]
    selected = [(n, f) for n, f in CHECKS if not wanted or any(w in n for w in wanted)]
    failures = []
    print(f"running {len(selected)} extra checks\n")
    for name, fn in selected:
        llm_mod.release_all()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, traceback.format_exc(limit=4)))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    llm_mod.release_all()
    print("\n" + "=" * 72)
    print(f"{len(selected) - len(failures)} passed, {len(failures)} failed")
    for name, tb in failures:
        print(f"\n--- {name}\n{tb}")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
