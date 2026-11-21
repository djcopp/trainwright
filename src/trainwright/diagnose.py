"""Automatic training-curve diagnosis: turn a run's history into findings + next experiments.

Each finding has a ``code`` (stable identifier), a ``severity`` (critical / warning / info),
human-readable ``message`` with the evidence, and ``suggestions``: concrete config overrides
with the reason for each. ``next_experiments`` turns the top suggestions into ready-to-run
commands with a hypothesis, so the next run is a deliberate experiment rather than a guess.
"""

from __future__ import annotations

import math
from typing import Any

import yaml

from .leaderboard import best_run
from .project import Project
from .tracking import Run

AUG_LEVELS = ["none", "light", "medium", "strong"]


def _finite(xs: list[Any]) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def _better(a: float, b: float, mode: str) -> bool:
    return a > b if mode == "max" else a < b


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)


class _Suggest:
    """Builds overrides relative to the run's current config."""

    def __init__(self, exp: dict[str, Any], modality: str):
        self.exp = exp
        self.modality = modality
        self.params = exp.get("model", {}).get("params", {})

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.exp
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def s(self, key: str, value: Any, why: str) -> dict[str, Any]:
        if isinstance(value, float):
            value = float(f"{value:.3g}")
        return {"override": f"{key}={value}", "why": why}

    def regularise(self) -> list[dict[str, Any]]:
        out = []
        wd = self.get("optim.weight_decay", 1e-4) or 0.0
        out.append(
            self.s(
                "optim.weight_decay",
                max(1e-4, wd * 10),
                "stronger weight decay shrinks weights and reduces variance",
            )
        )
        if "dropout" in self.params or self.modality in ("tabular", "text", "image"):
            d = float(self.params.get("dropout", 0.1))
            if d < 0.5:
                out.append(
                    self.s(
                        "model.params.dropout",
                        min(0.5, round(d + 0.15, 2)),
                        "more dropout regularises the head/body",
                    )
                )
        if self.modality == "image":
            cur = self.get("preprocess.augment", "light")
            if cur in AUG_LEVELS and cur != "strong":
                out.append(
                    self.s(
                        "preprocess.augment",
                        AUG_LEVELS[AUG_LEVELS.index(cur) + 1],
                        "stronger augmentation is the most effective regulariser for images",
                    )
                )
            if not self.get("train.label_smoothing"):
                out.append(
                    self.s(
                        "train.label_smoothing",
                        0.1,
                        "label smoothing discourages over-confident fits",
                    )
                )
        if self.modality == "tabular" and int(self.params.get("hidden", 256)) > 64:
            out.append(
                self.s(
                    "model.params.hidden",
                    int(self.params.get("hidden", 256)) // 2,
                    "a smaller network has less capacity to memorise",
                )
            )
        if self.modality == "text":
            out.append(
                self.s(
                    "preprocess.min_token_freq",
                    int(self.get("preprocess.min_token_freq", 2)) + 2,
                    "rarer n-grams are mostly memorisation fodder",
                )
            )
        return out

    def capacity(self) -> list[dict[str, Any]]:
        out = []
        if self.modality == "tabular":
            out.append(
                self.s(
                    "model.params.hidden",
                    int(self.params.get("hidden", 256)) * 2,
                    "a wider network can fit more structure",
                )
            )
            if self.params.get("num_embedding", "none") == "none":
                out.append(
                    self.s(
                        "model.params.num_embedding",
                        "periodic",
                        "periodic numeric embeddings help MLPs model sharp/non-monotonic feature effects",
                    )
                )
        elif self.modality == "image":
            bb = self.params.get("backbone", "resnet18")
            nxt = {
                "resnet18": "resnet50",
                "mobilenet_v3_small": "mobilenet_v3_large",
                "mobilenet_v3_large": "efficientnet_b0",
                "efficientnet_b0": "convnext_tiny",
                "resnet34": "resnet50",
                "resnet50": "convnext_tiny",
            }.get(bb)
            if nxt:
                out.append(self.s("model.params.backbone", nxt, "a stronger pretrained backbone"))
            if self.get("optim.backbone_lr_mult", 0.1) < 0.3:
                out.append(
                    self.s(
                        "optim.backbone_lr_mult", 0.3, "let the backbone adapt more to your domain"
                    )
                )
        elif self.modality == "text":
            out.append(
                self.s(
                    "model.params.embed_dim",
                    int(self.params.get("embed_dim", 128)) * 2,
                    "larger embeddings",
                )
            )
        d = float(self.params.get("dropout", 0.0) or 0.0)
        if d > 0.15:
            out.append(
                self.s(
                    "model.params.dropout",
                    round(d / 2, 2),
                    "less regularisation while underfitting",
                )
            )
        return out


def diagnose_history(
    history: list[dict[str, Any]],
    summary: dict[str, Any],
    exp: dict[str, Any],
    metric: str,
    mode: str,
    modality: str,
    task: str,
    n_classes: int | None = None,
    baseline: dict[str, Any] | None = None,
    run_id: str = "<run>",
) -> dict[str, Any]:
    S = _Suggest(exp, modality)
    findings: list[dict[str, Any]] = []

    def add(
        code: str,
        severity: str,
        message: str,
        suggestions: list[dict[str, Any]] | None = None,
        **evidence: Any,
    ) -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "evidence": evidence,
                "suggestions": suggestions or [],
            }
        )

    n = len(history)
    stop = summary.get("stop_reason", "") or ""
    se_raw = (summary.get("val_se") or {}).get(metric)
    se = float(se_raw) if isinstance(se_raw, (int, float)) and se_raw is not None else float("nan")
    lr = S.get("optim.lr", 1e-3)
    if n == 0:
        add(
            "no_epochs",
            "critical",
            f"No epoch completed ({stop}).",
            [S.s("optim.lr", lr / 10, "a smaller LR avoids immediate divergence")],
        )
        return _wrap(findings, run_id)

    tl = [r.get("train_loss") for r in history]
    vl = [r.get("val_loss") for r in history]
    vm = [r.get(f"val_{metric}") for r in history]
    best_epoch = int(summary.get("best_epoch") or 0)
    bounded = task == "classification" and metric not in ("log_loss",)

    # 1. divergence ---------------------------------------------------------
    if stop.startswith("diverged") or len(_finite(tl)) < len(tl):
        add(
            "diverged",
            "critical",
            f"Training diverged ({stop or 'non-finite loss'}).",
            [
                S.s("optim.lr", lr / 10, "the step size is too large for this loss surface"),
                S.s("train.grad_clip", 0.5, "clip exploding gradients"),
                S.s("train.precision", "32", "rule out mixed-precision overflow"),
            ],
            lr=lr,
        )

    ftl, fvl = _finite(tl), _finite(vl)
    # 2. loss not decreasing ------------------------------------------------
    if len(ftl) >= 3 and ftl[0] > 0:
        drop = (ftl[0] - min(ftl)) / abs(ftl[0])
        if drop < 0.05:
            add(
                "loss_flat",
                "critical",
                f"Training loss barely moved ({_fmt(ftl[0])} -> {_fmt(min(ftl))}, "
                f"{drop:.1%}). The model is not learning: LR too low/high, frozen params, or features carry no signal.",
                [
                    S.s("optim.lr", lr * 3, "a larger step size if optimisation is just slow"),
                    S.s("optim.lr", lr / 10, "or a smaller one if updates overshoot"),
                ],
                relative_drop=round(drop, 4),
            )

    # 3. collapse to one class ----------------------------------------------
    if task == "classification" and n_classes:
        bal = summary.get("val", {}).get("balanced_accuracy")
        if isinstance(bal, float) and bal <= 1.0 / n_classes + 0.02:
            add(
                "collapsed",
                "critical",
                f"Validation balanced accuracy {bal:.3f} is chance level (1/{n_classes}); "
                "the model predicts (almost) a single class.",
                [
                    S.s(
                        "train.class_weights",
                        "balanced",
                        "re-weight the loss so minority classes matter",
                    ),
                    S.s("optim.lr", lr / 3, "collapse often follows an unstable start"),
                ],
                balanced_accuracy=bal,
            )

    # 4. overfitting ----------------------------------------------------------
    if len(fvl) >= 5:
        i_min = min(range(len(fvl)), key=lambda i: fvl[i])
        rise = (fvl[-1] - fvl[i_min]) / max(1e-12, abs(fvl[i_min]))
        loss_overfit = (
            i_min + 1 <= 0.7 * len(fvl) and rise > 0.10 and ftl[-1] < ftl[min(i_min, len(ftl) - 1)]
        )
        fvm_all = _finite(vm)
        metric_degraded = True
        if len(fvm_all) == len(fvl) and fvm_all:
            best_m = max(fvm_all) if mode == "max" else min(fvm_all)
            last_m = fvm_all[-1]
            drop = (best_m - last_m) if mode == "max" else (last_m - best_m)
            band = 2 * se if math.isfinite(se) else (0.005 if bounded else 0.01 * abs(best_m))
            metric_degraded = drop > band
        if loss_overfit and not metric_degraded:
            add(
                "overconfident",
                "info",
                f"Val loss rose {rise:.0%} after epoch {i_min + 1} but val {metric} did not get worse: the model is "
                "becoming over-confident (larger logits), not losing accuracy. This mainly hurts calibration; "
                "temperature scaling (applied automatically at evaluation/export) or label smoothing addresses it.",
                [S.s("train.label_smoothing", 0.1, "softer targets curb over-confidence")]
                if not S.get("train.label_smoothing")
                else [],
                val_loss_min_epoch=i_min + 1,
            )
        elif loss_overfit:
            add(
                "overfitting",
                "warning",
                f"Val loss bottomed at epoch {i_min + 1} ({_fmt(fvl[i_min])}) and rose {rise:.0%} by the end while "
                "train loss kept falling. The checkpoint is from the best epoch, but the model is memorising.",
                S.regularise(),
                val_loss_min_epoch=i_min + 1,
                val_loss_rise=round(rise, 4),
            )

    # 5. generalisation gap at the best epoch --------------------------------
    b = next((r for r in history if r["epoch"] == best_epoch), history[-1])
    tr_b, va_b = b.get(f"train_{metric}"), b.get(f"val_{metric}")
    if (
        isinstance(tr_b, (int, float))
        and isinstance(va_b, (int, float))
        and math.isfinite(tr_b)
        and math.isfinite(va_b)
    ):
        gap = (tr_b - va_b) if mode == "max" else (va_b - tr_b)
        rel = gap / max(1e-9, abs(va_b if mode == "max" else tr_b))
        big = gap > 0.08 if bounded else rel > 0.3
        if big and not any(f["code"] == "overfitting" for f in findings):
            add(
                "generalisation_gap",
                "warning",
                f"At the best epoch train {metric}={_fmt(tr_b)} vs val {_fmt(va_b)} (gap {_fmt(gap)}), measured "
                "with dropout/augmentation on, so the true gap is larger.",
                S.regularise(),
                gap=round(gap, 4),
            )

    # 6. not converged ---------------------------------------------------------
    fvm = _finite(vm)
    if "max_epochs" in stop and fvm and best_epoch >= n - max(1, n // 10):
        q = max(1, n // 4)
        early, late = fvm[-q - 1] if len(fvm) > q else fvm[0], fvm[-1]
        rel_gain = abs(late - early) / max(1e-9, abs(early))
        sev = "warning" if rel_gain > 0.01 else "info"
        add(
            "not_converged",
            sev,
            f"Best epoch is the last one ({best_epoch}/{n}) and val {metric} was still improving "
            f"({_fmt(early)} -> {_fmt(late)} over the last {q} epochs).",
            [
                S.s(
                    "train.max_epochs",
                    int(S.get("train.max_epochs", n)) * 2,
                    "more epochs at the same schedule",
                )
            ],
            relative_gain_last_quarter=round(rel_gain, 4),
        )
    if "time budget" in stop:
        from ._utils import parse_duration

        tb = parse_duration(S.get("train.time_budget")) or 600.0
        add(
            "time_budget_hit",
            "warning" if best_epoch >= n - 1 else "info",
            f"Training was cut by the time budget after {n} epochs (best epoch {best_epoch}); the LR schedule "
            "never finished annealing.",
            [
                S.s(
                    "train.max_epochs",
                    max(1, n),
                    "match max_epochs to what fits in the budget so the LR schedule completes",
                ),
                S.s(
                    "train.time_budget",
                    f"{max(1, round(tb * 2 / 60))}m",
                    "or give the run more wall-clock time",
                ),
            ],
        )

    # 7. instability -------------------------------------------------------------
    if len(fvm) >= 6:
        diffs = [fvm[i + 1] - fvm[i] for i in range(len(fvm) - 1)]
        flips = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
        amp = sorted(abs(d) for d in diffs)[len(diffs) // 2]
        scale = max(1e-9, max(fvm) - min(fvm))
        noise_floor = 2 * se if math.isfinite(se) else 0.0
        if (
            flips >= 0.6 * (len(diffs) - 1)
            and amp > noise_floor
            and (amp > 0.01 if bounded else amp / scale > 0.25)
        ):
            add(
                "unstable",
                "warning",
                f"Val {metric} oscillates (direction flipped {flips}x in {len(diffs)} epochs, median step {_fmt(amp)}).",
                [
                    S.s("optim.lr", lr / 3, "smaller steps reduce epoch-to-epoch noise"),
                    S.s(
                        "train.batch_size",
                        int(S.get("train.batch_size", 64)) * 2,
                        "larger batches give less noisy gradients",
                    ),
                ],
                flips=flips,
            )

    # 8. early best ---------------------------------------------------------------
    if "early stopping" in stop and best_epoch <= 2 and n >= 4:
        add(
            "early_peak",
            "warning",
            f"Best epoch was {best_epoch}; validation got worse immediately afterwards. LR is likely too high "
            "or the model is far too large for the data.",
            [
                S.s("optim.lr", lr / 3, "slower updates preserve what was learned"),
                *S.regularise()[:1],
            ],
        )

    # 9. imbalance -------------------------------------------------------------
    val = summary.get("val", {})
    acc, mf1 = val.get("accuracy"), val.get("macro_f1")
    if (
        isinstance(acc, float)
        and isinstance(mf1, float)
        and acc - mf1 > 0.12
        and S.get("train.class_weights") != "balanced"
    ):
        add(
            "minority_classes_weak",
            "warning",
            f"Accuracy {acc:.3f} but macro-F1 {mf1:.3f}: minority classes are handled much worse than the majority.",
            [S.s("train.class_weights", "balanced", "weight the loss by inverse class frequency")],
        )

    # 10. calibration --------------------------------------------------------------
    cal = summary.get("calibration") or {}
    ece = cal.get("ece_after", val.get("ece"))
    if isinstance(ece, float) and ece > 0.1:
        add(
            "poorly_calibrated",
            "info",
            f"Expected calibration error {ece:.3f}: predicted confidences are not trustworthy probabilities. "
            "Matters if you threshold on confidence or route low-confidence cases to an LLM/human.",
            [S.s("train.label_smoothing", 0.1, "often improves calibration")]
            if not S.get("train.label_smoothing")
            else [],
        )

    # 10b. noisy validation ----------------------------------------------------------
    if math.isfinite(se) and (se > 0.01 if bounded else False):
        add(
            "noisy_validation",
            "info",
            f"The val {metric} has a bootstrap standard error of ±{se:.3f} ({summary.get('n_val', '?')} examples): "
            f"differences smaller than ~{2 * se:.3f} between runs are noise. Don't chase them; prefer the simpler/"
            "cheaper model among tied runs, or enlarge the validation set (split.val) before fine-tuning further.",
            se=round(se, 5),
        )

    # 11. baseline comparison ----------------------------------------------------------
    score = val.get(metric)
    if baseline is None:
        add(
            "no_baselines",
            "info",
            "No baseline runs on this split yet, so there is no floor to compare against. "
            "Run `trainwright baseline`.",
        )
    elif isinstance(score, (int, float)) and math.isfinite(score):
        bscore = baseline.get(f"val_{metric}")
        if isinstance(bscore, (int, float)):
            delta = (score - bscore) if mode == "max" else (bscore - score)
            bse = baseline.get("val_se") or 0.0
            tie_band = 2 * math.sqrt(
                (se if math.isfinite(se) else 0.0) ** 2 + (bse if math.isfinite(bse) else 0.0) ** 2
            )
            if tie_band > 0 and abs(delta) <= tie_band:
                add(
                    "tied_with_baseline",
                    "warning" if delta < 0 else "info",
                    f"Statistically tied with baseline {baseline['id']} ({baseline.get('model')}): "
                    f"{_fmt(score)} vs {_fmt(bscore)} (difference {_fmt(delta)}, noise band ±{_fmt(tie_band)}). "
                    "If the baseline is cheaper to serve, it may be the better choice; otherwise push the neural "
                    "model further before claiming a win.",
                    S.capacity()[:2],
                    baseline=baseline["id"],
                    delta=round(delta, 5),
                )
            elif delta < 0:
                sugg = S.capacity() + [
                    S.s("train.max_epochs", int(S.get("train.max_epochs", 50)) * 2, "train longer")
                ]
                add(
                    "baseline_wins",
                    "warning",
                    f"Baseline {baseline['id']} ({baseline.get('model')}) scores val {metric}={_fmt(bscore)} vs this run "
                    f"{_fmt(score)}. The neural model is not yet earning its keep.",
                    sugg,
                    baseline=baseline["id"],
                    delta=round(delta, 5),
                )
            else:
                add(
                    "beats_baseline",
                    "info",
                    f"Beats the best baseline {baseline['id']} ({baseline.get('model')}) by {_fmt(delta)} val {metric}.",
                    baseline=baseline["id"],
                    delta=round(delta, 5),
                )
    return _wrap(findings, run_id)


def _wrap(findings: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    sev = {f["severity"] for f in findings}
    verdict = (
        "failed" if "critical" in sev else "needs_attention" if "warning" in sev else "healthy"
    )
    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    nxt = []
    for f in findings:
        if f["severity"] == "info":
            continue
        for s in f["suggestions"][:2]:
            nxt.append(
                {
                    "command": f"trainwright train --from-run {run_id} -o {s['override']} "
                    f'--hypothesis "{f["code"]}: {s["why"]}"',
                    "addresses": f["code"],
                    "why": s["why"],
                }
            )
        if len(nxt) >= 3:
            break
    return {"verdict": verdict, "findings": findings, "next_experiments": nxt[:3]}


def diagnose_run(project: Project, run: Run) -> dict[str, Any]:
    cfg = project.config
    summary = run.summary
    if run.meta.get("kind") == "final":
        return {"verdict": "final", "findings": [], "next_experiments": []}
    if summary.get("kind") == "baseline":
        return {"verdict": "baseline", "findings": [], "next_experiments": []}
    exp = yaml.safe_load(run.config_path.read_text()) if run.config_path.exists() else {}
    from .data.split import load_split

    split = load_split(project) or {}
    baseline = best_run(project, kind="baseline")
    if baseline is not None and baseline.get("model") == "baseline:dummy":
        # a dummy is not a meaningful bar to clear; still use it if it's all we have
        pass
    return diagnose_history(
        run.history,
        summary,
        exp,
        cfg.metric,
        cfg.metric_mode,
        cfg.modality,
        cfg.task,
        n_classes=len(split.get("classes", [])) or None,
        baseline=baseline,
        run_id=run.id,
    )
