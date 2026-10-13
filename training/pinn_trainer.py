"""Reusable Adam -> L-BFGS trainer for physics-informed neural networks.

The trainer is deliberately independent of a particular PDE.  A notebook or
application supplies batch, loss and fixed-diagnostic callbacks.  This keeps the
optimizer/checkpoint logic reusable for the later 3D and meta-learning tasks.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol

import torch
from torch import nn

from .checkpointing import (
    atomic_torch_save,
    config_fingerprint,
    cpu_state_dict,
    load_torch_checkpoint,
)


class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


LossFunction = Callable[
    [nn.Module, Mapping[str, Any], Mapping[str, Any], bool],
    tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, float]],
]
BatchSampler = Callable[[Mapping[str, Any], int], Mapping[str, Any]]
DiagnosticFunction = Callable[[nn.Module], Mapping[str, float]]
MetricLogger = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class AdamPhase:
    name: str
    steps: int
    learning_rate: float
    batch_size: int
    inertia: float = 1.0
    pde_factor: float = 1.0
    weight_decay: float = 1.0e-8
    warmup_fraction: float = 0.04
    minimum_lr_fraction: float = 0.05
    grad_clip: float = 2.0
    report_every: int = 100
    diagnostic_every: int = 200
    checkpoint_every: int = 500

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("Adam steps, batch_size and learning_rate must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 < self.minimum_lr_fraction <= 1:
            raise ValueError("minimum_lr_fraction must be in (0, 1]")


@dataclass(frozen=True)
class LBFGSPhase:
    name: str = "lbfgs_refinement"
    max_iterations: int = 140
    learning_rate: float = 0.6
    history_size: int = 25
    tolerance_grad: float = 1.0e-8
    tolerance_change: float = 1.0e-10
    line_search_fn: str | None = "strong_wolfe"
    grad_clip: float | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.learning_rate <= 0 or self.history_size < 1:
            raise ValueError("L-BFGS iteration, learning-rate and history values must be positive")


class PINNTrainer:
    """Two-phase PINN trainer with fixed-set selection and full checkpoint state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: LossFunction,
        batch_sampler: BatchSampler,
        diagnostic_fn: DiagnosticFunction,
        output_dir: str | Path,
        config: Mapping[str, Any],
        device: torch.device | str,
        stateful: Mapping[str, Stateful] | None = None,
        metric_logger: MetricLogger | None = None,
        selection_key: str = "fixed_pde_score",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.batch_sampler = batch_sampler
        self.diagnostic_fn = diagnostic_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.config = dict(config)
        self.fingerprint = config_fingerprint(self.config)
        self.stateful = dict(stateful or {})
        self.metric_logger = metric_logger
        self.selection_key = selection_key
        self.history: list[dict[str, Any]] = []
        self.best_score = math.inf
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_diagnostic: dict[str, float] | None = None
        self.adam_checkpoint = self.output_dir / "best_adam_checkpoint.pt"
        self.latest_checkpoint = self.output_dir / "latest_training_checkpoint.pt"
        self.lbfgs_checkpoint = self.output_dir / "final_lbfgs_checkpoint.pt"
        self.history_path = self.output_dir / "training_history.json"

    def _stateful_state(self) -> dict[str, Mapping[str, Any]]:
        return {name: item.state_dict() for name, item in self.stateful.items()}

    def _restore_stateful(self, payload: Mapping[str, Any]) -> None:
        for name, state in payload.get("stateful", {}).items():
            if name in self.stateful:
                self.stateful[name].load_state_dict(state)

    def _emit(self, row: Mapping[str, Any]) -> None:
        clean = dict(row)
        self.history.append(clean)
        if self.metric_logger is not None:
            self.metric_logger(clean)

    def _save_history(self) -> None:
        self.history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "fingerprint": self.fingerprint,
            "config": self.config,
            "model": cpu_state_dict(self.model),
            "stateful": self._stateful_state(),
            "history": self.history,
            "best_score": self.best_score,
            "best_state": self.best_state,
            "best_diagnostic": self.best_diagnostic,
        }
        payload.update(extra)
        return payload

    def _consider_best(self, diagnostic: Mapping[str, float], phase: str) -> bool:
        if self.selection_key not in diagnostic:
            raise KeyError(f"Diagnostic is missing selection key {self.selection_key!r}")
        score = float(diagnostic[self.selection_key])
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite fixed diagnostic score during {phase}")
        if score >= self.best_score:
            return False
        self.best_score = score
        self.best_state = cpu_state_dict(self.model)
        self.best_diagnostic = {name: float(value) for name, value in diagnostic.items()}
        return True

    @staticmethod
    def _lr_multiplier(step: int, phase: AdamPhase) -> float:
        warmup = max(1, int(phase.warmup_fraction * phase.steps))
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(phase.steps - warmup - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        floor = phase.minimum_lr_fraction
        return floor + (1.0 - floor) * cosine

    def train_adam(
        self,
        phases: list[AdamPhase],
        *,
        resume: bool = True,
    ) -> tuple[list[dict[str, Any]], float]:
        """Run resampled Adam phases with cosine scheduling and gradient clipping."""
        if not phases:
            raise ValueError("At least one Adam phase is required")
        start_phase = 0
        start_step = 0
        optimizer_state = None
        scheduler_state = None
        final_optimizer_state = None
        final_scheduler_state = None
        if resume and self.latest_checkpoint.is_file():
            checkpoint = load_torch_checkpoint(self.latest_checkpoint, map_location=self.device)
            if checkpoint.get("fingerprint") != self.fingerprint:
                raise RuntimeError("Checkpoint/config mismatch; use a new output directory")
            self.model.load_state_dict(checkpoint["model"])
            self._restore_stateful(checkpoint)
            self.history = list(checkpoint.get("history", []))
            self.best_score = float(checkpoint.get("best_score", math.inf))
            self.best_state = checkpoint.get("best_state")
            self.best_diagnostic = checkpoint.get("best_diagnostic")
            start_phase = int(checkpoint.get("phase_index", 0))
            start_step = int(checkpoint.get("next_step", 0))
            optimizer_state = checkpoint.get("optimizer")
            scheduler_state = checkpoint.get("scheduler")
            final_optimizer_state = optimizer_state
            final_scheduler_state = scheduler_state
            if checkpoint.get("adam_completed", False):
                return self.history, self.best_score

        if self.best_state is None:
            initial = self.diagnostic_fn(self.model)
            self._consider_best(initial, "initial")

        for phase_index in range(start_phase, len(phases)):
            phase = phases[phase_index]
            phase_map = asdict(phase)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=phase.learning_rate,
                weight_decay=phase.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step, selected=phase: self._lr_multiplier(step, selected),
            )
            first_step = start_step if phase_index == start_phase else 0
            if first_step and optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                if scheduler_state is not None:
                    scheduler.load_state_dict(scheduler_state)

            window_start = time.perf_counter()
            window_step = first_step
            for step in range(first_step, phase.steps):
                batch = self.batch_sampler(phase_map, step)
                optimizer.zero_grad(set_to_none=True)
                total, terms, weights = self.loss_fn(
                    self.model, batch, phase_map, True
                )
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite Adam loss at {phase.name}:{step}")
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), phase.grad_clip
                )
                optimizer.step()
                scheduler.step()

                report = step % phase.report_every == 0 or step == phase.steps - 1
                if report:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    elapsed = time.perf_counter() - window_start
                    denominator = max(step - window_step + 1, 1)
                    row = {
                        "optimizer_phase": "adam",
                        "phase": phase.name,
                        "phase_index": phase_index,
                        "step": step,
                        "total": float(total.detach()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "grad_norm": float(grad_norm.detach()),
                        "seconds_per_step": elapsed / denominator,
                        **{name: float(value.detach()) for name, value in terms.items()},
                        **{f"weight_{name}": float(value) for name, value in weights.items()},
                    }
                    self._emit(row)
                    window_start = time.perf_counter()
                    window_step = step + 1

                diagnose = (
                    step % phase.diagnostic_every == 0
                    or step == phase.steps - 1
                )
                if diagnose:
                    diagnostic = self.diagnostic_fn(self.model)
                    improved = self._consider_best(diagnostic, phase.name)
                    self._emit({
                        "optimizer_phase": "diagnostic",
                        "phase": phase.name,
                        "phase_index": phase_index,
                        "step": step,
                        "improved": improved,
                        **{name: float(value) for name, value in diagnostic.items()},
                    })
                    if improved:
                        atomic_torch_save(
                            self._checkpoint_payload(
                                optimizer_phase="adam",
                                phase_index=phase_index,
                                step=step,
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),
                                adam_completed=False,
                            ),
                            self.adam_checkpoint,
                        )

                if step and step % phase.checkpoint_every == 0:
                    atomic_torch_save(
                        self._checkpoint_payload(
                            optimizer_phase="adam",
                            phase_index=phase_index,
                            next_step=step + 1,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            adam_completed=False,
                        ),
                        self.latest_checkpoint,
                    )

            start_step = 0
            final_optimizer_state = optimizer.state_dict()
            final_scheduler_state = scheduler.state_dict()
            optimizer_state = scheduler_state = None
            atomic_torch_save(
                self._checkpoint_payload(
                    optimizer_phase="adam",
                    phase_index=phase_index + 1,
                    next_step=0,
                    optimizer=final_optimizer_state,
                    scheduler=final_scheduler_state,
                    adam_completed=False,
                ),
                self.latest_checkpoint,
            )

        if self.best_state is None:
            raise RuntimeError("Adam finished without a valid fixed-set checkpoint")
        self.model.load_state_dict(self.best_state)
        final_diagnostic = self.diagnostic_fn(self.model)
        atomic_torch_save(
            self._checkpoint_payload(
                optimizer_phase="adam",
                phase_index=len(phases),
                next_step=0,
                optimizer=final_optimizer_state,
                scheduler=final_scheduler_state,
                final_diagnostic=final_diagnostic,
                adam_completed=True,
            ),
            self.adam_checkpoint,
        )
        atomic_torch_save(
            self._checkpoint_payload(
                optimizer_phase="adam",
                phase_index=len(phases),
                next_step=0,
                optimizer=final_optimizer_state,
                scheduler=final_scheduler_state,
                final_diagnostic=final_diagnostic,
                adam_completed=True,
            ),
            self.latest_checkpoint,
        )
        self._save_history()
        return self.history, self.best_score

    def refine_lbfgs(
        self,
        phase: LBFGSPhase,
        *,
        fixed_batch: Mapping[str, Any],
        restore_best_adam: bool = True,
    ) -> tuple[list[dict[str, Any]], float]:
        """Refine with deterministic float32 L-BFGS and retain the better fixed score."""
        if restore_best_adam:
            adam = load_torch_checkpoint(self.adam_checkpoint, map_location=self.device)
            if adam.get("fingerprint") != self.fingerprint:
                raise RuntimeError("Adam checkpoint/config mismatch")
            self.model.load_state_dict(adam["model"])
            self._restore_stateful(adam)
            self.history = list(adam.get("history", self.history))
            self.best_score = float(adam["best_score"])
            self.best_state = adam["best_state"]
            self.best_diagnostic = adam.get("best_diagnostic")

        if any(parameter.dtype != torch.float32 for parameter in self.model.parameters()):
            raise TypeError("L-BFGS refinement requires float32 model parameters")

        before_score = self.best_score
        before_state = self.best_state
        before_diagnostic = self.best_diagnostic
        phase_map = asdict(phase)

        # Update adaptive weights once, then freeze them throughout all repeated
        # closure evaluations so L-BFGS sees one deterministic objective.
        with torch.enable_grad():
            self.loss_fn(self.model, fixed_batch, phase_map, True)

        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=phase.learning_rate,
            max_iter=phase.max_iterations,
            history_size=phase.history_size,
            tolerance_grad=phase.tolerance_grad,
            tolerance_change=phase.tolerance_change,
            line_search_fn=phase.line_search_fn,
        )
        closure_calls = 0
        start = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            # Explicitly avoid autocast: PINN derivatives and the line search are
            # substantially more stable in float32.
            with torch.autocast(device_type=self.device.type, enabled=False):
                total, terms, weights = self.loss_fn(
                    self.model, fixed_batch, phase_map, False
                )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite L-BFGS loss at closure call {closure_calls}"
                )
            total.backward()
            if phase.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), phase.grad_clip
                )
            else:
                squared = torch.zeros((), device=self.device)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        squared = squared + parameter.grad.detach().square().sum()
                grad_norm = squared.sqrt()
            self._emit({
                "optimizer_phase": "lbfgs",
                "phase": phase.name,
                "closure_call": closure_calls,
                "total": float(total.detach()),
                "grad_norm": float(grad_norm.detach()),
                **{name: float(value.detach()) for name, value in terms.items()},
                **{f"weight_{name}": float(value) for name, value in weights.items()},
            })
            closure_calls += 1
            return total

        optimizer.step(closure)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        candidate_diagnostic = dict(self.diagnostic_fn(self.model))
        candidate_score = float(candidate_diagnostic[self.selection_key])
        accepted = math.isfinite(candidate_score) and candidate_score < before_score
        if accepted:
            self.best_score = candidate_score
            self.best_state = cpu_state_dict(self.model)
            self.best_diagnostic = {
                name: float(value) for name, value in candidate_diagnostic.items()
            }
        else:
            if before_state is None:
                raise RuntimeError("No Adam state available for non-degrading fallback")
            self.model.load_state_dict(before_state)
            self.best_score = before_score
            self.best_state = before_state
            self.best_diagnostic = before_diagnostic

        final_diagnostic = dict(self.diagnostic_fn(self.model))
        self._emit({
            "optimizer_phase": "lbfgs_summary",
            "phase": phase.name,
            "closure_calls": closure_calls,
            "elapsed_seconds": elapsed,
            "candidate_score": candidate_score,
            "accepted": accepted,
            **{name: float(value) for name, value in final_diagnostic.items()},
        })
        atomic_torch_save(
            self._checkpoint_payload(
                optimizer_phase="lbfgs",
                optimizer=optimizer.state_dict(),
                lbfgs_completed=True,
                lbfgs_accepted=accepted,
                lbfgs_candidate_score=candidate_score,
                lbfgs_closure_calls=closure_calls,
                elapsed_seconds=elapsed,
                final_diagnostic=final_diagnostic,
            ),
            self.lbfgs_checkpoint,
        )
        self._save_history()
        return self.history, self.best_score

    def restore(self, path: str | Path) -> dict[str, Any]:
        """Restore model, adaptive state and history from a trainer checkpoint."""
        payload = load_torch_checkpoint(path, map_location=self.device)
        if payload.get("fingerprint") != self.fingerprint:
            raise RuntimeError("Checkpoint/config mismatch")
        self.model.load_state_dict(payload["model"])
        self._restore_stateful(payload)
        self.history = list(payload.get("history", []))
        self.best_score = float(payload.get("best_score", math.inf))
        self.best_state = payload.get("best_state")
        self.best_diagnostic = payload.get("best_diagnostic")
        return payload
