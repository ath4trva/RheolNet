from __future__ import annotations

import torch

from training.pinn_trainer import AdamPhase, LBFGSPhase, PINNTrainer


class ScaleState:
    def __init__(self) -> None:
        self.value = 1.0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = float(state["value"])


def test_adam_lbfgs_checkpoint_round_trip(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(1, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
    state = ScaleState()
    fixed_x = torch.linspace(-1, 1, 41).reshape(-1, 1)

    def sampler(phase, step):
        generator = torch.Generator().manual_seed(1000 + step)
        x = 2 * torch.rand((32, 1), generator=generator) - 1
        return {"x": x, "target": 1.5 * x - 0.2}

    def loss_fn(net, batch, phase, update):
        error = (net(batch["x"]) - batch["target"]).square().mean()
        if update:
            state.value = 0.99 * state.value + 0.01 * float(error.detach())
        total = state.value * error
        return total, {"fit": error}, {"fit": state.value}

    def diagnostic(net):
        with torch.no_grad():
            score = (net(fixed_x) - (1.5 * fixed_x - 0.2)).square().mean().sqrt()
        return {"fixed_pde_score": float(score)}

    config = {"test": "quadratic", "seed": 7}
    trainer = PINNTrainer(
        model=model,
        loss_fn=loss_fn,
        batch_sampler=sampler,
        diagnostic_fn=diagnostic,
        output_dir=tmp_path,
        config=config,
        device="cpu",
        stateful={"scale": state},
    )
    trainer.train_adam([
        AdamPhase(
            name="adam",
            steps=40,
            learning_rate=2e-2,
            batch_size=32,
            report_every=10,
            diagnostic_every=10,
            checkpoint_every=20,
        )
    ], resume=False)
    adam_score = trainer.best_score
    fixed_batch = {"x": fixed_x, "target": 1.5 * fixed_x - 0.2}
    trainer.refine_lbfgs(
        LBFGSPhase(max_iterations=20, learning_rate=0.8, history_size=10),
        fixed_batch=fixed_batch,
    )
    assert trainer.best_score <= adam_score
    assert trainer.adam_checkpoint.is_file()
    assert trainer.lbfgs_checkpoint.is_file()
    assert any(row["optimizer_phase"] == "adam" for row in trainer.history)
    assert any(row["optimizer_phase"] == "lbfgs" for row in trainer.history)

    probe = torch.tensor([[-0.75], [0.0], [0.5]])
    expected = model(probe).detach().clone()
    fresh = torch.nn.Sequential(torch.nn.Linear(1, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
    reloaded = PINNTrainer(
        model=fresh,
        loss_fn=loss_fn,
        batch_sampler=sampler,
        diagnostic_fn=diagnostic,
        output_dir=tmp_path,
        config=config,
        device="cpu",
        stateful={"scale": ScaleState()},
    )
    payload = reloaded.restore(trainer.lbfgs_checkpoint)
    assert payload["lbfgs_completed"] is True
    assert torch.equal(expected, fresh(probe).detach())
