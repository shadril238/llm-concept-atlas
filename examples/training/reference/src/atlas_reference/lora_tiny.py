"""Train a visible low-rank task with base/fine-tuned/adapter comparisons."""

import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[5]


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    W = torch.eye(8)
    train = torch.randn(48, 8)
    test = torch.randn(16, 8)
    target = W + torch.arange(1.0, 9.0).reshape(8, 1) @ torch.ones(1, 8) * 0.025
    y = train @ target.T
    full = torch.nn.Parameter(W.clone())
    A = torch.nn.Parameter(torch.randn(1, 8) * 0.1)
    B = torch.nn.Parameter(torch.zeros(8, 1))
    full_optimizer = torch.optim.Adam([full], lr=0.04)
    adapter_optimizer = torch.optim.Adam([A, B], lr=0.04)
    initial_A_gradient = initial_B_gradient = None
    for step in range(160):
        full_optimizer.zero_grad()
        adapter_optimizer.zero_grad()
        full_loss = ((train @ full.T - y) ** 2).mean()
        adapter_loss = ((train @ (W + B @ A).T - y) ** 2).mean()
        full_loss.backward()
        adapter_loss.backward()
        if step == 0:
            initial_A_gradient = A.grad.abs().sum().item()
            initial_B_gradient = B.grad.abs().sum().item()
            assert initial_A_gradient == 0 and initial_B_gradient > 0
        full_optimizer.step()
        adapter_optimizer.step()
    expected = test @ target.T
    mse = lambda weights: ((test @ weights.T - expected) ** 2).mean().item()
    assert torch.equal(W, torch.eye(8))
    assert mse(W + B @ A) < mse(W)
    factorized = test @ W.T + (test @ A.T) @ B.T
    merged = test @ (W + B @ A).T
    assert torch.allclose(factorized, merged, atol=1e-6)
    path = ROOT / "examples/artifacts/lora-tiny"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"A": A.detach(), "B": B.detach()}, path / "adapter.pt")
    reloaded = torch.load(path / "adapter.pt", weights_only=True)
    assert torch.allclose(test @ (W + reloaded["B"] @ reloaded["A"]).T, merged)
    print(
        json.dumps(
            {
                "base_mse": round(mse(W), 6),
                "full_mse": round(mse(full), 6),
                "adapter_mse": round(mse(W + B @ A), 6),
                "base_frozen": True,
                "first_A_gradient": initial_A_gradient,
                "first_B_gradient": round(initial_B_gradient, 6),
                "full_parameters": 64,
                "adapter_parameters": 16,
                "merge_reload": True,
            }
        )
    )


if __name__ == "__main__":
    main()
