import math

import pytest
import torch

from experiments.student.augment_ar_histories import substitute_history


def context():
    generator = torch.Generator().manual_seed(7)
    embedding = torch.randn(9, 8, generator=generator)
    alpha = torch.tensor([1.25])
    x = torch.randn(1, 9, 8, generator=generator)
    positions = torch.arange(3, 7).float()[:, None]
    frequencies = torch.exp(torch.arange(0, 8, 2).float() * -(math.log(10000) / 8))
    pe = torch.empty(4, 8)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
    x[0, 5:] = embedding[torch.tensor([0, 1, 2, 3])] + alpha * pe
    mask = torch.ones(1, 2, 9, 9, dtype=torch.bool)
    mask[:, :, :2, :2] = False
    for row in range(2, 9):
        mask[:, :, row, :row + 1] = False
    logits = torch.randn(5, 9, generator=generator)
    logits[:, -1] = 1000  # EOS must never become a continuing-history input.
    return {"inputs": x, "mask": mask, "prediction_start": 4, "teacher_logits": logits}, embedding, alpha


def test_substitutions_preserve_prompt_positions_mask_and_seed():
    ctx, embedding, alpha = context()
    original = {key: value.clone() if torch.is_tensor(value) else value for key, value in ctx.items()}
    changed, rows = substitute_history(ctx, embedding, alpha, torch.Generator().manual_seed(42))
    repeated, repeated_rows = substitute_history(ctx, embedding, alpha, torch.Generator().manual_seed(42))
    assert rows == repeated_rows and torch.equal(changed, repeated)
    assert rows and torch.equal(changed[:, :5], original["inputs"][:, :5])
    touched = {row["row"] for row in rows}
    for index in range(5, 9):
        assert torch.equal(changed[:, index], ctx["inputs"][:, index]) == (index not in touched)
    assert all(row["new_token"] < 8 and row["new_token"] != row["old_token"] for row in rows)
    for key, value in ctx.items():
        assert torch.equal(value, original[key]) if torch.is_tensor(value) else value == original[key]


def test_rejects_wrong_teacher_embedding_or_noncausal_history():
    ctx, embedding, alpha = context()
    with pytest.raises(ValueError, match="reconstruction"):
        substitute_history(ctx, embedding + 0.01, alpha, torch.Generator())
    ctx["mask"][0, 0, 5, 8] = False
    with pytest.raises(ValueError, match="causal"):
        substitute_history(ctx, embedding, alpha, torch.Generator())
