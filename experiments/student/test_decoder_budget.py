import pytest

from experiments.student.init_budget_decoder import actual_fft_budget, choose_plan


def test_budget_accounts_for_fft_size_change_and_rejects_tap_switch():
    row = {
        "outer_channels": 384,
        "inner_channels": 240,
        "keep": list(range(240)),
        "kernel": 11,
        "dilation": 5,
        "budget": {"actual_n1_input_time": 2360},
        "baseline_spectral_MACs": 1495203840,
    }
    result = actual_fft_budget(row)
    assert result["expected_fft_sizes"] == [128, 256]
    assert result["after"] == 913766400
    assert result["saved"] == 581437440
    with pytest.raises(ValueError, match="TapConv1d"):
        actual_fft_budget(dict(row, inner_channels=48, keep=list(range(48))))
    with pytest.raises(ValueError, match="channel IDs"):
        actual_fft_budget(dict(row, inner_channels=240, keep=list(range(239))))


def test_plan_selects_lower_worst_error_at_same_required_work_reduction():
    choices = []
    for pair, errors in [(0, {60: 0.20, 72: 0.10}), (1, {60: 0.12, 72: 0.11})]:
        for inner, error in errors.items():
            choices.append(
                {
                    "stage": 2,
                    "original_branch": 1,
                    "pair": pair,
                    "outer_channels": 96,
                    "inner_channels": inner,
                    "keep": list(range(inner)),
                    "relative_pair_output_RMSE": error,
                    "kernel": 7,
                    "dilation": 1,
                    "budget": {"actual_n1_input_time": 100},
                    "baseline_spectral_MACs": 7133184,
                }
            )
    selected, baseline, removed = choose_plan({"choices": choices}, 0.3125, 0.23)
    assert [row["inner_channels"] for row in selected] == [72, 60]
    assert baseline == 14266368
    assert removed / baseline == 0.3125
    with pytest.raises(ValueError, match="operation"):
        choose_plan({"choices": choices}, 0.5, 0.23)
