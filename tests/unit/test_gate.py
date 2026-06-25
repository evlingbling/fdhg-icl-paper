from fdhg.gate import classification_gate, regression_gate


def test_classification_gate_selects_consistent_gain() -> None:
    assert classification_gate(
        [0.70, 0.71, 0.69, 0.72],
        [0.72, 0.73, 0.70, 0.74],
    )


def test_classification_gate_rejects_one_seed_failure() -> None:
    assert not classification_gate(
        [0.70, 0.71],
        [0.72, 0.70],
    )


def test_regression_gate_selects_consistent_reduction() -> None:
    assert regression_gate(
        [1.0, 1.1, 0.9, 1.2],
        [0.9, 1.0, 0.8, 1.1],
    )


def test_regression_gate_rejects_tie() -> None:
    assert not regression_gate(
        [1.0, 1.1],
        [1.0, 1.0],
    )
