"""callqa: a calibrated judge for scoring recorded calls against an editable rubric."""
from .calibrate import Calibration, HumanPass, calibrate, load_human
from .checks import checks_for_rubric, run_checks
from .gate import evaluate, gate, golden_from_human, self_test
from .judge import AnthropicJudge, CallVerdict, ModelJudge
from .kappa import quadratic_weighted_kappa
from .outcome import check_outcomes, load_ledger
from .passk import passk_table, passk_table_with_checks
from .rubric import Rubric, load_rubric, validate_rubric
from .stub import rule_based_model
from .transcript import Call, load_calls

__all__ = [
    "AnthropicJudge", "Call", "CallVerdict", "Calibration", "HumanPass",
    "ModelJudge", "Rubric", "calibrate", "check_outcomes", "checks_for_rubric",
    "evaluate", "gate", "golden_from_human", "load_calls", "load_human",
    "load_ledger", "load_rubric", "passk_table", "passk_table_with_checks",
    "quadratic_weighted_kappa", "rule_based_model", "run_checks", "self_test",
    "validate_rubric",
]
