"""
The `overall_metrics` sheet of the uncertainties files: how well each uncertainty method separates the correct from the
wrong predictions of a dataset (AUROC and AURC), independently of any decision threshold.
"""
import pandas as pd

from pipeline.common import role_of_method
from metrics.aurc import AURC
from metrics.roc_auc import ROCAUC


def _safe_metric(metric, default: float, **kwargs) -> float:
    """metric(**kwargs), or `default` if it is not computable (e.g. only one class among the targets)."""
    try:
        return metric(**kwargs)
    except Exception:
        return default


def _overall_metrics_row(output_df: pd.DataFrame, uncertainty_col: str, aurc: AURC, roc_auc: ROCAUC) -> dict:
    """One row of the overall_metrics sheet for a single uncertainty column, evaluated against the exec/
    predicted_sql of the run that column is tied to (see config.UNCERTAINTY_COLUMN_ROLE)."""
    role = role_of_method(uncertainty_col)
    exec_col = f"{role}_exec"
    predicted_sql_col = f"{role}_predicted_sql"

    target = [1 - (0 if pd.isna(v) else v) for v in output_df[exec_col]]

    estimator_with_replaced_nones, target_with_replaced_nones = [], []
    estimator_with_removed_nones, target_with_removed_nones = [], []
    for target_value, uncertainty, predicted_sql in zip(
        target, output_df[uncertainty_col].to_list(), output_df[predicted_sql_col].to_list()
    ):
        if pd.isna(predicted_sql):
            continue
        target_with_replaced_nones.append(target_value)
        if not pd.isna(uncertainty):
            estimator_with_removed_nones.append(uncertainty)
            target_with_removed_nones.append(target_value)
            estimator_with_replaced_nones.append(uncertainty)
        else:
            estimator_with_replaced_nones.append(1 - target_value)

    auroc = _safe_metric(roc_auc, 0.0, target=target_with_replaced_nones, estimator=estimator_with_replaced_nones)
    auroc_no_nones = _safe_metric(roc_auc, 0.0, target=target_with_removed_nones, estimator=estimator_with_removed_nones)
    aurc_val = _safe_metric(aurc, 1.0, target=target_with_replaced_nones, estimator=estimator_with_replaced_nones)
    aurc_no_nones = _safe_metric(aurc, 1.0, target=target_with_removed_nones, estimator=estimator_with_removed_nones)

    exec_numeric = pd.to_numeric(output_df[exec_col], errors="coerce")

    return {
        "uncertainty_method": uncertainty_col,
        "n_rows": len(output_df),
        "null_predictions_count": int(output_df[predicted_sql_col].isna().sum()),
        "null_uncertainty_count": int(output_df[uncertainty_col].isna().sum()),
        "exec_accuracy_avg": float(exec_numeric.mean()) if exec_numeric.notna().any() else None,
        "auroc": auroc_no_nones,
        "aurc": aurc_no_nones,
        "auroc_with_nones": auroc,
        "aurc_with_nones": aurc_val,
    }


def build_overall_metrics_sheet(output_df: pd.DataFrame, uncertainty_columns: list[str]) -> pd.DataFrame:
    """One row per uncertainty-method column that has at least one non-null value."""
    aurc, roc_auc = AURC(), ROCAUC()
    rows = [
        _overall_metrics_row(output_df, col, aurc, roc_auc)
        for col in uncertainty_columns
        if output_df[col].notna().any()
    ]
    return pd.DataFrame(rows)
