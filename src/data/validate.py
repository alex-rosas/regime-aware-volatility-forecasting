import pandas as pd


def validate_mxn(df: pd.DataFrame) -> dict:
    """
    Validate the MXN/USD FIX rate series.
    Expected column: 'MXN_USD'
    """
    failures = []

    num_rows = len(df)
    if num_rows < 6000:
        failures.append(
            f"Too few rows: got {num_rows}, expected >= 6000."
        )

    null_rows = df['MXN_USD'].isnull().sum()
    if null_rows > 0:
        failures.append(
            f"{null_rows} null values found in 'MXN_USD'."
        )

    invalid_values = ((df['MXN_USD'] < 3) | (df['MXN_USD'] > 30)).sum()
    if invalid_values > 0:
        failures.append(
            f"{invalid_values} values outside plausible range [3, 30] in 'MXN_USD'."
        )

    return {"passed": len(failures) == 0, "failures": failures}


def validate_ipc(df: pd.DataFrame) -> dict:
    """
    Validate the IPC (S&P/BMV) index series.
    Expected column: 'IPC'
    """
    failures = []

    num_rows = len(df)
    if num_rows < 6000:
        failures.append(
            f"Too few rows: got {num_rows}, expected >= 6000."
        )

    null_rows = df['IPC'].isnull().sum()
    if null_rows > 0:
        failures.append(
            f"{null_rows} null values found in 'IPC'."
        )

    # A stock index cannot be zero or negative
    non_positive = (df['IPC'] <= 0).sum()
    if non_positive > 0:
        failures.append(
            f"{non_positive} non-positive values found in 'IPC' — index cannot be zero or negative."
        )

    return {"passed": len(failures) == 0, "failures": failures}


def validate_macro(df: pd.DataFrame) -> dict:
    """
    Validate macro indicators: VIX, DFF, T10Y2Y.
    Some nulls are acceptable (FRED has occasional gaps).
    """
    failures = []

    # VIX: must be positive and below 90
    if 'VIXCLS' in df.columns:
        invalid_vix = ((df['VIXCLS'] < 0) | (df['VIXCLS'] > 90)).sum()
        if invalid_vix > 0:
            failures.append(
                f"{invalid_vix} VIX values outside plausible range [0, 90]."
            )
    else:
        failures.append("Column 'VIXCLS' not found in macro DataFrame.")

    # DFF: Fed Funds Rate — must be non-negative
    if 'DFF' in df.columns:
        negative_dff = (df['DFF'] < 0).sum()
        if negative_dff > 0:
            failures.append(
                f"{negative_dff} negative values in 'DFF' — Fed Funds Rate cannot be negative."
            )
    else:
        failures.append("Column 'DFF' not found in macro DataFrame.")

    # T10Y2Y: yield spread — CAN be negative (inverted curve), no bounds check needed
    if 'T10Y2Y' not in df.columns:
        failures.append("Column 'T10Y2Y' not found in macro DataFrame.")

    return {"passed": len(failures) == 0, "failures": failures}


def run_all(
    mxn_df: pd.DataFrame,
    ipc_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    raise_on_failure: bool = False,
) -> dict:
    """
    Run all validation suites and return a combined report.

    Parameters
    ----------
    mxn_df, ipc_df, macro_df : pd.DataFrame
        The three raw DataFrames to validate.
    raise_on_failure : bool
        If True, raises ValueError when any dataset fails.
        Use False in notebooks (inspect freely).
        Use True in pipeline scripts (hard stop on bad data).

    Returns
    -------
    dict with keys 'mxn', 'ipc', 'macro' (each a result dict)
    and 'all_passed' (bool summary).
    """
    results = {
        "mxn":   validate_mxn(mxn_df),
        "ipc":   validate_ipc(ipc_df),
        "macro": validate_macro(macro_df),
    }

    # Separate dataset results from summary keys before iterating
    dataset_results = {k: v for k, v in results.items() if isinstance(v, dict)}
    all_passed = all(r["passed"] for r in dataset_results.values())
    results["all_passed"] = all_passed

    if not all_passed and raise_on_failure:
        failed = [name for name, r in dataset_results.items() if not r["passed"]]
        raise ValueError(
            f"Validation failed for: {failed}. Check results dict for details."
        )

    return results