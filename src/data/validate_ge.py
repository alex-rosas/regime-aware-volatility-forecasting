from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.data_context import AbstractDataContext


def build_mxn_suite(context: AbstractDataContext) -> ExpectationSuite:
    """
    Expectation suite for the MXN/USD raw CSV.

    Expectations:
    - Column MXN_USD exists
    - No nulls in MXN_USD
    - Values between 3 and 30 (historical MXN/USD range)
    - Row count > 6,000
    """
    suite = context.add_expectation_suite("mxn_usd_suite")

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "MXN_USD"}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "MXN_USD"}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "MXN_USD", "min_value": 3, "max_value": 30}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 6000}
        )
    )

    return suite


def build_ipc_suite(context: AbstractDataContext) -> ExpectationSuite:
    """
    Expectation suite for the IPC index raw CSV.

    Expectations:
    - Column IPC exists
    - No nulls in IPC
    - Values strictly positive
    - Row count > 6,000
    """
    suite = context.add_expectation_suite("ipc_suite")

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "IPC"}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "IPC"}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "IPC", "min_value": 0}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 6000}
        )
    )

    return suite


def build_macro_suite(context: AbstractDataContext) -> ExpectationSuite:
    """
    Expectation suite for the macro indicators raw CSV (VIXCLS, DFF, T10Y2Y).

    Expectations:
    - Columns VIXCLS, DFF, T10Y2Y all exist
    - VIXCLS between 0 and 90
    - DFF >= 0 (federal funds rate is never negative in this series)
    - T10Y2Y: no range constraint — yield curve can and does invert
      (negative values observed during 2006-2007 and 2022-2023)
    - Nulls allowed for all three — FRED has gaps on non-business days
    - Row count > 6,000
    """
    suite = context.add_expectation_suite("macro_suite")

    # column existence — all three must be present
    for col in ["VIXCLS", "DFF", "T10Y2Y"]:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": col}
            )
        )

    # VIXCLS: fear index, always positive, historically below 90
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "VIXCLS", "min_value": 0, "max_value": 90}
        )
    )

    # DFF: federal funds rate, non-negative
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "DFF", "min_value": 0}
        )
    )

    # T10Y2Y: no range constraint — can be negative (inverted yield curve)

    # row count
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 6000}
        )
    )

    return suite