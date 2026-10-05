"""L06: build controlled A/B blocks; reuse L03/L04 measurements and analysis."""
from common.service import validate_design


def build_matrix(design):
    """12 rows with L04's schema. Long input; seeds 11,22,33.
    For 11/33: budget2048 low/high then budget512 low/high; reverse budgets for 22.
    No other service parameter changes. Re-measure A in this lab.
    """
    validate_design(design)
    # TODO L06-1: implement block order; freeze and save before any real measurement.
    raise NotImplementedError("L06-1: construct budget experiment")

