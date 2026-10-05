"""L07: implement a held-out design without re-tuning rates, N or SLO."""
from common.service import validate_design


def build_matrix(design):
    """Same schema and block order as L06; replace ONLY input_len with heldout_input.
    Default 1536+64<=2048; reduced contexts: 1408+64<=1536 or 896+64<=1024.
    Use unique L07 run_ids, distinct from L06.
    """
    validate_design(design)
    # TODO L07-1: build your held-out plan. Reuse L06 only if controls stay fixed.
    raise NotImplementedError("L07-1: implement held-out matrix")

