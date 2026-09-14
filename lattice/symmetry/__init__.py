"""Symmetry group data and generators (Oh / Dic / C4 little groups).

Re-exports the public names of the four modules below, in their historical
order so that name precedence is unchanged (``refRotateDict`` and the
``split_*`` helpers exist in more than one module; the last import wins, exactly
as the previous ``import *`` chain did).
"""

from .group_generator import (
    C4_generator1,
    C4_generator2,
    Dic2_generator,
    Dic3_generator,
    Dic4_generator,
    Fermion_generator,
    OhD_generator,
    falvor2irrep,
    falvor3irrep,
    irrep_generators,
    refRotateDict,
)
from .utils import (
    antisymmetric_tensor,
    are_collinear,
    are_collinear_and_normalize,
    check_and_normalize_arrays,
    generate_hardcoded_code,
    multiplicationTable,
    normalize_array,
    select_nonzero_vector,
    split_expression,
    split_first_term,
    split_mul,
)
from .gen_hardcoded_rep import (
    genIrrepOhD,
    genLittleGroupIrrep,
    genMatrixGroupOhD,
    genR_ref,
    gen_connection,
    littleGroup,
    momentunSymplify,
    reductionToLittleGroup,
    wignerRotate,
)
from .hardcoded_rep import (
    C4_irreps1,
    C4_irreps2,
    Dic2_irreps,
    Dic3_irreps,
    Dic4_irreps,
    Fermion_rep,
    OD_irreps,
    OhD_Multiply_table,
    OhD_inv,
    OhD_irreps,
    OhD_mul,
    gauge_link,
    group_element,
    irrep_row_connection_dict,
    little_group_irreps,
    little_group_reduction_map,
    little_group_reduction_map_Dic2,
    little_group_reduction_map_Dic3,
    little_group_reduction_map_Dic4,
)
