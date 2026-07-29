import panel as pn
pn.extension('plotly', 'mathjax')

from .explorer import (
    ParamSpec,
    ParamState,
    ParamController,
    DecompositionSpec,
    DecompositionExplorer,
)
from .spectrangular import plot_spectrambisonics_2d, CIRCULAR_2D
from .spectrospherics import (
    plot_spherical_harmonic,
    plot_spectrambisonics_3d,
    plot_euler_noncommutativity,
    SPHERICAL_3D,
)
from .polar3d import (
    plot_polar_2d_vs_3d,
    plot_dead_knob,
    plot_same_energy,
)
from .wigner import (
    plot_wigner_factorization,
    plot_small_d,
    plot_wigner_homomorphism,
    plot_field_rotation,
    plot_wigner_manifold,
    plot_orbit_stabilizer,
)
from .lie import (
    plot_angle_noncommutativity,
    plot_tangent_space,
    plot_exponential_map,
    plot_bracket,
    plot_commutator_loop,
    plot_parallel_transport,
    plot_hairy_ball,
    plot_wigner_mixing,
    plot_loop_rephasing,
    plot_orbit_invariants,
)