"""Interactive visualisations for the Wigner D-matrix notebook.

One panel per property of ``D^l``, in the order of ``2_wigner_d_matrix.ipynb`` :

=================================  ==========================================
``plot_wigner_factorization``      D = diag(e^-i m' a) . d(b) . diag(e^-i m g)
``plot_small_d``                   the small d-matrix : real, orthogonal, one-parameter
``plot_wigner_homomorphism``       D(R1) D(R2) = D(R1 R2) : it is a *representation*
``plot_field_rotation``            its role : rotating a sound field is a matrix product
``plot_wigner_manifold``           a 3-dimensional submanifold of a huge matrix space
``plot_orbit_stabilizer``          orbits, stabilizers, and the invariant count
=================================  ==========================================

The complex matrices follow the notebook's convention,
``D^j(a,b,g) = <jm'| e^{-i a Jz} e^{-i b Jy} e^{-i g Jz} |jm>``.
The *real* matrices are the ones that actually act on ambisonic coefficients : they
are fitted directly against this package's own real SH convention (SN3D, ACN order
inside a degree), so whatever ``utils.spherical_harmonic`` does, they match it.
"""

import functools

import numpy as np
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import expm

from .lie import (J, rot, rotvec, bracket, wigner_D, hat,
                  BLUE, RED, GREEN, AMBER, PURPLE, GREY, _formula)
from .utils import spherical_harmonic
from .spectrospherics import EULER_FIELD_PRESETS, _coeffs_from_dict

pn.extension('mathjax', 'plotly')


# --------------------------------------------------------------------------- #
#  the real Wigner matrices, fitted against this package's own SH convention   #
# --------------------------------------------------------------------------- #

def fibonacci_sphere(n=800):
    """A quasi-uniform set of directions, used as the fitting grid."""
    i = np.arange(n) + .5
    zenith = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** .5) * i
    return np.stack([np.sin(zenith) * np.cos(azimuth),
                     np.sin(zenith) * np.sin(azimuth),
                     np.cos(zenith)], -1)


_FIT_PTS = fibonacci_sphere(800)


def sh_matrix(l, dirs):
    """The (n_dirs, 2l+1) matrix of real harmonics of degree l, in ACN order."""
    az = np.arctan2(dirs[..., 1], dirs[..., 0])
    ze = np.arccos(np.clip(dirs[..., 2], -1, 1))
    return np.stack([spherical_harmonic(az, ze, l, m) for m in range(-l, l + 1)], -1)


def real_wigner(l, R):
    """The real Wigner matrix : the unique D with  Y_m(R^-1 u) = sum_m' D[m',m] Y_m'(u).

    Fitted by least squares on a quasi-uniform grid, so it inherits the package's
    SH convention exactly. Because SN3D gives every harmonic of a degree the same
    norm, the result comes out orthogonal to machine precision."""
    A = sh_matrix(l, _FIT_PTS)
    B = sh_matrix(l, _FIT_PTS @ R)          # row i of PTS @ R is R^-1 u_i
    return np.linalg.lstsq(A, B, rcond=None)[0]


def real_wigner_z(l, alpha):
    """D of a rotation about z, in closed form : each pair (-m, m) is a planar
    rotation by m * alpha, and m = 0 is untouched. This is the 2D spectrangular
    picture, repeated once per order."""
    d = 2 * l + 1
    Z = np.eye(d)
    for m in range(1, l + 1):
        i, k = l - m, l + m                                  # indices of -m and +m
        c, s = np.cos(m * alpha), np.sin(m * alpha)
        Z[i, i], Z[i, k], Z[k, i], Z[k, k] = c, s, -s, c
    return Z


def real_wigner_euler(l, alpha, beta, gamma):
    """D of the ZYZ rotation R(alpha, beta, gamma)."""
    return real_wigner(l, rot(2, alpha) @ rot(1, beta) @ rot(2, gamma))


@functools.lru_cache(maxsize=32)
def real_generators(l, eps=1e-4):
    """dD^l(J_a) in the real basis, by central differences. Antisymmetric, and they
    reproduce the so(3) brackets -- which is all the orbit dimension needs."""
    return tuple((real_wigner(l, expm(eps * J[a])) - real_wigner(l, expm(-eps * J[a])))
                 / (2 * eps) for a in range(3))


def _haar_rotvecs(n, seed=0):
    """n uniformly distributed rotations, as rotation vectors (via unit quaternions)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.sign(q[:, :1])                                   # same hemisphere
    angle = 2 * np.arccos(np.clip(q[:, 0], -1, 1))
    s = np.maximum(np.sqrt(1 - q[:, 0] ** 2), 1e-12)
    return q[:, 1:] / s[:, None] * angle[:, None]


@functools.lru_cache(maxsize=8)
def rotation_cloud(l, n=1200):
    """``n`` Wigner matrices of degree ``l``, for uniformly drawn rotations.

    Built once per degree from the generators -- D(exp(w^)) = exp(sum_a w_a G_a),
    the representation property again -- which is a few hundred times faster than
    fitting each matrix, and plenty accurate for a point cloud."""
    G = np.array(real_generators(l))
    return np.array([expm(np.tensordot(w, G, axes=1)) for w in _haar_rotvecs(n)])


def field_from_coeffs(coeffs_by_degree, dirs):
    """Evaluate sum_l sum_m a_l^m Y_l^m on a set of directions."""
    out = np.zeros(dirs.shape[:-1])
    for l, a in coeffs_by_degree.items():
        out = out + sh_matrix(l, dirs) @ np.asarray(a, float)
    return out


def _split_by_degree(flat):
    """An ACN-ordered flat coefficient list -> {l: vector}."""
    out, i = {}, 0
    l = 0
    while i < len(flat):
        out[l] = np.array(flat[i:i + 2 * l + 1], float)
        i += 2 * l + 1
        l += 1
    return out


# ---- shared plotting helpers ---------------------------------------------- #

#: NB the zenith range deliberately stops just short of the two poles. At
#: cos(zenith) = -1 exactly, scipy's ``assoc_legendre_p(n, 0, -1)`` returns +1
#: instead of (-1)^n (checked on scipy 1.15.2), so ``utils.spherical_harmonic``
#: comes out with the wrong sign on the south-pole row for every odd degree.
#: Anything that samples the sphere with ``linspace(0, pi, ...)`` inherits it.
_POLE_EPS = 1e-4
_GRID_AZ, _GRID_ZE = np.meshgrid(np.linspace(-np.pi, np.pi, 96),
                                 np.linspace(_POLE_EPS, np.pi - _POLE_EPS, 49), indexing='ij')
_GRID_DIRS = np.stack([np.sin(_GRID_ZE) * np.cos(_GRID_AZ),
                       np.sin(_GRID_ZE) * np.sin(_GRID_AZ),
                       np.cos(_GRID_ZE)], -1)


def _field_surface(f, scale, name, dirs=None, color=None, opacity=1.0, grow=1.0):
    """The balloon r = |f|, blue where f < 0 and red where f > 0.

    ``color`` paints it in one flat colour instead, which is what a field drawn *on
    top of* another one needs -- with ``grow`` a few percent above 1 so the two do
    not z-fight when they happen to be the same surface."""
    dirs = _GRID_DIRS if dirs is None else dirs
    r = grow * np.abs(f) / scale
    paint = (dict(surfacecolor=np.sign(f), colorscale=[[0, BLUE], [1, RED]], cmin=-1, cmax=1)
             if color is None else
             dict(surfacecolor=np.zeros_like(r), colorscale=[[0, color], [1, color]],
                  cmin=0, cmax=1))
    return go.Surface(x=r * dirs[..., 0], y=r * dirs[..., 1], z=r * dirs[..., 2],
                      showscale=False, name=name, opacity=opacity, **paint)


def closest_rotation(l, a, b, n_starts=8, iters=30):
    """``min over R of ||D(R)a - b||``, and the ``D`` that achieves it.

    Two stages, and both are needed. A cloud of ~1200 rotations samples the
    3-dimensional group only every ~17 degrees, which on its own leaves a residual as
    large as the effect being measured. Refining from it with Gauss-Newton -- using
    d/dw_k D(R)a = G_k D(R)a -- converges quadratically, but D^l oscillates l times
    faster than that sampling, so a single start falls into a local minimum for
    l >= 3. Hence the refinement is run from the best ``n_starts`` candidates and the
    best result kept.

    Two details the naive version gets wrong, both on *symmetric* multiplets, which
    are exactly the interesting ones. The Jacobian is rank-deficient there -- a
    stabilizer direction moves nothing -- so the least-squares step must truncate its
    null direction (``rcond``) or it comes back with a step of 1e8 ; and near a
    stationary point the residual is second-order, so the step has to be capped and
    backtracked or it never lands. Without the two, the search misses even the
    identity : ``a`` itself is reported 0.028 away from ``a``."""
    cloud = rotation_cloud(l)
    coarse = np.linalg.norm(cloud @ a - b, axis=1)
    G = real_generators(l)
    best_val, best_D = np.inf, cloud[int(np.argmin(coarse))]
    for start in np.argsort(coarse)[:n_starts]:
        D = cloud[start]
        for _ in range(iters):
            res = b - D @ a
            cur = np.linalg.norm(res)
            Jac = np.stack([g @ (D @ a) for g in G], axis=-1)       # (2l+1, 3)
            step, *_ = np.linalg.lstsq(Jac, res, rcond=1e-8)
            norm = np.linalg.norm(step)
            if not np.all(np.isfinite(step)) or norm < 1e-13:
                break
            if norm > .5:                       # a linearization is only local
                step *= .5 / norm
            for _ in range(12):                 # backtrack until it actually improves
                D_try = expm(sum(s * g for s, g in zip(step, G))) @ D
                if np.linalg.norm(b - D_try @ a) < cur:
                    D = D_try
                    break
                step = step * .5
            else:
                break
        val = float(np.linalg.norm(D @ a - b))
        if val < best_val:
            best_val, best_D = val, D
    return best_val, best_D


def _matrix_heatmap(M, m_index, zmin=None, zmax=None, colorscale='RdBu', reverse=True):
    """``m_index`` is the actual order of the rows/columns : the complex matrices of
    ``lie.angular_momentum`` run m = l ... -l, the real ACN ones run m = -l ... l.
    Passing it keeps the axes labelled by m rather than by position."""
    return go.Heatmap(z=M, x=m_index, y=m_index, zmin=zmin, zmax=zmax, colorscale=colorscale,
                      reversescale=reverse, showscale=False,
                      hovertemplate="m'=%{y}, m=%{x}<br>%{z:.4f}<extra></extra>")


# --------------------------------------------------------------------------- #
#  1 -- the factorization  D = phase . d(beta) . phase                        #
# --------------------------------------------------------------------------- #

def plot_wigner_factorization():
    """Where the three Euler angles go.

    alpha and gamma enter only through diagonal phases : they cannot move energy
    between orders. Everything that actually *mixes* the multiplet sits in the small
    d-matrix, which depends on beta alone -- which is why |D| never moves when you
    drag alpha or gamma."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=6, value=3)
    al = pn.widgets.FloatSlider(name="α  (rad)", start=0, end=2 * np.pi, step=.01, value=1.0)
    be = pn.widgets.FloatSlider(name="β  (rad)", start=0, end=np.pi, step=.01, value=0.9)
    ga = pn.widgets.FloatSlider(name="γ  (rad)", start=0, end=2 * np.pi, step=.01, value=0.6)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, al, be, ga):
        m = np.arange(l, -l - 1, -1.0)
        D = wigner_D(l, al, be, ga)
        d = wigner_D(l, 0, be, 0)
        La = np.diag(np.exp(-1j * m * al))
        Lg = np.diag(np.exp(-1j * m * ga))

        fig = make_subplots(rows=1, cols=4, horizontal_spacing=.055,
                            subplot_titles=("|D<sup>ℓ</sup>(α,β,γ)|",
                                            "arg D<sup>ℓ</sup>(α,β,γ)",
                                            "the small d<sup>ℓ</sup>(β)  (real)",
                                            "|D| − |d| :  α, γ change nothing"))
        fig.add_trace(_matrix_heatmap(np.abs(D), m, 0, 1, 'Blues', False), row=1, col=1)
        fig.add_trace(_matrix_heatmap(np.angle(D), m, -np.pi, np.pi, 'HSV', False), row=1, col=2)
        fig.add_trace(_matrix_heatmap(d.real, m, -1, 1), row=1, col=3)
        fig.add_trace(_matrix_heatmap(np.abs(D) - np.abs(d), m, -1, 1), row=1, col=4)
        for c in range(1, 5):
            fig.update_xaxes(title="m", row=1, col=c, dtick=max(1, l // 2))
            fig.update_yaxes(title="m'" if c == 1 else "", row=1, col=c, dtick=max(1, l // 2))
        fig.update_layout(width=1180, height=340, uirevision='constant',
                          margin=dict(t=52, b=42, l=40, r=10))
        pane.object = fig

        read.object = (
            "| property | value |\n|---|---|\n"
            f"| ‖D − diag(e^−im'α) · d(β) · diag(e^−imγ)‖ | {np.abs(D - La @ d @ Lg).max():.2e} |\n"
            f"| ‖ |D| − |d(β)| ‖   (α and γ are invisible here) | "
            f"{np.abs(np.abs(D) - np.abs(d)).max():.2e} |\n"
            f"| max imaginary part of d(β) | {np.abs(d.imag).max():.2e}  (d is **real**) |\n"
            f"| ‖D†D − 𝟙‖   (unitary) | {np.abs(D.conj().T @ D - np.eye(2*l+1)).max():.1e} |\n"
            f"| ‖d(β)ᵀd(β) − 𝟙‖   (orthogonal) | "
            f"{np.abs(d.real.T @ d.real - np.eye(2*l+1)).max():.1e} |\n"
            f"| in the **real** basis : ‖D(R_z(α)) − Givens blocks of angle mα‖ | "
            f"{np.abs(real_wigner(l, rot(2, al)) - real_wigner_z(l, al)).max():.1e} |\n\n"
            "*This is the whole content of the formula D = e^(−im'α) d(β) e^(−imγ). The two "
            "z-rotations are **diagonal** : they multiply each order by a phase and can never "
            "move energy between orders — that is the 2D spectrangular situation, one angle per "
            "|m|, everything commuting. All the difficulty of 3D lives in the middle factor, the "
            "small d-matrix, and it depends on β alone. Drag α and γ : the fourth panel stays "
            "flat at zero. Drag β : the matrix fills in. The last row says the same thing in the "
            "**real** basis you actually store ambisonic coefficients in : a yaw is exactly a "
            "stack of planar rotations, one per pair (−m, +m), each turning at its own rate mα.*")

    pn.bind(update, l.param.value_throttled, al.param.value_throttled,
            be.param.value_throttled, ga.param.value_throttled, watch=True)
    update(l.value, al.value, be.value, ga.value)

    return pn.Column(
        pn.pane.Markdown("## Where the three angles go : D = phase · d(β) · phase"),
        pn.Row(l, al, be, ga), pane,
        _formula(r"$$D^j_{m'm}(\alpha,\beta,\gamma) \;=\; e^{-im'\alpha}\, "
                 r"d^j_{m'm}(\beta)\, e^{-im\gamma}$$"),
        read)


# --------------------------------------------------------------------------- #
#  2 -- the small d-matrix                                                     #
# --------------------------------------------------------------------------- #

def plot_small_d():
    """The small d-matrix, the only part that mixes.

    It is real, orthogonal, and a one-parameter subgroup : d(b1) d(b2) = d(b1+b2).
    At beta = 0 it is the identity (no mixing), at beta = pi it is the antidiagonal
    (m -> -m, the field turned upside down), and in between it spreads one order over
    all the others."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=6, value=3)
    be = pn.widgets.FloatSlider(name="β  (rad)", start=0, end=np.pi, step=.01, value=0.8)
    m_sel = pn.widgets.IntSlider(name="follow the order m", start=-6, end=6, value=0)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, be, m_sel):
        m_sel = int(np.clip(m_sel, -l, l))
        m = np.arange(l, -l - 1, -1.0)
        d = wigner_D(l, 0, be, 0).real
        betas = np.linspace(0, np.pi, 181)
        curves = np.array([wigner_D(l, 0, b, 0).real[:, l - m_sel] for b in betas])

        fig = make_subplots(rows=1, cols=2, column_widths=[.42, .58],
                            subplot_titles=(f"d<sup>ℓ</sup>(β = {be:.2f})",
                                            f"the column m = {m_sel} as β sweeps 0 → π"))
        fig.add_trace(_matrix_heatmap(d, m, -1, 1), row=1, col=1)
        fig.update_xaxes(title="m", row=1, col=1, dtick=1)
        fig.update_yaxes(title="m'", row=1, col=1, dtick=1)
        for k, mm in enumerate(m.astype(int)):
            fig.add_trace(go.Scatter(x=betas, y=curves[:, k], mode='lines',
                                     name=f"m' = {mm}", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[be, be], y=[-1, 1], mode='lines', showlegend=False,
                                 line=dict(color='black', width=1, dash='dot')), row=1, col=2)
        fig.update_xaxes(title="β", row=1, col=2)
        fig.update_yaxes(title=f"d(β)[m', m={m_sel}]", row=1, col=2, range=[-1.05, 1.05])
        fig.update_layout(width=1080, height=400, uirevision='constant',
                          margin=dict(t=52, b=42, l=50, r=10),
                          legend=dict(font=dict(size=10)))
        pane.object = fig

        d0 = wigner_D(l, 0, 0., 0).real
        dpi = wigner_D(l, 0, np.pi, 0).real
        b1, b2 = 0.31, 0.47
        comp = np.abs(wigner_D(l, 0, b1, 0).real @ wigner_D(l, 0, b2, 0).real
                      - wigner_D(l, 0, b1 + b2, 0).real).max()
        anti = np.abs(np.abs(dpi) - np.eye(2 * l + 1)[::-1]).max()
        col = d[:, l - m_sel]
        read.object = (
            "| property | value |\n|---|---|\n"
            f"| ‖d(0) − 𝟙‖   (no tilt, no mixing) | {np.abs(d0 - np.eye(2*l+1)).max():.1e} |\n"
            f"| ‖ \\|d(π)\\| − antidiagonal‖   (upside down : m → −m) | {anti:.1e} |\n"
            f"| ‖d(β₁)d(β₂) − d(β₁+β₂)‖   (a one-parameter subgroup) | {comp:.1e} |\n"
            f"| ‖d(−β) − d(β)ᵀ‖   (the inverse is the transpose) | "
            f"{np.abs(wigner_D(l, 0, -be, 0).real - d.T).max():.1e} |\n"
            f"| ‖column m={m_sel}‖²   (energy is only redistributed) | "
            f"{float(col @ col):.10f} |\n"
            f"| orders reached from m={m_sel} at this β | "
            f"{int(np.sum(np.abs(col) > 1e-9))} / {2*l+1} |\n\n"
            "*Follow one column : it starts as a single spike (β = 0, the order is alone), "
            "spreads over **every** other order as β grows, and lands on the mirrored order at "
            "β = π. The sum of squares along the column never moves — a tilt redistributes a "
            "degree's energy among its orders without ever creating or destroying any. This is "
            "the precise sense in which an individual coefficient a_ℓ^m has no rotation-invariant "
            "meaning, while ‖a_ℓ‖ does. Note also that these are one-parameter subgroups : "
            "tilting by β₁ then β₂ is tilting by β₁+β₂, because both are rotations about the "
            "*same* axis — the commuting case of the previous notebook.*")

    pn.bind(update, l.param.value_throttled, be.param.value_throttled,
            m_sel.param.value_throttled, watch=True)
    update(l.value, be.value, m_sel.value)

    return pn.Column(
        pn.pane.Markdown("## The small d-matrix : the only factor that mixes"),
        pn.Row(l, be, m_sel), pane,
        _formula(r"$$d^j_{m'm}(\beta) \;=\; \left\langle j m' \right| "
                 r"e^{-i\beta J_y} \left| j m \right\rangle, \qquad "
                 r"d(\beta_1)\, d(\beta_2) = d(\beta_1 + \beta_2)$$"),
        read)


# --------------------------------------------------------------------------- #
#  3 -- it is a representation                                                 #
# --------------------------------------------------------------------------- #

def plot_wigner_homomorphism():
    """The defining property : D turns composition of rotations into matrix products.

    D(R1) D(R2) = D(R1 R2), to machine precision, in every degree. Two things follow
    at once : the order of the factors matters exactly as much as the order of the
    rotations, and adding Euler angles is meaningless."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=5, value=3)
    a1 = pn.widgets.FloatSlider(name="R₁ : α₁", start=0, end=2 * np.pi, step=.01, value=0.9)
    b1 = pn.widgets.FloatSlider(name="R₁ : β₁", start=0, end=np.pi, step=.01, value=1.1)
    g1 = pn.widgets.FloatSlider(name="R₁ : γ₁", start=0, end=2 * np.pi, step=.01, value=0.4)
    a2 = pn.widgets.FloatSlider(name="R₂ : α₂", start=0, end=2 * np.pi, step=.01, value=2.1)
    b2 = pn.widgets.FloatSlider(name="R₂ : β₂", start=0, end=np.pi, step=.01, value=0.7)
    g2 = pn.widgets.FloatSlider(name="R₂ : γ₂", start=0, end=2 * np.pi, step=.01, value=1.5)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, a1, b1, g1, a2, b2, g2):
        R1 = rot(2, a1) @ rot(1, b1) @ rot(2, g1)
        R2 = rot(2, a2) @ rot(1, b2) @ rot(2, g2)
        D1, D2 = real_wigner(l, R1), real_wigner(l, R2)
        D12, D21 = real_wigner(l, R1 @ R2), real_wigner(l, R2 @ R1)
        D_sum = real_wigner_euler(l, a1 + a2, b1 + b2, g1 + g2)

        fig = make_subplots(rows=1, cols=4, horizontal_spacing=.055,
                            subplot_titles=("D(R₁) D(R₂)", "D(R₁R₂)",
                                            "D(R₁)D(R₂) − D(R₁R₂)", "D(R₂R₁) − D(R₁R₂)"))
        m = np.arange(-l, l + 1)                              # real matrices are ACN-ordered
        for k, M in enumerate([D1 @ D2, D12, D1 @ D2 - D12, D21 - D12]):
            fig.add_trace(_matrix_heatmap(M, m, -1, 1), row=1, col=k + 1)
            fig.update_xaxes(title="m", row=1, col=k + 1, dtick=max(1, l // 2))
            fig.update_yaxes(title="m'" if k == 0 else "", row=1, col=k + 1, dtick=max(1, l // 2))
        fig.update_layout(width=1180, height=340, uirevision='constant',
                          margin=dict(t=52, b=42, l=40, r=10))
        pane.object = fig

        read.object = (
            "| statement | error |\n|---|---|\n"
            f"| **D(R₁) D(R₂) = D(R₁R₂)**   (the representation property) | "
            f"{np.abs(D1 @ D2 - D12).max():.2e} |\n"
            f"| D(R₂) D(R₁) = D(R₁R₂)   (the wrong order) | "
            f"{np.abs(D2 @ D1 - D12).max():.2e} |\n"
            f"| D(α₁+α₂, β₁+β₂, γ₁+γ₂) = D(R₁R₂)   (Euler angles do **not** add) | "
            f"{np.abs(D_sum - D12).max():.2e} |\n"
            f"| D(R)⁻¹ = D(R)ᵀ = D(R⁻¹) | "
            f"{np.abs(np.linalg.inv(D1) - real_wigner(l, R1.T)).max():.2e} |\n"
            f"| D(I) = 𝟙 | {np.abs(real_wigner(l, np.eye(3)) - np.eye(2*l+1)).max():.2e} |\n"
            f"| angle between R₁R₂ and R₂R₁ | "
            f"{np.degrees(np.linalg.norm(rotvec(R1 @ R2 @ (R2 @ R1).T))):.2f}° |\n\n"
            "*This one line — D(R₁)D(R₂) = D(R₁R₂) — is what makes the Wigner matrix worth "
            "having : composing rotations of a sound field becomes multiplying matrices, so a "
            "whole trajectory of rotations can be accumulated once and applied once. It also "
            "transports every pathology of SO(3) into the coefficients unchanged : the third row "
            "is not a rounding error but the statement that Euler angles are coordinates on a "
            "curved group, not a vector space you may add in. Set β₁ = β₂ = 0 and the third row "
            "collapses to zero — with a single axis left, the angles do add.*")

    pn.bind(update, l.param.value_throttled, a1.param.value_throttled, b1.param.value_throttled,
            g1.param.value_throttled, a2.param.value_throttled, b2.param.value_throttled,
            g2.param.value_throttled, watch=True)
    update(l.value, a1.value, b1.value, g1.value, a2.value, b2.value, g2.value)

    return pn.Column(
        pn.pane.Markdown("## D is a *representation* : composition becomes matrix product"),
        pn.Row(pn.Column(a1, b1, g1), pn.Column(a2, b2, g2), l), pane,
        _formula(r"$$D^\ell(R_1)\, D^\ell(R_2) \;=\; D^\ell(R_1 R_2), \qquad "
                 r"D^\ell(R)^{-1} = D^\ell(R)^\top = D^\ell(R^{-1})$$"),
        read)


# --------------------------------------------------------------------------- #
#  4 -- the role : rotating a sound field is a matrix product                  #
# --------------------------------------------------------------------------- #

def plot_field_rotation():
    """What the matrix is *for*.

    The left sphere is the field. The middle one is the honest but expensive way of
    rotating it -- resample it at the rotated directions. The right one never touches
    the sphere at all : it multiplies the coefficient vector of every degree by
    D^l(R). The two agree to machine precision, and that is the entire point."""

    field_sel = pn.widgets.Select(name="sound field", options=list(EULER_FIELD_PRESETS))
    al = pn.widgets.FloatSlider(name="α  (rad)", start=0, end=2 * np.pi, step=.01, value=1.2)
    be = pn.widgets.FloatSlider(name="β  (rad)", start=0, end=np.pi, step=.01, value=0.8)
    ga = pn.widgets.FloatSlider(name="γ  (rad)", start=0, end=2 * np.pi, step=.01, value=0.5)

    pane, bars, read = pn.pane.Plotly(), pn.pane.Plotly(), pn.pane.Markdown()

    def update(field_name, al, be, ga):
        coeffs = _split_by_degree(_coeffs_from_dict(EULER_FIELD_PRESETS[field_name]))
        R = rot(2, al) @ rot(1, be) @ rot(2, ga)

        f0 = field_from_coeffs(coeffs, _GRID_DIRS)
        f_resampled = field_from_coeffs(coeffs, _GRID_DIRS @ R)          # f(R^-1 u)
        rotated = {l: real_wigner(l, R) @ a for l, a in coeffs.items()}
        f_matrix = field_from_coeffs(rotated, _GRID_DIRS)
        scale = max(np.abs(f0).max(), 1e-9)

        fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                            subplot_titles=("the field  f(u)",
                                            "resampled :  f(R⁻¹u)",
                                            "by the matrix :  Σ (D a)ₘ Yₘ(u)"))
        for k, (f, nm) in enumerate([(f0, 'f'), (f_resampled, 'resampled'), (f_matrix, 'matrix')]):
            fig.add_trace(_field_surface(f, scale, nm), row=1, col=k + 1)
        ax = dict(range=[-1.05, 1.05], showticklabels=False, title='',
                  backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True)
        cam = dict(eye=dict(x=1.6, y=1.6, z=1.0))
        fig.update_layout(width=1150, height=400, uirevision='constant',
                          margin=dict(t=46, b=0, l=0, r=0),
                          scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube', camera=cam),
                          scene2=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube', camera=cam),
                          scene3=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube', camera=cam))
        pane.object = fig

        degrees = sorted(coeffs)
        labels = [f"{l},{m}" for l in degrees for m in range(-l, l + 1)]
        before = np.concatenate([coeffs[l] for l in degrees])
        after = np.concatenate([rotated[l] for l in degrees])
        bfig = make_subplots(rows=1, cols=2, column_widths=[.62, .38],
                             subplot_titles=("the coefficients themselves",
                                             "energy per degree ‖a<sub>ℓ</sub>‖²"))
        bfig.add_trace(go.Bar(x=labels, y=before, name='before', marker_color=GREY), row=1, col=1)
        bfig.add_trace(go.Bar(x=labels, y=after, name='after', marker_color=BLUE), row=1, col=1)
        bfig.add_trace(go.Bar(x=degrees, y=[coeffs[l] @ coeffs[l] for l in degrees],
                              name='before', marker_color=GREY, showlegend=False), row=1, col=2)
        bfig.add_trace(go.Bar(x=degrees, y=[rotated[l] @ rotated[l] for l in degrees],
                              name='after', marker_color=BLUE, showlegend=False), row=1, col=2)
        bfig.update_xaxes(title="ℓ, m", row=1, col=1, type='category')
        bfig.update_xaxes(title="ℓ", row=1, col=2, dtick=1)
        bfig.update_layout(width=1000, height=300, barmode='group', uirevision='constant',
                           margin=dict(t=46, b=40, l=50, r=10),
                           legend=dict(x=0, y=1.25, orientation='h', font=dict(size=10)))
        bars.object = bfig

        per_deg = " , ".join(f"ℓ={l}: {abs(coeffs[l] @ coeffs[l] - rotated[l] @ rotated[l]):.1e}"
                             for l in degrees)
        read.object = (
            "| check | value |\n|---|---|\n"
            f"| max │resampled − matrix│ over the sphere | "
            f"{np.abs(f_resampled - f_matrix).max():.2e} |\n"
            f"| peak of the field (for scale) | {np.abs(f0).max():.4f} |\n"
            f"| energy drift per degree | {per_deg} |\n"
            f"| total energy before / after | "
            f"{float(before @ before):.10f} / {float(after @ after):.10f} |\n"
            f"| degrees mixed by the rotation | none — D is block-diagonal in ℓ |\n\n"
            "*The middle and right spheres are computed by completely different means : one "
            "evaluates spherical harmonics at rotated directions, the other multiplies "
            "(2ℓ+1)×(2ℓ+1) matrices and never looks at the sphere. They agree to 1e-15. That is "
            "the role of the Wigner matrix in an ambisonic pipeline — rotating a sound field is "
            "not resampling, it is a small block-diagonal matrix product, one block per degree, "
            "cheap enough to run per audio buffer. Watch the bars : individual coefficients are "
            "shuffled beyond recognition, the energy of each degree is untouched, and no energy "
            "ever crosses from one degree to another — the degrees are irreducible.*")

    pn.bind(update, field_sel, al.param.value_throttled, be.param.value_throttled,
            ga.param.value_throttled, watch=True)
    update(field_sel.value, al.value, be.value, ga.value)

    return pn.Column(
        pn.pane.Markdown("## What it is for : rotating a field is a matrix product"),
        pn.Row(field_sel, al, be, ga), pane, bars,
        _formula(r"$$f(R^{-1}u) \;=\; \sum_\ell \sum_m \left(D^\ell(R)\, "
                 r"\mathbf{a}_\ell\right)_m Y_\ell^m(u)$$"),
        read)


# --------------------------------------------------------------------------- #
#  5 -- a 3-dimensional submanifold in a huge matrix space                     #
# --------------------------------------------------------------------------- #

def plot_wigner_manifold():
    """"D^l is a 3-dimensional submanifold embedded into a huge matrix space".

    Every rotation gives a (2l+1)x(2l+1) matrix -- a point in a space of dimension
    (2l+1)^2 -- but the set of all of them only has three degrees of freedom. Pick any
    three matrix entries and watch a 3-parameter surface, never a cloud filling space."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=5, value=2)
    i1 = pn.widgets.IntSlider(name="entry 1 : row m'", start=-5, end=5, value=0)
    j1 = pn.widgets.IntSlider(name="entry 1 : column m", start=-5, end=5, value=0)
    i2 = pn.widgets.IntSlider(name="entry 2 : row m'", start=-5, end=5, value=1)
    j2 = pn.widgets.IntSlider(name="entry 2 : column m", start=-5, end=5, value=-1)
    i3 = pn.widgets.IntSlider(name="entry 3 : row m'", start=-5, end=5, value=-1)
    j3 = pn.widgets.IntSlider(name="entry 3 : column m", start=-5, end=5, value=1)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, i1, j1, i2, j2, i3, j3):
        d = 2 * l + 1
        idx = [(int(np.clip(i, -l, l)) + l) * d + (int(np.clip(j, -l, l)) + l)
               for i, j in [(i1, j1), (i2, j2), (i3, j3)]]
        mats = rotation_cloud(l)
        cloud = mats.reshape(len(mats), -1)
        P = cloud[:, idx]

        # local dimension : the rank of the three tangent directions at a random point
        G = real_generators(l)
        R0 = rot(2, .7) @ rot(1, 1.1) @ rot(2, .3)
        D0 = real_wigner(l, R0)
        tangents = np.stack([(g @ D0).ravel() for g in G], -1)
        sv = np.linalg.svd(tangents, compute_uv=False)
        rank = int(np.sum(sv > 1e-6 * sv[0]))
        sv_cloud = np.linalg.svd(cloud - cloud.mean(0), compute_uv=False)

        fig = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                            specs=[[{"type": "scene"}, {"type": "xy"}]],
                            subplot_titles=(f"{len(mats)} rotations, seen through 3 matrix entries",
                                            "singular values of the cloud (its *linear* span)"))
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',
                                   marker=dict(size=2.5, color=P[:, 2], colorscale='Viridis',
                                               showscale=False), name='D(R)'), row=1, col=1)
        ax = dict(backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True)
        fig.add_trace(go.Scatter(y=sv_cloud, mode='markers+lines', name='singular value',
                                 marker=dict(size=5, color=BLUE)), row=1, col=2)
        fig.update_yaxes(type='log', title='singular value', row=1, col=2)
        fig.update_xaxes(title='index', row=1, col=2)
        fig.update_layout(width=1080, height=460, uirevision='constant',
                          margin=dict(t=50, b=40, l=10, r=10),
                          scene=dict(xaxis=dict(title=f"D[{i1},{j1}]", **ax),
                                     yaxis=dict(title=f"D[{i2},{j2}]", **ax),
                                     zaxis=dict(title=f"D[{i3},{j3}]", **ax),
                                     aspectmode='cube'))
        pane.object = fig

        n_lin = int(np.sum(sv_cloud > 1e-8 * sv_cloud[0]))
        read.object = (
            "| space | dimension |\n|---|---|\n"
            f"| all (2ℓ+1)×(2ℓ+1) matrices | {d*d} |\n"
            f"| the orthogonal group O({d}) that D lives inside | {d*(d-1)//2} |\n"
            f"| **the image D^ℓ(SO(3)), as a manifold** | **{rank}** — always 3 |\n"
            f"| its *linear* span (nonzero singular values) | {n_lin} |\n"
            f"| singular values of the 3 tangent directions | {np.round(sv, 4)} |\n"
            f"| ‖[G_x,G_y] − G_z‖   (the tangent space *is* the algebra) | "
            f"{np.abs(bracket(G[0], G[1]) - G[2]).max():.1e} |\n\n"
            "*Two different numbers, and the difference is the whole point. **Linearly** the "
            "cloud spans everything — all (2ℓ+1)² directions have a nonzero singular value, which "
            "is Peter–Weyl telling you that the entries of an irreducible representation are "
            "linearly independent functions on the group (see *7_peter_weyl.ipynb*). **As a "
            "manifold** it is 3-dimensional : pick any three entries and you get a surface, never "
            "a solid, because only α, β, γ are free. The set is curved, not flat, which is why no "
            "linear coordinate system on those (2ℓ+1)² numbers can parametrize it. The tangent "
            "space at every point is spanned by exactly the three generators, and they close on "
            "the same brackets as in *B_so3_lie.ipynb* — the curvature that left a residual spin "
            "after a closed pointing loop is here too, in every degree.*")

    pn.bind(update, l.param.value_throttled, i1.param.value_throttled, j1.param.value_throttled,
            i2.param.value_throttled, j2.param.value_throttled, i3.param.value_throttled,
            j3.param.value_throttled, watch=True)
    update(l.value, i1.value, j1.value, i2.value, j2.value, i3.value, j3.value)

    return pn.Column(
        pn.pane.Markdown("## A 3-dimensional submanifold in a (2ℓ+1)²-dimensional matrix space"),
        pn.Row(pn.Column(l), pn.Column(i1, j1), pn.Column(i2, j2), pn.Column(i3, j3)),
        pane,
        _formula(r"$$D^\ell : SO(3) \hookrightarrow O(2\ell+1) \subset "
                 r"\mathbb{R}^{(2\ell+1)^2}, \qquad \dim \mathrm{im}\, D^\ell = 3$$"),
        read)


# --------------------------------------------------------------------------- #
#  6 -- orbits, stabilizers, invariants                                        #
# --------------------------------------------------------------------------- #

_MULTIPLET_PRESETS = {
    "zonal  Y_ℓ⁰  (axisymmetric)": "zonal",
    "sectoral  Y_ℓ^ℓ": "sectoral",
    "two orders  Y_ℓ⁰ + Y_ℓ^ℓ": "mixed",
    "generic (random)": "generic",
}

#: The hover ghost is re-sent over the websocket on every mouse move, so this panel
#: draws its spheres on a quarter of the usual grid : 64 x 33 is still smooth and
#: keeps one update around 25 kB instead of 110 kB.
_ORB_AZ, _ORB_ZE = np.meshgrid(np.linspace(-np.pi, np.pi, 64),
                               np.linspace(_POLE_EPS, np.pi - _POLE_EPS, 33), indexing='ij')
_ORB_DIRS = np.stack([np.sin(_ORB_ZE) * np.cos(_ORB_AZ),
                      np.sin(_ORB_ZE) * np.sin(_ORB_AZ),
                      np.cos(_ORB_ZE)], -1)

#: the three one-parameter subgroups, drawn as loops through the multiplet
_LOOP_AXES = [("x", RED), ("y", GREEN), ("z", BLUE)]

_REACH_TOL = 5e-3       # below this, a rotation really does land on the target

_SUB = "₀₁₂₃₄₅₆₇₈₉"
_SUPD = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _SUP(m):
    """``m`` as a superscript, for labelling Y_l^m inside a plot."""
    return ("⁻" if m < 0 else "") + "".join(_SUPD[int(c)] for c in str(abs(m)))


#: the sphere on the left : plain axes, the shape is the whole message
_ORB_AX = dict(showticklabels=False, title='', backgroundcolor='rgb(242,242,242)',
               gridcolor='white', showbackground=True)

#: the orbit on the right : here the axes are the point, so they keep their labels
_ORB_AX2 = dict(backgroundcolor='rgb(246,246,246)', gridcolor='white', showbackground=True,
                zerolinecolor='rgb(200,200,200)', tickfont=dict(size=9))


def _preset_vector(l, kind):
    a = np.zeros(2 * l + 1)
    if kind == "zonal":
        a[l] = 1.
    elif kind == "sectoral":
        a[2 * l] = 1.
    elif kind == "mixed":
        a[l], a[2 * l] = 1., .8
    else:
        a = np.random.default_rng(3).normal(size=2 * l + 1)
    return a / np.linalg.norm(a)


def _subgroup_loop(l, a, k, n=145):
    """``{ D(exp(theta e_k)) a }`` for a full turn about the axis ``e_k``.

    The matrices are chained from a single small step instead of exponentiating each
    angle : one ``expm`` and n matrix products, orthogonal to ~1e-14 all the way
    round, which is far below anything visible in a plot."""
    G = real_generators(l)[k]
    step = expm((2 * np.pi / (n - 1)) * G)
    out, D = [a.copy()], np.eye(len(a))
    for _ in range(n - 1):
        D = step @ D
        out.append(D @ a)
    return np.array(out)


def _orbit_net(l, a, n_meridian=12, n_beta=25, n_parallel=5, n_alpha=49):
    """The orbit as a lat-long net rather than as dust.

    Every point is ``D(R(alpha, beta)) a``, the multiplet aimed at the direction
    ``R z`` : the meridians sweep the tilt, the parallels sweep the yaw. Scattered
    dots say nothing about the shape of a set ; the same points joined along the two
    parameters they came from show it for what it is -- a closed surface. For an
    axisymmetric multiplet that surface *is* the whole orbit, since the third angle
    does nothing.

    Returned as a list of ``(points, labels)`` curves, one per line to draw."""
    _, Gy, Gz = real_generators(l)
    betas = np.linspace(0, np.pi, n_beta)
    ring = np.linspace(0, 2 * np.pi, n_alpha)
    Eb = [expm(b * Gy) for b in betas]
    Er = [expm(t * Gz) for t in ring]

    curves = []
    for alpha in np.linspace(0, 2 * np.pi, n_meridian, endpoint=False):
        Ea = expm(alpha * Gz)
        curves.append((np.array([Ea @ E @ a for E in Eb]),
                       [(np.degrees(alpha), np.degrees(b)) for b in betas]))
    for beta in np.linspace(0, np.pi, n_parallel + 2)[1:-1]:
        Ebeta = expm(beta * Gy)
        curves.append((np.array([E @ Ebeta @ a for E in Er]),
                       [(np.degrees(t), np.degrees(beta)) for t in ring]))
    return curves




def _euler_D(l, alpha, beta, gamma):
    """``D`` of the ZYZ rotation R(alpha, beta, gamma), from the generators."""
    Gx, Gy, Gz = real_generators(l)
    return expm(alpha * Gz) @ expm(beta * Gy) @ expm(gamma * Gz)


def _euler_route(l, a, alpha, beta, gamma, n=25):
    """The path from ``a`` to ``D(alpha, beta, gamma) a``, one Euler angle at a time.

    Read right to left, as the matrices act : first the spin gamma, then the tilt
    beta, then the yaw alpha. Drawn as a route rather than a jump because that is
    what makes the stabilizer visible -- on an axisymmetric multiplet the first leg
    has no length at all, however far gamma is pushed."""
    _, Gy, Gz = real_generators(l)
    Dg, Db = expm(gamma * Gz), expm(beta * Gy)
    legs = ((Gz, gamma, np.eye(2 * l + 1)),      # the spin, acting first
            (Gy, beta, Dg),                      # then the tilt, on top of it
            (Gz, alpha, Db @ Dg))                # then the yaw, on top of both
    # the new angle multiplies on the *left* of what is already applied, or the legs
    # do not join up and the last one does not land on D(alpha, beta, gamma) a
    return [np.array([expm(t * G) @ done @ a for t in np.linspace(0, ang, n)])
            if abs(ang) > 1e-12 else np.empty((0, 2 * l + 1))
            for G, ang, done in legs]


def plot_orbit_stabilizer():
    """Orbits, stabilizers and invariants, measured rather than asserted -- and
    navigable, because a cloud of dots on unlabelled axes explains nothing.

    The right-hand plot is the set of *all* rotations of one multiplet, drawn in the
    3 principal directions of that set (the axes say how much of the set each one
    carries -- at l = 1 they carry all of it, higher up they do not). The orbit is
    drawn as the net it is : every point is the multiplet aimed at one direction,
    meridians sweeping the tilt and parallels the yaw. Three coloured loops give it a
    skeleton -- turning about x, about y, about z -- and **a loop that shrinks to a
    single dot is the stabilizer**, the whole circle of rotations that all return the
    identical coefficient vector.

    The three Euler sliders walk one rotation across that net. The amber route shows
    how it got there, one angle at a time, and the left-hand sphere draws the result
    solid inside the ghost of the multiplet it started from : where the solid shape
    fills the ghost exactly, the rotation did nothing.

    The green diamonds are single harmonics Y_l^m that some rotation of this
    multiplet *is*. Only those are drawn on the orbit -- a harmonic no rotation
    reaches is not a point of the orbit and has no business being pictured on it.
    Every harmonic's distance to the orbit is in the bar chart instead, measured with
    ``closest_rotation`` : zero means it lies on the orbit, anything else is a shape
    that no aiming of this multiplet will ever produce."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=5, value=2)
    preset = pn.widgets.Select(name="multiplet", options=list(_MULTIPLET_PRESETS))
    sym = pn.widgets.FloatSlider(name="break the symmetry (mix in a random multiplet)",
                                 start=0, end=1, step=.01, value=0.)
    alpha = pn.widgets.FloatSlider(name="yaw α about z (°)", start=0, end=360, step=1, value=0)
    beta = pn.widgets.FloatSlider(name="tilt β about y (°)", start=0, end=180, step=1, value=0)
    gamma = pn.widgets.FloatSlider(name="spin γ about z, first (°)", start=0, end=360,
                                   step=1, value=0)

    pane_field = pn.pane.Plotly(config={'displayModeBar': False})
    pane_orbit = pn.pane.Plotly()
    pane_reach = pn.pane.Plotly(config={'displayModeBar': False})
    where = pn.pane.Markdown()
    read = pn.pane.Markdown()

    #: whatever survives between a structural rebuild and a slider move
    st = {}

    def _rebuild(l, preset, sym):
        """Everything that depends on the multiplet but not on the three angles."""
        a = _preset_vector(l, _MULTIPLET_PRESETS[preset])
        if sym > 0:
            a = a + sym * np.random.default_rng(11).normal(size=2 * l + 1)
            a /= np.linalg.norm(a)
        d = 2 * l + 1

        T = np.stack([g @ a for g in real_generators(l)], -1)   # the tangent directions
        sv = np.linalg.svd(T, compute_uv=False)
        orbit_dim = int(np.sum(sv > 1e-5 * max(sv[0], 1e-12)))
        n_inv = d - orbit_dim

        cloud = rotation_cloud(l) @ a
        centre = cloud.mean(0)
        S, Vt = np.linalg.svd(cloud - centre, full_matrices=False)[1:]
        axes3, var = Vt[:3], S ** 2 / max((S ** 2).sum(), 1e-30)
        project = lambda V: (np.atleast_2d(V) - centre) @ axes3.T

        # ---- the orbit, as a net rather than as dust ------------------------
        # When it is a surface the net *is* the orbit ; when it is a volume the net
        # is only the zero-spin slice through it, and a dense one just tangles.
        net = (_orbit_net(l, a) if orbit_dim < 3 else
               _orbit_net(l, a, n_meridian=6, n_parallel=3))
        traces = []
        if orbit_dim == 3:
            P = project(cloud)
            traces.append(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',
                marker=dict(size=1.5, color=GREY, opacity=.18),
                name='rotations off the net (the spin γ)', hoverinfo='skip'))
        for j, (pts, _lab) in enumerate(net):
            P = project(pts)
            traces.append(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                line=dict(color=GREY, width=1.5), opacity=.65 if orbit_dim < 3 else .35,
                hoverinfo='skip', legendgroup='net', showlegend=(j == 0),
                name='the orbit : every aiming of the multiplet' if orbit_dim < 3 else
                     'aimings with no spin (one slice of the orbit)'))
        for k, (nm, color) in enumerate(_LOOP_AXES):
            loop = _subgroup_loop(l, a, k)
            collapsed = float(np.abs(loop - a).max()) < _REACH_TOL
            P = project(loop)
            traces.append(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2],
                mode='markers' if collapsed else 'lines',
                line=dict(color=color, width=7),
                marker=dict(size=15, color=color, symbol='circle-open', line=dict(width=3)),
                hoverinfo='skip',
                name=f"about {nm}" + ("  ⟵ COLLAPSED to a point" if collapsed else "")))

        # ---- which single harmonics this multiplet can actually be turned into
        reach = np.array([closest_rotation(l, a, e, n_starts=6, iters=25)[0]
                          for e in np.eye(d)])
        hit = [m for m, r in zip(range(-l, l + 1), reach) if r < _REACH_TOL]
        if hit:
            P = project(np.eye(d)[[m + l for m in hit]])
            traces.append(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers+text',
                marker=dict(size=8, symbol='diamond', color=GREEN),
                text=[f"Y{_SUP(m)}" for m in hit], textposition='top center',
                textfont=dict(size=10), hoverinfo='skip',
                name='single harmonics this multiplet can be turned into'))
        P0 = project(a)
        traces.append(go.Scatter3d(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode='markers',
                                   marker=dict(size=7, color='black'), hoverinfo='skip',
                                   name='a itself (all three angles at 0)'))

        st.update(a=a, l=l, d=d, project=project, orbit_dim=orbit_dim,
                  Y=sh_matrix(l, _ORB_DIRS), traces=traces, var=var,
                  title=f"every rotation of this multiplet (ℓ = {l})")

        # ---- the honest answer about the harmonics, as distances -------------
        names = [f"Y{_SUP(m)}" for m in range(-l, l + 1)]
        bars = go.Figure(go.Bar(
            x=names, y=reach, marker_color=PURPLE,
            hovertemplate="%{x} : %{y:.3f} away<extra></extra>"))
        # a bar of height zero draws nothing, and 'nothing' is the one answer here
        # that must not be silent : mark the reachable ones explicitly.
        on_orbit = [n for n, r in zip(names, reach) if r < _REACH_TOL]
        if on_orbit:
            bars.add_trace(go.Scatter(
                x=on_orbit, y=[0] * len(on_orbit), mode='markers+text',
                marker=dict(size=11, color=GREEN, symbol='diamond'),
                text=['on the orbit'] * len(on_orbit), textposition='top center',
                textfont=dict(size=9, color=GREEN), cliponaxis=False,
                hovertemplate="%{x} : a rotation of a lands exactly here<extra></extra>"))
        bars.update_layout(
            width=520, height=210, uirevision='constant',
            title=dict(text="how far each single harmonic is from this orbit<br>"
                            "<sub>0 = some rotation of the multiplet <b>is</b> that harmonic ;"
                            " anything else = it never will be</sub>",
                       x=.5, font=dict(size=12)),
            margin=dict(t=58, b=30, l=50, r=10), showlegend=False,
            yaxis=dict(title="‖D(R)a − Yℓᵐ‖, best over all R", range=[0, max(reach.max(), .1) * 1.15]))
        pane_reach.object = bars

        axisymmetric = orbit_dim < 3
        expected = 2 if (l == 1 or (_MULTIPLET_PRESETS[preset] == "zonal" and sym == 0)) else 3
        read.object = (
            "| quantity | measured | theory |\n|---|---|---|\n"
            f"| dim ℋ_ℓ = 2ℓ+1 | {d} | {d} |\n"
            f"| singular values of (Gₓa, G_ya, G_za) | {np.round(sv, 5)} | — |\n"
            f"| **dim orbit** | **{orbit_dim}** | {expected} |\n"
            f"| **dim stabilizer** = 3 − dim orbit | **{3 - orbit_dim}** | "
            f"{3 - expected} {'(SO(2) : an axis of symmetry)' if axisymmetric else '(finite)'} |\n"
            f"| **n_invariants** = (2ℓ+1) − dim orbit | **{n_inv}** | "
            f"{d - expected}"
            f"{'  (generic : 2ℓ−2)' if not axisymmetric and l >= 2 else '  (a symmetric point : more invariants than the generic 2ℓ−2)' if l >= 2 else ''} |\n"
            f"| ‖a‖ along the orbit (an invariant) | "
            f"{np.linalg.norm(cloud, axis=1).min():.8f} … "
            f"{np.linalg.norm(cloud, axis=1).max():.8f} | 1 |\n"
            f"| the orbit spans | {int(np.sum(S > 1e-6 * S[0]))} of {d} directions, "
            f"{100*var[:3].sum():.0f} % of it in the 3 drawn | — |\n"
            f"| single harmonics it can be turned into | "
            f"**{', '.join('Y%s' % _SUP(m) for m in hit) if hit else 'none'}** "
            f"| {'all of them' if len(hit) == d else 'closest miss : %.3f' % np.sort(reach)[len(hit)]} |\n\n"
            "*The three coloured loops are the whole point. Each is one axis turned through a "
            "full circle, so each is a one-parameter subgroup drawn where it actually goes. On "
            "the **zonal** preset the z loop is not a loop at all : it collapses to a single "
            "dot, because every one of those rotations returns the identical coefficient vector. "
            "That collapse **is** dim Stab = 1, and the third singular value dropping to zero is "
            "the same fact in arithmetic. Push the **spin γ** slider on that preset : the route "
            "never leaves the black dot, and the solid shape never leaves its ghost.*\n\n"
            "*Set **ℓ = 1** for the picture with nothing hidden : there the orbit is exactly a "
            "sphere and 2ℓ+1 = 3, so the projection throws nothing away at all. The x and y loops "
            "are great circles, the z loop is the pole they meet at, and all three harmonics sit "
            "on the surface — every dipole really is a rotated Y₁⁰. Every degree above that is "
            "the same story seen through a lossy window ; the percentages on the axes say how "
            "lossy.*\n\n"
            "*Nudge the symmetry slider and the collapsed loop opens up : the orbit gains its "
            "third dimension and one invariant is spent on orientation. Everything the section "
            "above asserts — dim 𝒪 = 3 − dim Stab, n_inv = (2ℓ+1) − dim 𝒪 — is measured here, "
            "not assumed.*\n\n"
            "*Only the harmonics the orbit really passes through are drawn on it. The rest are "
            "not points of this orbit at all, so putting them in the picture would only invite "
            "the eye to place them on a surface they are not on — the bar chart gives their "
            "distance instead. The sectoral preset is the one to try : a yaw of 90°/ℓ carries "
            "Yℓ^ℓ exactly onto Yℓ^−ℓ, two bars at zero, and the blue z loop visibly runs through "
            "both diamonds.*")

        _move(alpha.value, beta.value, gamma.value)

    def _move(al_deg, be_deg, ga_deg):
        """Only the three angles changed : redraw the marker, the route and the shape."""
        if not st:
            return
        l, a, project = st['l'], st['a'], st['project']
        al, be, ga = np.radians([al_deg, be_deg, ga_deg])
        vec = _euler_D(l, al, be, ga) @ a
        moved = float(np.linalg.norm(vec - a))

        # -- the orbit, with the route walked across it --------------------- #
        # Always the same four traces, empty ones included : Panel keeps one data
        # source per trace *by position*, so a trace appearing or vanishing shifts
        # every index after it and forces the whole figure to re-sync. Fixed count,
        # and only the arrays that actually changed travel.
        route = _euler_route(l, a, al, be, ga)
        extra = []
        for leg, nm, dash in zip(route, ("spin γ", "tilt β", "yaw α"),
                                 ('dot', 'solid', 'solid')):
            P = project(leg) if len(leg) > 1 else np.empty((0, 3))
            extra.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                                      line=dict(color=AMBER, width=5, dash=dash),
                                      name=f"the route : {nm}", hoverinfo='skip',
                                      legendgroup='route', showlegend=(nm == "yaw α")))
        Pm = project(vec)
        extra.append(go.Scatter3d(x=Pm[:, 0], y=Pm[:, 1], z=Pm[:, 2], mode='markers',
                                  marker=dict(size=10, color=AMBER, symbol='circle',
                                              line=dict(color='black', width=1)),
                                  name='where those angles land', hoverinfo='skip'))
        fig = go.Figure(st['traces'] + extra)
        var = st['var']
        fig.update_layout(
            width=660, height=560, uirevision='constant',
            title=dict(text=st['title'] + "<br><sub>the amber route walks the three sliders, "
                                          "one angle at a time</sub>", x=.5, font=dict(size=13)),
            margin=dict(t=70, b=0, l=0, r=0),
            legend=dict(x=0, y=1, font=dict(size=10), bgcolor='rgba(255,255,255,.7)'),
            scene=dict(xaxis=dict(title=f"PC1 · {100*var[0]:.0f} %", **_ORB_AX2),
                       yaxis=dict(title=f"PC2 · {100*var[1]:.0f} %", **_ORB_AX2),
                       zaxis=dict(title=f"PC3 · {100*var[2]:.0f} %", **_ORB_AX2),
                       aspectmode='data', camera=dict(eye=dict(x=1.7, y=1.7, z=1.0))))
        pane_orbit.object = fig

        # -- the shape : the multiplet as a ghost, the rotation solid inside - #
        f0, f1 = st['Y'] @ a, st['Y'] @ vec
        scale = max(np.abs(f0).max(), np.abs(f1).max(), 1e-9)
        # The ghost keeps its true size and the solid is drawn a hair inside it : the
        # two are the same surface whenever the rotation does nothing, and coincident
        # geometry would z-fight into speckle. Painting the solid *smaller* also stops
        # the translucent shell from washing over it when the two shapes differ.
        shape = go.Figure([
            _field_surface(f0, scale, 'the multiplet a', dirs=_ORB_DIRS,
                           color=GREY, opacity=.18),
            _field_surface(f1, scale, 'this rotation of it', dirs=_ORB_DIRS, grow=.97),
        ])
        shape.update_layout(
            width=520, height=430, uirevision='constant', showlegend=False,
            title=dict(text=f"α = {al_deg:.0f}°, β = {be_deg:.0f}°, γ = {ga_deg:.0f}°"
                            f" &nbsp;·&nbsp; ‖D(R)a − a‖ = <b>{moved:.3f}</b><br>"
                            "<sub>solid = the rotated multiplet · ghost = where it started"
                            "</sub>", x=.5, font=dict(size=13)),
            margin=dict(t=68, b=0, l=0, r=0),
            scene=dict(xaxis=dict(range=[-1.13, 1.13], **_ORB_AX),
                       yaxis=dict(range=[-1.13, 1.13], **_ORB_AX),
                       zaxis=dict(range=[-1.13, 1.13], **_ORB_AX),
                       aspectmode='cube', camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))))
        pane_field.object = shape

        where.object = (
            f"**‖D(R)a − a‖ = {moved:.3f}**"
            + ("  ·  *these angles change nothing at all : the rotation is in the stabilizer*"
               if moved < _REACH_TOL else
               "  ·  *same energy, same shape, re-aimed — the orbit is where all of these live*"))

    pn.bind(_rebuild, l.param.value_throttled, preset, sym.param.value_throttled, watch=True)
    pn.bind(_move, alpha.param.value_throttled, beta.param.value_throttled,
            gamma.param.value_throttled, watch=True)
    _rebuild(l.value, preset.value, sym.value)

    return pn.Column(
        pn.pane.Markdown("## Orbits, stabilizers, invariants — measured"),
        pn.Row(l, preset, sym),
        pn.Row(alpha, beta, gamma),
        pn.Row(pn.Column(pane_field, where, pane_reach), pane_orbit),
        _formula(r"$$\dim \mathcal{O}_{\mathbf{a}} = 3 - \dim \mathrm{Stab}(\mathbf{a}), "
                 r"\qquad \underbrace{2\ell+1}_{\dim \mathcal{H}_\ell} = "
                 r"n_{\mathrm{inv}} + \dim \mathcal{O}_{\mathbf{a}}$$"),
        read)
