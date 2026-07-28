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

from .lie import J, rot, rotvec, bracket, wigner_D, BLUE, RED, GREY, _formula
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


def _field_surface(f, scale, name):
    r = np.abs(f) / scale
    return go.Surface(x=r * _GRID_DIRS[..., 0], y=r * _GRID_DIRS[..., 1],
                      z=r * _GRID_DIRS[..., 2], surfacecolor=np.sign(f),
                      cmin=-1, cmax=1, showscale=False, name=name,
                      colorscale=[[0, BLUE], [1, RED]])


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
    "generic (random)": "generic",
    "two orders  Y_ℓ⁰ + Y_ℓ^ℓ": "mixed",
}


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


def plot_orbit_stabilizer():
    """Orbits, stabilizers and invariants, measured rather than asserted.

    The orbit's dimension is the rank of the three tangent directions ``G_a . a`` ;
    the stabilizer's dimension is what is left of the three rotations, and the number
    of independent invariants is the codimension. The zonal preset is the interesting
    one : it is axisymmetric, so one whole rotation does nothing to it."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=5, value=2)
    preset = pn.widgets.Select(name="multiplet", options=list(_MULTIPLET_PRESETS))
    tilt = pn.widgets.FloatSlider(name="tilt the multiplet away from the preset",
                                  start=0, end=1, step=.01, value=0.)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, preset, tilt):
        a = _preset_vector(l, _MULTIPLET_PRESETS[preset])
        if tilt > 0:
            a = a + tilt * np.random.default_rng(11).normal(size=2 * l + 1)
            a /= np.linalg.norm(a)

        G = real_generators(l)
        T = np.stack([g @ a for g in G], -1)                 # the three tangent directions
        sv = np.linalg.svd(T, compute_uv=False)
        orbit_dim = int(np.sum(sv > 1e-5 * max(sv[0], 1e-12)))
        stab_dim = 3 - orbit_dim
        n_inv = (2 * l + 1) - orbit_dim

        # the orbit itself, projected on its own three leading directions
        cloud = rotation_cloud(l) @ a
        C = cloud - cloud.mean(0)
        S, Vt = np.linalg.svd(C, full_matrices=False)[1:]
        P = C @ Vt[:3].T

        f = sh_matrix(l, _GRID_DIRS) @ a

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=(f"the field of this multiplet (ℓ = {l})",
                                            f"its orbit, projected on its 3 leading directions"))
        fig.add_trace(_field_surface(f, max(np.abs(f).max(), 1e-9), 'field'), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',
                                   marker=dict(size=2, color=P[:, 2], colorscale='Viridis',
                                               showscale=False), name='orbit'), row=1, col=2)
        ax = dict(showticklabels=False, title='', backgroundcolor='rgb(240,240,240)',
                  gridcolor='white', showbackground=True)
        fig.update_layout(width=1080, height=440, uirevision='constant',
                          margin=dict(t=48, b=0, l=0, r=0),
                          scene=dict(xaxis=dict(range=[-1.05, 1.05], **ax),
                                     yaxis=dict(range=[-1.05, 1.05], **ax),
                                     zaxis=dict(range=[-1.05, 1.05], **ax), aspectmode='cube',
                                     camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))),
                          scene2=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube'))
        pane.object = fig

        axisymmetric = (l == 1) or (_MULTIPLET_PRESETS[preset] == "zonal" and tilt == 0)
        expected = 2 if axisymmetric else 3
        read.object = (
            "| quantity | measured | theory |\n|---|---|---|\n"
            f"| dim ℋ_ℓ = 2ℓ+1 | {2*l+1} | {2*l+1} |\n"
            f"| singular values of (Gₓa, G_ya, G_za) | {np.round(sv, 5)} | — |\n"
            f"| **dim orbit** | **{orbit_dim}** | {expected} |\n"
            f"| **dim stabilizer** = 3 − dim orbit | **{stab_dim}** | "
            f"{3 - expected} {'(SO(2) : an axis of symmetry)' if axisymmetric else '(finite)'} |\n"
            f"| **n_invariants** = (2ℓ+1) − dim orbit | **{n_inv}** | "
            f"{2*l + 1 - expected}"
            f"{'  (generic : 2ℓ−2)' if not axisymmetric and l >= 2 else '  (a symmetric point : more invariants than the generic 2ℓ−2)' if l >= 2 else ''} |\n"
            f"| ‖a‖ along the orbit (an invariant) | "
            f"{np.linalg.norm(cloud, axis=1).min():.8f} … "
            f"{np.linalg.norm(cloud, axis=1).max():.8f} | 1 |\n"
            f"| linear span of the orbit cloud | {int(np.sum(S > 1e-6 * S[0]))} of {2*l+1} "
            f"directions | — |\n\n"
            "*The rank of those three tangent vectors is the whole story. For a **zonal** "
            "harmonic one of the three rotations does nothing at all — the third singular value "
            "drops to zero, the stabilizer is the SO(2) of spins about its axis, and the orbit is "
            "only a 2-sphere. Nudge the tilt slider and the symmetry breaks : the third direction "
            "comes alive, the orbit fills its three dimensions, and one further invariant is lost "
            "to orientation. Everything the section above asserts — dim 𝒪 = 3 − dim Stab, "
            "n_invariants = (2ℓ+1) − dim 𝒪 — is being measured here, not assumed.*")

    pn.bind(update, l.param.value_throttled, preset, tilt.param.value_throttled, watch=True)
    update(l.value, preset.value, tilt.value)

    return pn.Column(
        pn.pane.Markdown("## Orbits, stabilizers, invariants — measured"),
        pn.Row(l, preset, tilt), pane,
        _formula(r"$$\dim \mathcal{O}_{\mathbf{a}} = 3 - \dim \mathrm{Stab}(\mathbf{a}), "
                 r"\qquad \underbrace{2\ell+1}_{\dim \mathcal{H}_\ell} = "
                 r"n_{\mathrm{inv}} + \dim \mathcal{O}_{\mathbf{a}}$$"),
        read)
