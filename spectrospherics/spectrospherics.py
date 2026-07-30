from functools import lru_cache

import panel as pn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from .utils import spherical_harmonic, plot_spherical, plot_ambisonics_3d, generate_harmonic, sph2car
from .explorer import ParamSpec, DecompositionSpec, DecompositionExplorer

def get_harmonic_plot(n, m):
    [azimuth, zenith] = np.mgrid[-np.pi:np.pi:0.05, 0:np.pi:0.01]
    out = spherical_harmonic(azimuth, zenith, n, m)
    return plot_spherical(out, azimuth=azimuth, zenith=zenith)


# --------------------------------------------------------------------------- #
#  reading one real spherical harmonic                                        #
# --------------------------------------------------------------------------- #

#: scipy's ``assoc_legendre_p(n, 0, x)`` returns +1 at x = -1 instead of (-1)^n,
#: so every grid that samples zenith = pi exactly gets a wrong south-pole row.
_POLE_EPS = 1e-4

_BLUE, _RED = 'rgb(59,130,246)', 'rgb(239,68,68)'


def _sign_changes(values, closed=False):
    """Number of sign changes along a sampled curve, ignoring exact zeros."""
    s = np.sign(values[np.abs(values) > 1e-9 * max(np.abs(values).max(), 1e-12)])
    if s.size < 2:
        return 0
    if closed:
        s = np.append(s, s[0])
    return int(np.sum(s[1:] != s[:-1]))


def _fibonacci_dirs(n=6000):
    i = np.arange(n) + .5
    ze = np.arccos(1 - 2 * i / n)
    az = np.pi * (1 + 5 ** .5) * i
    return az, ze


def _harmonic_classification(n, m):
    if m == 0:
        return "zonal" if n else "omni", "no azimuthal dependence : a stack of parallels"
    if abs(m) == n:
        return "sectoral", "no nodal parallel : orange-slice sectors, all the energy on the equator"
    return "tesseral", "a checkerboard of parallels and meridians"


def plot_spherical_harmonic():
    """A reading guide for a single real spherical harmonic Y_n^m.

    Six linked views of the same function : the usual balloon r = |Y| ; the same
    thing painted on the unit sphere, where the nodal lines are visible ; the
    equirectangular map, where the (n - |m|) parallels and the |m| meridians can
    simply be counted ; the two separable factors whose product *is* the harmonic ;
    and the horizontal / median polar cuts an audio engineer actually listens to.

    The old single-balloon figure is still one call away, as ``get_harmonic_plot``."""

    n = pn.widgets.IntSlider(name="degree  n  (ambisonic order)", start=0, end=8, value=2)
    m = pn.widgets.IntSlider(name="order  m", start=-8, end=8, value=1)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    az = np.linspace(-np.pi, np.pi, 145)
    ze = np.linspace(_POLE_EPS, np.pi - _POLE_EPS, 73)
    AZ, ZE = np.meshgrid(az, ze, indexing='ij')
    DIRS = np.stack([np.sin(ZE) * np.cos(AZ), np.sin(ZE) * np.sin(AZ), np.cos(ZE)], -1)
    _fib_az, _fib_ze = _fibonacci_dirs()

    def update(n, m):
        m = int(np.clip(m, -n, n))
        Y = spherical_harmonic(AZ, ZE, n, m)
        peak = max(np.abs(Y).max(), 1e-12)

        fig = make_subplots(
            rows=2, cols=3, row_heights=[.56, .44], vertical_spacing=.12,
            horizontal_spacing=.07,
            specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}, {"type": "polar"}]],
            subplot_titles=("r = |Y| , coloured by sign", "Y painted on the sphere",
                            "the map : count the nodal lines",
                            "zenith factor : P<sub>n</sub><sup>|m|</sup>(cos θ)",
                            "azimuth factor : cos/sin(mφ)",
                            "polar cuts"))

        # (1) the balloon
        r = np.abs(Y) / peak
        fig.add_trace(go.Surface(x=r * DIRS[..., 0], y=r * DIRS[..., 1], z=r * DIRS[..., 2],
                                 surfacecolor=np.sign(Y), cmin=-1, cmax=1, showscale=False,
                                 colorscale=[[0, _BLUE], [1, _RED]],
                                 hovertemplate="|Y| = %{surfacecolor}<extra></extra>"),
                      row=1, col=1)
        # (2) the same function painted on the unit sphere : the nodal lines are the pale bands
        fig.add_trace(go.Surface(x=DIRS[..., 0], y=DIRS[..., 1], z=DIRS[..., 2],
                                 surfacecolor=Y, cmin=-peak, cmax=peak, showscale=False,
                                 colorscale='RdBu', reversescale=True), row=1, col=2)
        # (3) the equirectangular map, with the zero contour drawn on top
        fig.add_trace(go.Heatmap(z=Y.T, x=np.degrees(az), y=np.degrees(ze),
                                 zmin=-peak, zmax=peak, colorscale='RdBu', reversescale=True,
                                 showscale=False,
                                 hovertemplate="φ=%{x:.0f}°, θ=%{y:.0f}°<br>Y=%{z:.3f}<extra></extra>"),
                      row=1, col=3)
        fig.add_trace(go.Contour(z=Y.T, x=np.degrees(az), y=np.degrees(ze),
                                 contours=dict(start=0, end=0, size=1, coloring='none'),
                                 line=dict(color='black', width=1.2), showscale=False,
                                 hoverinfo='skip'), row=1, col=3)

        # (4) & (5) the two factors whose product is the harmonic
        az0 = 0. if m >= 0 else np.pi / (2 * abs(m))          # where the azimuth factor is 1
        zenith_factor = spherical_harmonic(az0, ze, n, m)
        azimuth_factor = np.cos(m * az) if m >= 0 else np.sin(abs(m) * az)
        fig.add_trace(go.Scatter(x=np.degrees(ze), y=zenith_factor, mode='lines',
                                 line=dict(color=_BLUE, width=2.5), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=np.degrees(ze), y=np.zeros_like(ze), mode='lines',
                                 line=dict(color='grey', width=1, dash='dot'),
                                 showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.degrees(az), y=azimuth_factor, mode='lines',
                                 line=dict(color=_RED, width=2.5), showlegend=False),
                      row=2, col=2)
        fig.add_trace(go.Scatter(x=np.degrees(az), y=np.zeros_like(az), mode='lines',
                                 line=dict(color='grey', width=1, dash='dot'),
                                 showlegend=False, hoverinfo='skip'), row=2, col=2)

        # (6) the two cuts, as polar directivities, split by sign
        t = np.linspace(0, 2 * np.pi, 361)
        horiz = spherical_harmonic(t, np.full_like(t, np.pi / 2), n, m)
        med_az = np.where(t <= np.pi, 0., np.pi)
        med_ze = np.where(t <= np.pi, t, 2 * np.pi - t)
        med = spherical_harmonic(med_az, np.clip(med_ze, _POLE_EPS, np.pi - _POLE_EPS), n, m)
        for vals, name, dash in [(horiz, "horizontal plane (θ=90°)", 'solid'),
                                 (med, "median plane (φ=0)", 'dot')]:
            for sign, color in [(1, _RED), (-1, _BLUE)]:
                rr = np.where(np.sign(vals) == sign, np.abs(vals), np.nan)
                fig.add_trace(go.Scatterpolar(r=rr, theta=np.degrees(t), mode='lines',
                                              line=dict(color=color, width=2.5, dash=dash),
                                              name=f"{name} {'+' if sign > 0 else '−'}",
                                              showlegend=False), row=2, col=3)

        ax = dict(range=[-1.05, 1.05], showticklabels=False, title='', showbackground=True,
                  backgroundcolor='rgb(240,240,240)', gridcolor='white')
        scene = dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube',
                     camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)))
        fig.update_layout(width=1180, height=720, uirevision='constant',
                          margin=dict(t=54, b=40, l=40, r=20),
                          scene=scene, scene2=scene,
                          polar=dict(radialaxis=dict(showticklabels=False, range=[0, peak * 1.05]),
                                     angularaxis=dict(rotation=90, direction='counterclockwise')))
        fig.update_xaxes(title="azimuth φ (°)", row=1, col=3, dtick=90)
        fig.update_yaxes(title="zenith θ (°)", row=1, col=3, autorange='reversed', dtick=45)
        fig.update_xaxes(title="θ (°)", row=2, col=1, dtick=45)
        fig.update_xaxes(title="φ (°)", row=2, col=2, dtick=90)
        pane.object = fig

        # ---- the numbers -------------------------------------------------- #
        # kind, kind_note = _harmonic_classification(n, m)
        # acn = n * n + n + m
        fib = spherical_harmonic(_fib_az, _fib_ze, n, m)
        # self_norm = float(np.mean(fib ** 2))                  # (1/4pi) * integral of Y^2
        cross = 0.
        for nn in range(0, max(n, 3) + 1):
            for mm in range(-nn, nn + 1):
                if (nn, mm) != (n, m):
                    cross = max(cross, abs(float(np.mean(
                        fib * spherical_harmonic(_fib_az, _fib_ze, nn, mm)))))
        # antipode = spherical_harmonic(_fib_az + np.pi, np.pi - _fib_ze, n, m)

        # count the nodal lines on the map instead of asserting the rule
        # ze_cut = ze[np.argmax(np.abs(zenith_factor))]         # a latitude where Y does not vanish
        # n_meridian = _sign_changes(spherical_harmonic(az[:-1], ze_cut, n, m), closed=True)
        # n_parallel = _sign_changes(zenith_factor)

        # read.object = (
        #     f"### Y<sub>{n}</sub><sup>{m}</sup> — **{kind}** ({kind_note})\n\n"
        #     "| property | value |\n|---|---|\n"
        #     f"| ambisonic channel (ACN = n² + n + m) | **{acn}** |\n"
        #     f"| nodal parallels : counted / n − \\|m\\| | **{n_parallel}** / {n - abs(m)} |\n"
        #     f"| sign changes around the equator : counted / 2\\|m\\| | "
        #     f"**{n_meridian}** / {2 * abs(m)} |\n"
        #     f"| peak \\|Y\\| | {np.abs(Y).max():.4f}  (SN3D keeps every harmonic ≤ 1) |\n"
        #     f"| (1/4π)∫Y² dΩ | {self_norm:.6f}   vs   1/(2n+1) = {1/(2*n+1):.6f} |\n"
        #     f"| largest overlap with another harmonic | {cross:.2e}  (orthogonal — the "
        #     f"residual is the {_fib_az.size}-point quadrature, not the harmonics) |\n"
        #     f"| parity Y(−u) − (−1)ⁿY(u) | {np.abs(antipode - (-1)**n * fib).max():.1e} |\n"
        #     f"| degrees of freedom at this degree | 2n+1 = {2*n+1} |\n\n"
        #     "*The two bottom-left curves are the whole definition : a harmonic is the **product** "
        #     "of a profile in zenith and an oscillation in azimuth, "
        #     f"Y = N · P(cos θ) · {'cos' if m >= 0 else 'sin'}({abs(m)}φ). "
        #     "Read the map like a score : |m| tells you how many times the sign alternates as you "
        #     "walk around the equator (2|m| zero crossings, |m| nodal meridians), and n − |m| tells "
        #     "you how many nodal parallels cut it horizontally. Their sum is always n. "
        #     "Sweep m from −n to n and watch the energy migrate from the poles (m = 0, zonal) to "
        #     "the equator (|m| = n, sectoral) : that is why the sectoral harmonics carry the "
        #     "horizontal resolution of an ambisonic mix, and the zonal ones the vertical.*")

    def _sync(degree, order):
        m.start, m.end = -degree, degree          # the order is meaningless outside [-n, n]
        update(degree, order)

    pn.bind(_sync, n.param.value_throttled, m.param.value_throttled, watch=True)
    _sync(n.value, m.value)

    return pn.Column(
        pn.pane.Markdown("## Reading a spherical harmonic"),
        pn.Row(n, m),
        pane
        # pn.Row(pane, read)
    )


# --------------------------------------------------------------------------- #
#  Spherical-harmonic synthesis (the 3D counterpart of spectrangular)         #
# --------------------------------------------------------------------------- #

#: The sphere is sampled once and the harmonic basis cached per order, so an
#: animated frame is a single matrix product instead of ~n² Legendre evaluations.
#: The grid is deliberately modest : every frame republishes the whole surface, and
#: at 0.05 rad (126 x 63 = 7938 points) that is ~250 kB per frame, enough to swamp
#: the websocket and leave the scene impossible to drag. 64 x 33 is a quarter of it.
_SPH_AZ, _SPH_ZE = np.meshgrid(np.linspace(-np.pi, np.pi, 64),
                               np.linspace(_POLE_EPS, np.pi - _POLE_EPS, 33),
                               indexing='ij')
_SPH_DIRS = np.stack([np.sin(_SPH_ZE) * np.cos(_SPH_AZ),
                      np.sin(_SPH_ZE) * np.sin(_SPH_AZ),
                      np.cos(_SPH_ZE)], -1)


@lru_cache(maxsize=None)
def _spherical_basis(order):
    """ACN-ordered stack ``(n_coeffs, *grid)`` of the real harmonics up to ``order``."""
    return np.stack([spherical_harmonic(_SPH_AZ, _SPH_ZE, n, m)
                     for n in range(order + 1) for m in range(-n, n + 1)])


def from_spherical_harmonics(coeffs, order=None, normalize=True):
    """Sum the ACN-ordered ``coeffs`` into a field on the sampled sphere.

    ``normalize`` divides by the peak, which keeps the balloon filling the frame
    while a shape morphs -- at the cost of hiding any change that is *only* gain.
    Animating a₀⁰ on its own is exactly that case : it scales the whole field, so
    normalised it produces a perfectly motionless sphere."""
    if order is None:
        order = int(round(np.sqrt(len(coeffs)))) - 1
    basis = _spherical_basis(order)
    if len(coeffs) != len(basis):
        raise ValueError("expected %d coefficients for order %d, got %d"
                         % (len(basis), order, len(coeffs)))
    out = np.tensordot(np.asarray(coeffs, dtype=float), basis, axes=1)
    peak = np.abs(out).max()
    return out / peak if (normalize and peak > 0) else out


#: with every slider in [-1, 1] a field rarely peaks above ~2, so this holds the
#: un-normalised balloon in view without the axes rescaling under the animation
_SPH_RANGE = 2.2


def _balloon(f, span):
    """r = |f| coloured by sign, in a layout that is identical frame to frame.

    Both halves of that matter for animation : a constant ``uirevision`` is what
    lets Plotly keep the camera the user dragged, and a byte-identical layout is
    what stops Panel resending it 15 times a second."""
    r = np.abs(f)
    axis = dict(range=[-span, span], showticklabels=False, title='', showbackground=True,
                backgroundcolor='rgb(235,235,235)', gridcolor='white',
                zerolinecolor='white')
    fig = go.Figure(go.Surface(
        x=r * _SPH_DIRS[..., 0], y=r * _SPH_DIRS[..., 1], z=r * _SPH_DIRS[..., 2],
        surfacecolor=np.sign(f), cmin=-1, cmax=1, showscale=False,
        colorscale=[[0, _BLUE], [1, _RED]],
        hovertemplate="|f| = %{customdata:.3f}<extra></extra>", customdata=r))
    # No fixed width : the pane is stretch_width, and a hard-coded one would park
    # the scene against the left edge of a container wider than itself. autosize
    # hands the width back to the pane, which is what keeps it centred on redraw.
    fig.update_layout(autosize=True, height=520, uirevision='constant', showlegend=False,
                      margin=dict(t=10, b=10, l=10, r=10),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode='cube',
                                 camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))))
    return fig


def plot_spherical_fn(*coeffs, order=None, polar=False, normalize=True):
    """Balloon of the field synthesised from ``coeffs``.

    ``polar=True`` reads them as the hyperspherical parametrisation of each
    degree (one radius then 2n angles), i.e. the 3D analogue of the
    amplitude/phase pair used by the circular explorer."""
    if order is None:
        raise ValueError(order)
    if polar:
        coeffs = sph2car(*coeffs)
    out = from_spherical_harmonics(coeffs, order=order, normalize=normalize)
    return _balloon(out, 1.05 if normalize else _SPH_RANGE)


# --------------------------------------------------------------------------- #
#  Declarative decomposition description (fed to the generic explorer)        #
# --------------------------------------------------------------------------- #

def _build_spherical_params(structural):
    """One row per degree n, carrying its 2n+1 coefficients a_n^m, m = -n..n.
    Rows stack top -> bottom, so the pyramid reads like the ACN layout."""
    order = structural["order"]
    rows = []
    for n in range(order + 1):
        rows.append([
            ParamSpec(f"c{n}_{m}", rf"$a_{{{n}}}^{{{m}}}$",
                      value=1.0 if n == 0 else 0.0, start=-1.0, end=1.0)
            for m in range(-n, n + 1)
        ])
    return rows


def _render_spherical(params, structural):
    order = structural["order"]
    coeffs = [params[f"c{n}_{m}"] for n in range(order + 1) for m in range(-n, n + 1)]
    return plot_spherical_fn(*coeffs, order=order,
                             normalize=structural["fit to frame"])


SPHERICAL_3D = DecompositionSpec(
    name="Spherical harmonics (3D)",
    structural={"order": (1, 5, 3), "fit to frame": True},
    build_params=_build_spherical_params,
    render=_render_spherical,
)




def plot_spectrambisonics_3d():
    """Coefficient-by-coefficient explorer for a 3D ambisonic field.

    Untick *fit to frame* to see gain as well as shape : normalised, a field that
    is only being scaled -- a₀⁰ moved on its own, say -- is a sphere that never
    changes."""
    return DecompositionExplorer(SPHERICAL_3D).layout




# --------------------------------------------------------------------------- #
#  Polar-like decompositions                                                  #
# --------------------------------------------------------------------------- #

def _build_pairwise_polar_params(structural):
    """One row per degree n, carrying its 2n+1 coefficients a_n^m, m = -n..n.
    Rows stack top -> bottom, so the pyramid reads like the ACN layout."""
    order = structural["order"]
    rows = []
    for n in range(order + 1):
        if n == 0: 
            rows.append([
                ParamSpec(f"c0_0", rf"$a_0^0$", value=0., start=-1., end=1.)
            ])
        else:
            c0 = ParamSpec(f"c{n}_0", rf"$c_0^0$", value=1.0 if n == 1 else 0.0, start=-1.0, end=1.0)
            rows.append(
                sum([
                    [ParamSpec(f"c{n}_{m}", rf"$c_{{{n}}}^{{{m}}}$",
                        value=0.0, start=-1.0, end=1.0), 
                    ParamSpec(f"phi{n}_{m}", rf"$phi_{{{n}}}^{{{m}}}$",
                        value=0.0, start=-1.0, end=1.0)]
                    for m in range(1, n + 1)], [c0])
            )
    return rows

def _pairwise_polar_transform(params: dict, structural: dict) -> dict:
    order = structural["order"]
    new_params = {}
    for n in range(order+1):
        if n == 0: 
            new_params['c0_0'] = params['c0_0']
        else:
            for m in range(n+1):
                if m == 0:
                    new_params[f'c{n}_{m}'] = params[f'c{n}_{m}']
                else:
                    c = params[f'c{n}_{m}']
                    phi = params[f'phi{n}_{m}']
                    new_params[f'c{n}_{m}'] = float(c * np.cos(m * 2 * np.pi * phi))
                    new_params[f'c{n}_-{m}'] = float(c * np.sin(m * 2 *np.pi * phi))
    return new_params

PAIRWISE_POLAR_3D = DecompositionSpec(
    name = "Pairwse Polar Coefficients (3D)",
    structural={"order": (1, 5, 3), "fit to frame": True},
    build_params=_build_pairwise_polar_params,
    transform=_pairwise_polar_transform,
    render=_render_spherical,
)


def plot_pairwise_polar_spectrambisonics():
    """Coefficient-by-coefficient explorer for a 3D ambisonic field.

    Untick *fit to frame* to see gain as well as shape : normalised, a field that
    is only being scaled -- a₀⁰ moved on its own, say -- is a sphere that never
    changes."""
    return DecompositionExplorer(PAIRWISE_POLAR_3D).layout



def _build_hyperspheric_params(structural):
    """One row per degree n, carrying its 2n+1 coefficients a_n^m, m = -n..n.
    Rows stack top -> bottom, so the pyramid reads like the ACN layout."""
    order = structural["order"]
    rows = []
    for n in range(order + 1):
        if n == 0: 
            rows.append([
                ParamSpec(f"r0", rf"$r_0$", value=0., start=0., end=1.)
            ])
        else:
            rows.append(
                [ParamSpec(f"r{n}", rf"$r_{n}", value=1.0 if n == 1 else 0.0, start=0., end=1.0)] + 
                [ParamSpec(f"phi{n}_{m}", rf"$\phi_{{{n}}}^{{{m}}}$", value=0.0, start=0, end=1.0) 
                    for m in range(0, 2*n+1)]
            )
    return rows

def _hyperspheric_transform(params: dict, structural: dict) -> dict:
    order = structural["order"]
    new_params = {}
    for n in range(order+1):
        if n == 0: 
            new_params['c0_0'] = params['r0']
        else:
            r = params[f'r{n}']
            phis = [params.get(f'phi{n}_{m}') for m in range(2*n+1)]
            phis = np.clip(np.array([phis[i] * np.pi for i in range(len(phis)-1)] + [phis[-1] * 2 * np.pi]), 1e-7, None)
            m_label = -n
            for m in range(2*n+1):
                new_params[f'c{n}_{m_label}'] = r * np.prod(np.sin(phis[:m])) * np.cos(phis[m])
                m_label += 1
    return new_params

HYPERSPHERIC_3D = DecompositionSpec(
    name = "Pairwse Polar Coefficients (3D)",
    structural={"order": (1, 5, 3), "fit to frame": True},
    build_params=_build_hyperspheric_params,
    transform=_hyperspheric_transform,
    render=_render_spherical,
)


def plot_hyperspherics_spectrambisonics():
    """Coefficient-by-coefficient explorer for a 3D ambisonic field.

    Untick *fit to frame* to see gain as well as shape : normalised, a field that
    is only being scaled -- a₀⁰ moved on its own, say -- is a sphere that never
    changes."""
    return DecompositionExplorer(HYPERSPHERIC_3D).layout





# --------------------------------------------------------------------------- #
#  Hyperspheric angle tracking                                                #
# --------------------------------------------------------------------------- #

from .lie import (BLUE, RED, GREEN, AMBER, PURPLE,
                                 _arrow, _frame_axes, _scene, _sphere, hat)

# --------------------------------------------------------------------------- #
#  The chart : hyperangles -> degree-1 coefficients                           #
# --------------------------------------------------------------------------- #

#: ACN order of a degree-1 block, and the axis each coefficient carries in SN3D.
ACN_M = (-1, 0, 1)
ACN_AXIS = ('y', 'z', 'x')
#: one colour per hyperangle. Blue/red stay reserved for the sign of the field,
#: as everywhere else in the package, so the knobs take the remaining three.
ANGLE_COLOR = (GREEN, PURPLE, AMBER)


def _m(m):
    """``-1``, ``0``, ``+1`` -- with a sign only where there is one to show."""
    return f"{m:+d}" if m else "0"


def angle_scales(redundant=True):
    """Radian span of each normalised slider : ``pi`` except the last one."""
    return np.array([np.pi, 2 * np.pi] if not redundant else [np.pi, np.pi, 2 * np.pi])


def to_radians(s, redundant=True):
    """Normalised slider values ``[0, 1]`` -> angles, as the explorer scales them."""
    return np.asarray(s, float)[:len(angle_scales(redundant))] * angle_scales(redundant)


def hyper_to_acn1(r, phis):
    """Degree-1 ACN coefficients ``(a_1^-1, a_1^0, a_1^+1)`` from ``r`` and angles.

    ``len(phis) == 3`` is the explorer's redundant chart (a trailing cosine on the
    last coefficient), ``len(phis) == 2`` the standard norm-preserving one."""
    phis = np.asarray(phis, float)
    c, prod = np.zeros(3), 1.0
    for k in range(3):
        c[k] = r * prod * (np.cos(phis[k]) if k < len(phis) else 1.0)
        if k < len(phis):
            prod *= np.sin(phis[k])
    return c


def jacobian_acn1(r, phis):
    """``dc/dphi`` of :func:`hyper_to_acn1`, shape ``(3, len(phis))``.

    Written as an explicit product rather than with ``cot phi_i`` so that it stays
    exact where a sine vanishes -- which is exactly where the interesting
    degeneracies are."""
    phis = np.asarray(phis, float)
    K = len(phis)
    Jac = np.zeros((3, K))
    for k in range(3):
        tail = np.cos(phis[k]) if k < K else 1.0     # the coefficient's own factor
        for i in range(K):
            if i > k:
                continue                              # phi_i does not appear in c_k
            if i == k:
                head = np.prod(np.sin(phis[:k]))
                Jac[k, i] = -r * head * np.sin(phis[k])
            else:
                head = np.prod([np.sin(phis[j]) for j in range(k) if j != i])
                Jac[k, i] = r * head * np.cos(phis[i]) * tail
    return Jac


def acn1_to_vector(c):
    """The dipole direction : ACN ``(m=-1, 0, +1)`` is ``(y, z, x)``, so reorder."""
    c = np.asarray(c, float)
    return np.array([c[2], c[0], c[1]])


def vector_to_acn1(v):
    """Inverse of :func:`acn1_to_vector`."""
    v = np.asarray(v, float)
    return np.array([v[1], v[2], v[0]])


def acn1_rotation_block(R3):
    """Degree-1 real-SH rotation block : the 3x3 with axes permuted to (y, z, x).

    Same object as ``torchspatialaudio.utils.sh_rotation._R1_from_cartesian`` --
    kept local so this script stands on its own."""
    p = [1, 2, 0]
    return np.asarray(R3, float)[np.ix_(p, p)]


def rodrigues(axis, angle):
    """``exp(angle * axis^)`` for a unit ``axis``."""
    K = hat(np.asarray(axis, float))
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# --------------------------------------------------------------------------- #
#  The incidence of one hyperangle : rotation vs gain                         #
# --------------------------------------------------------------------------- #

def generators(r, phis):
    """Per-angle decomposition of ``dv/dphi_k`` into a rotation and a gain.

    Returns one dict per angle with ``velocity``, ``radial`` (the gain rate,
    ``d|v|/dphi_k``), ``tangential``, ``omega`` (angular velocity), ``rate``
    (``|omega|``, in radians of field rotation per radian of knob) and ``axis``."""
    c = hyper_to_acn1(r, phis)
    v = acn1_to_vector(c)
    Jv = np.stack([acn1_to_vector(col) for col in jacobian_acn1(r, phis).T], axis=1)
    rho = np.linalg.norm(v)
    out = []
    for k in range(Jv.shape[1]):
        d = Jv[:, k]
        if rho < 1e-12:                       # a collapsed field has no direction
            out.append(dict(velocity=d, radial=0.0, tangential=d * 0,
                            omega=np.zeros(3), rate=0.0, axis=np.zeros(3)))
            continue
        vhat = v / rho
        radial = float(vhat @ d)
        omega = np.cross(v, d) / rho ** 2
        rate = float(np.linalg.norm(omega))
        out.append(dict(velocity=d, radial=radial, tangential=d - radial * vhat,
                        omega=omega, rate=rate,
                        axis=omega / rate if rate > 1e-12 else np.zeros(3)))
    return out


def sweep(r, s, k, redundant=True, n=241):
    """Sweep the ``k``-th normalised angle over its whole range, others frozen.

    Returns ``(s_k, V, rates, radials)`` : the swept values, the curve ``v(s_k)``
    the dipole traces, and the two rates along it."""
    s = list(s)
    grid = np.linspace(0.0, 1.0, n)
    V, rates, radials = [], [], []
    for t in grid:
        s_t = list(s)
        s_t[k] = float(t)
        phis = to_radians(s_t, redundant)
        V.append(acn1_to_vector(hyper_to_acn1(r, phis)))
        g = generators(r, phis)[k]
        rates.append(g['rate'] * angle_scales(redundant)[k])      # per slider unit
        radials.append(g['radial'] * angle_scales(redundant)[k])
    return grid, np.asarray(V), np.asarray(rates), np.asarray(radials)


def finite_check(r, s, k, redundant=True, delta=1e-4):
    """Compare the generator against the finite motion of the same knob.

    ``angle/step`` is the angle of the *exact* minimal rotation taking ``v`` to
    the stepped ``v``, divided by the step -- it must agree with ``rate``.
    ``gain`` is the norm ratio : anything but 1 means the knob is not a rotation.
    ``rot_residual`` is how far the pure rotation ``exp(delta * omega^) v`` lands
    from the true ``v(phi + delta)``, per unit step : it converges to ``|radial|``,
    i.e. it measures exactly the gain the knob leaks.  ``full_residual`` adds the
    gain back and must vanish with the step, confirming the split is complete."""
    phis = to_radians(s, redundant)
    v0 = acn1_to_vector(hyper_to_acn1(r, phis))
    step = np.zeros(len(phis))
    step[k] = delta
    v1 = acn1_to_vector(hyper_to_acn1(r, phis + step))
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    g = generators(r, phis)[k]
    cos = float(np.clip((v0 @ v1) / max(n0 * n1, 1e-15), -1.0, 1.0))
    turned = (rodrigues(g['axis'], g['rate'] * delta) @ v0) if g['rate'] > 1e-12 else v0
    gained = turned * (1.0 + delta * g['radial'] / max(n0, 1e-15))
    return dict(rate=g['rate'], angle_over_step=np.arccos(cos) / delta,
                gain=n1 / max(n0, 1e-15), radial=g['radial'],
                rot_residual=float(np.linalg.norm(turned - v1)) / max(delta, 1e-15),
                full_residual=float(np.linalg.norm(gained - v1)) / max(delta, 1e-15))


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #

_AZ, _ZE = np.mgrid[-np.pi:np.pi:61j, 0:np.pi:31j]
_DIRS = np.stack([np.sin(_ZE) * np.cos(_AZ),
                  np.sin(_ZE) * np.sin(_AZ),
                  np.cos(_ZE)], axis=-1)


#: the balloon is drawn at 60% of its true size, so the orbits -- which live on the
#: sphere of radius |v| -- stay outside it instead of being swallowed by it.
BALLOON_SCALE = 0.6


def _dipole_balloon(v, opacity=0.35):
    """The degree-1 field itself : ``r = |v . u|``, coloured by the sign of the lobe."""
    f = _DIRS @ np.asarray(v, float)
    rad = BALLOON_SCALE * np.abs(f)
    return go.Surface(x=rad * _DIRS[..., 0], y=rad * _DIRS[..., 1], z=rad * _DIRS[..., 2],
                      surfacecolor=np.sign(f), cmin=-1, cmax=1, opacity=opacity,
                      showscale=False, hoverinfo='skip', showlegend=False,
                      colorscale=[[0, BLUE], [1, RED]])


def _axis_line(axis, color, name, scale=1.35):
    """A rotation axis, drawn through the origin in both directions."""
    a = np.asarray(axis, float) * scale
    return go.Scatter3d(x=[-a[0], a[0]], y=[-a[1], a[1]], z=[-a[2], a[2]],
                        mode='lines', name=name, showlegend=True,
                        line=dict(width=4, color=color, dash='dash'),
                        hoverinfo='skip')


def figure_incidences(r, s, redundant=True, show_axes=True, show_orbits=True,
                      tangent_scale=0.25):
    """The main view : the dipole, and what each hyperangle does to it.

    Solid arrow = the field's pointing vector ``v``.  Coloured arrow = the
    velocity ``dv/dphi_k`` the knob imprints on it.  Dashed line = the axis of
    the rotation that velocity amounts to.  Thin curve = the orbit the knob
    sweeps out on its own."""
    phis = to_radians(s, redundant)
    c = hyper_to_acn1(r, phis)
    v = acn1_to_vector(c)
    gens = generators(r, phis)
    scales = angle_scales(redundant)

    # the reference sphere has radius r : an orbit that dips inside it is a knob
    # that is losing gain rather than rotating.
    traces = [_sphere(0.05, radius=max(r, 1e-3)), *_frame_axes(1.35), _dipole_balloon(v)]
    traces += _arrow((0, 0, 0), v, '#333333', 'v (pointing)', width=9)

    for k, g in enumerate(gens):
        color = ANGLE_COLOR[k]
        # per slider unit, not per radian : that is the quantity the knob shows
        rate, radial = g['rate'] * scales[k], g['radial'] * scales[k]
        tag = f"phi_{k}"
        if show_orbits:
            _, V, _, _ = sweep(r, s, k, redundant, n=181)
            traces.append(go.Scatter3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], mode='lines',
                                       name=f"orbit of {tag}", legendgroup=tag,
                                       line=dict(width=3, color=color), hoverinfo='skip'))
        traces += _arrow(v, tangent_scale * g['velocity'] * scales[k], color,
                         f"{tag} : {rate:.2f} rad rotation, {radial:+.2f} gain",
                         width=5)
        if show_axes and np.linalg.norm(g['axis']) > 0:
            traces.append(_axis_line(g['axis'], color, f"axis of {tag}"))

    fig = go.Figure(traces)
    coeff = "  ".join(f"a₁<sup>{_m(m)}</sup>={ci:+.3f}" for m, ci in zip(ACN_M, c))
    fig.update_layout(
        title=dict(text=f"degree-1 field : {coeff}   |v| = {np.linalg.norm(v):.3f}",
                   x=0.5, font=dict(size=13)),
        autosize=True, height=560, margin=dict(t=42, b=0, l=0, r=0),
        scene=_scene(1.4), uirevision='constant',
        legend=dict(x=0, y=1, font=dict(size=10)))
    return fig


def figure_sweeps(r, s, redundant=True, n=241):
    """Static summary : the orbits, and the rotation / gain rate along each of them.

    The middle panel is where the degeneracies show : the ``phi_1`` rate collapses
    to zero as ``phi_0`` approaches a pole, and (redundant chart) ``phi_2`` barely
    rotates anything at all while the bottom panel shows it draining the gain."""
    scales = angle_scales(redundant)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.62, 0.38],
                        specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                               [None, {"type": "xy"}]],
                        subplot_titles=("orbits of each hyperangle",
                                        "rotation rate |ω| (rad per slider unit)",
                                        "gain rate d|v| (per slider unit)"),
                        vertical_spacing=0.12)

    fig.add_trace(_sphere(0.05, radius=max(r, 1e-3)), row=1, col=1)
    for tr in _frame_axes(1.3):
        fig.add_trace(tr, row=1, col=1)
    v0 = acn1_to_vector(hyper_to_acn1(r, to_radians(s, redundant)))
    for tr in _arrow((0, 0, 0), v0, '#333333', 'v', width=8):
        fig.add_trace(tr, row=1, col=1)

    for k in range(len(scales)):
        color = ANGLE_COLOR[k]
        grid, V, rates, radials = sweep(r, s, k, redundant, n=n)
        fig.add_trace(go.Scatter3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], mode='lines',
                                   name=f"phi_{k}", legendgroup=f"phi_{k}",
                                   line=dict(width=4, color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=grid, y=rates, mode='lines', name=f"phi_{k}",
                                 legendgroup=f"phi_{k}", showlegend=False,
                                 line=dict(color=color)), row=1, col=2)
        fig.add_trace(go.Scatter(x=grid, y=radials, mode='lines', name=f"phi_{k}",
                                 legendgroup=f"phi_{k}", showlegend=False,
                                 line=dict(color=color)), row=2, col=2)

    chart = "redundant (explorer)" if redundant else "proper"
    fig.update_layout(title=dict(text=f"hyperangle incidences on degree 1 — {chart} chart, "
                                      f"r = {r:.2f}, s = {tuple(round(x, 2) for x in s)}",
                                 x=0.5, font=dict(size=13)),
                      height=620, width=1080, margin=dict(t=70, b=40, l=10, r=10),
                      scene=_scene(1.4), uirevision='constant',
                      legend=dict(x=0, y=1, font=dict(size=11)))
    fig.update_xaxes(title_text="slider value", row=2, col=2)
    fig.update_yaxes(rangemode='tozero', row=1, col=2)
    return fig


def _build_params(structural):
    """Row 1 : the radius.  Row 2 : the hyperangles, in normalised slider units."""
    n_angles = 3 if structural["redundant angle"] else 2
    return [
        [ParamSpec("r", r"$r$", value=1.0, start=0.0, end=1.0)],
        [ParamSpec(f"s{k}", rf"$\phi_{k}$", value=0.5 if k == 0 else 0.0,
                   start=0.0, end=1.0) for k in range(n_angles)],
    ]


def _transform(params, structural):
    """Sliders -> the geometry the figure is drawn from (the explorer's own idiom :
    the knobs the user turns and the quantities plotted need not be the same)."""
    n_angles = 3 if structural["redundant angle"] else 2
    return dict(r=params["r"], s=[params[f"s{k}"] for k in range(n_angles)])


def _render(geo, structural):
    return figure_incidences(geo["r"], geo["s"],
                             redundant=structural["redundant angle"],
                             show_axes=structural["rotation axes"],
                             show_orbits=structural["orbits"])


HYPERANGLE_DEGREE1 = DecompositionSpec(
    name="Hyperangle incidences on degree 1",
    structural={"redundant angle": True, "rotation axes": True, "orbits": True},
    build_params=_build_params,
    transform=_transform,
    render=_render,
)


def plot_hyperangles_degree1():
    """Interactive view of what each hyperangle does to the degree-1 field.

    Watch the ``phi_1`` arrow shrink to nothing as ``phi_0`` goes to 0 or 1 (the
    chart's gimbal lock), and untick *redundant angle* to see the third knob --
    the one that only shrinks the field -- disappear."""
    return DecompositionExplorer(HYPERANGLE_DEGREE1).layout



# --------------------------------------------------------------------------- #
#  Euler-angle non-commutativity demonstration                                #
# --------------------------------------------------------------------------- #

#: A few example sound fields, given as {(degree l, order m): coefficient}.
#: The first-order ones are axisymmetric (they reveal *pointing* differences);
#: the higher-order ones are not, so they also reveal *roll* differences.
EULER_FIELD_PRESETS = {
    "1st-order pointer (→ +x)": {(0, 0): 0.6, (1, 1): 1.0},
    "Tilted dipole (x + z)":    {(1, 1): 1.0, (1, 0): 1.0},
    "Quadrupole  Y₂²":          {(2, 2): 1.0},
    "Lopsided beam (mixed)":    {(0, 0): 0.4, (1, 1): 1.0, (2, 1): 0.7, (3, 2): 0.5},
    "Pointer + side flag":      {(1, 1): 1.0, (2, -2): 0.8},
}

# a shared camera so the three spheres start from the exact same viewpoint
_EULER_CAMERA = dict(eye=dict(x=1.7, y=1.7, z=1.1))


def _coeffs_from_dict(field):
    """Flatten a {(l, m): coeff} field into an ACN-ordered coefficient vector."""
    order = max(l for (l, _m) in field)
    coeffs = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            coeffs.append(float(field.get((l, m), 0.0)))
    return coeffs


def _field_surface(field, azimuth, zenith, title):
    """Small Plotly sphere (radius = |g|, colour = sign of g) for the triptych."""
    r = np.abs(field)
    x = r * np.sin(zenith) * np.cos(azimuth)
    y = r * np.sin(zenith) * np.sin(azimuth)
    z = r * np.cos(zenith)
    surface = go.Surface(
        x=x, y=y, z=z, surfacecolor=np.sign(field),
        colorscale=[[0, 'rgb(59,130,246)'], [1, 'rgb(239,68,68)']],
        cmin=-1, cmax=1, showscale=False,
    )
    axis = dict(range=[-1, 1], showbackground=True, showticklabels=False,
                backgroundcolor='rgb(230,230,230)', gridcolor='white',
                zerolinecolor='white', title='')
    fig = go.Figure(surface)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        width=360, height=360, margin=dict(t=34, b=0, l=0, r=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode='cube',
                   camera=_EULER_CAMERA),
        uirevision='constant',
    )
    return fig


def plot_euler_noncommutativity():
    """Interactive panel showing that applying the *same* two rotations in a
    different order yields a *different* rotated sound field -- because SO(3) is
    non-commutative. The two panels apply a yaw about Z and a pitch about Y in
    the two possible orders; the mismatch angle quantifies how far apart the two
    results are (it vanishes only in the degenerate/commuting cases)."""

    field_sel = pn.widgets.Select(name="sound field", options=list(EULER_FIELD_PRESETS))
    yaw = pn.widgets.FloatSlider(name="yaw α about Z (°)", start=0, end=360, step=1, value=90)
    pitch = pn.widgets.FloatSlider(name="pitch β about Y (°)", start=0, end=180, step=1, value=90)

    # plotting grid + the unit direction of every grid point (precomputed once)
    az = np.linspace(-np.pi, np.pi, 72)
    ze = np.linspace(0, np.pi, 37)
    AZ, ZE = np.meshgrid(az, ze, indexing='ij')
    dirs = np.stack([np.sin(ZE) * np.cos(AZ),
                     np.sin(ZE) * np.sin(AZ),
                     np.cos(ZE)], axis=-1).reshape(-1, 3)

    orig_pane = pn.pane.Plotly()
    left_pane = pn.pane.Plotly()
    right_pane = pn.pane.Plotly()
    mismatch = pn.pane.Markdown()

    def _rotate_field(coeffs, rot):
        # the value seen at output direction d is the source field read at R^{-1} d
        src = rot.apply(dirs, inverse=True).reshape(AZ.shape + (3,))
        src_az = np.arctan2(src[..., 1], src[..., 0])
        src_ze = np.arccos(np.clip(src[..., 2], -1.0, 1.0))
        return generate_harmonic(src_az, src_ze, *coeffs)

    def update(field_name, a_deg, b_deg):
        coeffs = _coeffs_from_dict(EULER_FIELD_PRESETS[field_name])
        Rz = Rotation.from_euler('z', np.radians(a_deg))
        Ry = Rotation.from_euler('y', np.radians(b_deg))
        R_left = Ry * Rz          # yaw first, then pitch
        R_right = Rz * Ry         # pitch first, then yaw

        orig_pane.object = _field_surface(
            generate_harmonic(AZ, ZE, *coeffs), AZ, ZE, "original")
        left_pane.object = _field_surface(
            _rotate_field(coeffs, R_left), AZ, ZE, "yaw → pitch   (Rᵧ·R𝓏)")
        right_pane.object = _field_surface(
            _rotate_field(coeffs, R_right), AZ, ZE, "pitch → yaw   (R𝓏·Rᵧ)")

        angle = np.degrees((R_left * R_right.inv()).magnitude())
        note = "they commute here 🡒 identical fields" if angle < 0.5 else \
               "the two fields are genuinely different orientations"
        mismatch.object = (
            f"### Orientation mismatch between the two orders: **{angle:.1f}°**\n"
            f"{note}. &nbsp; (Try α or β = 0° or 180° to reach a commuting case.)")

    pn.bind(update, field_sel, yaw.param.value_throttled, pitch.param.value_throttled,
            watch=True)
    update(field_sel.value, yaw.value, pitch.value)   # initial render

    return pn.Column(
        pn.pane.Markdown(
            "## Euler-angle non-commutativity\n"
            "The same yaw (about **Z**) and pitch (about **Y**) applied in the two "
            "possible orders give **different** rotated sound fields."),
        pn.Row(field_sel, yaw, pitch),
        pn.Row(orig_pane, left_pane, right_pane),
        mismatch,
    )