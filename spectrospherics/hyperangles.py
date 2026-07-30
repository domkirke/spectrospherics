"""What each hyperangle does to *one* degree, at any degree.

The degree-1 panel in :mod:`spectrospherics.spectrospherics` had it easy : the
three coefficients of ``l = 1`` *are* a vector, so every change of them at fixed
norm is a rotation, and the only question was which one.  From ``l = 2`` up that
stops being true, and this module is about what replaces it.

Count the dimensions.  A degree lives in ``R^(2l+1)``; the hyperspherical chart
gives it a radius and ``2l`` angles.  But SO(3) is only 3-dimensional, so the
orbit of a field under rotation is a 3-dimensional surface (2-dimensional if the
field happens to be axisymmetric) inside a space of dimension ``2l+1``.  For
``l = 1`` the sphere of radius ``r`` *is* the orbit -- 2 angles, 2-dimensional
orbit of the axisymmetric dipole, they match.  For ``l >= 2`` the chart has more
angles than SO(3) has directions to offer, so most of what a hyperangle does
cannot be a rotation at all : it *deforms* the pattern.

So the incidence of a knob splits into three orthogonal pieces.  With
``c`` the degree's coefficients, ``d = dc/dphi_k`` the knob's velocity, and
``G_x, G_y, G_z`` the so(3) generators in the real SH basis
(:func:`spectrospherics.wigner.real_generators`, so ``c -> D^l(R) c``) :

    gain        (c_hat . d) c_hat          changes the level, not the shape
    rotation    A omega,  A = [G_a c]      moves along the SO(3) orbit
    deformation d - gain - rotation        leaves the orbit : a new shape

``omega`` is the least-squares fit, i.e. the angular velocity of the rotation
that best accounts for the knob, and the three pieces are mutually orthogonal
(the generators are antisymmetric, so ``A`` is already orthogonal to ``c``),
which means their squared norms partition the knob's motion into percentages
that add to 100.

What comes out :

* ``l = 1`` : deformation is exactly zero and ``omega`` reduces to the
  cross-product formula of the degree-1 panel -- verified in :func:`self_check`.
* ``l >= 2`` : the rotation share collapses.  Only 3 directions out of ``2l`` can
  be rotations even in principle, and the chart is not aligned with them, so a
  typical knob is mostly deformation : turning it does not move the sound field,
  it *changes* it.
* the axisymmetric fields (``c`` along ``Y_l^0`` and its rotations) are where the
  orbit drops to 2 dimensions : one whole rotation direction dies, which is the
  higher-degree version of the degree-1 gimbal lock.
* the explorer's redundant last angle keeps behaving as it does at degree 1 : it
  mostly drains gain.

Usage
-----
    from spectrospherics.hyperangles import plot_hyperangles_degree
    plot_hyperangles_degree().servable()          # degree picked in the panel

    from spectrospherics.hyperangles import report, self_check
    report(l=3)                                   # the numbers, one degree at a time
    self_check()
"""

import functools

import numpy as np
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .explorer import DecompositionExplorer, DecompositionSpec, ParamSpec
from .lie import (BLUE, RED, GREEN, AMBER, PURPLE, GREY,
                  _arrow, _frame_axes, _scene, _sphere)
from .spectrospherics import BALLOON_SCALE, _axis_line
from .wigner import fibonacci_sphere, real_generators, sh_matrix

pn.extension('mathjax', 'plotly')

#: one colour per hyperangle, cycled if a degree has more knobs than colours.
KNOB_COLORS = (GREEN, PURPLE, AMBER, 'rgb(6,182,212)', 'rgb(217,70,239)',
               'rgb(132,204,22)', 'rgb(249,115,22)', 'rgb(99,102,241)',
               'rgb(20,184,166)', 'rgb(190,24,93)', 'rgb(163,163,163)')

#: colours of the three pieces a knob's motion splits into.
ROT_COLOR, GAIN_COLOR, DEF_COLOR = GREEN, GREY, RED

MAX_DEGREE = 5


def knob_color(k):
    return KNOB_COLORS[k % len(KNOB_COLORS)]


def knob_label(k):
    return f"φ{'₀₁₂₃₄₅₆₇₈₉'[k]}" if k < 10 else f"φ_{k}"


# --------------------------------------------------------------------------- #
#  The chart : hyperangles -> the 2l+1 coefficients of one degree             #
# --------------------------------------------------------------------------- #

def n_angles(l, redundant=True):
    """``2l`` angles parametrise ``2l+1`` coefficients; the explorer uses one more."""
    return 2 * l + (1 if redundant else 0)


def chart_scales(l, redundant=True):
    """Radian span of each normalised slider : ``pi`` everywhere, ``2 pi`` for the last."""
    scales = np.full(n_angles(l, redundant), np.pi)
    scales[-1] = 2 * np.pi
    return scales


def chart_angles(l, s, redundant=True):
    """Normalised slider values in ``[0, 1]`` -> angles, as the explorer scales them."""
    scales = chart_scales(l, redundant)
    return np.asarray(s, float)[:len(scales)] * scales


def hyper_to_acn(l, r, phis):
    """The ACN coefficients ``(a_l^-l ... a_l^+l)`` of one degree from ``r`` and angles.

    ``len(phis) == 2l`` is the standard norm-preserving chart, ``2l+1`` the
    explorer's redundant one (a trailing cosine on the last coefficient).  This is
    :func:`spectrospherics.spectrospherics.hyper_to_acn1` for any degree."""
    phis = np.asarray(phis, float)
    N, K = 2 * l + 1, len(phis)
    c, prod = np.zeros(N), 1.0
    for k in range(N):
        c[k] = r * prod * (np.cos(phis[k]) if k < K else 1.0)
        if k < K:
            prod *= np.sin(phis[k])
    return c


def jacobian_acn(l, r, phis):
    """``dc/dphi``, shape ``(2l+1, len(phis))``.

    Kept as an explicit product rather than ``cot phi_i`` so it stays exact where
    a sine vanishes -- which is where the degeneracies live."""
    phis = np.asarray(phis, float)
    N, K = 2 * l + 1, len(phis)
    Jac = np.zeros((N, K))
    for k in range(N):
        tail = np.cos(phis[k]) if k < K else 1.0
        for i in range(min(k, K - 1) + 1):
            if i == k:
                Jac[k, i] = -r * np.prod(np.sin(phis[:k])) * np.sin(phis[k])
            else:
                head = np.prod([np.sin(phis[j]) for j in range(k) if j != i])
                Jac[k, i] = r * head * np.cos(phis[i]) * tail
    return Jac


# --------------------------------------------------------------------------- #
#  The incidence of one hyperangle : rotation vs gain vs deformation          #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=MAX_DEGREE + 1)
def _generators(l):
    """``(3, 2l+1, 2l+1)`` stack of the so(3) generators acting on ACN coefficients."""
    return np.array(real_generators(l))


def orbit_tangent(l, c):
    """``A = [G_x c, G_y c, G_z c]``, whose column span is the tangent to the SO(3)
    orbit through ``c`` -- at most 3-dimensional, and always orthogonal to ``c``."""
    return np.stack([Ga @ c for Ga in _generators(l)], axis=1)


def orbit_dimension(l, c, tol=1e-8):
    """Rank of :func:`orbit_tangent` : 3 generically, 2 for an axisymmetric field
    (a whole rotation direction is stabilised), 0 for the zero field."""
    if np.linalg.norm(c) < tol:
        return 0
    sv = np.linalg.svd(orbit_tangent(l, c), compute_uv=False)
    return int(np.sum(sv > tol * max(sv[0], tol)))


def incidences(l, r, phis):
    """Split every hyperangle's velocity into gain + rotation + deformation.

    One dict per angle : ``velocity`` (``dc/dphi_k``), ``gain`` (the rate ``d|c|``),
    ``omega`` / ``rate`` / ``axis`` (the best-fit angular velocity, in radians of
    field rotation per radian of knob), ``deformation`` (the leftover vector) and
    ``shares``, the three orthogonal pieces as fractions of the knob's motion
    energy (they sum to 1)."""
    c = hyper_to_acn(l, r, phis)
    Jac = jacobian_acn(l, r, phis)
    A = orbit_tangent(l, c)
    rho = np.linalg.norm(c)
    out = []
    for k in range(Jac.shape[1]):
        d = Jac[:, k]
        speed2 = float(d @ d)
        if rho < 1e-12 or speed2 < 1e-24:
            out.append(dict(velocity=d, gain=0.0, omega=np.zeros(3), rate=0.0,
                            axis=np.zeros(3), rotation=d * 0, deformation=d * 0,
                            shares=(0.0, 0.0, 0.0), speed=np.sqrt(speed2)))
            continue
        chat = c / rho
        gain = float(chat @ d)
        tangential = d - gain * chat
        omega = np.linalg.lstsq(A, tangential, rcond=None)[0]
        rotation = A @ omega
        deformation = tangential - rotation
        rate = float(np.linalg.norm(omega))
        # the three pieces are mutually orthogonal, so their energies partition |d|^2
        shares = (float(rotation @ rotation) / speed2,
                  gain ** 2 / speed2,
                  float(deformation @ deformation) / speed2)
        out.append(dict(velocity=d, gain=gain, omega=omega, rate=rate,
                        axis=omega / rate if rate > 1e-12 else np.zeros(3),
                        rotation=rotation, deformation=deformation,
                        shares=shares, speed=np.sqrt(speed2)))
    return out


def chart_rotation_share(l, r, phis):
    """How much of the *whole* chart's motion is rotation, at this point.

    ``1`` means every knob is a rigid rotation of the field (only ``l = 1`` in the
    proper chart manages that); at high degree this drops towards ``3/(2l)`` and
    below, because that is all the room SO(3) leaves."""
    inc = incidences(l, r, phis)
    energy = sum(g['speed'] ** 2 for g in inc)
    if energy < 1e-24:
        return 0.0
    return sum(g['shares'][0] * g['speed'] ** 2 for g in inc) / energy


def sweep(l, r, s, k, redundant=True, n=161):
    """Sweep the ``k``-th normalised angle over its range, the others frozen.

    Returns the swept values, the tracked main-lobe direction, and the rotation /
    gain / deformation rates along the way (per slider unit, not per radian)."""
    s = list(s)
    grid = np.linspace(0.0, 1.0, n)
    coeffs, rates, gains, defs = [], [], [], []
    scale = chart_scales(l, redundant)[k]
    for t in grid:
        s_t = list(s)
        s_t[k] = float(t)
        phis = chart_angles(l, s_t, redundant)
        coeffs.append(hyper_to_acn(l, r, phis))
        g = incidences(l, r, phis)[k]
        rates.append(g['rate'] * scale)
        gains.append(g['gain'] * scale)
        defs.append(np.linalg.norm(g['deformation']) * scale)
    return (grid, lobe_track(l, np.asarray(coeffs)),
            np.asarray(rates), np.asarray(gains), np.asarray(defs))


# --------------------------------------------------------------------------- #
#  Following the pattern : where the main lobe points                         #
# --------------------------------------------------------------------------- #

_LOBE_DIRS = fibonacci_sphere(1200)


@functools.lru_cache(maxsize=MAX_DEGREE + 1)
def _lobe_basis(l):
    return sh_matrix(l, _LOBE_DIRS)


def main_lobe(l, c):
    """Direction of the strongest positive lobe of the degree-``l`` field.

    For ``l = 1`` this is the pointing vector itself; for higher degrees it is the
    only bit of "where the field looks" that survives, and following it is what
    makes a rotation visible on a plot."""
    f = _lobe_basis(l) @ np.asarray(c, float)
    return _LOBE_DIRS[int(np.argmax(f))]


def lobe_track(l, C, cap=0.9):
    """Main lobe along a sequence of coefficient vectors, tracked for continuity.

    A degree-``l`` pattern has up to ``l`` lobes of similar strength, so a plain
    argmax hops between them and draws a curve that is not there.  After the first
    point the search is restricted to a cap around the previous direction, which
    follows *one* lobe."""
    B = _lobe_basis(l)
    F = B @ np.asarray(C, float).T                     # (n_dirs, n_steps)
    track = [_LOBE_DIRS[int(np.argmax(F[:, 0]))]]
    for j in range(1, F.shape[1]):
        near = (_LOBE_DIRS @ track[-1]) > cap
        idx = np.flatnonzero(near)
        pick = idx[int(np.argmax(F[idx, j]))] if idx.size else int(np.argmax(F[:, j]))
        track.append(_LOBE_DIRS[pick])
    return np.asarray(track)


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #

_AZ, _ZE = np.mgrid[-np.pi:np.pi:61j, 0:np.pi:31j]
_GRID_DIRS = np.stack([np.sin(_ZE) * np.cos(_AZ),
                       np.sin(_ZE) * np.sin(_AZ),
                       np.cos(_ZE)], axis=-1)


@functools.lru_cache(maxsize=MAX_DEGREE + 1)
def _grid_basis(l):
    return sh_matrix(l, _GRID_DIRS.reshape(-1, 3)).reshape(_GRID_DIRS.shape[:2] + (2 * l + 1,))


def _degree_balloon(l, c, opacity=0.35, scale=BALLOON_SCALE):
    """The degree's own pattern : ``r = |f|``, coloured by the sign of the lobe."""
    f = _grid_basis(l) @ np.asarray(c, float)
    rad = scale * np.abs(f)
    return go.Surface(x=rad * _GRID_DIRS[..., 0], y=rad * _GRID_DIRS[..., 1],
                      z=rad * _GRID_DIRS[..., 2], surfacecolor=np.sign(f),
                      cmin=-1, cmax=1, opacity=opacity, showscale=False,
                      hoverinfo='skip', showlegend=False,
                      colorscale=[[0, BLUE], [1, RED]])


def figure_incidences(l, r, s, redundant=True, show_axes=True, show_orbits=True):
    """One degree, and what each of its hyperangles does to it.

    Left : the pattern, the main-lobe track each knob sweeps out, and the axis of
    the rotation each knob best approximates.  Right : the split of every knob's
    motion into rotation / gain / deformation.  The deformation bar is the one to
    watch -- it is empty at ``l = 1`` and dominant almost everywhere above."""
    phis = chart_angles(l, s, redundant)
    c = hyper_to_acn(l, r, phis)
    inc = incidences(l, r, phis)
    scales = chart_scales(l, redundant)
    lobe = main_lobe(l, c)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.63, 0.37],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("", "what each knob actually does"))

    fig.add_trace(_sphere(0.05, radius=max(r, 1e-3)), row=1, col=1)
    for tr in _frame_axes(1.3):
        fig.add_trace(tr, row=1, col=1)
    fig.add_trace(_degree_balloon(l, c), row=1, col=1)
    for tr in _arrow((0, 0, 0), lobe * max(np.linalg.norm(c), 1e-3), '#333333',
                     'main lobe', width=8):
        fig.add_trace(tr, row=1, col=1)

    rot, gain, deform, labels = [], [], [], []
    for k, g in enumerate(inc):
        color, tag = knob_color(k), knob_label(k)
        labels.append(tag)
        rot.append(100 * g['shares'][0])
        gain.append(100 * g['shares'][1])
        deform.append(100 * g['shares'][2])
        if show_orbits:
            _, track, _, _, _ = sweep(l, r, s, k, redundant, n=121)
            fig.add_trace(go.Scatter3d(x=track[:, 0], y=track[:, 1], z=track[:, 2],
                                       mode='lines', name=f"lobe under {tag}",
                                       legendgroup=tag, line=dict(width=3, color=color),
                                       hoverinfo='skip'), row=1, col=1)
        if show_axes and np.linalg.norm(g['axis']) > 0:
            fig.add_trace(_axis_line(g['axis'], color,
                                     f"{tag} : {g['rate'] * scales[k]:.2f} rad"),
                          row=1, col=1)

    for name, vals, color in (("rotation", rot, ROT_COLOR), ("gain", gain, GAIN_COLOR),
                              ("deformation", deform, DEF_COLOR)):
        fig.add_trace(go.Bar(x=labels, y=vals, name=name, marker_color=color,
                             hovertemplate="%{x} : %{y:.1f}% " + name + "<extra></extra>"),
                      row=1, col=2)

    dim = orbit_dimension(l, c)
    share = 100 * chart_rotation_share(l, r, phis)
    fig.update_layout(
        title=dict(text=f"degree {l} : |c| = {np.linalg.norm(c):.3f}, "
                        f"SO(3) orbit is {dim}-dimensional inside R<sup>{2 * l + 1}</sup>, "
                        f"chart is {share:.0f}% rotation",
                   x=0.5, font=dict(size=13)),
        autosize=True, height=560, margin=dict(t=60, b=10, l=0, r=0),
        scene=_scene(1.4), uirevision='constant', barmode='stack',
        legend=dict(x=0, y=1, font=dict(size=10)))
    fig.update_yaxes(title_text="% of the knob's motion", range=[0, 100], row=1, col=2)
    return fig


def figure_sweeps(l, r, s, redundant=True, n=161):
    """Static summary : the lobe tracks, and the three rates along each full sweep.

    At ``l = 1`` the deformation panel is flat zero.  Above it, it is usually the
    largest of the three -- the knob is reshaping the field, not turning it."""
    scales = chart_scales(l, redundant)
    fig = make_subplots(rows=3, cols=2, column_widths=[0.55, 0.45],
                        specs=[[{"type": "scene", "rowspan": 3}, {"type": "xy"}],
                               [None, {"type": "xy"}], [None, {"type": "xy"}]],
                        subplot_titles=("main lobe swept by each hyperangle",
                                        "rotation rate |ω| (rad per slider unit)",
                                        "gain rate d|c| (per slider unit)",
                                        "deformation rate (per slider unit)"),
                        vertical_spacing=0.09)

    fig.add_trace(_sphere(0.05, radius=1.0), row=1, col=1)
    for tr in _frame_axes(1.3):
        fig.add_trace(tr, row=1, col=1)
    c0 = hyper_to_acn(l, r, chart_angles(l, s, redundant))
    for tr in _arrow((0, 0, 0), main_lobe(l, c0), '#333333', 'main lobe', width=8):
        fig.add_trace(tr, row=1, col=1)

    for k in range(len(scales)):
        color, tag = knob_color(k), knob_label(k)
        grid, track, rates, gains, defs = sweep(l, r, s, k, redundant, n=n)
        fig.add_trace(go.Scatter3d(x=track[:, 0], y=track[:, 1], z=track[:, 2],
                                   mode='lines', name=tag, legendgroup=tag,
                                   line=dict(width=4, color=color)), row=1, col=1)
        for row, y in ((1, rates), (2, gains), (3, defs)):
            fig.add_trace(go.Scatter(x=grid, y=y, mode='lines', name=tag,
                                     legendgroup=tag, showlegend=False,
                                     line=dict(color=color)), row=row, col=2)

    chart = "redundant (explorer)" if redundant else "proper"
    fig.update_layout(
        title=dict(text=f"hyperangle incidences on degree {l} — {chart} chart, "
                        f"r = {r:.2f}", x=0.5, font=dict(size=13)),
        height=720, width=1080, margin=dict(t=70, b=40, l=10, r=10),
        scene=_scene(1.25), uirevision='constant',
        legend=dict(x=0, y=1, font=dict(size=11)))
    fig.update_xaxes(title_text="slider value", row=3, col=2)
    fig.update_yaxes(rangemode='tozero', row=1, col=2)
    fig.update_yaxes(rangemode='tozero', row=3, col=2)
    return fig


# --------------------------------------------------------------------------- #
#  The interactive panel : one degree at a time                                #
# --------------------------------------------------------------------------- #

#: how many angle sliders to put on one row before wrapping.
_PER_ROW = 6


def _build_params(structural):
    """Row 1 : the radius.  Then the hyperangles, wrapped into rows of six."""
    l = structural["degree"]
    K = n_angles(l, structural["redundant angle"])
    # keys carry the degree, so switching degree does not smear one degree's
    # angles onto another's sliders -- but returning to a degree restores them
    rows = [[ParamSpec(f"r{l}", r"$r$", value=1.0, start=0.0, end=1.0)]]
    specs = [ParamSpec(f"s{l}_{k}", rf"$\phi_{{{k}}}$",
                       value=0.5 if k == 0 else 0.0, start=0.0, end=1.0)
             for k in range(K)]
    rows += [specs[i:i + _PER_ROW] for i in range(0, len(specs), _PER_ROW)]
    return rows


def _transform(params, structural):
    l = structural["degree"]
    K = n_angles(l, structural["redundant angle"])
    return dict(l=l, r=params[f"r{l}"], s=[params[f"s{l}_{k}"] for k in range(K)])


def _render(geo, structural):
    return figure_incidences(geo["l"], geo["r"], geo["s"],
                             redundant=structural["redundant angle"],
                             show_axes=structural["rotation axes"],
                             show_orbits=structural["lobe tracks"])


HYPERANGLE_DEGREE = DecompositionSpec(
    name="Hyperangle incidences, one degree at a time",
    structural={"degree": (1, MAX_DEGREE, 2), "redundant angle": True,
                "rotation axes": True, "lobe tracks": True},
    build_params=_build_params,
    transform=_transform,
    render=_render,
)


def plot_hyperangles_degree():
    """What the hyperangles do to a single degree, for any degree up to 5.

    Set *degree* to 1 to recover the dipole panel : deformation stays empty and
    every knob is a rotation.  Raise it and watch the deformation bars take over --
    the chart has ``2l`` angles to spend and SO(3) only ever offers 3 directions.
    ``r`` and the angles are remembered per degree, so you can go back and forth."""
    return DecompositionExplorer(HYPERANGLE_DEGREE).layout


# --------------------------------------------------------------------------- #
#  Text report & self-check                                                    #
# --------------------------------------------------------------------------- #

def report(l=2, r=1.0, s=None, redundant=True):
    """Print the incidence of every hyperangle of degree ``l`` at one chart point."""
    K = n_angles(l, redundant)
    if s is None:                       # a generic-looking point, nothing special
        s = [0.35, 0.25, 0.15] + [0.2] * (K - 3)
    s = (list(s) + [0.0] * K)[:K]
    scales = chart_scales(l, redundant)
    phis = chart_angles(l, s, redundant)
    c = hyper_to_acn(l, r, phis)
    chart = "redundant (explorer)" if redundant else "proper"
    print(f"\ndegree {l} : {2 * l + 1} coefficients, {K} angles ({chart} chart), r = {r:g}")
    print("coeffs     : " + ", ".join(f"a{l}^{m:+d}={ci:+.3f}"
                                      for m, ci in zip(range(-l, l + 1), c)))
    print(f"|c| = {np.linalg.norm(c):.4f},  SO(3) orbit dimension = "
          f"{orbit_dimension(l, c)} (of {2 * l + 1}),  "
          f"chart rotation share = {100 * chart_rotation_share(l, r, phis):.1f}%")
    print("\n  knob   rot rate   axis (x, y, z)              gain rate   deform rate"
          "   rotation / gain / deformation")
    for k, g in enumerate(incidences(l, r, phis)):
        ax, sh = g['axis'], g['shares']
        print(f"  {knob_label(k):5s}  {g['rate'] * scales[k]:8.4f}   "
              f"({ax[0]:+.3f}, {ax[1]:+.3f}, {ax[2]:+.3f})   "
              f"{g['gain'] * scales[k]:+9.4f}   "
              f"{np.linalg.norm(g['deformation']) * scales[k]:11.4f}   "
              f"{100 * sh[0]:5.1f}% / {100 * sh[1]:5.1f}% / {100 * sh[2]:5.1f}%")


def report_all(r=1.0, redundant=True, degrees=range(1, MAX_DEGREE + 1)):
    """The same report for every degree, plus the trend the counting predicts."""
    for l in degrees:
        report(l, r, None, redundant)
    print("\nrotation share of the chart, degree by degree "
          "(3 SO(3) directions out of 2l angles) :")
    for l in degrees:
        K = n_angles(l, redundant)
        s = [0.35, 0.25, 0.15] + [0.2] * (K - 3)
        phis = chart_angles(l, s[:K], redundant)
        print(f"  l = {l} : {100 * chart_rotation_share(l, r, phis):5.1f}%   "
              f"(3 / 2l = {100 * min(1.0, 3 / (2 * l)):5.1f}%)")


def self_check(verbose=True):
    """Verify the chart, the split, and that degree 1 reproduces the dipole panel."""
    from .spectrospherics import (generators as generators_deg1, hyper_to_acn1,
                                  jacobian_acn1, acn1_to_vector, angle_scales)
    rng = np.random.default_rng(0)
    worst = dict(chart1=0.0, jac=0.0, split=0.0, brackets=0.0, deg1_omega=0.0,
                 deg1_deform=0.0, first_order=0.0)
    best = dict(deform_l2=0.0)

    for l in range(1, MAX_DEGREE + 1):
        G = _generators(l)
        # the generators must close under the so(3) brackets : [Gx, Gy] = Gz etc.
        for a, b, cc in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            worst['brackets'] = max(worst['brackets'], float(np.abs(
                G[a] @ G[b] - G[b] @ G[a] - G[cc]).max()))
        for redundant in (True, False):
            K = n_angles(l, redundant)
            for _ in range(20):
                r = float(rng.uniform(0.3, 1.0))
                phis = rng.uniform(0.15, np.pi - 0.15, K)
                phis[-1] *= 2 if redundant else 2
                c = hyper_to_acn(l, r, phis)

                # 1. the general chart is the degree-1 one when l == 1
                if l == 1:
                    worst['chart1'] = max(worst['chart1'], float(np.abs(
                        c - hyper_to_acn1(r, phis)).max()))
                    worst['chart1'] = max(worst['chart1'], float(np.abs(
                        jacobian_acn(l, r, phis) - jacobian_acn1(r, phis)).max()))
                    worst['chart1'] = max(worst['chart1'], float(np.abs(
                        chart_scales(1, redundant) - angle_scales(redundant)).max()))

                # 2. analytic Jacobian vs central differences
                h = 1e-6
                num = np.stack([(hyper_to_acn(l, r, phis + h * e) -
                                 hyper_to_acn(l, r, phis - h * e)) / (2 * h)
                                for e in np.eye(K)], axis=1)
                worst['jac'] = max(worst['jac'], float(np.abs(
                    num - jacobian_acn(l, r, phis)).max()))

                for k, g in enumerate(incidences(l, r, phis)):
                    d = g['velocity']
                    # 3. the split is complete and orthogonal : the energies partition
                    parts = g['rotation'] + g['gain'] * c / np.linalg.norm(c) + g['deformation']
                    worst['split'] = max(worst['split'], float(np.abs(parts - d).max()),
                                         abs(sum(g['shares']) - 1.0))
                    # 4. omega really generates the rotation part, to first order
                    eps = 1e-6
                    D = np.eye(2 * l + 1) + eps * np.tensordot(g['omega'], G, axes=1)
                    worst['first_order'] = max(worst['first_order'], float(np.abs(
                        (D @ c - c) / eps - g['rotation']).max()))
                    # 5. degree 1 : no deformation, and omega is the cross product
                    if l == 1:
                        worst['deg1_deform'] = max(worst['deg1_deform'],
                                                   float(np.abs(g['deformation']).max()))
                        w1 = generators_deg1(r, phis)[k]['omega']
                        worst['deg1_omega'] = max(worst['deg1_omega'],
                                                  float(np.abs(g['omega'] - w1).max()))
                    if l == 2:
                        best['deform_l2'] = max(best['deform_l2'],
                                                float(np.linalg.norm(g['deformation'])))
    if verbose:
        print("\nself-check")
        print(f"  general chart == degree-1 chart          : {worst['chart1']:.2e}")
        print(f"  analytic Jacobian vs finite differences  : {worst['jac']:.2e}")
        print(f"  so(3) brackets of the real generators    : {worst['brackets']:.2e}")
        print(f"  split is complete and orthogonal         : {worst['split']:.2e}")
        print(f"  omega generates the rotation part        : {worst['first_order']:.2e}")
        print(f"  degree 1 : deformation is exactly zero   : {worst['deg1_deform']:.2e}")
        print(f"  degree 1 : omega == cross-product omega  : {worst['deg1_omega']:.2e}")
        print(f"  degree 2 : deformation is NOT zero       : {best['deform_l2']:.2e} "
              f"(must be large)")
    assert max(worst.values()) < 1e-5, worst
    assert best['deform_l2'] > 1e-3, best
    return worst
