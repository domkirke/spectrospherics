"""Why a polar decomposition does not survive the jump to three dimensions.

Three interactions for ``1_3d_issues.ipynb``, each attached to one claim of the
notebook :

===============================  ===========================================
``plot_polar_2d_vs_3d``          the baseline : 2 = 1 + 1 works, 2l+1 = (2l-2) + 3 does not
``plot_dead_knob``               the stabilizer : a rotation control that does nothing
``plot_same_energy``             one modulus cannot be the shape
===============================  ===========================================
"""

import numpy as np
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.linalg import expm

from .lie import rot, hat, BLUE, RED, GREEN, GREY, AMBER, _formula
from .wigner import (real_wigner, real_wigner_z, sh_matrix, rotation_cloud,
                     real_generators, closest_rotation, _GRID_DIRS, _field_surface)

pn.extension('mathjax', 'plotly')


def _pair_norms(a, l):
    """The norms of the (-m, +m) pairs : what a yaw preserves, one per |m|."""
    return np.array([abs(a[l])] + [float(np.hypot(a[l - m], a[l + m]))
                                   for m in range(1, l + 1)])


def _best_alignment(l, a, b, n_starts=40, iters=30):
    """min over R of ||D(R) a - b||, and the rotated vector that achieves it."""
    val, D = closest_rotation(l, a, b, n_starts=n_starts, iters=iters)
    return val, D @ a


def _orbit_dimensions(l, a, tol=1e-6):
    """``(dim orbit, dead axis)`` at ``a``, measured rather than asserted.

    Turning ``a`` about the axis w moves it at the rate ``(sum_k w_k G_k) a``, so the
    three columns ``G_k a`` span the directions the orbit can go. Whatever they fail
    to span is an axis whose rotation does not move ``a`` at all -- the stabilizer,
    read off the last right-singular vector."""
    J = np.stack([g @ a for g in real_generators(l)], axis=-1)      # (2l+1, 3)
    s, Vt = np.linalg.svd(J)[1:]
    dim_orbit = int((s > max(s[0], 1e-12) * tol).sum())
    return dim_orbit, (Vt[-1] if dim_orbit < 3 else None)


def _ghost_surface(f, scale, grow=1.06):
    """The 'after' field as a translucent single-colour shell.

    Blown up by a few percent on purpose : when the knob is dead the two surfaces
    are the *same* surface, and coincident geometry would z-fight into a speckled
    mess instead of reading as one shape wrapped in the other."""
    ghost = _field_surface(f, scale, 'after', color=AMBER, opacity=.5, grow=grow)
    ghost.update(hoverinfo='skip')
    return ghost


def _segment(p, q, color, name, dash='solid', width=5, showlegend=True):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode='lines',
                        line=dict(color=color, width=width, dash=dash),
                        name=name, showlegend=showlegend, hoverinfo='name')


def _sphere_scene(camera=(1.6, 1.6, 1.0), rng=1.05):
    ax = dict(range=[-rng, rng], showticklabels=False, title='', showbackground=True,
              backgroundcolor='rgb(242,242,242)', gridcolor='white')
    return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube',
                camera=dict(eye=dict(x=camera[0], y=camera[1], z=camera[2])))


# --------------------------------------------------------------------------- #
#  1 -- the baseline : what exactly does 2D have that 3D has not               #
# --------------------------------------------------------------------------- #

def plot_polar_2d_vs_3d():
    """The 2D recipe and the 3D attempt, side by side, driven by the same knobs.

    Left : a circular mode. Its orbit under rotation is a *circle*, so one radius is
    the whole shape and one angle is the whole orientation -- 2 = 1 + 1.
    Right : a degree-l multiplet. Yaw alone still behaves (it rotates each (-m, +m)
    pair by m*alpha, l little 2D phases at once), but the moment beta leaves zero the
    pairs are mixed, and the only survivor is the total norm."""

    n = pn.widgets.IntSlider(name="2D : circular order n", start=1, end=5, value=3)
    l = pn.widgets.IntSlider(name="3D : degree ℓ", start=1, end=4, value=2)
    al = pn.widgets.FloatSlider(name="yaw α about z (°)  — both sides", start=0, end=360,
                                step=1, value=60)
    be = pn.widgets.FloatSlider(name="tilt β about y (°)  — 3D only", start=0, end=180,
                                step=1, value=0)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()
    theta = np.linspace(0, 2 * np.pi, 361)

    def update(n, l, al_deg, be_deg):
        a_deg, b_deg = np.radians(al_deg), np.radians(be_deg)

        # ---- 2D : c = r e^{i phi}, a rotation is exactly a phase shift ------
        r0, phi0 = 1.0, np.radians(25.)
        phi = phi0 + n * a_deg
        f2d = r0 * np.cos(n * theta - phi)
        circle = np.array([r0 * np.cos(theta), r0 * np.sin(theta)])

        # ---- 3D : the same experiment on a multiplet ------------------------
        a0 = np.zeros(2 * l + 1)
        a0[l], a0[2 * l] = .8, 1.                       # a zonal + sectoral mixture
        a0 /= np.linalg.norm(a0)
        yaw_only = real_wigner_z(l, a_deg) @ a0
        full = real_wigner(l, rot(2, a_deg) @ rot(1, b_deg)) @ a0
        f3d = sh_matrix(l, _GRID_DIRS) @ full
        cloud = rotation_cloud(l) @ a0
        C = cloud - cloud.mean(0)
        Vt = np.linalg.svd(C, full_matrices=False)[2]
        P, here = C @ Vt[:3].T, (full - cloud.mean(0)) @ Vt[:3].T

        fig = make_subplots(
            rows=2, cols=2, row_heights=[.5, .5], vertical_spacing=.11,
            specs=[[{"type": "polar"}, {"type": "scene"}],
                   [{"type": "xy"}, {"type": "scene"}]],
            subplot_titles=("2D : the mode on the circle", "3D : the field on the sphere",
                            "2D : its orbit is a circle — one radius",
                            "3D : its orbit is a ≤3-dimensional surface"))
        for sign, color in [(1, RED), (-1, BLUE)]:
            rr = np.where(np.sign(f2d) == sign, np.abs(f2d), np.nan)
            fig.add_trace(go.Scatterpolar(r=rr, theta=np.degrees(theta), mode='lines',
                                          line=dict(color=color, width=3),
                                          showlegend=False), row=1, col=1)
        fig.add_trace(_field_surface(f3d, max(np.abs(f3d).max(), 1e-9), 'field'), row=1, col=2)
        fig.add_trace(go.Scatter(x=circle[0], y=circle[1], mode='lines', showlegend=False,
                                 line=dict(color=GREY, width=2, dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[r0 * np.cos(phi)], y=[r0 * np.sin(phi)], mode='markers',
                                 marker=dict(size=13, color=AMBER), showlegend=False,
                                 hovertemplate="c = r e^{iφ}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[0, r0 * np.cos(phi)], y=[0, r0 * np.sin(phi)], mode='lines',
                                 line=dict(color=AMBER, width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',
                                   marker=dict(size=1.8, color=GREY, opacity=.45),
                                   showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter3d(x=[here[0]], y=[here[1]], z=[here[2]], mode='markers',
                                   marker=dict(size=7, color=AMBER), showlegend=False,
                                   name='here'), row=2, col=2)

        fig.update_layout(width=1080, height=760, uirevision='constant',
                          margin=dict(t=50, b=40, l=40, r=20),
                          polar=dict(radialaxis=dict(showticklabels=False, range=[0, 1.05]),
                                     angularaxis=dict(rotation=0)),
                          scene=_sphere_scene(),
                          scene2=dict(aspectmode='data',
                                      xaxis=dict(title='', showticklabels=False),
                                      yaxis=dict(title='', showticklabels=False),
                                      zaxis=dict(title='', showticklabels=False)))
        fig.update_xaxes(title="Re c", row=2, col=1, range=[-1.2, 1.2], zeroline=True)
        fig.update_yaxes(title="Im c", row=2, col=1, range=[-1.2, 1.2], zeroline=True)
        pane.object = fig

        pairs0, pairs_yaw = _pair_norms(a0, l), _pair_norms(yaw_only, l)
        pairs_full = _pair_norms(full, l)
        n_inv = 1 if l <= 1 else 2 * l - 2
        orbit = 2 if l == 1 else 3
        read.object = (
            "| | 2D : a mode | 3D : a degree |\n|---|---|---|\n"
            f"| numbers in it | 2 (Re c, Im c) | 2ℓ+1 = {2*l+1} |\n"
            f"| rotation group | SO(2), abelian | SO(3), **not** abelian |\n"
            f"| the orbit | a circle, dim 1 | dim {orbit} (≤ 3) |\n"
            f"| invariants (the shape) | 1 : the radius \\|c\\| | {n_inv} "
            f"{'(energy only)' if l <= 1 else '(energy + 2ℓ−3 nonlinear ones)'} |\n"
            f"| bookkeeping | 2 = 1 + 1 ✔ | {2*l+1} = {n_inv} + {orbit}"
            f"{' ✔' if n_inv + orbit == 2*l+1 else ' ✘'} |\n"
            f"| \\|c\\| / ‖a‖ after the yaw | {abs(r0 - r0):.1e} | "
            f"{abs(np.linalg.norm(a0) - np.linalg.norm(full)):.1e} |\n"
            f"| per-pair norms after **yaw only** | phase shifts by nα | "
            f"drift {np.abs(pairs0 - pairs_yaw).max():.1e} |\n"
            f"| per-pair norms after **yaw + tilt** | — | "
            f"drift **{np.abs(pairs0 - pairs_full).max():.3f}** |\n\n"
            "*Left, the whole 2D story in one picture : the coefficient lives on a circle, "
            "rotating slides it along that circle, and the radius never moves. Two numbers split "
            "cleanly into one shape and one orientation, so modulus and phase **are** the "
            "decomposition.*\n\n"
            "*Right, the same experiment. With β = 0 the analogy holds better than one might "
            f"expect : a yaw rotates each (−m, +m) pair by mα, so it is ℓ = {l} little 2D phases "
            "running at once and every pair norm is preserved (row 7). Now push **β** off zero : "
            "the pair norms start drifting (row 8), the orbit is a surface rather than a circle, "
            f"and the only quantity left standing is the single total norm. The {2*l+1} numbers "
            f"do split — into {n_inv} shape invariants and {orbit} orientation directions — but "
            "that is no longer one modulus and one angle, and no relabelling will make it so.*")

    pn.bind(update, n.param.value_throttled, l.param.value_throttled,
            al.param.value_throttled, be.param.value_throttled, watch=True)
    update(n.value, l.value, al.value, be.value)

    return pn.Column(
        pn.pane.Markdown("## The 2D recipe, and the same experiment in 3D"),
        pn.Row(n, l, al, be),
        pn.Row(pane, pn.Column(
            _formula(r"$$c = |c|\,e^{i\varphi} \quad\Longrightarrow\quad "
                     r"2 = \underbrace{1}_{\text{shape}} + \underbrace{1}_{\text{orientation}}"
                     r"\qquad\text{but}\qquad 2\ell+1 = "
                     r"\underbrace{(2\ell-2)}_{\text{shape}} + \underbrace{3}_{\text{orientation}}$$"),
            read)))


# --------------------------------------------------------------------------- #
#  2 -- the stabilizer : a control that does nothing                           #
# --------------------------------------------------------------------------- #

#: below this, a rotation is doing nothing at all to the coefficients
_DEAD_TOL = 1e-6

_KNOB_Z = "z — the chart's axis"
#: n is where the tilt aimed the shape ; for the axisymmetric shape that *is* its
#: symmetry axis, for the lopsided one it is only a reference direction.
_KNOB_N = "n — the tilted axis"


def _dead_knob_shape(l, kind):
    """The starting shape, before any tilt : axisymmetric about z, or lopsided."""
    if kind.startswith("axisymmetric"):
        a = np.zeros(2 * l + 1)
        a[l] = 1.                                        # Y_l^0
        return a
    a = np.random.default_rng(7).normal(size=2 * l + 1)
    return a / np.linalg.norm(a)


def plot_dead_knob():
    """The stabilizer, felt rather than defined.

    One shape, one rotation knob, and the question 'did anything happen ?'. The
    field before the turn is drawn solid, the field after it as an amber shell on
    top : when the knob is dead the shell sits exactly on the shape and nothing is
    visible but the tint. A body-fixed mark, drawn before and after with the arc it
    travelled, shows that the rotation really did take place -- so the field not
    moving is a fact about the *shape*, not a knob that failed to be turned.

    The two things worth trying. Take the axisymmetric shape and turn the z knob :
    dead, for every angle, the whole circle of rotations landing on one single
    coefficient vector. Then tilt the shape : the z knob wakes up, yet the knob
    about the shape's own axis n is dead exactly as before. The dead direction did
    not disappear, it *followed the shape* -- which is why no cleverer choice of
    Euler axes can get rid of it, and why dim O = 3 - dim Stab."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=5, value=3)
    kind = pn.widgets.RadioButtonGroup(
        name="the shape", options=["axisymmetric  Yℓ⁰", "lopsided"], value="axisymmetric  Yℓ⁰",
        button_style="outline", width=230)
    tilt = pn.widgets.FloatSlider(name="1. tilt the shape away from z (°)",
                                  start=0, end=90, step=1, value=0)
    axis = pn.widgets.RadioButtonGroup(name="knob axis", options=[_KNOB_Z, _KNOB_N],
                                       value=_KNOB_Z, button_style="outline", width=300)
    knob = pn.widgets.FloatSlider(name="2. turn the knob (°)", start=0, end=360,
                                  step=1, value=90)

    pane3d, pane2d, read = pn.pane.Plotly(), pn.pane.Plotly(), pn.pane.Markdown()
    sweep_deg = np.linspace(0, 360, 181)
    ez = np.array([0., 0., 1.])

    def update(l, kind, tilt_deg, axis_name, knob_deg):
        a0 = _dead_knob_shape(l, kind)
        Rt = rot(1, np.radians(tilt_deg))
        Dt = real_wigner(l, Rt)
        before = Dt @ a0                                 # the shape, tilted into place
        n = Rt @ ez                                      # the axis it was tilted onto
        phi = np.radians(knob_deg)

        # D is a homomorphism, so a turn about n is the turn about z conjugated by the
        # tilt : D(Rt Rz Rt^-1) = Dt D_z Dt^T. That keeps the whole 360 deg sweep below
        # in closed form instead of 181 least-squares fits.
        on_z = axis_name == _KNOB_Z
        k = ez if on_z else n
        after = real_wigner_z(l, phi) @ before if on_z else Dt @ real_wigner_z(l, phi) @ a0

        res_z, res_n = [], []
        for p in np.radians(sweep_deg):
            W = real_wigner_z(l, p)
            res_z.append(np.linalg.norm(W @ before - before))
            res_n.append(np.linalg.norm(Dt @ W @ a0 - before))
        res_z, res_n = np.array(res_z), np.array(res_n)

        Y = sh_matrix(l, _GRID_DIRS)
        f0, f1 = Y @ before, Y @ after
        scale = max(np.abs(f0).max(), np.abs(f1).max(), 1e-9)
        moved = float(np.linalg.norm(after - before))
        dead = moved < _DEAD_TOL
        dim_orbit, dead_axis = _orbit_dimensions(l, before)

        # ---- the sphere : one scene, everything superimposed ----------------
        mark0 = Rt @ np.array([1., 0., 0.])              # a mark painted on the shape
        Rk = expm(phi * hat(k))
        arc = 1.25 * np.array([expm(s * hat(k)) @ mark0
                               for s in np.linspace(0, phi, 80)])
        traces = [_field_surface(f0, scale, 'the field, before'),
                  _ghost_surface(f1, scale),
                  _segment(-1.3 * k, 1.3 * k, GREY, "the knob's axis", dash='dash', width=4)]
        if dead_axis is not None:
            traces.append(_segment(-1.15 * dead_axis, 1.15 * dead_axis, GREEN,
                                   "the dead axis (measured)", width=8))
        traces += [
            go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode='lines',
                         line=dict(color=AMBER, width=4, dash='dot'),
                         name='the turn the knob made', hoverinfo='name'),
            _segment([0, 0, 0], 1.25 * mark0, GREY, 'a mark on the shape, before', width=7),
            _segment([0, 0, 0], 1.25 * (Rk @ mark0), AMBER, '… the same mark, after', width=7),
        ]
        verdict = ("DEAD — the field did not move at all" if dead
                   else f"alive — the field moved by {moved:.3f}")
        fig3d = go.Figure(traces)
        fig3d.update_layout(
            width=620, height=520, uirevision='constant',
            title=dict(text=f"turning {knob_deg:.0f}° about {axis_name.split(' —')[0]} : "
                            f"<b>{verdict}</b><br>"
                            "<sub>solid = the field before · amber shell = the field after "
                            "(drawn 6 % larger so the two do not fight)</sub>",
                       x=.5, font=dict(size=13)),
            margin=dict(t=76, b=0, l=0, r=0), scene=_sphere_scene(rng=1.3),
            legend=dict(x=0, y=0, font=dict(size=10), bgcolor='rgba(255,255,255,.6)'))
        pane3d.object = fig3d

        # ---- the two diagnostics, kept in their own (xy-only) figure --------
        fig2d = make_subplots(rows=2, cols=1, vertical_spacing=.22,
                              subplot_titles=("how far the knob moves the field, over a whole turn",
                                              "the coefficients themselves"))
        for res, color, nm in [(res_z, BLUE, "about z"), (res_n, AMBER, "about n")]:
            fig2d.add_trace(go.Scatter(x=sweep_deg, y=res, mode='lines', name=nm,
                                       line=dict(color=color, width=3)), row=1, col=1)
        fig2d.add_trace(go.Scatter(x=[knob_deg], y=[moved], mode='markers', name='you are here',
                                   marker=dict(size=11, color=RED)), row=1, col=1)
        m = np.arange(-l, l + 1)
        fig2d.add_trace(go.Bar(x=m, y=before, name='before', marker_color=GREY), row=2, col=1)
        fig2d.add_trace(go.Bar(x=m, y=after, name='after', marker_color=BLUE), row=2, col=1)
        top = max(res_z.max(), res_n.max())
        for res, color, nm in [(res_z, BLUE, "z"), (res_n, AMBER, "n")]:
            if res.max() < _DEAD_TOL:                    # say it, rather than leaving a flat line
                fig2d.add_annotation(x=180, y=0, text=f"flat at zero : the {nm} knob is dead "
                                                      "for <i>every</i> angle",
                                     showarrow=False, yshift=14, row=1, col=1,
                                     font=dict(size=11, color=color))
        fig2d.update_xaxes(title="knob angle (°)", dtick=90, row=1, col=1)
        fig2d.update_yaxes(title="‖a after − a before‖", row=1, col=1,
                           range=[-.02 * max(top, .1), max(top * 1.15, .1)])
        fig2d.update_xaxes(title="m", dtick=1, row=2, col=1)
        fig2d.update_layout(width=560, height=520, barmode='group', uirevision='constant',
                            margin=dict(t=46, b=40, l=60, r=10),
                            legend=dict(orientation='h', y=1.14, x=0, font=dict(size=10)))
        pane2d.object = fig2d

        flat_z, flat_n = res_z.max() < _DEAD_TOL, res_n.max() < _DEAD_TOL
        angle_kn = np.degrees(np.arccos(np.clip(abs(float(k @ n)), 0, 1)))
        read.object = (
            "| | |\n|---|---|\n"
            f"| the knob moved the field by | **{moved:.2e}** |\n"
            f"| the most it moves it, anywhere in the 360° | about z : "
            f"**{res_z.max():.2e}**{' (dead everywhere)' if flat_z else ''} &nbsp;·&nbsp; "
            f"about n : **{res_n.max():.2e}**{' (dead everywhere)' if flat_n else ''} |\n"
            f"| dim 𝒪 / dim Stab, measured on the generators | **{dim_orbit} / "
            f"{3 - dim_orbit}** |\n"
            f"| the dead axis | "
            f"{'none — every rotation moves this shape' if dead_axis is None else np.round(dead_axis, 3)}"
            f"{'' if dead_axis is None else f', {np.degrees(np.arccos(np.clip(abs(float(dead_axis @ n)), 0, 1))):.1f}° away from n'} |\n"
            f"| the knob's axis is | {angle_kn:.0f}° away from n |\n\n"
            "*Start with the axisymmetric shape, tilt 0, and turn the z knob. The mark swings "
            "round — the rotation really happened — and the amber shell stays exactly on the "
            "shape. The blue curve says it is not a lucky angle : the field does not move for "
            "**any** amount of turn, so a whole circle of distinct rotations produces one single "
            "coefficient vector. A parameter spent on that knob has been spent on nothing.*\n\n"
            "*Now tilt the shape. The z knob wakes up (the blue curve lifts off zero), and the "
            "tempting conclusion is that the tilt destroyed the symmetry. It did not : switch the "
            "knob to **n**, the axis the tilt aimed the shape along, and it is dead again — the "
            "amber curve flat at zero for every angle, the green measured dead axis lying exactly "
            "on n, the two overlapping in the sphere. The shape is still "
            "axisymmetric — it simply is not symmetric about **z** any more. The dead direction "
            "was never a property of the coordinates ; it is carried by the shape, which is why "
            "no cleverer choice of Euler axes removes it.*\n\n"
            "*Switch to the lopsided shape and, for ℓ ≥ 2, nothing is dead at all : dim 𝒪 = 3, "
            "the generic case. (At ℓ = 1 there is no such thing — every dipole is a rotated "
            "Y₁⁰, so the stabilizer is 1-dimensional whatever you do.) That is the whole "
            "content of dim 𝒪 = 3 − dim Stab : the orbit is three-dimensional **except** on the "
            "symmetric shapes, where rotation coordinates quietly stop being coordinates.*")

    pn.bind(update, l.param.value_throttled, kind, tilt.param.value_throttled,
            axis, knob.param.value_throttled, watch=True)
    update(l.value, kind.value, tilt.value, axis.value, knob.value)

    return pn.Column(
        pn.pane.Markdown("## The dead knob : when a rotation control does nothing"),
        pn.Row(l, pn.Column(pn.pane.Markdown("the shape", margin=(0, 0, -10, 10)), kind),
               tilt),
        pn.Row(pn.Column(pn.pane.Markdown("the knob turns about…", margin=(0, 0, -10, 10)), axis),
               knob),
        pn.Row(pane3d, pane2d),
        _formula(r"$$\mathrm{Stab}(\mathbf{a}) = \{R : D^\ell(R)\,\mathbf{a} = \mathbf{a}\},"
                 r"\qquad \dim \mathcal{O} = 3 - \dim \mathrm{Stab}$$"),
        read)


# --------------------------------------------------------------------------- #
#  3 -- one modulus is not the shape                                           #
# --------------------------------------------------------------------------- #

def plot_same_energy():
    """Two fields with exactly the same energy, and no rotation between them.

    The second field starts as a rotated copy of the first, so a rotation aligns them
    perfectly. The morph slider then bends it -- keeping the norm *identical* -- and
    the best alignment over the whole rotation group degrades. Same modulus, different
    shape : one number cannot be the shape."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=2, end=5, value=2)
    morph = pn.widgets.FloatSlider(name="bend the second field (norm kept fixed)",
                                   start=0, end=1, step=.01, value=0.35)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, morph):
        rng = np.random.default_rng(4)
        a = rng.normal(size=2 * l + 1)
        a /= np.linalg.norm(a)
        u = rng.normal(size=2 * l + 1)
        u -= (u @ a) * a
        u /= np.linalg.norm(u)
        R = rot(2, .9) @ rot(1, 1.2) @ rot(2, .4)
        b = real_wigner(l, R) @ (np.cos(morph * np.pi / 2) * a + np.sin(morph * np.pi / 2) * u)

        gap, aligned = _best_alignment(l, a, b)

        fA = sh_matrix(l, _GRID_DIRS) @ a
        fB = sh_matrix(l, _GRID_DIRS) @ b
        fAl = sh_matrix(l, _GRID_DIRS) @ aligned
        scale = max(np.abs(fA).max(), np.abs(fB).max(), 1e-9)

        fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                            subplot_titles=("field A", "field B  (same ‖a‖)",
                                            "A, turned as close to B as possible"))
        for k, f in enumerate([fA, fB, fAl]):
            fig.add_trace(_field_surface(f, scale, ''), row=1, col=k + 1)
        fig.update_layout(width=1120, height=390, uirevision='constant',
                          margin=dict(t=44, b=0, l=0, r=0),
                          scene=_sphere_scene(), scene2=_sphere_scene(), scene3=_sphere_scene())
        pane.object = fig

        read.object = (
            "| | field A | field B |\n|---|---|---|\n"
            f"| ‖a‖ — the only 2D-style modulus | {np.linalg.norm(a):.10f} | "
            f"{np.linalg.norm(b):.10f} |\n"
            f"| best ‖D(R)a − b‖ over all rotations | — | **{gap:.4f}** |\n"
            f"| are they the same shape ? | — | "
            f"**{'yes — a rotation carries A onto B' if gap < 1e-5 else 'NO — no rotation does it'}** |\n\n"
            "*Both fields carry exactly the same energy, to ten decimals. At morph = 0 the second "
            "one really is the first, merely turned, and the search finds the rotation that "
            "brings them together — the third sphere lands on the second. Push the morph and the "
            "norm does not move by a digit while no rotation can bring them together any more : the two "
            "fields are genuinely different objects that no aiming can reconcile.*\n\n"
            "*That is the whole objection to a 3D modulus-and-phase. In 2D the radius labels the "
            "orbit completely — same radius means same mode up to a phase. Here 'same ‖a‖' says "
            f"almost nothing : for ℓ = {l} the orbit is pinned down by {2*l-2} invariants, and the "
            "norm is only the first of them. The rest are the nonlinear ones — for ℓ = 2, the "
            "prolate-versus-oblate contraction tr Q³ alongside the energy tr Q².*")

    pn.bind(update, l.param.value_throttled, morph.param.value_throttled, watch=True)
    update(l.value, morph.value)

    return pn.Column(
        pn.pane.Markdown("## Same energy, different shape : one modulus is not enough"),
        pn.Row(l, morph), pane,
        _formula(r"$$\Vert \mathbf{a}_\ell \Vert \text{ equal} \;\;\not\Longrightarrow\;\; "
                 r"\exists\, R : D^\ell(R)\,\mathbf{a} = \mathbf{b}$$"),
        read)
