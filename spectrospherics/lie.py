"""Interactive visualisations for the SO(3) / Lie-algebra notebook.

One panel per mathematical notion of ``B_so3_lie.ipynb`` :

======================================  ==========================================
``plot_tangent_space``                  §1  so(3) = antisymmetric matrices, w^ v = w x v
``plot_exponential_map``                §1  exp : so(3) -> SO(3), Rodrigues' formula
``plot_bracket``                        §2  [a^, b^] = (a x b)^, and why order matters
``plot_commutator_loop``                §3  K = Rx(e)Ry(d)Rx(-e)Ry(-d) = Rz(e*d) + O(3)
``plot_parallel_transport``             §4  holonomy on the sphere = enclosed area
``plot_hairy_ball``                     §4  no continuous section S^2 -> SO(3)
``plot_wigner_mixing``                  §5  D^l(a,b,g), L_z diagonal vs L_y mixing
``plot_loop_rephasing``                 §5  a closed pointing loop re-phases a multiplet
``plot_orbit_invariants``               §6  2l+1 = (invariants) + (orbit), for l = 2
======================================  ==========================================

Every function returns a Panel object : use ``.servable()`` in a served notebook,
or ``.show()`` to open it in a browser.
"""

import numpy as np
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import expm, logm

pn.extension('mathjax', 'plotly')


# --------------------------------------------------------------------------- #
#  a small shared so(3) toolbox                                               #
# --------------------------------------------------------------------------- #

BLUE, RED, GREEN, AMBER, PURPLE, GREY = ('rgb(59,130,246)', 'rgb(239,68,68)',
                                         'rgb(16,185,129)', 'rgb(245,158,11)',
                                         'rgb(139,92,246)', 'rgb(120,120,120)')


def hat(w):
    """The so(3) isomorphism  w -> w^ , with  w^ v = w x v ."""
    wx, wy, wz = w
    return np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]])


J = [hat(e) for e in np.eye(3)]                     # Jx, Jy, Jz


def bracket(A, B):
    return A @ B - B @ A


def rot(axis, angle):
    """exp(angle * J_axis), axis in {0, 1, 2}. Rodrigues, since J^3 = -J."""
    Ja = J[axis]
    return np.eye(3) + np.sin(angle) * Ja + (1 - np.cos(angle)) * (Ja @ Ja)


def rotvec(M):
    """axis x angle of a rotation matrix, read off log M in so(3)."""
    w = logm(M).real
    return np.array([w[2, 1], w[0, 2], w[1, 0]])


def unit(az_deg, el_deg):
    """Unit vector from azimuth / elevation in degrees."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def min_rotation(a, b):
    """The minimal rotation taking the unit vector ``a`` onto the unit vector ``b``
    (rotation in the plane they span). Undefined for b = -a."""
    k, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-12:
        raise ValueError("antipodal vectors : the minimal rotation is undefined")
    K = hat(k)
    return np.eye(3) + K + K @ K / (1 + c)


def angular_momentum(l):
    """The (2l+1)-dimensional irrep of so(3), |l,m> basis ordered m = l ... -l."""
    m = np.arange(l, -l - 1, -1.0)
    d = 2 * l + 1
    Lz = np.diag(m).astype(complex)
    Lp, Lm = np.zeros((d, d), complex), np.zeros((d, d), complex)
    for i in range(d):
        for k in range(d):
            if m[i] == m[k] + 1:
                Lp[i, k] = np.sqrt(l * (l + 1) - m[k] * (m[k] + 1))
            if m[i] == m[k] - 1:
                Lm[i, k] = np.sqrt(l * (l + 1) - m[k] * (m[k] - 1))
    return (Lp + Lm) / 2, (Lp - Lm) / 2j, Lz


def wigner_D(l, alpha, beta, gamma):
    """D^l(alpha, beta, gamma) = e^{-i a Lz} e^{-i b Ly} e^{-i g Lz}  (ZYZ)."""
    _, Ly, Lz = angular_momentum(l)
    return expm(-1j * alpha * Lz) @ expm(-1j * beta * Ly) @ expm(-1j * gamma * Lz)


# ---- plotly helpers -------------------------------------------------------- #

def _sphere(opacity=0.12, radius=1.0):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:31j]
    return go.Surface(x=radius * np.cos(u) * np.sin(v),
                      y=radius * np.sin(u) * np.sin(v),
                      z=radius * np.cos(v),
                      opacity=opacity, showscale=False, hoverinfo='skip',
                      showlegend=False, colorscale=[[0, 'grey'], [1, 'grey']])


def _arrow(origin, vec, color, name, width=6, head=0.22, showlegend=True, dash=None):
    """A 3D arrow = a line plus a cone head. Returns a list of traces."""
    origin, vec = np.asarray(origin, float), np.asarray(vec, float)
    tip = origin + vec
    n = np.linalg.norm(vec)
    traces = [go.Scatter3d(x=[origin[0], tip[0]], y=[origin[1], tip[1]],
                           z=[origin[2], tip[2]], mode='lines', name=name,
                           legendgroup=name, showlegend=showlegend,
                           line=dict(width=width, color=color, dash=dash))]
    if n > 1e-9:
        traces.append(go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]],
                              u=[head * vec[0]], v=[head * vec[1]], w=[head * vec[2]],
                              sizemode='absolute', sizeref=head * n, anchor='tip',
                              showscale=False, legendgroup=name, showlegend=False,
                              hoverinfo='skip',
                              colorscale=[[0, color], [1, color]]))
    return traces


def _frame_axes(scale=1.35):
    """The three (dotted) coordinate axes of the ambient space."""
    out = []
    for e, c, nm in zip(np.eye(3), ['#888', '#888', '#888'], ['x', 'y', 'z']):
        out.append(go.Scatter3d(x=[0, scale * e[0]], y=[0, scale * e[1]],
                                z=[0, scale * e[2]], mode='lines+text',
                                text=['', nm], textposition='top center',
                                line=dict(width=2, color=c, dash='dot'),
                                hoverinfo='skip', showlegend=False))
    return out


def _scene(rng=1.4, eye=(1.5, 1.5, 1.0)):
    ax = dict(range=[-rng, rng], showticklabels=False, title='',
              backgroundcolor='rgb(240,240,240)', gridcolor='white',
              zerolinecolor='white', showbackground=True)
    return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube',
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])))


def _formula(latex):
    """A static display formula.

    It must go through ``pn.pane.LaTeX`` and *not* through a Markdown pane :
    Panel renders Markdown with markdown-it, which eats TeX escapes before MathJax
    ever sees them (``\\,`` -> ``,``, ``\\;`` -> ``;``, ``\\{`` -> ``{``) and turns a
    pair of subscripts on one line into an <em> block. The live read-outs below are
    therefore written in plain Unicode, and only these fixed formulas are LaTeX."""
    return pn.pane.LaTeX(latex, styles={'font-size': '15px'}, width=520)


def _figure(traces, title, width=720, height=520, rng=1.4, eye=(1.5, 1.5, 1.0)):
    fig = go.Figure(traces)
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=13)),
                      width=width, height=height, margin=dict(t=42, b=0, l=0, r=0),
                      scene=_scene(rng, eye), uirevision='constant',
                      legend=dict(x=0, y=1, font=dict(size=11)))
    return fig


# --------------------------------------------------------------------------- #
#  a light entry point, for 1_3d_issues.ipynb                                  #
# --------------------------------------------------------------------------- #

#: a small rigid body : nose, wing, up -- enough to see both pointing and roll
_BODY = ((np.array([1., 0., 0.]), RED, "nose  +x"),
         (np.array([0., 1., 0.]), GREEN, "wing  +y"),
         (np.array([0., 0., 1.]), BLUE, "up  +z"))


def plot_angle_noncommutativity():
    """Two turns, taken in the two possible orders -- and the leftover of a round trip.

    The same x-turn and y-turn are applied in both orders : the object does not end up
    in the same place. Then the round trip x, y, -x, -y is played : it returns to the
    starting *instructions* but not to the starting orientation, and what is left over
    is a turn about the third axis -- the Lie bracket [Jx, Jy] = Jz, made visible.

    The full story (generators, BCH, holonomy) is in *B_so3_lie.ipynb*."""

    eps = pn.widgets.FloatSlider(name="ε : turn about x (°)", start=0, end=180, step=1, value=40)
    dlt = pn.widgets.FloatSlider(name="δ : turn about y (°)", start=0, end=180, step=1, value=40)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def _draw(fig, R, col, legend=False):
        for v, color, name in _BODY:
            for tr in _arrow([0, 0, 0], .85 * v, 'rgba(140,140,140,.55)', 'start',
                             width=3, head=.11, showlegend=False):
                fig.add_trace(tr, row=1, col=col)
            for tr in _arrow([0, 0, 0], R @ v, color, name, width=7, head=.2,
                             showlegend=legend):
                fig.add_trace(tr, row=1, col=col)

    def update(eps_deg, dlt_deg):
        e, d = np.radians(eps_deg), np.radians(dlt_deg)
        X, Y = rot(0, e), rot(1, d)
        first, second = X @ Y, Y @ X                       # the rightmost turn acts first
        loop = X @ Y @ rot(0, -e) @ rot(1, -d)

        fig = make_subplots(rows=1, cols=3, horizontal_spacing=.02,
                            specs=[[{"type": "scene"}] * 3],
                            subplot_titles=("Rx(ε) · Ry(δ)", "Ry(δ) · Rx(ε)",
                                            "there and back : Rx(ε)Ry(δ)Rx(−ε)Ry(−δ)"))
        for k, R in enumerate([first, second, loop]):
            _draw(fig, R, k + 1, legend=(k == 0))
        ax = dict(range=[-1.15, 1.15], showticklabels=False, title='', showbackground=True,
                  backgroundcolor='rgb(242,242,242)', gridcolor='white')
        scene = dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode='cube',
                     camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)))
        fig.update_layout(width=1040, height=380, uirevision='constant',
                          margin=dict(t=44, b=0, l=0, r=0),
                          scene=scene, scene2=scene, scene3=scene,
                          legend=dict(x=0, y=.05, font=dict(size=11)))
        pane.object = fig

        gap = np.degrees(np.linalg.norm(rotvec(first @ second.T)))
        w = rotvec(loop)
        residual = np.degrees(np.linalg.norm(w))
        axis = np.round(w / np.linalg.norm(w), 3) if residual > 1e-9 else "—"
        read.object = (
            "| | |\n|---|---|\n"
            f"| the two orders differ by | **{gap:.1f}°** |\n"
            f"| the round trip leaves | **{residual:.1f}°** about {axis} |\n"
            f"| ε·δ , the small-angle prediction | {np.degrees(e * d):.1f}° about z |\n\n"
            "*Both panels on the left were given the **same two turns** in a different order, and "
            "they disagree. On the right the object is turned and then turned back : the "
            "instructions cancel exactly, yet the grey starting arrows and the coloured ones no "
            "longer coincide — a leftover **spin about z**, out of two turns that never mentioned "
            "z.*\n\n"
            "*The first two numbers are always equal, and not by luck : the disagreement between "
            "the two orders **is** the round trip, since (RxRy)(RyRx)⁻¹ = Rx Ry Rx⁻¹ Ry⁻¹. "
            "Failing to commute and failing to close are one and the same fact.*\n\n"
            "*That leftover is what the Lie bracket measures. Rotations form a **Lie group** — a "
            "group whose elements can also be differentiated — and [A,B] = AB − BA is exactly the "
            "residue of the round trip taken by two infinitesimal turns. Bring both sliders down "
            "to 10–20° and the measurement meets the ε·δ prediction; open them up and it drifts, "
            "because the bracket is only the first term. It never vanishes for SO(3) "
            "([Jx, Jy] = Jz) — which is what a 2D phase never had to worry about : with a single "
            "axis there is nothing to fail to commute with. Set either slider to 0° (or both to "
            "180°) and everything collapses back to the commuting case.*")

    pn.bind(update, eps.param.value_throttled, dlt.param.value_throttled, watch=True)
    update(eps.value, dlt.value)

    return pn.Column(
        pn.Row(eps, dlt),
        pane,
        _formula(r"$$[A,B] = AB - BA \qquad R_x(\epsilon)R_y(\delta)R_x(-\epsilon)"
                 r"R_y(-\delta) \;\approx\; R_z(\epsilon\delta)$$"),
        read)


# --------------------------------------------------------------------------- #
#  §1 -- the tangent space :  an infinitesimal rotation is a cross product      #
# --------------------------------------------------------------------------- #

def plot_tangent_space():
    """so(3) as the tangent space at the identity.

    A curve R(t) = exp(t w^) through the identity moves a point v0 along a circle;
    its velocity is *always* w x v, i.e. the antisymmetric matrix w^ applied to v.
    The three visible consequences of antisymmetry are printed live : the velocity
    is orthogonal to the position (the norm is conserved -- the motion stays on the
    sphere) and orthogonal to the axis' component."""

    az = pn.widgets.FloatSlider(name="axis azimuth (°)", start=-180, end=180, value=35)
    el = pn.widgets.FloatSlider(name="axis elevation (°)", start=-90, end=90, value=40)
    t = pn.widgets.FloatSlider(name="time t  (angle swept, rad)", start=0, end=2 * np.pi,
                               step=.01, value=1.1)
    v_az = pn.widgets.FloatSlider(name="point v₀ azimuth (°)", start=-180, end=180, value=-60)
    v_el = pn.widgets.FloatSlider(name="point v₀ elevation (°)", start=-90, end=90, value=10)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(az, el, t, v_az, v_el):
        w = unit(az, el)
        A = hat(w)
        v0 = unit(v_az, v_el)
        v = expm(t * A) @ v0
        speed = A @ v                                     # = w x v
        orbit = np.array([expm(s * A) @ v0 for s in np.linspace(0, 2 * np.pi, 160)])

        traces = [_sphere()] + _frame_axes()
        traces += _arrow([0, 0, 0], w, GREEN, "axis ω  (the generator)")
        traces += _arrow([0, 0, 0], v, BLUE, "position  v(t) = exp(t ω̂) v₀")
        traces += _arrow(v, speed, RED, "velocity  v̇ = ω̂ v = ω × v")
        traces.append(go.Scatter3d(x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2],
                                   mode='lines', name='orbit of v₀',
                                   line=dict(width=3, color=BLUE, dash='dot')))
        traces.append(go.Scatter3d(x=[v0[0]], y=[v0[1]], z=[v0[2]], mode='markers',
                                   marker=dict(size=5, color='black'), name='v₀'))
        pane.object = _figure(traces, "the tangent space at the identity : v̇ = ω × v")

        R_t = expm(t * A)
        read.object = (
            "| check | value |\n|---|---|\n"
            f"| antisymmetry ‖Aᵀ + A‖ | {np.abs(A.T + A).max():.1e} |\n"
            f"| v · v̇   (the norm cannot change) | {float(v @ speed):.1e} |\n"
            f"| ω · v̇   (no motion along the axis) | {float(w @ speed):.1e} |\n"
            f"| ‖v(t)‖ | {np.linalg.norm(v):.6f} |\n"
            f"| det exp(tA) , ‖RᵀR − I‖ | {np.linalg.det(R_t):.6f} , "
            f"{np.abs(R_t.T @ R_t - np.eye(3)).max():.1e} |\n\n"
            "*Antisymmetry is exactly what makes the velocity orthogonal to the position : "
            "vᵀA v = 0 for every v, so d‖v‖²/dt = 0 and the motion is confined to the sphere. "
            "That is the infinitesimal version of RᵀR = I — the constraint of the group, "
            "differentiated. The generator ω is the only thing left standing : the whole "
            "tangent space at the identity is these three numbers.*")

    pn.bind(update, az.param.value_throttled, el.param.value_throttled,
            t.param.value_throttled, v_az.param.value_throttled,
            v_el.param.value_throttled, watch=True)
    update(az.value, el.value, t.value, v_az.value, v_el.value)

    return pn.Column(
        pn.pane.Markdown("## §1 — the tangent space : an infinitesimal rotation *is* a cross product"),
        pn.Row(pn.Column(az, el, t), pn.Column(v_az, v_el)),
        pn.Row(pane, pn.Column(
            _formula(r"$$\dot v \;=\; \hat\omega\, v \;=\; \omega \times v, "
                     r"\qquad \hat\omega^\top = -\hat\omega$$"), read)))


# --------------------------------------------------------------------------- #
#  §1 -- the exponential map and Rodrigues' formula                            #
# --------------------------------------------------------------------------- #

def plot_exponential_map():
    """Rodrigues' formula, drawn term by term.

    Split the vector into the part along the axis (untouched) and the part
    orthogonal to it (rotated inside its plane) :
    ``R v = v_par + cos(th) v_perp + sin(th) (n x v)`` -- which is exactly
    ``exp(th n^) = I + sin(th) n^ + (1 - cos(th)) n^^2`` applied to v."""

    az = pn.widgets.FloatSlider(name="axis azimuth (°)", start=-180, end=180, value=30)
    el = pn.widgets.FloatSlider(name="axis elevation (°)", start=-90, end=90, value=45)
    th = pn.widgets.FloatSlider(name="angle θ (rad)", start=0, end=2 * np.pi, step=.01, value=1.2)
    v_az = pn.widgets.FloatSlider(name="vector v azimuth (°)", start=-180, end=180, value=-70)
    v_el = pn.widgets.FloatSlider(name="vector v elevation (°)", start=-90, end=90, value=5)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(az, el, th, v_az, v_el):
        n = unit(az, el)
        N = hat(n)
        v = unit(v_az, v_el)
        v_par = float(n @ v) * n
        v_perp = v - v_par
        cross = np.cross(n, v)
        Rv = v_par + np.cos(th) * v_perp + np.sin(th) * cross
        R = expm(th * N)
        rodrigues = np.eye(3) + np.sin(th) * N + (1 - np.cos(th)) * (N @ N)

        circle = np.array([v_par + np.cos(s) * v_perp + np.sin(s) * cross
                           for s in np.linspace(0, 2 * np.pi, 160)])

        traces = [_sphere()] + _frame_axes()
        traces += _arrow([0, 0, 0], n, GREEN, "axis n")
        traces += _arrow([0, 0, 0], v, 'black', "v")
        traces += _arrow([0, 0, 0], v_par, PURPLE, "v∥ = (n·v) n   (invariant)")
        traces += _arrow(v_par, v_perp, BLUE, "v⊥   (rotated)")
        traces += _arrow(v_par, cross, AMBER, "n × v   (the other axis of the plane)")
        traces += _arrow([0, 0, 0], Rv, RED, "R v")
        traces.append(go.Scatter3d(x=circle[:, 0], y=circle[:, 1], z=circle[:, 2],
                                   mode='lines', name='the circle v⊥ travels',
                                   line=dict(width=3, color=RED, dash='dot')))
        pane.object = _figure(traces, "Rodrigues :  R v = v∥ + cos θ · v⊥ + sin θ · (n × v)")

        read.object = (
            "| check | value |\n|---|---|\n"
            f"| ‖exp(θn̂) − Rodrigues‖ | {np.abs(R - rodrigues).max():.1e} |\n"
            f"| ‖R v − (v∥ + cos θ · v⊥ + sin θ · n×v)‖ | {np.abs(R @ v - Rv).max():.1e} |\n"
            f"| angle between v and R v | "
            f"{np.degrees(np.arccos(np.clip(v @ Rv, -1, 1))):.2f}° |\n"
            f"| ‖n̂³ + n̂‖   (why the series closes) | {np.abs(N @ N @ N + N).max():.1e} |\n"
            f"| (n·v) , preserved by R | {float(n @ v):.6f} , {float(n @ Rv):.6f} |\n\n"
            "*The series exp(θn̂) = Σ θᵏ n̂ᵏ/k! only ever produces n̂ and n̂², because n̂³ = −n̂ : "
            "the odd powers resum into sin θ, the even ones into 1 − cos θ. Everything happens "
            "inside the single plane spanned by v⊥ and n×v, while v∥ never moves — which is why "
            "one axis and one angle are enough, and why θ → θ + 2π changes nothing : this "
            "one-parameter subgroup is a circle.*")

    pn.bind(update, az.param.value_throttled, el.param.value_throttled,
            th.param.value_throttled, v_az.param.value_throttled,
            v_el.param.value_throttled, watch=True)
    update(az.value, el.value, th.value, v_az.value, v_el.value)

    return pn.Column(
        pn.pane.Markdown("## §1 — the exponential map, term by term"),
        pn.Row(pn.Column(az, el, th), pn.Column(v_az, v_el)),
        pn.Row(pane, pn.Column(
            _formula(r"$$e^{\theta \hat n} \;=\; I + \sin\theta\, \hat n "
                     r"+ (1-\cos\theta)\,\hat n^2$$"), read)))


# --------------------------------------------------------------------------- #
#  §2 -- the bracket is the cross product, and it measures the disorder        #
# --------------------------------------------------------------------------- #

def plot_bracket():
    """[a^, b^] = (a x b)^ , and the two orderings of two finite rotations.

    The left half shows the algebra (the bracket of two generators is their cross
    product) ; the right half shows the group (the same two rotations applied in
    the two orders send a test point to two different places)."""

    a_az = pn.widgets.FloatSlider(name="a : azimuth (°)", start=-180, end=180, value=0)
    a_el = pn.widgets.FloatSlider(name="a : elevation (°)", start=-90, end=90, value=0)
    a_n = pn.widgets.FloatSlider(name="‖a‖  (rotation angle, rad)", start=0, end=np.pi, step=.01, value=1.0)
    b_az = pn.widgets.FloatSlider(name="b : azimuth (°)", start=-180, end=180, value=90)
    b_el = pn.widgets.FloatSlider(name="b : elevation (°)", start=-90, end=90, value=0)
    b_n = pn.widgets.FloatSlider(name="‖b‖  (rotation angle, rad)", start=0, end=np.pi, step=.01, value=1.0)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(a_az, a_el, a_n, b_az, b_el, b_n):
        a, b = a_n * unit(a_az, a_el), b_n * unit(b_az, b_el)
        c = np.cross(a, b)
        Ra, Rb = expm(hat(a)), expm(hat(b))
        u0 = np.array([0., 0., 1.])
        p_ab, p_ba = Ra @ Rb @ u0, Rb @ Ra @ u0
        mismatch = rotvec(Ra @ Rb @ (Rb @ Ra).T)

        traces = [_sphere()] + _frame_axes()
        traces += _arrow([0, 0, 0], a, BLUE, "a  (generator)")
        traces += _arrow([0, 0, 0], b, GREEN, "b  (generator)")
        traces += _arrow([0, 0, 0], c, AMBER, "a × b   =   [â, b̂]")
        traces += _arrow([0, 0, 0], u0, 'black', "test point u₀")
        traces += _arrow([0, 0, 0], p_ab, RED, "Rₐ R_b u₀")
        traces += _arrow([0, 0, 0], p_ba, PURPLE, "R_b Rₐ u₀", dash='dash')
        pane.object = _figure(traces, "the bracket (algebra) and the two orderings (group)")

        ab = hat(a) @ hat(b)
        read.object = (
            "| check | value |\n|---|---|\n"
            f"| ‖[â,b̂] − (a×b)^‖ | {np.abs(bracket(hat(a), hat(b)) - hat(c)).max():.1e} |\n"
            f"| ‖âb̂ + (âb̂)ᵀ‖   (the *product* leaves the algebra) | "
            f"{np.abs(ab + ab.T).max():.3f} |\n"
            f"| ‖[â,b̂] + [â,b̂]ᵀ‖   (the *bracket* does not) | "
            f"{np.abs(bracket(hat(a), hat(b)) + bracket(hat(a), hat(b)).T).max():.1e} |\n"
            f"| ‖a × b‖ | {np.linalg.norm(c):.4f} |\n"
            f"| angle between Rₐ R_b u₀ and R_b Rₐ u₀ | "
            f"{np.degrees(np.arccos(np.clip(p_ab @ p_ba, -1, 1))):.2f}° |\n"
            f"| the mismatch rotation Rₐ R_b (R_b Rₐ)⁻¹ : angle | "
            f"{np.degrees(np.linalg.norm(mismatch)):.2f}° |\n\n"
            "*Align **a** and **b** (same azimuth and elevation) and everything collapses at "
            "once : a × b = 0, the bracket vanishes, and the two orderings agree — that is the "
            "abelian one-axis subgroup, the 2D case. Separate the axes and the cross product "
            "grows, and so does the disagreement : the bracket is not a metaphor for the "
            "disorder, it is its measure. Note also that âb̂ is not antisymmetric — the product "
            "of two generators is not a generator — while the bracket always is : that is why "
            "the algebra is closed under [·,·] and not under the matrix product.*")

    pn.bind(update, a_az.param.value_throttled, a_el.param.value_throttled,
            a_n.param.value_throttled, b_az.param.value_throttled,
            b_el.param.value_throttled, b_n.param.value_throttled, watch=True)
    update(a_az.value, a_el.value, a_n.value, b_az.value, b_el.value, b_n.value)

    return pn.Column(
        pn.pane.Markdown("## §2 — the bracket : the cross product, and the price of disorder"),
        pn.Row(pn.Column(a_az, a_el, a_n), pn.Column(b_az, b_el, b_n)),
        pn.Row(pane, pn.Column(
            _formula(r"$$[\hat a, \hat b] \;=\; \hat a \hat b - \hat b \hat a "
                     r"\;=\; \widehat{a \times b}$$"), read)))


# --------------------------------------------------------------------------- #
#  §3 -- the commutator loop : the demonstration itself                        #
# --------------------------------------------------------------------------- #

_SCAL_EPS = np.logspace(-3, -0.1, 40)


def _loop(eps, dlt):
    return rot(0, eps) @ rot(1, dlt) @ rot(0, -eps) @ rot(1, -dlt)


_SCAL_ANG = np.array([np.linalg.norm(rotvec(_loop(e, e))) for e in _SCAL_EPS])


def plot_commutator_loop():
    """The demonstration of §3, made interactive.

    Drive the closed rectangle ``+eps x, +dlt y, -eps x, -dlt y`` and watch the
    residual rotation. Two things can be tuned : the size of the loop (the residual
    follows the *area* eps*dlt, not the perimeter), and the subdivision n -- driving
    the n-times smaller loop n^2 times converges to exactly exp([A,B])."""

    eps = pn.widgets.FloatSlider(name="ε  : x-leg (rad)", start=0.0, end=1.5, step=.01, value=0.6)
    dlt = pn.widgets.FloatSlider(name="δ  : y-leg (rad)", start=0.0, end=1.5, step=.01, value=0.6)
    sub = pn.widgets.IntSlider(name="subdivision n  (loop /n, driven n² times)", start=1, end=12, value=1)
    u_az = pn.widgets.FloatSlider(name="test direction : azimuth (°)", start=-180, end=180, value=0)
    u_el = pn.widgets.FloatSlider(name="test direction : elevation (°)", start=-90, end=90, value=0)

    pane3d, pane2d, read = pn.pane.Plotly(), pn.pane.Plotly(), pn.pane.Markdown()

    def update(eps, dlt, sub, u_az, u_el):
        u0 = unit(u_az, u_el)
        e, d = eps / sub, dlt / sub
        legs = [(1, -d), (0, -e), (1, +d), (0, +e)]       # rightmost factor first

        C, path, corners = np.eye(3), [], []
        for _ in range(sub * sub):                        # drive the small loop n² times
            for axis, angle in legs:
                for s in np.linspace(0, angle, max(6, int(60 / sub))):
                    path.append(expm(s * J[axis]) @ C @ u0)
                C = expm(angle * J[axis]) @ C
                corners.append(C @ u0)
        path, corners = np.array(path), np.array(corners)

        K = C
        w = rotvec(K)
        angle = np.linalg.norm(w)
        target = expm(eps * dlt * J[2])                   # exp([A,B])
        A, B = eps * J[0], dlt * J[1]
        bch3 = bracket(A, B) + .5 * bracket(A + B, bracket(A, B))

        traces = [_sphere()] + _frame_axes()
        traces.append(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines',
                                   name='trajectory of u₀', line=dict(width=5, color=BLUE)))
        traces.append(go.Scatter3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                                   mode='markers', name='leg ends',
                                   marker=dict(size=3, color=BLUE)))
        traces += _arrow([0, 0, 0], u0, 'black', "start  u₀")
        traces += _arrow([0, 0, 0], K @ u0, RED, "end  K u₀")
        traces += _arrow([0, 0, 0], target @ u0, GREEN, "exp([A,B]) u₀", dash='dash')
        if angle > 1e-9:
            n = w / angle
            traces.append(go.Scatter3d(x=[-n[0], n[0]], y=[-n[1], n[1]], z=[-n[2], n[2]],
                                       mode='lines', name='residual axis  log K',
                                       line=dict(width=4, color=AMBER, dash='dash')))
        pane3d.object = _figure(traces, f"n² = {sub*sub} turn(s) around the loop of area ε·δ/n²",
                                width=640, height=520)

        fig = go.Figure([
            go.Scatter(x=_SCAL_EPS, y=_SCAL_ANG, mode='markers', name='‖log K(ε,ε)‖',
                       marker=dict(size=5, color=BLUE)),
            go.Scatter(x=_SCAL_EPS, y=_SCAL_EPS ** 2, mode='lines', name='ε² (the area law)',
                       line=dict(color=RED, dash='dash')),
            go.Scatter(x=[max(eps, 1e-3)], y=[max(angle, 1e-12)], mode='markers',
                       name='current setting', marker=dict(size=12, color=AMBER, symbol='x')),
        ])
        fig.update_layout(width=430, height=430, uirevision='constant',
                          margin=dict(t=42, b=40, l=40, r=10),
                          title=dict(text="the residual follows the area, not the perimeter",
                                     x=.5, font=dict(size=13)),
                          xaxis=dict(type='log', title='ε = δ (rad)'),
                          yaxis=dict(type='log', title='residual angle (rad)'),
                          legend=dict(x=0, y=1, font=dict(size=10)))
        pane2d.object = fig

        ratio = (f"{angle / (eps * dlt):.5f}" if eps * dlt > 1e-12 else
                 "— (a leg is zero : nothing left to fail to commute)")
        axis_txt = str(np.round(w / angle, 4)) if angle > 1e-9 else "—"
        read.object = (
            "| quantity | value |\n|---|---|\n"
            f"| residual angle ‖log K‖ | {angle:.6f} rad  ({np.degrees(angle):.3f}°) |\n"
            f"| predicted εδ | {eps*dlt:.6f} rad  ({np.degrees(eps*dlt):.3f}°) |\n"
            f"| ratio | {ratio} |\n"
            f"| residual axis | {axis_txt} |\n"
            f"| ‖log K − [A,B]‖ | {np.abs(logm(K).real - bracket(A, B)).max():.2e} |\n"
            f"| ‖log K − ([A,B] + ½[A+B,[A,B]])‖ | {np.abs(logm(K).real - bch3).max():.2e} |\n"
            f"| ‖K − exp([A,B])‖ | {np.abs(K - target).max():.2e} |\n"
            f"| gap between u₀ and K u₀ | "
            f"{np.degrees(np.arccos(np.clip(u0 @ (K @ u0), -1, 1))):.3f}° |\n\n"
            "*Halve ε **and** δ and the residual is divided by four : it is an **area**. "
            "Push the subdivision slider and the last error rows collapse — the n²-th power of "
            "the n-times smaller loop converges to exp([A,B]) exactly, which is the precise "
            "sense in which the bracket *is* the non-commutativity. Set either leg to zero and "
            "everything vanishes : one axis alone commutes with itself.*")

    pn.bind(update, eps.param.value_throttled, dlt.param.value_throttled,
            sub.param.value_throttled, u_az.param.value_throttled,
            u_el.param.value_throttled, watch=True)
    update(eps.value, dlt.value, sub.value, u_az.value, u_el.value)

    return pn.Column(
        pn.pane.Markdown("## §3 — the commutator loop : a closed set of commands, an open rotation"),
        pn.Row(pn.Column(eps, dlt, sub), pn.Column(u_az, u_el)),
        pn.Row(pane3d, pane2d),
        _formula(r"$$K = e^{A}e^{B}e^{-A}e^{-B} = I + [A,B] + O(\eta^3), \qquad "
                 r"[A,B] = \varepsilon\delta\, J_z$$"),
        read)


# --------------------------------------------------------------------------- #
#  §4 -- holonomy : parallel transport around a spherical rectangle            #
# --------------------------------------------------------------------------- #

def _transport(path, v0):
    """Discrete Levi-Civita transport of the tangent vector ``v0`` along a path of
    unit vectors : at each step, apply the minimal rotation taking u_k to u_{k+1}."""
    v, out = np.array(v0, float), [np.array(v0, float)]
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        if float(np.dot(a, b)) < 1 - 1e-14:
            v = min_rotation(a, b) @ v
        out.append(v.copy())
    return np.array(out)


def plot_parallel_transport():
    """The same residual rotation, seen as the curvature of the sphere.

    Transport a tangent vector around a closed latitude/longitude rectangle : it
    comes back rotated by the *enclosed area*, exactly as the commutator loop of §3
    came back rotated by the area eps*dlt of its command rectangle."""

    th0 = pn.widgets.FloatSlider(name="start colatitude θ₀ (°)", start=5, end=170, value=60)
    dth = pn.widgets.FloatSlider(name="Δθ  (°)", start=-80, end=80, value=35)
    dph = pn.widgets.FloatSlider(name="Δφ  (°)", start=-180, end=180, value=60)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def _pt(theta, phi):
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    def update(th0, dth, dph):
        t0, dt, dp = np.radians(th0), np.radians(dth), np.radians(dph)
        t1 = np.clip(t0 + dt, 1e-3, np.pi - 1e-3)
        N = 220                                          # fine enough for a clean 1e-6 residual
        path = np.concatenate([
            [_pt(t0, p) for p in np.linspace(0, dp, N)],
            [_pt(t, dp) for t in np.linspace(t0, t1, N)],
            [_pt(t1, p) for p in np.linspace(dp, 0, N)],
            [_pt(t, 0) for t in np.linspace(t1, t0, N)]])

        base = path[0]
        # a unit tangent at the base point : the local "east" direction
        east = np.cross([0, 0, 1.], base)
        east /= np.linalg.norm(east)
        vs = _transport(path, east)
        v_end = vs[-1]

        # signed holonomy angle, measured in the tangent plane at the base point
        north = np.cross(base, east)
        ang = np.arctan2(float(v_end @ north), float(v_end @ east))
        area = dp * (np.cos(t0) - np.cos(t1))            # exact solid angle enclosed

        traces = [_sphere(0.10)] + _frame_axes()
        traces.append(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines',
                                   name='closed path', line=dict(width=5, color=BLUE)))
        step = max(1, len(path) // 28)
        for k in range(0, len(path), step):
            traces += _arrow(path[k], .28 * vs[k], GREY, "transported vector",
                             width=3, head=.1, showlegend=(k == 0))
        traces += _arrow(base, .45 * east, 'black', "initial vector", width=7, head=.14)
        traces += _arrow(base, .45 * v_end, RED, "returned vector", width=7, head=.14)
        pane.object = _figure(traces, "parallel transport around a closed spherical rectangle")

        read.object = (
            "| quantity | value |\n|---|---|\n"
            f"| holonomy angle (measured) | {ang:.6f} rad  ({np.degrees(ang):.3f}°) |\n"
            f"| enclosed area Δφ · (cos θ₀ − cos θ₁) | {area:.6f} sr |\n"
            f"| Gauss–Bonnet residual, (angle + area) mod 2π | "
            f"{np.angle(np.exp(1j * (ang + area))):.2e} |\n"
            f"| ‖returned vector‖   (transport is an isometry) | {np.linalg.norm(v_end):.6f} |\n"
            f"| returned · base   (still tangent) | {float(v_end @ base):.1e} |\n\n"
            "*The path is closed, the vector is not : it comes back turned by exactly the area "
            "it enclosed. Shrink Δθ and Δφ together and the defect falls quadratically — the "
            "same O(η²) law as the commutator [A,B] of §3, and for the same reason : this sphere "
            "is the base of the fibration SO(2) → SO(3) → S², pointing is the loop, and the roll "
            "is the residue. Flip the sign of Δφ and the spin reverses : holonomy is a *signed* "
            "area, just as [A,B] = −[B,A]. Take Δθ or Δφ to zero — a degenerate rectangle "
            "encloses nothing, and the vector comes home untouched.*")

    pn.bind(update, th0.param.value_throttled, dth.param.value_throttled,
            dph.param.value_throttled, watch=True)
    update(th0.value, dth.value, dph.value)

    return pn.Column(
        pn.pane.Markdown("## §4 — holonomy : the residual rotation is the enclosed area"),
        pn.Row(th0, dth, dph),
        pn.Row(pane, pn.Column(
            _formula(r"$$\text{holonomy} \;=\; -\iint_{\Sigma} K \, dA \;=\; "
                     r"-\,\text{enclosed area} \qquad (K \equiv 1)$$"), read)))


# --------------------------------------------------------------------------- #
#  §4 -- the hairy ball : no continuous "direction without roll"               #
# --------------------------------------------------------------------------- #

def plot_hairy_ball():
    """Why the roll cannot be legislated away.

    A decomposition "direction + no roll" is a continuous section of
    SO(2) -> SO(3) -> S^2, i.e. a continuous non-vanishing tangent frame on the
    sphere. Here is the most natural candidate -- transport a reference tangent
    vector by the *minimal* rotation from a reference direction n0 -- together with
    the point where it necessarily breaks down."""

    n_az = pn.widgets.FloatSlider(name="reference direction n₀ : azimuth (°)", start=-180, end=180, value=0)
    n_el = pn.widgets.FloatSlider(name="reference direction n₀ : elevation (°)", start=-90, end=90, value=90)
    ring = pn.widgets.FloatSlider(name="ring around the defect : angular radius (°)",
                                  start=5, end=80, value=25)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def _section(U, n0, t0):
        """Minimal-rotation section applied to the tangent t0 at n0, vectorised."""
        K = np.cross(n0, U)                                   # (N,3)
        c = U @ n0                                            # (N,)
        kt = np.cross(K, t0)
        kkt = np.cross(K, kt)
        return t0 + kt + kkt / (1 + c)[:, None]

    def update(n_az, n_el, ring):
        n0 = unit(n_az, n_el)
        t0 = np.cross(n0, [0, 0, 1.]) if abs(n0[2]) < .99 else np.cross(n0, [1., 0, 0])
        t0 /= np.linalg.norm(t0)
        defect = -n0

        # a lattice of directions, the antipode of n0 removed
        th, ph = np.meshgrid(np.linspace(.18, np.pi - .18, 13),
                             np.linspace(0, 2 * np.pi, 25)[:-1], indexing='ij')
        U = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)],
                     -1).reshape(-1, 3)
        keep = (U @ n0) > -0.985
        U = U[keep]
        T = _section(U, n0, t0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)

        # the same field sampled on a small ring around the defect: it winds twice
        r = np.radians(ring)
        e1 = np.cross(n0, [0, 0, 1.]) if abs(n0[2]) < .99 else np.cross(n0, [1., 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n0, e1)
        phis = np.linspace(0, 2 * np.pi, 129)              # closed ring
        Ur = np.array([-np.cos(r) * n0 + np.sin(r) * (np.cos(p) * e1 + np.sin(p) * e2)
                       for p in phis])
        Tr = _section(Ur, n0, t0)
        Tr /= np.linalg.norm(Tr, axis=1, keepdims=True)
        # winding of the field along the ring, read in a frame that itself turns once:
        # the index of the defect is that winding, and Poincare-Hopf forces it to be 2.
        f1 = np.cross(np.cross(Ur, e1), Ur)                # tangential part of e1 ...
        f2 = np.cross(Ur, f1)                              # ... and its tangential normal
        f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
        f2 /= np.linalg.norm(f2, axis=1, keepdims=True)
        loc = np.unwrap(np.arctan2(np.einsum('ij,ij->i', Tr, f2),
                                   np.einsum('ij,ij->i', Tr, f1)))
        # the ring is described around -n0, i.e. clockwise seen from outside at the
        # defect : the extra sign restores the usual orientation, and the index is +2
        winding = -(loc[-1] - loc[0]) / (2 * np.pi)
        Ur, Tr = Ur[::4], Tr[::4]                          # thin out for the display

        traces = [_sphere(0.18)]
        traces.append(go.Cone(x=U[:, 0], y=U[:, 1], z=U[:, 2],
                              u=.16 * T[:, 0], v=.16 * T[:, 1], w=.16 * T[:, 2],
                              sizemode='absolute', sizeref=.16, anchor='tail',
                              showscale=False, name='the combed hair',
                              colorscale=[[0, BLUE], [1, BLUE]], hoverinfo='skip'))
        traces.append(go.Cone(x=Ur[:, 0], y=Ur[:, 1], z=Ur[:, 2],
                              u=.20 * Tr[:, 0], v=.20 * Tr[:, 1], w=.20 * Tr[:, 2],
                              sizemode='absolute', sizeref=.20, anchor='tail',
                              showscale=False, name='ring around the defect',
                              colorscale=[[0, AMBER], [1, AMBER]], hoverinfo='skip'))
        traces.append(go.Scatter3d(x=Ur[:, 0], y=Ur[:, 1], z=Ur[:, 2], mode='lines',
                                   line=dict(width=3, color=AMBER, dash='dot'),
                                   name='ring', showlegend=False))
        traces += _arrow([0, 0, 0], n0, GREEN, "reference n₀")
        traces.append(go.Scatter3d(x=[defect[0]], y=[defect[1]], z=[defect[2]],
                                   mode='markers', name='the cowlick  (u = −n₀)',
                                   marker=dict(size=9, color=RED, symbol='x')))
        pane.object = _figure(traces, "combing the sphere : the defect moves, it never leaves")

        read.object = (
            "| quantity | value |\n|---|---|\n"
            f"| defect position (u = −n₀) | {np.round(defect, 3)} |\n"
            f"| winding of the field around the ring | {winding:+.2f}  ≈  {round(winding):+d} |\n"
            f"| Euler characteristic χ(S²) | +2 |\n"
            f"| ‖t(u)‖ away from the defect | {np.linalg.norm(T, axis=1).min():.4f} … "
            f"{np.linalg.norm(T, axis=1).max():.4f} |\n\n"
            "*Move n₀ : the bald spot follows, but it cannot be brushed away, and the winding "
            "stays pinned at 2 whatever the radius of the ring. Poincaré–Hopf forces the indices "
            "of the zeros of any tangent field to sum to χ(S²) = 2, so a nowhere-vanishing frame "
            "field on S² does not exist. Hence there is **no continuous way to attach a roll to "
            "a direction** : SO(3) is not S² × SO(2) (it is ℝP³, doubly covered by the unit "
            "quaternions), and a 3D 'phase' can never be reduced to a pointing direction.*")

    pn.bind(update, n_az.param.value_throttled, n_el.param.value_throttled,
            ring.param.value_throttled, watch=True)
    update(n_az.value, n_el.value, ring.value)

    return pn.Column(
        pn.pane.Markdown("## §4 — the hairy ball : the roll cannot be legislated away"),
        pn.Row(n_az, n_el, ring),
        pn.Row(pane, pn.Column(
            _formula(r"$$SO(2) \hookrightarrow SO(3) \xrightarrow{\ \pi\ } S^2, \qquad "
                     r"\nexists\ s : S^2 \to SO(3) \ \text{ continuous, } \ \pi \circ s "
                     r"= \mathrm{id}$$"), read)))


# --------------------------------------------------------------------------- #
#  §5 -- the Wigner-D matrix : what a rotation does to a multiplet             #
# --------------------------------------------------------------------------- #

def plot_wigner_mixing():
    """D^l(alpha, beta, gamma), as a picture.

    A rotation about z is diagonal -- each order m only picks up the phase
    exp(-i m alpha), which is the 2D story repeated 2l+1 times. A tilt beta is the
    Wigner-d matrix : it genuinely mixes the orders inside the degree, and only the
    norm of the multiplet survives."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=6, value=3)
    al = pn.widgets.FloatSlider(name="α  : yaw about z (rad)", start=0, end=2 * np.pi, step=.01, value=0.8)
    be = pn.widgets.FloatSlider(name="β  : tilt about y (rad)", start=0, end=np.pi, step=.01, value=0.0)
    ga = pn.widgets.FloatSlider(name="γ  : roll about z (rad)", start=0, end=2 * np.pi, step=.01, value=0.0)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, al, be, ga):
        D = wigner_D(l, al, be, ga)
        m = np.arange(l, -l - 1, -1)
        Lx, Ly, Lz = angular_momentum(l)

        a = np.zeros(2 * l + 1, complex)
        a[l] = 1.0                                        # start from the m = 0 order alone
        a[0] = 0.6
        b = D @ a

        fig = make_subplots(rows=1, cols=3, column_widths=[.38, .38, .24],
                            subplot_titles=("|D<sup>ℓ</sup>(α,β,γ)|",
                                            "arg D<sup>ℓ</sup>(α,β,γ)",
                                            "|a<sub>m</sub>| before / after"))
        fig.add_trace(go.Heatmap(z=np.abs(D), x=m, y=m, zmin=0, zmax=1, colorscale='Blues',
                                 showscale=False, hovertemplate="m'=%{y}, m=%{x}<br>|D|=%{z:.3f}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Heatmap(z=np.angle(D), x=m, y=m, zmin=-np.pi, zmax=np.pi,
                                 colorscale='HSV', showscale=False,
                                 hovertemplate="m'=%{y}, m=%{x}<br>arg=%{z:.3f}<extra></extra>"),
                      row=1, col=2)
        fig.add_trace(go.Bar(x=m, y=np.abs(a), name='before', marker_color=GREY), row=1, col=3)
        fig.add_trace(go.Bar(x=m, y=np.abs(b), name='after', marker_color=BLUE), row=1, col=3)
        for c in (1, 2):
            fig.update_xaxes(title="m", row=1, col=c, dtick=1)
            fig.update_yaxes(title="m'", row=1, col=c, dtick=1)
        fig.update_xaxes(title="m", row=1, col=3, dtick=1)
        fig.update_layout(width=1000, height=380, uirevision='constant',
                          margin=dict(t=50, b=40, l=40, r=10), barmode='group',
                          legend=dict(x=0, y=1.15, orientation='h', font=dict(size=10)))
        pane.object = fig

        off = np.abs(D - np.diag(np.diag(D))).max()
        read.object = (
            "| quantity | value |\n|---|---|\n"
            f"| off-diagonal weight of D | {off:.4f} "
            f"{'(diagonal : one phase per order)' if off < 1e-9 else '(the orders are mixed)'} |\n"
            f"| unitarity ‖D† D − 𝟙‖ | {np.abs(D.conj().T @ D - np.eye(2*l+1)).max():.1e} |\n"
            f"| ‖a‖ before / after | {np.linalg.norm(a):.6f} / {np.linalg.norm(b):.6f} |\n"
            f"| Casimir L² | {(Lx@Lx + Ly@Ly + Lz@Lz)[0,0].real:.3f} = ℓ(ℓ+1) = {l*(l+1)} |\n"
            f"| ‖[Lx,Ly] − i Lz‖   (the same algebra, in dimension {2*l+1}) | "
            f"{np.abs(bracket(Lx, Ly) - 1j * Lz).max():.1e} |\n\n"
            "*Set **β = 0** : the modulus panel becomes the identity and only the phase panel "
            "moves — each order turns at its own rate exp(−imα), which is exactly the 2D "
            "spectrangular picture, one planar rotation per |m|, all of them commuting. Now "
            "raise **β** : the matrix fills in, the orders are irreversibly mixed by the "
            "Wigner-d matrix d(β), and the only thing left standing is ‖a‖ (the Casimir). "
            "That one picture is why a per-order modulus/phase cannot survive in 3D — the "
            "quantity you would call a phase is only defined along a single axis.*")

    pn.bind(update, l.param.value_throttled, al.param.value_throttled,
            be.param.value_throttled, ga.param.value_throttled, watch=True)
    update(l.value, al.value, be.value, ga.value)

    return pn.Column(
        pn.pane.Markdown("## §5 — the Wigner-D matrix : a phase along z, a mixing along y"),
        pn.Row(l, al, be, ga),
        pane,
        _formula(r"$$D^\ell(\alpha,\beta,\gamma) = e^{-i\alpha L_z}\, e^{-i\beta L_y}\, "
                 r"e^{-i\gamma L_z}, \qquad [L_z, L_\pm] = \pm L_\pm$$"),
        read)


# --------------------------------------------------------------------------- #
#  §5 -- the closed pointing loop re-phases the multiplet                      #
# --------------------------------------------------------------------------- #

def plot_loop_rephasing():
    """The demonstration of §3, transported into the degree-l representation.

    The very same closed loop of *pointing* manoeuvres, applied to the ambisonic
    coefficients : the norm is untouched, but every order m comes back with a phase
    -m*eps*dlt, and orders that were empty are populated."""

    l = pn.widgets.IntSlider(name="degree ℓ", start=1, end=6, value=3)
    eps = pn.widgets.FloatSlider(name="ε  : x-leg (rad)", start=0, end=1.2, step=.01, value=0.3)
    dlt = pn.widgets.FloatSlider(name="δ  : y-leg (rad)", start=0, end=1.2, step=.01, value=0.2)
    reps = pn.widgets.IntSlider(name="number of times the loop is driven", start=1, end=20, value=1)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    def update(l, eps, dlt, reps):
        Lx, Ly, Lz = angular_momentum(l)
        Jl = [-1j * Lx, -1j * Ly, -1j * Lz]
        K = (expm(eps * Jl[0]) @ expm(dlt * Jl[1])
             @ expm(-eps * Jl[0]) @ expm(-dlt * Jl[1]))
        K = np.linalg.matrix_power(K, reps)
        m = np.arange(l, -l - 1, -1)

        a = np.zeros(2 * l + 1, complex)
        a[0], a[l], a[-1] = 1.0, 0.5, 0.8                 # orders m = +l, 0, -l
        b = K @ a
        phase = np.angle(np.diag(K))
        predicted = -m * eps * dlt * reps

        fig = make_subplots(rows=1, cols=2, column_widths=[.5, .5],
                            subplot_titles=("|a<sub>m</sub>| : the loop populates empty orders",
                                            "phase picked up by each order m"))
        fig.add_trace(go.Bar(x=m, y=np.abs(a), name='before', marker_color=GREY), row=1, col=1)
        fig.add_trace(go.Bar(x=m, y=np.abs(b), name='after', marker_color=BLUE), row=1, col=1)
        fig.add_trace(go.Scatter(x=m, y=phase, mode='markers', name='arg D(K)ₘₘ',
                                 marker=dict(size=10, color=BLUE)), row=1, col=2)
        fig.add_trace(go.Scatter(x=m, y=predicted, mode='lines', name='−m εδ  (the prediction)',
                                 line=dict(color=RED, dash='dash')), row=1, col=2)
        fig.update_xaxes(title="m", autorange='reversed', row=1, col=1)
        fig.update_xaxes(title="m", autorange='reversed', row=1, col=2)
        fig.update_yaxes(title="phase (rad)", row=1, col=2)
        fig.update_layout(width=980, height=380, barmode='group', uirevision='constant',
                          margin=dict(t=52, b=40, l=50, r=10),
                          legend=dict(x=0, y=1.18, orientation='h', font=dict(size=10)))
        pane.object = fig

        leaked = np.abs(b)[np.abs(a) < 1e-12]
        read.object = (
            "| quantity | value |\n|---|---|\n"
            f"| ‖a‖ before / after | {np.linalg.norm(a):.8f} / {np.linalg.norm(b):.8f} |\n"
            f"| largest phase error vs −m εδ | "
            f"{np.abs(np.angle(np.exp(1j*(phase - predicted)))).max():.2e} rad |\n"
            f"| energy leaked into the orders that were empty | "
            f"{np.linalg.norm(leaked)**2 if leaked.size else 0.:.3e} |\n"
            f"| ‖D(K) − exp(εδ · dD(Jz))‖ | "
            f"{np.abs(K - np.linalg.matrix_power(expm(eps*dlt*Jl[2]), reps)).max():.2e} |\n"
            f"| ‖D(K)† D(K) − 𝟙‖ | {np.abs(K.conj().T @ K - np.eye(2*l+1)).max():.1e} |\n\n"
            "*The sound field was pointed around and brought back : the total energy of the "
            "degree is exactly what it was, and yet the field is **not** the field one started "
            "from. The orders precess at 2ℓ+1 different rates (the straight line on the right), "
            "and energy has leaked into orders that were empty. No scalar phase could have "
            "bookkept that — which is the whole difficulty of a 3D magnitude–phase decomposition. "
            "Raise the repetition count to watch the drift accumulate.*")

    pn.bind(update, l.param.value_throttled, eps.param.value_throttled,
            dlt.param.value_throttled, reps.param.value_throttled, watch=True)
    update(l.value, eps.value, dlt.value, reps.value)

    return pn.Column(
        pn.pane.Markdown("## §5 — a closed *pointing* loop comes back re-phased"),
        pn.Row(l, eps, dlt, reps),
        pane,
        _formula(r"$$D^\ell\!\left(K(\varepsilon,\delta)\right) = \mathbb{1} "
                 r"+ \varepsilon\delta\, dD^\ell(J_z) + O(\eta^3) "
                 r"= \mathrm{diag}\left(e^{-i m \varepsilon\delta}\right) + O(\eta^3)$$"),
        read)


# --------------------------------------------------------------------------- #
#  §6 -- moduli versus group coordinates, on the degree-2 multiplet            #
# --------------------------------------------------------------------------- #

def plot_orbit_invariants():
    """The dimension count 5 = 2 + 3, made tangible.

    A degree-2 field is a symmetric traceless 3x3 matrix Q, i.e. f(u) = u^T Q u.
    Its two invariants are the eigenvalues (tr Q^2 and tr Q^3) -- the *moduli* --
    and its three Euler angles are the *group coordinates*. Moving the angles moves
    the field without moving a single invariant."""

    lam1 = pn.widgets.FloatSlider(name="λ₁  (modulus 1)", start=-1.5, end=1.5, step=.01, value=1.0)
    lam2 = pn.widgets.FloatSlider(name="λ₂  (modulus 2)", start=-1.5, end=1.5, step=.01, value=-0.3)
    al = pn.widgets.FloatSlider(name="α  (group coordinate)", start=0, end=2 * np.pi, step=.01, value=0.)
    be = pn.widgets.FloatSlider(name="β  (group coordinate)", start=0, end=np.pi, step=.01, value=0.)
    ga = pn.widgets.FloatSlider(name="γ  (group coordinate)", start=0, end=2 * np.pi, step=.01, value=0.)

    pane, read = pn.pane.Plotly(), pn.pane.Markdown()

    az_g, ze_g = np.meshgrid(np.linspace(-np.pi, np.pi, 84),
                             np.linspace(0, np.pi, 43), indexing='ij')
    dirs = np.stack([np.sin(ze_g) * np.cos(az_g), np.sin(ze_g) * np.sin(az_g),
                     np.cos(ze_g)], -1)

    def update(lam1, lam2, al, be, ga):
        lam = np.array([lam1, lam2, -lam1 - lam2])
        Rot = rot(2, al) @ rot(1, be) @ rot(2, ga)
        Q = Rot @ np.diag(lam) @ Rot.T
        f = np.einsum('...i,ij,...j->...', dirs, Q, dirs)

        r = np.abs(f) / max(np.abs(f).max(), 1e-9)
        surf = go.Surface(x=r * dirs[..., 0], y=r * dirs[..., 1], z=r * dirs[..., 2],
                          surfacecolor=np.sign(f), cmin=-1, cmax=1, showscale=False,
                          colorscale=[[0, BLUE], [1, RED]], name='f(u) = uᵀQu')
        traces = [surf]
        for k, col in enumerate([GREEN, AMBER, PURPLE]):
            axis_v = Rot[:, k] * (0.5 + abs(lam[k]))
            traces += _arrow([0, 0, 0], axis_v, col, f"eigenvector {k+1}  (λ = {lam[k]:+.2f})")
            traces += _arrow([0, 0, 0], -axis_v, col, "", showlegend=False)
        pane.object = _figure(traces, "a degree-2 field : two moduli (the eigenvalues), "
                                      "three angles (the eigenframe)", width=680, height=520)

        comps = np.array([Q[0, 0], Q[1, 1], Q[0, 1], Q[0, 2], Q[1, 2]])
        read.object = (
            "| quantity | value | moves with the angles ? |\n|---|---|---|\n"
            f"| the 5 components (Qxx, Qyy, Qxy, Qxz, Qyz) | {np.round(comps, 3)} | **yes** |\n"
            f"| eigenvalues λ | {np.round(np.sort(np.linalg.eigvalsh(Q))[::-1], 4)} | no |\n"
            f"| tr Q² = ‖a₂‖² | {np.trace(Q @ Q):.6f} | no |\n"
            f"| tr Q³ = 3 det Q | {np.trace(Q @ Q @ Q):.6f} | no |\n"
            f"| tr Q   (a degree-2 field is traceless) | {np.trace(Q):.1e} | no |\n\n"
            "*Move α, β, γ : all five coefficients dance, the field turns, and **not one "
            "invariant moves** — the state slides along its orbit. Move λ₁ or λ₂ : the shape "
            "itself changes, and no rotation will ever bring it back. That is the polar "
            "decomposition correctly generalised — except that the 'modulus' is a *pair* of "
            "numbers and the 'phase' is a whole element of SO(3), whose coordinates, by §3, do "
            "not simply add up. In general a degree ℓ carries 2ℓ−2 moduli, so ℓ = 1 is the only "
            "degree that a norm and a direction can describe.*")

    pn.bind(update, lam1.param.value_throttled, lam2.param.value_throttled,
            al.param.value_throttled, be.param.value_throttled, ga.param.value_throttled,
            watch=True)
    update(lam1.value, lam2.value, al.value, be.value, ga.value)

    return pn.Column(
        pn.pane.Markdown("## §6 — moduli versus group coordinates : the degree-2 case, 5 = 2 + 3"),
        pn.Row(pn.Column(lam1, lam2), pn.Column(al, be, ga)),
        pn.Row(pane, pn.Column(
            _formula(r"$$\underbrace{5}_{\dim\mathcal{H}_2} \;=\; "
                     r"\underbrace{2}_{\mathrm{tr}\,Q^2,\ \mathrm{tr}\,Q^3} \;+\; "
                     r"\underbrace{3}_{\alpha,\ \beta,\ \gamma}$$"), read)))
