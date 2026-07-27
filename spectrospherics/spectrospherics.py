import panel as pn
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

from .utils import spherical_harmonic, plot_spherical, plot_ambisonics_3d, generate_harmonic

def get_harmonic_plot(n, m):
    [azimuth, zenith] = np.mgrid[-np.pi:np.pi:0.05, 0:np.pi:0.01]
    out = spherical_harmonic(azimuth, zenith, n, m)
    return plot_spherical(out, azimuth=azimuth, zenith=zenith)


def plot_spherical_harmonic():
    n = pn.widgets.IntInput(name="degree", value=0)
    m = pn.widgets.IntInput(name="order", value=0)

    harmonic_pane = pn.pane.Plotly(get_harmonic_plot(n.value, m.value))

    def update_harmonic(n, m):
        harmonic_pane.object = get_harmonic_plot(n, m)

    pn.bind(update_harmonic, n, m, watch=True),
    return pn.Column(
        pn.Row(pn.Spacer(), n, m, pn.Spacer()),
        pn.Row(pn.Spacer(), harmonic_pane, pn.Spacer())
    )


def plot_spherical_harmonics():

    order = pn.widgets.IntSlider(value=3, start=1, end=5, name="Ambisonics Order")
    ambisonics_3d_explorer = pn.Column(
        order,
        pn.bind(plot_ambisonics_3d, order=order)
    )
    return ambisonics_3d_explorer


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