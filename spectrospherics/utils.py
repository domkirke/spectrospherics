import numpy as np
import plotly as pl
import plotly.graph_objects as go
import panel as pn
pn.extension('mathjax', 'plotly')
import math
import numpy as np
from scipy.special import assoc_legendre_p, assoc_legendre_p_all, factorial
from scipy.spatial.transform import Rotation
import plotly
import plotly.graph_objs as go


def circular_harmonic(theta, n):
    if n==0: 
        return np.ones_like(theta)
    elif n > 0: 
        return np.cos(n * theta)
    else:
        return np.sin(np.abs(n) * theta)


#: default real-SH normalization used across the package ("sn3d" or "n3d")
SH_CONVENTION = "sn3d"


def sh_normalization(n, m, convention=SH_CONVENTION):
    """Ambisonics normalization factor N_n^{|m|} for the *real* spherical
    harmonics (Condon-Shortley phase excluded, as is standard in ambisonics).

    - ``"sn3d"`` : Schmidt semi-normalized (the AmbiX standard). Every harmonic
      is bounded by 1 and the omni Y_0^0 == 1. Includes the azimuthal
      ``(2 - delta_{m,0})`` factor so the m=0 and m!=0 channels are balanced.
    - ``"n3d"``  : fully normalized / orthonormal, ``(1/4pi) int Y^2 dOmega = 1``;
      this is ``sqrt(2n+1)`` times SN3D.

    scipy's ``norm=True`` is neither: it normalizes only the Legendre part over
    ``x in [-1, 1]`` (a geodesy convention), which drops the azimuthal factor and
    unbalances the channels -- bad for ambisonics.
    """
    m = abs(m)
    sn3d = np.sqrt((2.0 - (m == 0)) * factorial(n - m) / factorial(n + m))
    conv = convention.lower()
    if conv == "sn3d":
        return sn3d
    if conv == "n3d":
        return np.sqrt(2 * n + 1) * sn3d
    raise ValueError(f"unknown SH convention: {convention!r} (use 'sn3d' or 'n3d')")


def spherical_harmonic(azimuth, zenith, n, m, convention=SH_CONVENTION):
    # unnormalized associated Legendre P_n^{|m|}, with the Condon-Shortley
    # phase (-1)^m removed, then scaled by the ambisonics normalization factor.
    mm = abs(m)
    legendre = ((-1.0) ** mm) * np.asarray(
        assoc_legendre_p(n, mm, np.cos(zenith), norm=False))[0]
    norm = sh_normalization(n, m, convention)
    if m >= 0:
        return norm * legendre * np.cos(mm * azimuth)
    return norm * legendre * np.sin(mm * azimuth)
            

def plot_spherical(f, azimuth, zenith, pane=None):
    r = np.abs(f)
    x = r * np.sin(zenith) * np.cos(azimuth)
    y = r * np.sin(zenith) * np.sin(azimuth)
    z = r * np.cos(zenith)

    surface = go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.sign(f),
        colorscale=[[0, 'rgb(59,130,246)'], [1, 'rgb(239,68,68)']],
        cmin=-1, cmax=1,
        showscale=False,
    )
    data = [surface]
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                range=[-1, 1],
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                range=[-1, 1],
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                range=[-1, 1],
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectmode='cube'
        ),
        uirevision='constant',
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        title="Spherical panning function",
        width=1200,
        height=500,
        margin=dict(t=50, b=50, r=50, l=50),
        )
    if pane is None:
        return fig
    pane.object = fig
    return pane

def rotate_zenith(azimuth, zenith, alpha, beta, gamma):
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    vecs = np.stack([x, y, z], axis=-1)
    rot = Rotation.from_euler('ZYZ', [alpha, beta, gamma])
    rotated = rot.apply(vecs.reshape(-1, 3), inverse=True).reshape(vecs.shape)
    return np.arccos(np.clip(rotated[..., 2], -1., 1.))


def plot_ambisonics_3d(order):
    min_height = 30
    coeffs = []
    slider_grid = pn.GridSpec(sizing_mode="stretch_width", min_height=min_height, nrows=order, ncols=2*order + 1)
    for n in range(order):
        val = 1. if n == 0 else 0.
        widgets=[pn.widgets.EditableFloatSlider(value=val, start=-1., end=1., step=0.01, name=f"$a_{n}^{i}$", height=min_height) for i in range(2 * n + 1)]
        coeffs.extend(widgets)
        for l in range(len(widgets)):
            slider_grid[n, l] = widgets[l]

    harmonic_pane = pn.pane.Plotly(plot_harmonic_fn(*[c.value for c in coeffs]))

    def update_harmonic_3d(*coeff_values):
        harmonic_pane.object = plot_harmonic_fn(*coeff_values)

    pn.bind(update_harmonic_3d, *coeffs, watch=True)

    return pn.Column(
        slider_grid, 
        harmonic_pane
    )


def generate_harmonic(azimuth, zenith, *coeffs, direction=0):  
    out = np.zeros(azimuth.shape)
    order = None
    for o in range(10):
        n_coeffs = sum([2*oo + 1 for oo in range(o+1)])
        if n_coeffs == len(coeffs):
            order = o 
            break
        if len(coeffs) < n_coeffs:
            break
    if order is None: 
        raise ValueError('invalid number of coeffs : %d'%len(coeffs))
    coeffs = list(coeffs)
    for i in range(order+1):
        for j, mm in enumerate(range(-i, i+1)):
            coeff = coeffs.pop(0)
            out = out + coeff * spherical_harmonic(azimuth, zenith, i, mm)
    peak = np.abs(out).max()
    return out / peak if peak > 0 else out


def plot_harmonic_fn(*coeffs):
    [azimuth, zenith] = np.mgrid[-np.pi:np.pi:0.05, 0:np.pi:0.05]
    coeffs_cart = sph2car(*coeffs)
    out = generate_harmonic(azimuth, zenith, *coeffs_cart)
    return plot_spherical(out, azimuth, zenith)

def order_from_coeffs(*coeffs):
    for o in range(10):
        n_coeffs = sum([2*oo + 1 for oo in range(o+1)])
        if n_coeffs == len(coeffs):
            return o
        if len(coeffs) < n_coeffs:
            return
    return

def sph2car(*coeffs):
    order = order_from_coeffs(*coeffs)
    if order is None: 
        raise ValueError("got invalid number of coeffs : %d"%len(coeffs))
    coeffs = list(coeffs)
    cartesian_coeffs = []
    for o in range(order+1):
        current_coeffs = np.array([coeffs.pop(0) for _ in range(2*o+1)])
        if o > 0:
            for i in range(2*o+1):
                c = current_coeffs[0]
                if i < 2*o: c = c * np.cos(current_coeffs[-1] * 2 * np.pi) 
                if i > 0 and i < 2*o: c = c * np.prod(np.sin(current_coeffs[1:i+1] * np.pi))
                if i > 0 and i == 2*o: c = c * np.prod(np.sin(current_coeffs[1:] * np.pi))
                cartesian_coeffs.append(float(c))
        else:
            cartesian_coeffs.extend(current_coeffs)
    return cartesian_coeffs
                

def invariant_plane_coeffs(*params, order=None):
    if order is None: raise ValueError(order)
    params = list(params)
    coeffs = []
    for l in range(order):
        r0 = params.pop(0)
        neg, pos = [], []
        for m in range(1, l + 1):
            r_m = params.pop(0)
            theta_m = params.pop(0)
            neg.append(r_m * np.cos(theta_m))
            pos.append(-r_m * np.sin(theta_m))
        coeffs.extend(reversed(neg))
        coeffs.append(r0)
        coeffs.extend(pos)
    return coeffs
