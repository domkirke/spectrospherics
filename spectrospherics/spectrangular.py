import numpy as np
from plotly import graph_objects as go

from .utils import circular_harmonic
from .explorer import ParamSpec, DecompositionSpec, DecompositionExplorer


# --------------------------------------------------------------------------- #
#  Circular-harmonic synthesis (the actual maths)                             #
# --------------------------------------------------------------------------- #

def from_circular_harmonics(amplitudes, phases, n_steps=100):
    assert len(amplitudes) == len(phases) + 1
    n = len(amplitudes)
    theta = np.linspace(0, 1., n_steps)
    a0, *amplitudes = amplitudes
    out = np.full_like(theta, a0)
    for n in range(len(phases)):
        a_n = amplitudes[n] * np.cos(2*np.pi*phases[n])
        b_n = amplitudes[n] * np.sin(2*np.pi*phases[n])
        out += a_n * circular_harmonic(theta * 2 * np.pi, n+1) + b_n * circular_harmonic(theta * 2 * np.pi, -(n+1))
    return theta, out


def plot_circular_fn(*args, order=None):
    if order is None: raise ValueError(order)
    amplitudes = np.array(args[:-order])
    phases =  np.array(args[-order:])
    theta, out = from_circular_harmonics(amplitudes, phases)
    fig = go.Figure(go.Scatterpolar(r=out, theta=theta * 360, mode="lines"))
    # autosize + a fixed height : the width belongs to the (stretch_width) pane, so
    # the plot keeps the same place and size on every redraw
    fig.update_layout(uirevision='constant', autosize=True, height=480,
        polar = dict(
            radialaxis = dict(range=[0, 5]),
            angularaxis = dict(showticklabels=False, ticks='')
        ))
    return fig


# --------------------------------------------------------------------------- #
#  Declarative decomposition description (fed to the generic explorer)        #
# --------------------------------------------------------------------------- #

def _build_circular_params(structural):
    """One row for the DC term, then one row per harmonic order carrying its
    amplitude and phase. Rows stack top -> bottom in the explorer."""
    order = structural["order"]
    rows = [[ParamSpec("a0", r"$a_0$", value=1.0, start=0.0, end=1.0)]]
    for n in range(1, order + 1):
        rows.append([
            ParamSpec(f"a{n}", rf"$a_{{{n}}}$", value=0.0, start=0.0, end=1.0),
            ParamSpec(f"phi{n}", rf"$\phi_{{{n}}}$", value=0.0, start=0.0, end=1.0),
        ])
    return rows


def _render_circular(params, structural):
    order = structural["order"]
    amplitudes = [params["a0"]] + [params[f"a{n}"] for n in range(1, order + 1)]
    phases = [params[f"phi{n}"] for n in range(1, order + 1)]
    return plot_circular_fn(*amplitudes, *phases, order=order)


CIRCULAR_2D = DecompositionSpec(
    name="Circular harmonics (2D)",
    structural={"order": (1, 7, 3)},
    build_params=_build_circular_params,
    render=_render_circular,
)


# --------------------------------------------------------------------------- #
#  Public entry point                                                          #
# --------------------------------------------------------------------------- #

def plot_spectrambisonics_2d():
    return DecompositionExplorer(CIRCULAR_2D).layout
