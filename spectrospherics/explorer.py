"""Reusable building blocks for interactive spectral-decomposition explorers.

The layout is intentionally decoupled from any particular decomposition so it
can be reused for the 2D circular harmonics, the 3D spherical harmonics, the
invariant-plane parametrisation, etc.  To wire a new decomposition you only need
to describe it with a :class:`DecompositionSpec` and hand it to a
:class:`DecompositionExplorer`.

Features provided for free by the framework:
  * parameter values are preserved across structural changes (e.g. raising the
    order from 2 to 4 keeps the coefficients already dialled in);
  * a randomization button;
  * a vertically-growing, scrollable coefficient area (top -> bottom).

Every control is manual.  Sliders report on ``value_throttled``, so a drag costs
one redraw when it ends rather than one per pixel -- which for a 3D surface is
the difference between a scene that responds and one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import panel as pn


# --------------------------------------------------------------------------- #
#  Parameter description & persistent state                                   #
# --------------------------------------------------------------------------- #

@dataclass
class ParamSpec:
    """Static description of a single scalar parameter."""

    key: str            # unique, *stable* identifier used for value preservation
    label: str          # display name (LaTeX allowed thanks to the mathjax ext.)
    value: float = 0.0  # default value
    start: float = -1.0
    end: float = 1.0
    step: float = 0.01


@dataclass
class ParamState:
    """What must survive a structural rebuild for one parameter."""

    value: float

    @classmethod
    def from_spec(cls, spec: ParamSpec) -> "ParamState":
        return cls(value=spec.value)


# --------------------------------------------------------------------------- #
#  Per-parameter controller                                                    #
# --------------------------------------------------------------------------- #

class ParamController:
    """Widget and behaviour for a single parameter.

    Exposes :attr:`value`, a :meth:`view` for layout, :meth:`randomize`, and
    :meth:`state` to snapshot itself across a rebuild."""

    def __init__(self, spec: ParamSpec, state: ParamState, on_change: Callable[[], None]):
        self.spec = spec
        self._on_change = on_change

        self.slider = pn.widgets.EditableFloatSlider(
            name=spec.label, start=spec.start, end=spec.end, step=spec.step,
            value=state.value, fixed_start=spec.start, fixed_end=spec.end,
            sizing_mode="stretch_width", margin=(2, 8),
        )
        # ``value_throttled`` and not ``value``: one redraw when the drag ends.
        self.slider.param.watch(lambda _event: self._on_change(), "value_throttled")

    @property
    def value(self) -> float:
        return self.slider.value

    def randomize(self) -> None:
        self.slider.value = float(np.random.uniform(self.spec.start, self.spec.end))

    def state(self) -> ParamState:
        return ParamState(value=self.slider.value)

    def view(self) -> pn.Row:
        return pn.Row(self.slider, sizing_mode="stretch_width", margin=(2, 4))


# --------------------------------------------------------------------------- #
#  Decomposition description                                                   #
# --------------------------------------------------------------------------- #

def identity_transform(params: dict, structural: dict) -> dict:
    """The default :attr:`DecompositionSpec.transform`: hand the sliders straight on."""
    return params


@dataclass
class DecompositionSpec:
    """Declarative description of one decomposition to explore.

    ``structural`` maps a structural control name either to ``(start, end,
    default)`` integer bounds (e.g. the ambisonics order) or to a plain boolean
    for an on/off choice.  ``build_params`` turns the current structural values
    into a list of *rows* of :class:`ParamSpec`; each row is laid out
    horizontally, rows stack top -> bottom.  ``render`` maps the flat
    ``{key: value}`` parameter dict (plus structural values) to a figure.

    ``transform`` sits between the two: the sliders' raw values go through it on
    their way to ``render``, so the controls the user turns and the quantities the
    figure is drawn from need not be the same thing.  It is the identity by
    default.  Signature and return are free-form -- ``(params, structural) ->
    whatever render takes as its first argument`` -- so it can rescale, reorder,
    add derived entries, or change representation entirely::

        # sliders in a hyperspherical chart, coefficients in the cartesian one
        def to_cartesian(params, structural):
            order = structural["order"]
            radii_and_angles = [params[k] for k in _chart_keys(order)]
            return dict(enumerate(sph2car(*radii_and_angles)))

    Only ``render`` sees the transformed values.  The sliders, the saved state and
    the randomize button all keep working on the raw ones, so a transform can be
    added or changed without touching anything the user interacts with.
    """

    name: str
    structural: Dict[str, tuple]
    build_params: Callable[[dict], List[List[ParamSpec]]]
    render: Callable[[dict, dict], object]
    transform: Callable[[dict, dict], object] = identity_transform


# --------------------------------------------------------------------------- #
#  The generic explorer                                                        #
# --------------------------------------------------------------------------- #

class DecompositionExplorer:
    """Assembles structural controls, parameter controllers and the plot into a
    single Panel layout for a given :class:`DecompositionSpec`."""

    def __init__(self, spec: DecompositionSpec, coeff_max_height: int = 380):
        self.spec = spec
        self._states: Dict[str, ParamState] = {}      # persistent value store
        self._controllers: Dict[str, ParamController] = {}
        self._warned = set()                          # failures already reported

        # -- structural controls (order, ...) ----------------------------- #
        self._structural: Dict[str, pn.widgets.Widget] = {}
        struct_widgets = []
        for key, setting in spec.structural.items():
            if isinstance(setting, bool):            # a plain on/off structural choice
                w = pn.widgets.Checkbox(name=key.capitalize(), value=setting)
                w.param.watch(self._on_structural, "value")
            else:
                start, end, default = setting
                w = pn.widgets.IntSlider(name=key.capitalize(), start=start, end=end,
                                         value=default)
                w.param.watch(self._on_structural, "value_throttled")
            self._structural[key] = w
            struct_widgets.append(w)

        # -- global actions ----------------------------------------------- #
        self.randomize_btn = pn.widgets.Button(name="🎲 Randomize", button_type="primary")
        self.randomize_btn.on_click(self._randomize)

        # -- panes --------------------------------------------------------- #
        self.coeff_area = pn.Column(sizing_mode="stretch_width", scroll=True,
                                    max_height=coeff_max_height)
        # ``responsive`` is what makes Plotly track the container it is given ; the
        # figures are built with autosize on and no width, so the pane owns the
        # width and the plot stays put instead of shifting on every redraw.
        self.plot_pane = pn.pane.Plotly(sizing_mode="stretch_width",
                                        config={"responsive": True})

        self._build_controllers()
        self.render()

        self.layout = pn.Column(
            pn.Row(*struct_widgets, sizing_mode="stretch_width"),
            pn.Row(self.randomize_btn),
            self.coeff_area,
            self.plot_pane,
            sizing_mode="stretch_width",
        )

    # -- structural / rebuild --------------------------------------------- #

    def _structural_values(self) -> dict:
        return {k: w.value for k, w in self._structural.items()}

    def _build_controllers(self) -> None:
        rows = self.spec.build_params(self._structural_values())
        self._controllers = {}
        view_rows = []
        for row in rows:
            row_views = []
            for sp in row:
                state = self._states.setdefault(sp.key, ParamState.from_spec(sp))
                ctrl = ParamController(sp, state, on_change=self.render)
                self._controllers[sp.key] = ctrl
                row_views.append(ctrl.view())
            view_rows.append(pn.Row(*row_views, sizing_mode="stretch_width"))
        self.coeff_area.objects = view_rows

    def _snapshot(self) -> None:
        """Persist the live controllers' values before a rebuild."""
        for key, ctrl in self._controllers.items():
            self._states[key] = ctrl.state()

    def _on_structural(self, event) -> None:
        self._snapshot()
        self._build_controllers()
        self.render()

    # -- global actions ---------------------------------------------------- #

    def _randomize(self, event=None) -> None:
        for ctrl in self._controllers.values():
            ctrl.randomize()
        self.render()

    # -- rendering --------------------------------------------------------- #

    def render(self) -> None:
        """Redraw the plot from the current control values, passed through the
        spec's ``transform`` on the way."""
        params = {k: c.value for k, c in self._controllers.items()}
        structural = self._structural_values()
        transform = self.spec.transform or identity_transform
        try:
            self.plot_pane.object = self.spec.render(transform(params, structural),
                                                     structural)
        except Exception as exc:            # a bad frame must not break the panel
            import warnings
            msg = f"render failed: {exc!r}"
            if msg not in self._warned:     # ...nor report itself on every drag
                self._warned.add(msg)
                warnings.warn(msg)

    # backwards-compatible alias
    def request_render(self) -> None:
        self.render()

    # -- Panel integration ------------------------------------------------- #

    def __panel__(self):
        return self.layout

    def servable(self, *args, **kwargs):
        return self.layout.servable(*args, **kwargs)
