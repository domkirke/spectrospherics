"""Reusable building blocks for interactive spectral-decomposition explorers.

The layout is intentionally decoupled from any particular decomposition so it
can be reused for the 2D circular harmonics, the 3D spherical harmonics, the
invariant-plane parametrisation, etc.  To wire a new decomposition you only need
to describe it with a :class:`DecompositionSpec` and hand it to a
:class:`DecompositionExplorer`.

Features provided for free by the framework:
  * parameter values are preserved across structural changes (e.g. raising the
    order from 2 to 4 keeps the coefficients already dialled in);
  * every parameter can be automated with a per-parameter LFO (diverse shapes);
  * a randomization button;
  * a vertically-growing, scrollable coefficient area (top -> bottom).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import panel as pn


# --------------------------------------------------------------------------- #
#  Low-frequency oscillators (automation)                                     #
# --------------------------------------------------------------------------- #

LFO_SHAPES = ["sine", "triangle", "saw", "square", "sample&hold", "smooth-noise"]


class LFO:
    """Stateful low-frequency oscillator producing values in ``[-1, 1]``.

    The state is only needed by the random shapes (``sample&hold`` and
    ``smooth-noise``), which redraw a target value once per cycle.
    """

    def __init__(self):
        self._cycle = None
        self._prev = 0.0
        self._next = float(np.random.uniform(-1.0, 1.0))

    def _refresh_random(self, cycle: int) -> None:
        if cycle != self._cycle:
            self._cycle = cycle
            self._prev = self._next
            self._next = float(np.random.uniform(-1.0, 1.0))

    def sample(self, shape: str, phase: float) -> float:
        """Return the oscillator value for an (absolute) ``phase`` in cycles."""
        p = phase % 1.0
        if shape == "sine":
            return float(np.sin(2 * np.pi * p))
        if shape == "triangle":
            return float(2.0 * abs(2.0 * (p - np.floor(p + 0.5))) - 1.0)
        if shape == "saw":
            return float(2.0 * p - 1.0)
        if shape == "square":
            return 1.0 if p < 0.5 else -1.0
        if shape in ("sample&hold", "smooth-noise"):
            self._refresh_random(int(np.floor(phase)))
            if shape == "sample&hold":
                return self._prev
            return float(self._prev + (self._next - self._prev) * p)
        return 0.0


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
    """Everything that must survive a structural rebuild for one parameter.

    ``value`` is the slider position, i.e. the *onset* the LFO oscillates around
    (the LFO never writes it back)."""

    value: float
    auto: bool = False
    shape: str = "sine"
    polarity: str = "bi"     # "bi": osc in [-1, 1]  |  "uni": osc in [0, 1]
    rate: float = 0.2
    depth: float = 0.5
    phase: float = 0.0

    @classmethod
    def from_spec(cls, spec: ParamSpec) -> "ParamState":
        return cls(value=spec.value)


# --------------------------------------------------------------------------- #
#  Per-parameter controller (slider + collapsible LFO settings)               #
# --------------------------------------------------------------------------- #

class ParamController:
    """Widgets and behaviour for a single parameter.

    Exposes :attr:`value`, a :meth:`view` for layout, :meth:`tick` to advance
    the automation, :meth:`randomize`, and :meth:`state` to snapshot itself.
    """

    def __init__(self, spec: ParamSpec, state: ParamState, on_manual: Callable[[], None]):
        self.spec = spec
        self._on_manual = on_manual
        self._lfo = LFO()

        # The slider is directly editable (number box) and acts as the LFO
        # onset: the LFO modulates around it at render time and never writes it.
        self.slider = pn.widgets.EditableFloatSlider(
            name=spec.label, start=spec.start, end=spec.end, step=spec.step,
            value=state.value, fixed_start=spec.start, fixed_end=spec.end,
            sizing_mode="stretch_width", margin=(2, 8),
        )
        self.auto = pn.widgets.Checkbox(name="LFO", value=state.auto, width=52, align="end")
        self.shape = pn.widgets.Select(
            name="", options=LFO_SHAPES, value=state.shape, width=104, margin=(2, 4))
        self.polarity = pn.widgets.RadioButtonGroup(
            options=["bi", "uni"], value=state.polarity, width=72, margin=(2, 4),
            button_style="outline")
        self.rate = pn.widgets.FloatInput(
            name="Hz", start=0.01, end=5.0, step=0.01, value=state.rate,
            width=68, margin=(2, 4))
        self.depth = pn.widgets.FloatInput(
            name="depth", start=0.0, end=1.0, step=0.01, value=state.depth,
            width=68, margin=(2, 4))
        self.phase = pn.widgets.FloatInput(
            name="phase", start=0.0, end=1.0, step=0.01, value=state.phase,
            width=68, margin=(2, 4))

        # compact single-row LFO settings, revealed only when automation is on
        self._settings = pn.Row(
            self.shape, self.polarity, self.rate, self.depth, self.phase,
            visible=state.auto, margin=(0, 0, 4, 20),
        )

        self.auto.param.watch(self._toggle_auto, "value")
        self.slider.param.watch(self._on_change, "value")

    # -- callbacks --------------------------------------------------------- #

    def _toggle_auto(self, event) -> None:
        # Show/hide the settings and redraw so the plot immediately reflects the
        # automation turning on (or snaps back to the onset when turning off).
        self._settings.visible = event.new
        self._on_manual()

    def _on_change(self, event) -> None:
        self._on_manual()

    # -- public API -------------------------------------------------------- #

    @property
    def value(self) -> float:
        """The onset value (slider position), unaffected by automation."""
        return self.slider.value

    def output(self, t: float) -> float:
        """The value to render at absolute time ``t`` (seconds): the onset when
        automation is off, otherwise the onset modulated by the LFO."""
        if not self.auto.value:
            return self.slider.value
        rng = self.spec.end - self.spec.start
        osc = self._lfo.sample(self.shape.value, t * self.rate.value + self.phase.value)
        if self.polarity.value == "uni":
            # one-sided: sweep upward from the onset, [0, depth*range]
            mod = self.depth.value * rng * (osc + 1.0) / 2.0
        else:
            # centered on the onset: +/- depth*range/2  (peak-to-peak == depth*range)
            mod = self.depth.value * (rng / 2.0) * osc
        return float(np.clip(self.slider.value + mod, self.spec.start, self.spec.end))

    def randomize(self) -> None:
        self.slider.value = float(np.random.uniform(self.spec.start, self.spec.end))

    def state(self) -> ParamState:
        return ParamState(
            value=self.slider.value, auto=self.auto.value, shape=self.shape.value,
            polarity=self.polarity.value, rate=self.rate.value,
            depth=self.depth.value, phase=self.phase.value,
        )

    def view(self) -> pn.Column:
        header = pn.Row(self.slider, self.auto, sizing_mode="stretch_width")
        return pn.Column(header, self._settings, sizing_mode="stretch_width",
                         margin=(2, 4))


# --------------------------------------------------------------------------- #
#  Decomposition description                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class DecompositionSpec:
    """Declarative description of one decomposition to explore.

    ``structural`` maps a structural control name to ``(start, end, default)``
    integer bounds (e.g. the ambisonics order).  ``build_params`` turns the
    current structural values into a list of *rows* of :class:`ParamSpec`; each
    row is laid out horizontally, rows stack top -> bottom.  ``render`` maps the
    flat ``{key: value}`` parameter dict (plus structural values) to a figure.
    """

    name: str
    structural: Dict[str, tuple]
    build_params: Callable[[dict], List[List[ParamSpec]]]
    render: Callable[[dict, dict], object]


# --------------------------------------------------------------------------- #
#  The generic explorer                                                        #
# --------------------------------------------------------------------------- #

class DecompositionExplorer:
    """Assembles structural controls, parameter controllers, automation and the
    plot into a single Panel layout for a given :class:`DecompositionSpec`."""

    def __init__(self, spec: DecompositionSpec, lfo_fps: int = 30,
                 coeff_max_height: int = 380):
        self.spec = spec
        self._states: Dict[str, ParamState] = {}      # persistent value/LFO store
        self._controllers: Dict[str, ParamController] = {}
        self._t0 = time.time()
        self._lfo_period = max(1, int(1000 / lfo_fps))
        self._dirty = False          # a change is waiting to be painted
        self._loop_running = False   # whether the frame loop is driving renders

        # -- structural controls (order, ...) ----------------------------- #
        self._structural: Dict[str, pn.widgets.IntSlider] = {}
        struct_widgets = []
        for key, (start, end, default) in spec.structural.items():
            w = pn.widgets.IntSlider(name=key.capitalize(), start=start, end=end,
                                     value=default)
            w.param.watch(self._on_structural, "value_throttled")
            self._structural[key] = w
            struct_widgets.append(w)

        # -- global actions ----------------------------------------------- #
        self.randomize_btn = pn.widgets.Button(name="🎲 Randomize", button_type="primary")
        self.randomize_btn.on_click(self._randomize)
        self.stop_btn = pn.widgets.Button(name="⏹ Stop LFOs")
        self.stop_btn.on_click(self._stop_all)

        # -- panes --------------------------------------------------------- #
        self.coeff_area = pn.Column(sizing_mode="stretch_width", scroll=True,
                                    max_height=coeff_max_height)
        self.plot_pane = pn.pane.Plotly(sizing_mode="stretch_width")

        self._build_controllers()
        self._flush()
        self._start_automation()

        self.layout = pn.Column(
            pn.Row(*struct_widgets, sizing_mode="stretch_width"),
            pn.Row(self.randomize_btn, self.stop_btn),
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
                ctrl = ParamController(sp, state, on_manual=self.request_render)
                self._controllers[sp.key] = ctrl
                row_views.append(ctrl.view())
            view_rows.append(pn.Row(*row_views, sizing_mode="stretch_width"))
        self.coeff_area.objects = view_rows

    def _snapshot(self) -> None:
        """Persist the live controllers' values/LFO config before a rebuild."""
        for key, ctrl in self._controllers.items():
            self._states[key] = ctrl.state()

    def _on_structural(self, event) -> None:
        self._snapshot()
        self._build_controllers()
        self.request_render()

    # -- global actions ---------------------------------------------------- #

    def _randomize(self, event=None) -> None:
        for ctrl in self._controllers.values():
            ctrl.randomize()
        self.request_render()

    def _stop_all(self, event=None) -> None:
        for ctrl in self._controllers.values():
            ctrl.auto.value = False

    # -- automation loop --------------------------------------------------- #

    def _start_automation(self) -> None:
        """Start the single frame-paced render loop. It requires a running event
        loop, which exists in a notebook or served app but not in a bare script;
        in that case we defer to session load (server) and otherwise fall back to
        synchronous rendering on every change."""
        try:
            pn.state.add_periodic_callback(self._tick, period=self._lfo_period)
            self._loop_running = True
        except RuntimeError:
            try:
                pn.state.onload(self._start_automation)
            except Exception:
                pass  # no session context (e.g. plain script): render synchronously

    def _tick(self) -> None:
        # The sole renderer: paint once per frame when something changed or an
        # LFO is animating. Funnelling every update through here means manual
        # edits and LFO frames never issue competing figure swaps (which the
        # Plotly frontend would drop or reorder) -- the latest state always wins.
        if self._dirty or any(c.auto.value for c in self._controllers.values()):
            self._flush()

    # -- rendering --------------------------------------------------------- #

    def request_render(self) -> None:
        """Mark the plot dirty; the frame loop repaints on its next tick. Without
        a running loop (bare script) we paint immediately instead."""
        self._dirty = True
        if not self._loop_running:
            self._flush()

    def _flush(self) -> None:
        self._dirty = False
        t = time.time() - self._t0
        params = {k: c.output(t) for k, c in self._controllers.items()}
        try:
            self.plot_pane.object = self.spec.render(params, self._structural_values())
        except Exception as exc:  # never let a bad frame kill the periodic loop
            import warnings
            warnings.warn(f"render failed: {exc!r}")

    # backwards-compatible alias
    def render(self) -> None:
        self.request_render()

    # -- Panel integration ------------------------------------------------- #

    def __panel__(self):
        return self.layout

    def servable(self, *args, **kwargs):
        return self.layout.servable(*args, **kwargs)
