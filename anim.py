"""
One test star, driven by the slider. The rest of the cluster is a static
bound background (fixed circular orbits, not affected by the slider) so
it's clear what "leaving the cluster" looks like against a stable backdrop.

    E_i = (1/2) m_i (v_i - v_cluster)^2  -  sum_{j != i} G m_i m_j / r_ij

For the test star, j ranges over the background cluster members only.
Simplifications (same as before):
  - all masses = 1, so this is a specific (per-mass) energy
  - cluster center is fixed, so v_cluster = 0 in this frame
  - the test star's speed is set directly from the slider and its own
    escape speed (kinematic shortcut, not a force-integrated N-body run)

Convention:
    E < 0   -> bound, test star stays with the cluster
    E >= 0  -> stripped, test star leaves

Two scenes:

1. EnergySweepDemo
   Renders to mp4, slider is animated by code.
       manim -pql energy_slider_cluster.py EnergySweepDemo

2. InteractiveEnergySlider
   Real draggable slider, live only (OpenGL renderer).
       manim -p --renderer=opengl energy_slider_cluster.py InteractiveEnergySlider
"""

import random
import numpy as np
from manim import *

G = 1.0
SOFTENING = 0.15
N_BACKGROUND = 14
TIME_SCALE = 0.35
BACKGROUND_OMEGA = 0.4  # fixed angular speed, background stars never strip


def init_background(n: int, seed: int = 1):
    random.seed(seed)
    stars = []
    for _ in range(n):
        stars.append(
            {"r": random.uniform(0.4, 1.3), "angle": random.uniform(0, TAU)}
        )
    return stars


def background_positions(bg_state) -> np.ndarray:
    return np.array(
        [[s["r"] * np.cos(s["angle"]), s["r"] * np.sin(s["angle"])] for s in bg_state]
    )


def test_star_binding(test_pos: np.ndarray, bg_positions: np.ndarray) -> float:
    diff = bg_positions - test_pos
    r = np.sqrt(np.sum(diff**2, axis=1) + SOFTENING**2)
    return float(np.sum(G / r))


def make_background_updater(bg_state, center):
    def updater(mob, dt):
        for dot, s in zip(mob, bg_state):
            s["angle"] += BACKGROUND_OMEGA * dt
            pos = center + s["r"] * np.array([np.cos(s["angle"]), np.sin(s["angle"]), 0])
            dot.move_to(pos)

    return updater


def make_test_star_updater(test_state, bg_state, center, get_slider, readout_state):
    def updater(mob, dt):
        bg_pos = background_positions(bg_state)
        binding = test_star_binding(test_state["pos"], bg_pos)
        v_esc = np.sqrt(2 * binding)

        slider = get_slider()
        speed = v_esc * (1 + 0.5 * slider)  # slider=0 -> speed=v_esc -> E=0
        energy = 0.5 * speed**2 - binding

        r_vec = test_state["pos"]
        r_norm = np.linalg.norm(r_vec) + 1e-6
        tangent = np.array([-r_vec[1], r_vec[0]]) / r_norm
        vel = tangent * speed
        test_state["pos"] = test_state["pos"] + vel * dt * TIME_SCALE

        bound = energy < 0
        mob.set_color(GOLD if bound else RED)
        mob.move_to(center + np.array([test_state["pos"][0], test_state["pos"][1], 0]))

        readout_state["E"] = energy
        readout_state["bound"] = bound

    return updater


def build_common_mobjects():
    track = NumberLine(x_range=[-1, 1, 0.5], length=6, include_numbers=True)
    track.to_edge(DOWN, buff=1.3)

    eq = MathTex(
        r"E = \tfrac{1}{2}m(\vec v - \vec v_{\text{cluster}})^2",
        r"-\sum_{j} G\frac{m\, m_j}{r_{j}}",
        font_size=34,
    ).to_edge(UP)

    return track, eq


def build_scene_contents(scene, slider, center=UP * 1.0):
    track, eq = build_common_mobjects()

    bg_state = init_background(N_BACKGROUND)
    background = VGroup(*[Dot(radius=0.06, color=BLUE_B) for _ in bg_state])
    background.add_updater(make_background_updater(bg_state, center))

    test_state = {"pos": np.array([1.0, 0.0])}
    readout_state = {"E": 0.0, "bound": True}
    test_star = Dot(radius=0.1, color=GOLD)
    test_star.add_updater(
        make_test_star_updater(test_state, bg_state, center, slider.get_value, readout_state)
    )

    e_label = always_redraw(
        lambda: MathTex(f"E = {readout_state['E']:.2f}", font_size=32).next_to(
            test_star, UR, buff=0.15
        )
    )
    status_label = always_redraw(
        lambda: Text(
            "bound" if readout_state["bound"] else "stripped",
            font_size=26,
            color=GOLD if readout_state["bound"] else RED,
        ).next_to(track, DOWN, buff=0.4)
    )

    return track, eq, background, test_star, e_label, status_label


class EnergySweepDemo(Scene):
    def construct(self):
        slider = ValueTracker(-1.0)
        track, eq, background, test_star, e_label, status_label = build_scene_contents(
            self, slider
        )

        handle = Dot(color=YELLOW, radius=0.12)
        handle.add_updater(lambda m: m.move_to(track.n2p(slider.get_value())))
        slider_readout = always_redraw(
            lambda: MathTex(f"{slider.get_value():.2f}").next_to(track, UP, buff=0.5)
        )

        self.add(eq, track, handle, slider_readout, background, test_star, e_label, status_label)
        self.wait(1)
        self.play(slider.animate.set_value(0.3), run_time=4, rate_func=linear)
        self.wait(2)


class InteractiveEnergySlider(Scene):
    def construct(self):
        self.slider = ValueTracker(-0.6)
        track, eq, background, test_star, e_label, status_label = build_scene_contents(
            self, self.slider
        )
        self.track = track

        self.handle = Dot(color=YELLOW, radius=0.14)
        self.handle.add_updater(lambda m: m.move_to(self.track.n2p(self.slider.get_value())))
        slider_readout = always_redraw(
            lambda: MathTex(f"{self.slider.get_value():.2f}").next_to(self.track, UP, buff=0.5)
        )

        self.add(eq, self.track, self.handle, slider_readout, background, test_star, e_label, status_label)
        self.interactive_embed()

    def on_mouse_drag(self, point, d_point, buttons, modifiers):
        x = self.track.p2n(point)
        x = max(-1.0, min(1.0, x))
        self.slider.set_value(x)