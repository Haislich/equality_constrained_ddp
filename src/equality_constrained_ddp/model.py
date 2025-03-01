import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle


# We shouldn't be able to instantiate this system
class BaseSystem(ABC):
    f: Callable

    def __init__(self, name: str, n: int, m: int, g: float = 9.81):
        r"""
        A base class for dynamic systems.

        This class serves as a foundational structure for modeling dynamic systems,
        such as the cart-pendulum. It defines common properties like the number of
        state variables, control inputs, and the gravitational constant.

        Attributes:
        ---

            - name (str): The name of the model or system.
            - n (int): The number of state variables defining the system's state space.
            - m (int): The number of control inputs or actuators in the system.
            - g (float): The gravitational constant, defaulting to 9.81 m/s^2.
        """
        self.name = name
        """Name of the model"""
        self.n = n
        """Number of state variables"""
        self.m = m
        """Number of control inputs"""
        self.g = g
        """Gravitational contant"""

    def setup_animation(self, N_sim, x, u, save_frames):
        grid = GridSpec(2, 2)
        if not save_frames:
            self.ax_large = plt.subplot(grid[:, 0])
            self.ax_small1 = plt.subplot(grid[0, 1])
            self.ax_small2 = plt.subplot(grid[1, 1])
        else:
            self.ax_large = plt.subplot(grid[:, :])

        self.x_max = max(x.min(), x.max(), key=abs)
        self.u_max = max(u.min(), u.max(), key=abs)
        self.N_sim = N_sim

    def update_small_axes(self, x, u, i):
        self.ax_small1.cla()
        self.ax_small1.axis((0, self.N_sim, -self.x_max * 1.1, self.x_max * 1.1))
        self.ax_small1.plot(x[:, :i].T)

        self.ax_small2.cla()
        self.ax_small2.axis((0, self.N_sim, -self.u_max * 1.1, self.u_max * 1.1))
        self.ax_small2.plot(u[:, :i].T)

    def animate(
        self,
        N_sim,
        x,
        u,
        show_trail=False,
        save_video=False,
        video_filename="animation.mp4",
        save_frames=False,
        frame_number=0,
        frame_folder="frames",
        x_pred=None,
    ):
        self.setup_animation(N_sim, x, u, save_frames)
        if x_pred is not None:
            x_pred.append(x_pred[-1])  # replicate last prediction to avoid crash

        frame_indices = (
            np.linspace(0, self.N_sim, frame_number, dtype=int)
            if save_frames and frame_number > 0
            else []
        )

        if save_frames and frame_number > 0:
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

        def update_frame(i):
            self.ax_large.cla()

            trail_length = show_trail * 10
            spacing = 10
            trail_indices = [
                i - j * spacing for j in range(trail_length) if i - j * spacing >= 0
            ]

            for idx, j in enumerate(trail_indices):
                alpha = 1.0 - (
                    idx / (len(trail_indices) + 1)
                )  # make older frames more faded
                alpha /= 4.0  # make the trail more faded
                self.draw_frame(self.ax_large, j, x, u, alpha=alpha, x_pred=x_pred)

            self.draw_frame(self.ax_large, i, x, u, alpha=1.0, x_pred=x_pred)
            if not save_frames:
                self.update_small_axes(x, u, i)

            if save_frames and i in frame_indices:
                plt.savefig(os.path.join(frame_folder, f"frame_{i}.png"))
            return (self.ax_large,)

        ani = FuncAnimation(
            plt.gcf(), update_frame, frames=self.N_sim + 1, repeat=True, interval=10
        )  # type: ignore

        if save_video:
            writer = FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
            ani.save(video_filename, writer=writer)
        else:
            plt.show()

    @abstractmethod
    def draw_frame(self, ax, i, x, u, alpha=1.0, x_pred=None): ...
    @abstractmethod
    def constraints(self, x: cs.MX, u: cs.MX) -> cs.DM | Any: ...


@dataclass
class CartPendulumParameters:
    l: float
    """Lenght of the pendulum"""

    m1: float
    """Mass of the cart"""

    m2: float
    """Mass of the pendulum"""

    b1: float
    """Damping coefficient for the cart"""

    b2: float
    """Damping coefficient for the pendulum"""


class CartPendulum(BaseSystem):
    def __init__(self):
        r"""
        A dynamic system representing a cart-pendulum model.

        This class models the dynamics of a cart-pendulum system, where a pendulum
        is attached to a cart that moves along a linear track.

        Attributes:
        ---
            - p (Parameters): The physical parameters of the system, including:
                - l (float): Length of the pendulum.
                - m1 (float): Mass of the cart.
                - m2 (float): Mass of the pendulum.
                - b1 (float): Damping coefficient for the cart.
                - b2 (float): Damping coefficient for the pendulum.
            - f1 (callable): Lambda function representing the horizontal acceleration
                of the cart $\ddot{x}$ based on the system state and control input.
            - f2 (callable): Lambda function representing the angular acceleration
                of the pendulum $\ddot{\theta}$ based on the system state and control input.
            - f (callable): Lambda function representing the full state-space dynamics
                of the system, combining $[x, \theta, \dot{x}, \dot{\theta}]$ and their derivatives.
        """
        super().__init__("cart_pendulum", 4, 1)
        self.p = CartPendulumParameters(l=1, m1=2, m2=1, b1=0, b2=0)
        self.f1 = (
            lambda x, u: (
                self.p.l * self.p.m2 * cs.sin(x[1]) * x[3] ** 2
                + u
                + self.p.m2 * self.g * cs.cos(x[1]) * cs.sin(x[1])
            )
            / (self.p.m1 + self.p.m2 * (1 - cs.cos(x[1]) ** 2))
            - self.p.b1 * x[2]
        )
        """Equation of motion of the cart, gives the horizontal acceleration of the cart."""
        self.f2 = (
            lambda x, u: -(
                self.p.l * self.p.m2 * cs.cos(x[1]) * cs.sin(x[1]) * x[3] ** 2
                + u * cs.cos(x[1])
                + (self.p.m1 + self.p.m2) * self.g * cs.sin(x[1])
            )
            / (self.p.l * self.p.m1 + self.p.l * self.p.m2 * (1 - cs.cos(x[1]) ** 2))
            - self.p.b2 * x[3]
        )
        """Euqation of motion of the pendulum, gives the horizontal acceleration of the cart."""

        self.f = lambda x, u: cs.vertcat(x[2:4], self.f1(x, u), self.f2(x, u))
        """State space dynamics of the entire system."""

    def draw_frame(self, ax, i, x, u, alpha=1.0, x_pred=None):
        ax.axis((-1.5, 1.5, -1.5, 1.5))
        ax.set_aspect("equal")

        if x_pred is not None:
            x_p = x_pred[i]
            tip_x_pred = x_p[0, :] + np.sin(x_p[1, :])
            tip_y_pred = -np.cos(x_p[1, :])
            ax.plot(tip_x_pred, tip_y_pred, color="orange", alpha=alpha)

        ax.plot(
            x[0, i]
            + np.array((self.p.l, self.p.l, -self.p.l, -self.p.l, +self.p.l)) / 4,
            np.array((self.p.l, -self.p.l, -self.p.l, self.p.l, self.p.l)) / 4,
            color="orange",
            alpha=alpha,
        )
        ax.add_patch(
            Circle(
                (x[0, i] + math.sin(x[1, i]), -math.cos(x[1, i])),
                self.p.l / 8,
                color="blue",
                alpha=alpha,
            )
        )
        ax.plot(
            np.array((x[0, i], x[0, i] + math.sin(x[1, i]))),
            np.array((0, -math.cos(x[1, i]))),
            color="black",
            alpha=alpha,
        )

    def constraints(self, x, u):
        r"""
        Defines the equality constraints for the cart-pendulum system.

        This method specifies conditions that must be satisfied during
        the trajectory optimization process. In this case, the constraint
        ensures that the pendulum reaches or maintains the upright position.
        """
        h2 = x[1] - cs.pi
        return cs.vertcat(h2)


@dataclass
class UavParameters:
    m: float
    """Mass of the UAV"""

    I: float
    """Moment of inertia of the UAV"""

    fr_x: float
    """Friction coefficient in the x-direction"""

    fr_z: float
    """Friction coefficient in the z-direction"""

    fr_theta: float
    """Friction coefficient for rotational motion"""

    width: float
    """Width of the UAV"""


class Uav(BaseSystem):
    def __init__(self):
        super().__init__("uav", 6, 2)
        self.p = UavParameters(
            m=1.0, I=0.01, fr_x=0.01, fr_z=0.01, fr_theta=0.01, width=0.2
        )

        self.f1 = lambda x, u: x[3]
        self.f2 = lambda x, u: x[4]
        self.f3 = lambda x, u: x[5]

        self.f4 = (
            lambda x, u: (-0 * self.p.fr_x * x[3] + (u[0] + u[1]) * cs.sin(x[2]))
            / self.p.m
        )
        """The horizontal acceleration of the UAV influenced by the total thrust, the orientation of the UAV and the friction in the horizontal direction."""

        self.f5 = (
            lambda x, u: (
                -0 * self.p.fr_z * x[4]
                - self.p.m * self.g
                + (u[0] + u[1]) * cs.cos(x[2])
            )
            / self.p.m
        )
        """The vertical acceleration of the UAV influenced by the thrust in the verstical direction, gravity and the friction in the vertical direction."""

        self.f6 = (
            lambda x, u: (
                -0 * self.p.fr_theta * x[5] + (self.p.width / 2) * (u[1] - u[0])
            )
            / self.p.I
        )
        """The angular acceleration of the UAV influenced by the difference in thrust on the two rotors, the width of the UAV, that acts as lever arm and the rotation friction. Finally the moment of inertia is used to normalized the torque into angular acceleration."""

        self.f = lambda x, u: cs.vertcat(
            self.f1(x, u),
            self.f2(x, u),
            self.f3(x, u),
            self.f4(x, u),
            self.f5(x, u),
            self.f6(x, u),
        )
        """State space dynamics of the entire system."""

    def draw_frame(self, ax, i, x, u, alpha=1.0, x_pred=None):
        ax.axis((-2, 2, -2, 2))
        ax.set_aspect("equal")

        uav_length = 0.5
        uav_width = self.p.width
        x_pos = x[0, i]
        z_pos = x[1, i]
        theta = x[2, i]

        body_x1 = (
            x_pos
            + (uav_length * math.cos(theta)) / 2
            - (uav_width * math.sin(theta)) / 2
        )
        body_z1 = (
            z_pos
            - (uav_length * math.sin(theta)) / 2
            - (uav_width * math.cos(theta)) / 2
        )
        body_x2 = (
            x_pos
            - (uav_length * math.cos(theta)) / 2
            - (uav_width * math.sin(theta)) / 2
        )
        body_z2 = (
            z_pos
            + (uav_length * math.sin(theta)) / 2
            - (uav_width * math.cos(theta)) / 2
        )
        body_x3 = (
            x_pos
            - (uav_length * math.cos(theta)) / 2
            + (uav_width * math.sin(theta)) / 2
        )
        body_z3 = (
            z_pos
            + (uav_length * math.sin(theta)) / 2
            + (uav_width * math.cos(theta)) / 2
        )
        body_x4 = (
            x_pos
            + (uav_length * math.cos(theta)) / 2
            + (uav_width * math.sin(theta)) / 2
        )
        body_z4 = (
            z_pos
            - (uav_length * math.sin(theta)) / 2
            + (uav_width * math.cos(theta)) / 2
        )

        ax.plot(
            [body_x1, body_x2, body_x3, body_x4, body_x1],
            [body_z1, body_z2, body_z3, body_z4, body_z1],
            color="blue",
            lw=2,
            alpha=alpha,
        )

        ax.add_patch(
            Circle(
                (x_pos, z_pos),
                uav_length / 10,
                color="green",
                alpha=alpha,
            )
        )

        if x_pred is not None:
            x_p = x_pred[i]
            uav_x_pred = x_p[0, :]
            uav_y_pred = x_p[1, :]
            ax.plot(uav_x_pred, uav_y_pred, color="orange", alpha=alpha)

        thrust_scale = 0.05
        thrust_length_left = max(u[1, i], 0.001) / self.p.m * thrust_scale
        thrust_length_right = max(u[0, i], 0.001) / self.p.m * thrust_scale

        left_x = x_pos - (uav_width / 2) * math.cos(theta)
        left_z = z_pos + (uav_width / 2) * math.sin(theta)
        right_x = x_pos + (uav_width / 2) * math.cos(theta)
        right_z = z_pos - (uav_width / 2) * math.sin(theta)

        thrust_x_left = -thrust_length_left * math.sin(theta)
        thrust_z_left = -thrust_length_left * math.cos(theta)
        thrust_x_right = -thrust_length_right * math.sin(theta)
        thrust_z_right = -thrust_length_right * math.cos(theta)

        ax.arrow(
            left_x,
            left_z,
            thrust_x_left,
            thrust_z_left,
            color="red",
            head_width=0.05,
            head_length=0.05,
            alpha=alpha,
        )
        ax.arrow(
            right_x,
            right_z,
            thrust_x_right,
            thrust_z_right,
            color="red",
            head_width=0.05,
            head_length=0.05,
            alpha=alpha,
        )

    def constraints(self, x, u):
        """
        Defines the equality constraints for the UAV model.

        This function specifies that the UAV must land at position (0, 0, 0),
        meaning its x-position, z-position, and orientation should be 0.
        """

        h1 = x[0]  # Enforce x-position to be 0
        h2 = x[1]  # Enforce z-position to be 0
        h3 = x[2]  # Enforce orientation to be 0
        return cs.vertcat(h1, h2, h3)
