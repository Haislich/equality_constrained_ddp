from typing import Tuple

import casadi as cs
import numpy as np
from numpy.typing import NDArray

from equality_constrained_ddp.model import BaseSystem, CartPendulum


class BoundConstrainedLagrangian:
    def __init__(
        self,
        alpha: float = 1.0,
        eta_zero: float = 1.0,
        eta_threshold: float = 0.6,
        omega_zero: float = 15.0,
        omega_threshold: float = 1.0,
        max_line_search_iters: int = 10,
        time_horizon: int = 100,
        integration_timestep: float = 0.01,
        model: BaseSystem = CartPendulum(),
    ) -> None:
        self.alpha = alpha
        self.eta = eta_zero
        self.eta_threshold = eta_threshold
        self.omega = omega_zero
        self.omega_threshold = omega_threshold
        self.max_line_search_iters = max_line_search_iters
        self.time_horizon = time_horizon
        self.integration_timestep = integration_timestep
        self.model = model

        self.n = model.n
        self.m = model.m
        opt = cs.Opti()
        self.X: cs.MX = opt.variable(self.n)  # symbolic state vector
        self.U: cs.MX = opt.variable(self.m)
        self.Q: NDArray[np.float32] = np.eye(self.n, dtype=np.float32)
        self.R: NDArray[np.float32] = np.eye(self.m, dtype=np.float32) * 0.01
        self.Q_terminal: NDArray[np.float32] = np.eye(self.n, dtype=np.float32) * 0
        # TODO: This should be embedded inside the models.
        if self.model.name == "cart_pendulum":
            self.x_target = np.array(
                [0, cs.pi, 0, 0]
            )  # upright position for cart-pendulum
        elif self.model.name == "pendubot":
            self.x_target = np.array([cs.pi, 0, 0, 0])
        elif self.model.name == "uav":
            self.x_target = np.array([1, 1, 0, 0, 0, 0])
        else:
            raise ValueError("Unrecognized model type")

        self.h_dim = self.model.constraints(self.X, self.U).shape[0]
        self.LAMBDA: cs.MX = cs.MX.sym("lambda", self.h_dim)  # type: ignore
        self.MU: cs.MX = cs.MX.sym("mu", 1)  # type: ignore
        self.J: cs.Function = cs.Function(
            "J", [self.X, self.U], [self.cost(self.X, self.U)], {"post_expand": True}
        )
        self.L_mu: cs.Function = cs.Function(
            "L_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [self.augmented_lagrangian_cost(self.X, self.U, self.LAMBDA, self.MU)],
            {"post_expand": True},
        )
        self.Lx_mu: cs.Function = cs.Function(
            "Lx_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [cs.jacobian(self.L_mu(self.X, self.U, self.LAMBDA, self.MU), self.X)],
            {"post_expand": True},
        )

        self.Lxx_mu: cs.Function = cs.Function(
            "Lxx_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [
                cs.jacobian(
                    cs.jacobian(
                        self.L_mu(self.X, self.U, self.LAMBDA, self.MU),
                        self.X,
                    ),
                    self.X,
                )
            ],
            {"post_expand": True},
        )
        self.F: cs.Function = cs.Function(
            "F",
            [self.X, self.U],
            [self.discrete_dynamics(self.X, self.U)],
            {"post_expand": True},
        )
        self.Fx: cs.Function = cs.Function(
            "Fx",
            [self.X, self.U],
            [cs.jacobian(self.discrete_dynamics(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        self.Fu: cs.Function = cs.Function(
            "Fu",
            [self.X, self.U],
            [cs.jacobian(self.discrete_dynamics(self.X, self.U), self.U)],
            {"post_expand": True},
        )
        self.V = np.zeros(self.time_horizon + 1)
        self.Vx = np.zeros((self.n, self.time_horizon + 1))
        self.Vxx = np.zeros((self.n, self.n, self.time_horizon + 1))

        self.k = [np.zeros((self.m, 1)) for _ in range(self.time_horizon)]
        self.K = [np.zeros((self.m, self.n)) for _ in range(self.time_horizon)]

    def running_cost(self, X: cs.MX, U: cs.MX):
        return (self.x_target - X).T @ self.Q @ (self.x_target - X) + U.T @ self.R @ U

    def terminal_cost(self, X: cs.MX):
        return (self.x_target - X).T @ self.Q_terminal @ (self.x_target - X)

    def cost(self, X: cs.MX, U: cs.MX):
        return self.running_cost(X, U) + self.terminal_cost(X)

    def augmented_lagrangian_cost(self, X: cs.MX, U: cs.MX, LAMBDA: cs.MX, MU: cs.MX):
        # Se calcoliamo solo una volta i constraints rimangono fissi
        # e non cambiano al cambiare delle variabili simboliche, in teoria va fatto
        # perche constraints e' una DM non una MX

        constraints = self.model.constraints(X, U)
        return (
            self.cost(X, U)
            + cs.dot(LAMBDA.T, constraints)
            + MU * 0.5 * cs.sumsqr(constraints)
        )

    def discrete_dynamics(self, X: cs.MX, U: cs.MX):
        return X + self.integration_timestep * self.model.f(X, U)

    def backward_pass(
        self,
        x: NDArray[np.float32],
        u: NDArray[np.float32],
        lam: NDArray[np.float32],
        mu: float,
    ):
        x_N = x[:, self.time_horizon]
        u_N = u[:, self.time_horizon - 1]

        self.V[self.time_horizon] = (
            self.L_mu.call((x_N, u_N, lam, mu))[0].full().squeeze()
        )
        self.Vx[:, self.time_horizon] = (
            self.Lx_mu.call((x_N, u_N, lam, mu))[0].full().squeeze()
        )
        self.Vxx[:, :, self.time_horizon] = (
            self.Lxx_mu.call((x_N, u_N, lam, mu))[0].full().squeeze()
        )
        for i in reversed(range(self.time_horizon)):
            x_i = x[:, i]
            u_i = u[:, i]

            fx_i = self.Fx.call((x_i, u_i))
            fu_i = self.Fx.call((x_i, u_i))

    def forward_pass(self): ...

    def solver(self):
        # Initialize the trajectories as described in Eq 1, remembering that we have
        # One less input w.r.t the state
        x = np.zeros([self.n, self.time_horizon + 1], dtype=np.float32)
        u = np.ones([self.m, self.time_horizon], dtype=np.float32)
        lam = np.zeros([self.h_dim], dtype=np.float32)
        mu = 1.1

        cost = 0
        for i in range(self.time_horizon):
            x[:, i + 1] = self.F.call((x[:, i], u[:, i]))[0].full().squeeze()
            cost += self.L_mu.call((x[:, i], u[:, i], lam, mu))[0].full().squeeze()

        while self.eta > self.eta_threshold and self.omega > self.omega_threshold:
            self.backward_pass(x, u, lam, mu)


BoundConstrainedLagrangian(time_horizon=3).solver()
