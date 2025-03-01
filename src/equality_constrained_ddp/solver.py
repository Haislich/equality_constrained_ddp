import casadi as cs
import numpy as np
from numpy.typing import NDArray

from equality_constrained_ddp.model import BaseSystem, CartPendulum


class BoundConstrainedLagrangian:
    def __init__(
        self,
        k: float,
        alpha: float,
        eta_zero: float,
        eta_threshold: float,
        omega_zero: float,
        omega_threshold: float,
        max_iterations: int,
        time_horizon: int,
        integration_timestep: float,
        model: BaseSystem = CartPendulum(),
    ) -> None:
        self.k = k
        self.alpha = alpha
        self.eta_zero = eta_zero
        self.eta_threshold = eta_threshold
        self.omega_zero = omega_zero
        self.omega_threshold = omega_threshold
        self.max_iterations = max_iterations
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
        self.Q_terminal: NDArray[np.float32] = np.eye(self.n, dtype=np.float32) * 10000
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

        self.J: cs.Function = cs.Function(
            "J", [self.X, self.U], [self.cost(self.X, self.U)], {"post_expand": True}
        )
        self.constraints = self.model.constraints(self.X, self.U)
        self.LAMBDA: cs.MX = cs.MX.sym("lambda", self.constraints.shape[0])  # type: ignore
        self.MU: cs.MX = cs.MX.sym("mu", 1)  # type: ignore

        self.L_mu: cs.Function = cs.Function(
            "L_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [self.augmented_lagrangian_cost(self.X, self.U, self.LAMBDA, self.MU)],
            {"post_expand": True},
        )

    def running_cost(self, X: cs.MX, U: cs.MX):
        return (self.x_target - X).T @ self.Q @ (self.x_target - X) + U.T @ self.R @ U

    def terminal_cost(self, X: cs.MX):
        return (self.x_target - X).T @ self.Q_terminal @ (self.x_target - X)

    def cost(self, X: cs.MX, U: cs.MX):
        return self.running_cost(X, U) + self.terminal_cost(X)

    def augmented_lagrangian_cost(self, X: cs.MX, U: cs.MX, LAMBDA: cs.MX, MU: cs.MX):
        # Se calcoliamo solo una volta i constraints rimangono fissi
        # e non cambiano al cambiare delle variabili simboliche

        constraints = self.model.constraints(self.X, self.U)
        return (
            self.cost(X, U)
            + cs.dot(LAMBDA.T, constraints)
            + MU * 0.5 * cs.sumsqr(constraints)
        )
