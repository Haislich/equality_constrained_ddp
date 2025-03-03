from typing import List, Tuple

import casadi as cs
import numpy as np
from numpy.typing import NDArray

from equality_constrained_ddp.model import BaseSystem, CartPendulum

np.set_printoptions(precision=5)


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
        *,
        Q_weight: float = 1,
        R_weight: float = 0.1,
        Q_terminal_weight: float = 0,
    ) -> None:
        self.alpha = alpha
        self.eta_zero = eta_zero
        self.eta_threshold = eta_threshold
        self.omega_zero = omega_zero
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
        self.LAMBDA: cs.MX = opt.parameter(self.n)
        self.MU: cs.MX = opt.parameter(1)
        self.Q: NDArray[np.float32] = np.eye(self.n, dtype=np.float32) * Q_weight
        self.R: NDArray[np.float32] = np.eye(self.m, dtype=np.float32) * R_weight
        self.Q_terminal: NDArray[np.float32] = (
            np.eye(self.n, dtype=np.float32) * Q_terminal_weight
        )
        if self.model.name == "cart_pendulum":
            self.x_target = cs.DM(
                [0, cs.pi, 0, 0]
            )  # upright position for cart-pendulum
        elif self.model.name == "pendubot":
            self.x_target = cs.DM([cs.pi, 0, 0, 0])
        elif self.model.name == "uav":
            self.x_target = cs.DM([1, 1, 0, 0, 0, 0])
        else:
            raise ValueError("Unrecognized model type")

        # region: Symbolic definition of the running cost
        self.L: cs.Function = cs.Function(
            "L",
            [self.X, self.U],
            [self.running_cost(self.X, self.U)],
            {"post_expand": True},
        )
        self.Lx: cs.Function = cs.Function(
            "Lx",
            [self.X, self.U],
            [cs.jacobian(self.running_cost(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        self.Lu: cs.Function = cs.Function(
            "Lu",
            [self.X, self.U],
            [cs.jacobian(self.running_cost(self.X, self.U), self.U)],
            {"post_expand": True},
        )
        self.Lxx: cs.Function = cs.Function(
            "Lxx",
            [self.X, self.U],
            [
                cs.jacobian(
                    cs.jacobian(
                        self.running_cost(self.X, self.U),
                        self.X,
                    ),
                    self.X,
                )
            ],
            {"post_expand": True},
        )
        self.Luu: cs.Function = cs.Function(
            "Luu",
            [self.X, self.U],
            [
                cs.jacobian(
                    cs.jacobian(
                        self.running_cost(self.X, self.U),
                        self.U,
                    ),
                    self.U,
                )
            ],
            {"post_expand": True},
        )
        self.Lux: cs.Function = cs.Function(
            "Lux",
            [self.X, self.U],
            [
                cs.jacobian(
                    cs.jacobian(
                        self.running_cost(self.X, self.U),
                        self.U,
                    ),
                    self.X,
                )
            ],
            {"post_expand": True},
        )
        # endregion
        # region: Symbolic definition of the terminal cost
        self.L_terminal: cs.Function = cs.Function(
            "L_terminal",
            [self.X],
            [self.terminal_cost(self.X)],
            {"post_expand": True},
        )
        self.Lx_terminal: cs.Function = cs.Function(
            "Lx_terminal",
            [self.X],
            [cs.jacobian(self.terminal_cost(self.X), self.X)],
            {"post_expand": True},
        )
        self.Lxx_terminal: cs.Function = cs.Function(
            "Lxx_terminal",
            [self.X],
            [
                cs.jacobian(
                    cs.jacobian(
                        self.terminal_cost(self.X),
                        self.X,
                    ),
                    self.X,
                )
            ],
            {"post_expand": True},
        )
        # endregion
        # Symbolic definition of the Augemented lagrangian cost
        self.L_mu: cs.Function = cs.Function(
            "L_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [self.augmented_lagrangian_cost(self.X, self.U, self.LAMBDA, self.MU)],
            {"post_expand": True},
        )
        self.Lx_mu: cs.Function = cs.Function(
            "Lx_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [
                cs.jacobian(
                    self.augmented_lagrangian_cost(
                        self.X, self.U, self.LAMBDA, self.MU
                    ),
                    self.X,
                )
            ],
            {"post_expand": True},
        )

        self.Lxx_mu: cs.Function = cs.Function(
            "Lxx_mu",
            [self.X, self.U, self.LAMBDA, self.MU],
            [
                cs.jacobian(
                    cs.jacobian(
                        self.augmented_lagrangian_cost(
                            self.X, self.U, self.LAMBDA, self.MU
                        ),
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

        self.H: cs.Function = cs.Function("H", [self.X], [self.constraint(self.X)])

        self.V = np.zeros(self.time_horizon + 1)
        self.Vx = np.zeros((self.n, self.time_horizon + 1))
        self.Vxx = np.zeros((self.n, self.n, self.time_horizon + 1))

        self.k: List[NDArray] = [
            np.zeros([self.m, 1]) for _ in range(self.time_horizon)
        ]
        self.K: List[NDArray] = [
            np.zeros([self.m, self.n]) for _ in range(self.time_horizon)
        ]

    def running_cost(self, X: cs.MX, U: cs.MX):
        return (self.x_target - X).T @ self.Q @ (self.x_target - X) + U.T @ self.R @ U

    def terminal_cost(self, X: cs.MX):
        return (self.x_target - X).T @ self.Q_terminal @ (self.x_target - X)

    def cost(self, X: cs.MX, U: cs.MX):
        return self.running_cost(X, U) + self.terminal_cost(X)

    def augmented_lagrangian_cost(self, X: cs.MX, U: cs.MX, LAMBDA: cs.MX, MU: cs.MX):
        return (
            self.cost(X, U)
            + (LAMBDA.T @ self.constraint(X))
            + MU * 0.5 * cs.sumsqr(self.constraint(X))
        )

    def discrete_dynamics(self, X: cs.MX, U: cs.MX):
        return X + self.integration_timestep * self.model.f(X, U)

    def constraint(self, X):
        return X - self.x_target

    def backward_pass(
        self,
        x: NDArray[np.float32],
        u: NDArray[np.float32],
    ):
        x_N = x[:, self.time_horizon]

        self.V[self.time_horizon] = self.L_terminal.call((x_N,))[0].full().squeeze()
        self.Vx[:, self.time_horizon] = (
            self.Lx_terminal.call((x_N,))[0].full().squeeze()
        )
        self.Vxx[:, :, self.time_horizon] = (
            self.Lxx_terminal.call((x_N,))[0].full().squeeze()
        )

        for i in reversed(range(self.time_horizon)):
            x_i = x[:, i]
            u_i = u[:, i]

            fx_i: NDArray = self.Fx.call((x_i, u_i))[0].full()
            fu_i: NDArray = self.Fu.call((x_i, u_i))[0].full()

            Qxx: NDArray = (
                self.Lxx.call((x_i, u_i)) + fx_i.T @ self.Vxx[:, :, i + 1] @ fx_i
            ).reshape((self.n, self.n))
            Qux: NDArray = (
                self.Lux.call((x_i, u_i)) + fu_i.T @ self.Vxx[:, :, i + 1] @ fx_i
            ).reshape((self.m, self.n))
            Quu: NDArray = (
                self.Luu.call((x_i, u_i)) + fu_i.T @ self.Vxx[:, :, i + 1] @ fu_i
            ).reshape((self.m, self.m))
            Qx: NDArray = (
                self.Lx.call((x_i, u_i)) + self.Vx[:, i + 1].T @ fx_i
            ).reshape((self.n,))
            Qu: NDArray = (
                self.Lu.call((x_i, u_i)) + self.Vx[:, i + 1].T @ fu_i
            ).reshape((self.m,))

            q: float = self.L.call((x_i, u_i)) + self.V[i + 1]

            # Regularize Quu to ensure invertibility
            # Quu_reg = Quu + 1e-5
            Quu_inv = np.linalg.inv(Quu)

            self.k[i] = -Quu_inv @ Qu
            self.K[i] = -Quu_inv @ Qux

            self.Vxx[:, :, i] = Qxx - self.K[i].T @ Quu @ self.K[i]
            self.Vx[:, i] = Qx - self.K[i].T @ Quu @ self.k[i]
            self.V[i] = (q - 0.5 * (self.k[i].T @ Quu @ self.k[i])).item()

    def forward_pass(
        self, x: NDArray[np.float32], u: NDArray[np.float32], lam, mu, alpha: float
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        # lam = 0
        x_new = np.zeros([self.n, self.time_horizon + 1], dtype=np.float32)
        u_new = np.ones([self.m, self.time_horizon], dtype=np.float32)
        # Set the initial condition
        x_new[:, 0] = x[:, 0]
        new_cost = 0.0

        for i in range(self.time_horizon):
            # Apply updated control policy
            # print(u[:, i], alpha * self.k[i], self.K[i], x[:, i], x_new[:, i])
            u_new[:, i] = (
                u[:, i] + alpha * self.k[i] + self.K[i] @ (x_new[:, i] - x[:, i])
            )

            # Integrate system dynamics (e.g., Euler method)
            x_new[:, i + 1] = (
                self.F.call((x_new[:, i], u_new[:, i]))[0].full().reshape((self.n,))
            )
            new_cost += (
                self.L_mu.call((x_new[:, i + 1], u_new[:, i], lam, mu))[0]
                .full()
                .squeeze()
            )

        return x_new, u_new, new_cost

    def solve(self):
        # Initialize the trajectories as described in Eq 1, remembering that we have
        # One less input w.r.t the state
        x = np.zeros([self.n, self.time_horizon + 1], dtype=np.float32)
        u = np.ones([self.m, self.time_horizon], dtype=np.float32)
        lam = np.zeros([self.n], dtype=np.float32)
        mu = 1.1
        eta = self.eta_zero
        omega = self.omega_zero
        k = 10
        alpha = 1.0

        # We get an initial guess about where we are currently cost-wise
        cost = 0
        for i in range(self.time_horizon):
            x[:, i + 1] = self.F.call((x[:, i], u[:, i]))[0].full().reshape((self.n,))
            cost += self.L_mu.call((x[:, i + 1], u[:, i], lam, mu))[0].full().item()
        iteration = 0
        # Then we try to improve with what we know
        while eta > self.eta_threshold and omega > self.omega_threshold:
            iteration += 1
            # Compute the gains and value function updates
            self.backward_pass(x, u)
            alpha = self.alpha
            beta = 0.5
            new_cost = float("inf")
            x_new = x.copy()
            u_new = u.copy()
            for _ in range(self.max_line_search_iters):
                new_cost = 0
                x_new, u_new, new_cost = self.forward_pass(x, u, lam, mu, alpha)
                # If cost improves, accept step
                if new_cost < cost:
                    cost = new_cost
                    break
                else:
                    alpha *= beta

            x = x_new
            u = u_new
            L_norms = np.array(
                [
                    self.Lx_mu.call((x[:, i + 1], u[:, i], lam, mu))[0].full().squeeze()
                    for i in range(self.time_horizon)
                ]
            )

            L_norm = np.linalg.norm(L_norms, np.inf)
            c = self.H.call((x[:, self.time_horizon],))[0].full().reshape((self.n,))
            # print(c)
            c_norm = np.linalg.norm(
                c,
                np.inf,
            )
            if L_norm < omega:
                if c_norm < eta:
                    lam += c * mu
                    eta /= np.pow(mu, alpha)
                    omega /= mu

                else:
                    mu *= k

            print(
                f"Iteration: {iteration:5d} | "
                f"L_norm:{L_norm:.4f} | "
                f"eta: {eta:.4f} | "
                f"omega: {omega:.4f} | "
                f"mu: {mu:.4f} | "
                f"lambda: {lam.tolist()} | "
                f"|| h ||: {c_norm} | "
                f"Alpha: {alpha:.5f} | "
            )


BoundConstrainedLagrangian(
    time_horizon=100, eta_zero=10, omega_zero=15, max_line_search_iters=10
).solve()
