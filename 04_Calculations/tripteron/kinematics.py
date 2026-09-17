"""Kinematics of the Tripteron CNC router.

The mechanism is a 3-PRRR parallel manipulator: every leg has one actuated
prismatic joint (the lead screw) followed by three revolute joints whose axes
are parallel to the actuator.  The end effector therefore translates without
rotating and each actuator drives one Cartesian axis.

The solution follows the multibody scheme used in the course:

    Phi(q, t) = 0      12 equations, 12 unknowns
    q  = [b1 b2 b3 thC1 thD1 thC2 thD2 thC3 thD3 Px Py Pz]

    * 9 loop closure equations (3 per leg)
    * 3 driving constraints P - f(t) = 0

Position is solved with Newton-Raphson, velocity with the Jacobian and
acceleration with the derivative of the Jacobian.  A closed-form inverse
kinematic solution is also implemented and used to verify the numerical one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import parameters as P

IDX = {"b1": 0, "b2": 1, "b3": 2, "thC1": 3, "thD1": 4,
       "thC2": 5, "thD2": 6, "thC3": 7, "thD3": 8, "Px": 9, "Py": 10, "Pz": 11}
LEG_IDX = {1: (0, 3, 4), 2: (1, 5, 6), 3: (2, 7, 8)}     # b, thetaC, thetaD


# ---------------------------------------------------------------------------
# Constraint equations
# ---------------------------------------------------------------------------
def constraints(q: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Phi(q, t): 9 loop closure equations + 3 driving constraints."""
    phi = np.zeros(12)
    p_tcp = q[9:12]
    for leg, (ib, ic, id_) in LEG_IDX.items():
        p_ax, q_ax = P.PLANE[leg]
        chain = (P.LINK_C * (math.cos(q[ic]) * p_ax + math.sin(q[ic]) * q_ax)
                 + P.LINK_D * (math.cos(q[id_]) * p_ax + math.sin(q[id_]) * q_ax))
        phi[3 * (leg - 1):3 * leg] = P.A[leg] + q[ib] * P.U[leg] + chain + P.E[leg] - p_tcp
    phi[9:12] = p_tcp - path
    return phi


def jacobian(q: np.ndarray) -> np.ndarray:
    """dPhi/dq (12 x 12)."""
    J = np.zeros((12, 12))
    for leg, (ib, ic, id_) in LEG_IDX.items():
        p_ax, q_ax = P.PLANE[leg]
        rows = slice(3 * (leg - 1), 3 * leg)
        J[rows, ib] = P.U[leg]
        J[rows, ic] = P.LINK_C * (-math.sin(q[ic]) * p_ax + math.cos(q[ic]) * q_ax)
        J[rows, id_] = P.LINK_D * (-math.sin(q[id_]) * p_ax + math.cos(q[id_]) * q_ax)
        J[rows, 9:12] = -np.eye(3)
    J[9:12, 9:12] = np.eye(3)
    return J


def jacobian_dot(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Time derivative of the Jacobian for the acceleration analysis."""
    Jd = np.zeros((12, 12))
    for leg, (ib, ic, id_) in LEG_IDX.items():
        p_ax, q_ax = P.PLANE[leg]
        rows = slice(3 * (leg - 1), 3 * leg)
        Jd[rows, ic] = P.LINK_C * qd[ic] * (-math.cos(q[ic]) * p_ax - math.sin(q[ic]) * q_ax)
        Jd[rows, id_] = P.LINK_D * qd[id_] * (-math.cos(q[id_]) * p_ax - math.sin(q[id_]) * q_ax)
    return Jd


# ---------------------------------------------------------------------------
# Closed-form inverse kinematics (used as a reference solution)
# ---------------------------------------------------------------------------
def inverse_kinematics(p_tcp: np.ndarray, elbow_up: bool = True) -> np.ndarray:
    """Analytic solution of q for a given TCP position (raises if unreachable)."""
    q = np.zeros(12)
    q[9:12] = p_tcp
    for leg, (ib, ic, id_) in LEG_IDX.items():
        p_ax, q_ax = P.PLANE[leg]
        r = p_tcp - P.E[leg] - P.A[leg]
        q[ib] = float(r @ P.U[leg])
        u, v = float(r @ p_ax), float(r @ q_ax)
        dist = math.hypot(u, v)
        if dist > P.LINK_C + P.LINK_D or dist < abs(P.LINK_C - P.LINK_D):
            raise ValueError(f"leg {leg}: position out of reach ({dist:.1f} mm)")
        cos_beta = (dist ** 2 - P.LINK_C ** 2 - P.LINK_D ** 2) / (2 * P.LINK_C * P.LINK_D)
        beta = math.acos(max(-1.0, min(1.0, cos_beta)))
        beta = -beta if elbow_up else beta
        gamma = math.atan2(P.LINK_D * math.sin(beta), P.LINK_C + P.LINK_D * math.cos(beta))
        q[ic] = math.atan2(v, u) - gamma
        q[id_] = q[ic] + beta
    return q


def reachable(p_tcp: np.ndarray) -> bool:
    """True when the pose is inside the workspace (reach and stroke limits)."""
    for leg, (ib, ic, id_) in LEG_IDX.items():
        p_ax, q_ax = P.PLANE[leg]
        r = p_tcp - P.E[leg] - P.A[leg]
        b = float(r @ P.U[leg])
        if not (0.0 <= b <= P.STROKE):
            return False
        dist = math.hypot(float(r @ p_ax), float(r @ q_ax))
        # a margin of 5 % of the link length keeps the leg away from the
        # singular fully stretched posture
        if dist > 0.95 * (P.LINK_C + P.LINK_D) or dist < 0.1 * P.LINK_C:
            return False
    return True


# ---------------------------------------------------------------------------
# Newton-Raphson position solution
# ---------------------------------------------------------------------------
def solve_position(path: np.ndarray, q0: np.ndarray, tol: float = 1e-10,
                   max_iter: int = 50) -> tuple[np.ndarray, int, float]:
    q = q0.copy()
    for k in range(1, max_iter + 1):
        phi = constraints(q, path)
        err = float(np.linalg.norm(phi))
        if err < tol:
            return q, k, err
        q = q - np.linalg.solve(jacobian(q), phi)
    return q, max_iter, float(np.linalg.norm(constraints(q, path)))


@dataclass
class Trajectory:
    """Prescribed motion of the TCP."""
    name: str
    position: callable
    velocity: callable
    acceleration: callable
    duration: float

    def sample(self, n: int = 201) -> np.ndarray:
        return np.linspace(0.0, self.duration, n)


def solve_trajectory(traj: Trajectory, n: int = 201) -> dict:
    """Position, velocity and acceleration of every kinematic variable."""
    t = traj.sample(n)
    Q = np.zeros((len(t), 12))
    Qd = np.zeros_like(Q)
    Qdd = np.zeros_like(Q)
    iters = np.zeros(len(t), dtype=int)
    residual = np.zeros(len(t))

    q = inverse_kinematics(traj.position(0.0))
    for k, tk in enumerate(t):
        q, it, res = solve_position(traj.position(tk), q)
        Q[k], iters[k], residual[k] = q, it, res

        J = jacobian(q)
        rhs = np.zeros(12)
        rhs[9:12] = traj.velocity(tk)          # -dPhi/dt
        qd = np.linalg.solve(J, rhs)
        Qd[k] = qd

        rhs2 = np.zeros(12)
        rhs2[9:12] = traj.acceleration(tk)
        Qdd[k] = np.linalg.solve(J, rhs2 - jacobian_dot(q, qd) @ qd)
    return dict(t=t, q=Q, qd=Qd, qdd=Qdd, iterations=iters, residual=residual)


# ---------------------------------------------------------------------------
# Trajectories used in the report
# ---------------------------------------------------------------------------
def straight_line(speed: float = 10.0, length: float = 120.0) -> Trajectory:
    """Case 1: constant speed cut along X (routing speed of the PDS)."""
    duration = length / speed
    return Trajectory(
        "Case 1 - straight cut at constant speed",
        lambda t: np.array([-length / 2 + speed * t, 0.0, 0.0]),
        lambda t: np.array([speed, 0.0, 0.0]),
        lambda t: np.zeros(3),
        duration,
    )


def accelerated_move(accel: float = 20.0, duration: float = 4.0) -> Trajectory:
    """Case 2: rapid move with constant acceleration and then deceleration."""
    half = duration / 2

    def pos(t):
        s = 0.5 * accel * t ** 2 if t <= half else (0.5 * accel * half ** 2
                                                   + accel * half * (t - half)
                                                   - 0.5 * accel * (t - half) ** 2)
        return np.array([0.0, s - 0.5 * accel * half ** 2, 0.0])

    def vel(t):
        v = accel * t if t <= half else accel * (duration - t)
        return np.array([0.0, v, 0.0])

    def acc(t):
        a = accel if t <= half else -accel
        return np.array([0.0, a, 0.0])

    return Trajectory("Case 2 - accelerated rapid move along Y", pos, vel, acc, duration)


def helical_path(radius: float = 60.0, feed_z: float = 1.0, turns: float = 1.0,
                 speed: float = 10.0) -> Trajectory:
    """Case 3: helical pocket, the most demanding path for the three legs."""
    omega = speed / radius
    duration = turns * 2 * math.pi / omega
    return Trajectory(
        "Case 3 - helical pocket",
        lambda t: np.array([radius * math.cos(omega * t), radius * math.sin(omega * t),
                            -feed_z * t]),
        lambda t: np.array([-radius * omega * math.sin(omega * t),
                            radius * omega * math.cos(omega * t), -feed_z]),
        lambda t: np.array([-radius * omega ** 2 * math.cos(omega * t),
                            -radius * omega ** 2 * math.sin(omega * t), 0.0]),
        duration,
    )


def workspace(step: float = 5.0, margin: float = 140.0) -> dict:
    """Sample the reachable positions of the TCP on a regular grid."""
    axis = np.arange(-margin, margin + step, step)
    pts = []
    for x in axis:
        for y in axis:
            for z in axis:
                p = np.array([x, y, z])
                if reachable(p):
                    pts.append(p)
    pts = np.array(pts)
    return dict(points=pts, axis=axis,
                bounds=dict(x=(pts[:, 0].min(), pts[:, 0].max()),
                            y=(pts[:, 1].min(), pts[:, 1].max()),
                            z=(pts[:, 2].min(), pts[:, 2].max())),
                volume_mm3=len(pts) * step ** 3)
