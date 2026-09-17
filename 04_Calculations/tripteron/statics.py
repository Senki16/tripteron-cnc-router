"""Static analysis of the Tripteron CNC router.

Every body of the machine (three sliders, six links and the end effector) is
written as a free body and the 60 equilibrium equations are assembled into one
linear system whose unknowns are the joint reactions and the three actuator
forces.

    * revolute joint  -> 3 force components + 2 moment components
                         (a revolute pair cannot carry a moment about its axis)
    * prismatic joint -> 2 force components + 3 moment components, plus the
                         actuator force along the axis of the screw

The machine is over-constrained in rotation: the three guides restrain the
rotation of the platform about three axes while only three constraints are
needed, so the system has three redundant reactions.  The solution reported is
the minimum-norm least-squares solution, which is the usual engineering choice
when the stiffness distribution is unknown; the residual of the equilibrium
equations is printed as a check.
"""

from __future__ import annotations

import math

import numpy as np

from . import kinematics as K
from . import parameters as P

M_SLIDER = 2.0          # nut, carriage plate and bearing block of one leg [kg]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def leg_points(q: np.ndarray, leg: int) -> dict[str, np.ndarray]:
    """Position of the joints of one leg for the kinematic state q."""
    ib, ic, id_ = K.LEG_IDX[leg]
    p_ax, q_ax = P.PLANE[leg]
    A = P.A[leg] + q[ib] * P.U[leg]                      # revolute at the slider
    B = A + P.LINK_C * (math.cos(q[ic]) * p_ax + math.sin(q[ic]) * q_ax)
    C = B + P.LINK_D * (math.cos(q[id_]) * p_ax + math.sin(q[id_]) * q_ax)
    return dict(slider=A, elbow=B, platform=C,
                c_mid=0.5 * (A + B), d_mid=0.5 * (B + C))


def solve(q: np.ndarray, f_cut: np.ndarray | None = None, gravity: bool = True) -> dict:
    """Joint reactions and actuator forces for one pose."""
    f_cut = np.zeros(3) if f_cut is None else np.asarray(f_cut, dtype=float)
    p_tcp = q[9:12]

    # ---- unknown layout ---------------------------------------------------
    # per leg: guide wrench (6: 3 forces incl. the actuator + 3 moments),
    #          pin A (5), pin B (5), pin P (5)
    names: list[str] = []
    for leg in (1, 2, 3):
        names += [f"guide{leg}_F{c}" for c in "xyz"] + [f"guide{leg}_M{c}" for c in "xyz"]
        for pin in ("A", "B", "P"):
            names += [f"{pin}{leg}_F{c}" for c in "xyz"] + [f"{pin}{leg}_M1", f"{pin}{leg}_M2"]
    index = {n: i for i, n in enumerate(names)}
    n_unknowns = len(names)

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def add_body(force_cols, moment_cols, ext_force, ext_moment):
        """6 equilibrium equations of one body.

        force_cols  : list of (column, 3x3 matrix) applying a force unknown
        moment_cols : list of (column, 3x3 matrix) applying a moment unknown
        """
        for k in range(3):
            row = np.zeros(n_unknowns)
            for col, mat in force_cols:
                row[col] = mat[k]
            rows.append(row)
            rhs.append(-ext_force[k])
        for k in range(3):
            row = np.zeros(n_unknowns)
            for col, mat in moment_cols:
                row[col] = mat[k]
            rows.append(row)
            rhs.append(-ext_moment[k])

    def pin_cols(pin: str, leg: int, point: np.ndarray, ref: np.ndarray, sign: float):
        """Columns of the force/moment unknowns of a revolute pair."""
        u = P.U[leg]
        p_ax, q_ax = P.PLANE[leg]
        r = point - ref
        fcols, mcols = [], []
        for k, c in enumerate("xyz"):
            col = index[f"{pin}{leg}_F{c}"]
            e = np.zeros(3); e[k] = sign
            fcols.append((col, e))
            mcols.append((col, _skew(r) @ e))
        for k, axis in enumerate((p_ax, q_ax), start=1):
            col = index[f"{pin}{leg}_M{k}"]
            mcols.append((col, sign * axis))
        return fcols, mcols

    def guide_cols(leg: int, point: np.ndarray, ref: np.ndarray, sign: float):
        r = point - ref
        fcols, mcols = [], []
        for k, c in enumerate("xyz"):
            e = np.zeros(3); e[k] = sign
            col = index[f"guide{leg}_F{c}"]
            fcols.append((col, e))
            mcols.append((col, _skew(r) @ e))
            col = index[f"guide{leg}_M{c}"]
            mcols.append((col, sign * e))
        return fcols, mcols

    g = np.array([0.0, 0.0, -P.G]) if gravity else np.zeros(3)

    for leg in (1, 2, 3):
        pts = leg_points(q, leg)

        # --- slider (guide reaction + pin A reaction) ----------------------
        ref = pts["slider"]
        fc, mc = guide_cols(leg, ref, ref, +1.0)
        fc2, mc2 = pin_cols("A", leg, pts["slider"], ref, -1.0)   # reaction from link C
        add_body(fc + fc2, mc + mc2, M_SLIDER * g, np.zeros(3))

        # --- link C (pin A at one end, pin B at the other) -----------------
        ref = pts["c_mid"]
        fc, mc = pin_cols("A", leg, pts["slider"], ref, +1.0)
        fc2, mc2 = pin_cols("B", leg, pts["elbow"], ref, -1.0)
        add_body(fc + fc2, mc + mc2, P.M_ARM * g, np.zeros(3))

        # --- link D (pin B and platform pin) -------------------------------
        ref = pts["d_mid"]
        fc, mc = pin_cols("B", leg, pts["elbow"], ref, +1.0)
        fc2, mc2 = pin_cols("P", leg, pts["platform"], ref, -1.0)
        add_body(fc + fc2, mc + mc2, P.M_ARM * g, np.zeros(3))

    # --- end effector ------------------------------------------------------
    fcols, mcols = [], []
    for leg in (1, 2, 3):
        pts = leg_points(q, leg)
        fc, mc = pin_cols("P", leg, pts["platform"], p_tcp, +1.0)
        fcols += fc
        mcols += mc
    add_body(fcols, mcols, P.M_EFFECTOR * g + f_cut, np.zeros(3))

    A_sys = np.array(rows)
    b_sys = np.array(rhs)
    x, *_ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    residual = float(np.linalg.norm(A_sys @ x - b_sys))
    rank = int(np.linalg.matrix_rank(A_sys))

    sol = {n: float(x[i]) for i, n in enumerate(names)}
    actuator = {leg: sol[f"guide{leg}_F{c}"] for leg, c in zip((1, 2, 3), "xyz")}

    # --- bending in the links ---------------------------------------------
    links = {}
    for leg in (1, 2, 3):
        pts = leg_points(q, leg)
        for label, pin, tip, root in (("C", "A", "elbow", "slider"), ("D", "B", "platform", "elbow")):
            f_pin = np.array([sol[f"{pin}{leg}_F{c}"] for c in "xyz"])
            lever = np.linalg.norm(pts[tip] - pts[root])
            f_perp = float(np.linalg.norm(f_pin - (f_pin @ P.U[leg]) * P.U[leg] * 0))
            # the load that bends the link is the component perpendicular to the
            # link axis; the axis lies in the plane of the leg
            axis = (pts[tip] - pts[root]) / lever
            f_bend = np.linalg.norm(f_pin - (f_pin @ axis) * axis)
            links[f"link_{label}{leg}"] = dict(
                length_mm=lever,
                axial_N=float(f_pin @ axis),
                transverse_N=float(f_bend),
                moment_Nmm=float(f_bend * lever),
            )
    return dict(actuator_forces_N=actuator, reactions=sol, links=links,
                residual=residual, rank=rank, unknowns=n_unknowns,
                equations=A_sys.shape[0])


def virtual_work_check(q: np.ndarray, f_cut: np.ndarray) -> dict:
    """Actuator forces from the principle of virtual work (reference solution).

    For the Tripteron the Jacobian that maps actuator rates to TCP velocity is
    the identity, so the force of actuator i is simply the component of the
    external load along its axis.
    """
    return {leg: float(-np.asarray(f_cut) @ P.U[leg]) for leg in (1, 2, 3)}


def forward_kinematics(b: np.ndarray) -> np.ndarray:
    """TCP position for the three actuator coordinates (the legs are decoupled)."""
    p = np.zeros(3)
    for k, leg in enumerate((1, 2, 3)):
        p[k] = float((P.A[leg] + P.E[leg]) @ P.U[leg]) + b[k]
    return p


def body_masses(q: np.ndarray) -> list[tuple[float, str, int]]:
    """(mass, body, leg) of every moving body."""
    out = [(P.M_EFFECTOR, "effector", 0)]
    for leg in (1, 2, 3):
        out += [(M_SLIDER, "slider", leg), (P.M_ARM, "link_C", leg), (P.M_ARM, "link_D", leg)]
    return out


def _body_positions(b: np.ndarray) -> np.ndarray:
    """Centre of mass of every moving body for the actuator coordinates b."""
    q = K.inverse_kinematics(forward_kinematics(b))
    pos = [q[9:12]]
    for leg in (1, 2, 3):
        pts = leg_points(q, leg)
        pos += [pts["slider"], pts["c_mid"], pts["d_mid"]]
    return np.array(pos)


def actuator_forces(q: np.ndarray, f_cut: np.ndarray | None = None,
                    gravity: bool = True, delta: float = 1e-4) -> dict:
    """Actuator forces from the principle of virtual work.

    The derivative of the centre of mass of every body with respect to each
    actuator coordinate is obtained from the closed-form kinematics, so the
    weight of the moving parts is included exactly.
    """
    f_cut = np.zeros(3) if f_cut is None else np.asarray(f_cut, dtype=float)
    b0 = q[:3].copy()
    masses = body_masses(q)
    g = np.array([0.0, 0.0, -P.G]) if gravity else np.zeros(3)
    pos0 = _body_positions(b0)

    forces = {}
    for i, leg in enumerate((1, 2, 3)):
        bp, bm = b0.copy(), b0.copy()
        bp[i] += delta
        bm[i] -= delta
        dpos = (_body_positions(bp) - _body_positions(bm)) / (2 * delta)
        work = float(f_cut @ dpos[0])                      # cutting load on the TCP
        for (m, _, _), dr in zip(masses, dpos):
            work += float((m * g) @ dr) * 1.0              # weight of every body
        forces[leg] = -work
    return forces


def section_properties() -> dict:
    """Square tube used for the arms."""
    a, t = P.ARM["side"], P.ARM["wall"]
    b = a - 2 * t
    area = a ** 2 - b ** 2
    inertia = (a ** 4 - b ** 4) / 12.0
    return dict(area_mm2=area, I_mm4=inertia, c_mm=a / 2, W_mm3=inertia / (a / 2))


def bending_stress(moment_Nmm: float) -> float:
    return moment_Nmm / section_properties()["W_mm3"]
