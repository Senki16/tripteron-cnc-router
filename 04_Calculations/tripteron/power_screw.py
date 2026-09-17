"""Lead screw of one axis: torque, efficiency, self-locking, stresses and buckling.

The formulation is the standard one for a trapezoidal (ACME type) power screw,
Shigley, *Mechanical Engineering Design*, chapter 8.  Every quantity is computed
from the geometry of the screw actually mounted on the machine:

    Tr8x8 (P = 2 mm, 4 starts)  ->  lead L = 8 mm
    mean (pitch) diameter  dm = d - p/2 = 7 mm
    root (minor) diameter  dr = d - p   = 6 mm
    thread half angle      alpha = 15 deg (30 deg included, metric trapezoidal)
"""

from __future__ import annotations

import math

from . import parameters as P


def geometry(screw: dict | None = None) -> dict:
    s = dict(P.SCREW if screw is None else screw)
    d, p = s["d_nominal"], s["pitch"]
    s["lead"] = p * s["starts"]
    s["dm"] = d - p / 2.0
    s["dr"] = d - p
    s["lambda_deg"] = math.degrees(math.atan(s["lead"] / (math.pi * s["dm"])))
    return s


def torque(load: float, screw: dict | None = None) -> dict:
    """Raising and lowering torque, efficiency and self-locking condition."""
    s = geometry(screw)
    dm, L, mu, a = s["dm"], s["lead"], s["mu"], math.radians(s["alpha_deg"])
    sec = 1.0 / math.cos(a)

    t_raise = load * dm / 2 * (L + math.pi * mu * dm * sec) / (math.pi * dm - mu * L * sec)
    t_lower = load * dm / 2 * (math.pi * mu * dm * sec - L) / (math.pi * dm + mu * L * sec)
    t_collar = s["mu_c"] * load * s["d_c"] / 2.0

    total_raise = t_raise + t_collar
    total_lower = t_lower + t_collar
    efficiency = load * L / (2 * math.pi * total_raise)
    self_locking_limit = L * math.cos(a) / (math.pi * dm)      # mu required
    return dict(
        load_N=load,
        lead_mm=L, dm_mm=dm, dr_mm=s["dr"], lead_angle_deg=s["lambda_deg"],
        T_raise_thread_Nmm=t_raise, T_lower_thread_Nmm=t_lower, T_collar_Nmm=t_collar,
        T_raise_Nmm=total_raise, T_lower_Nmm=total_lower,
        efficiency=efficiency,
        mu_required_for_self_locking=self_locking_limit,
        self_locking=s["mu"] >= self_locking_limit,
        motor_torque_ratio=total_raise / 1000.0 / P.MOTOR["holding_torque"],
    )


def stresses(load: float, torque_Nmm: float, screw: dict | None = None) -> dict:
    """Axial, torsional, bearing and thread shear stresses with the von Mises check."""
    s = geometry(screw)
    dr, dm, p = s["dr"], s["dm"], s["pitch"]
    area = math.pi * dr ** 2 / 4.0
    sigma_axial = load / area                                    # compression [MPa]
    tau_torsion = 16.0 * torque_Nmm / (math.pi * dr ** 3)        # [MPa]

    nt = s["nut_engaged_threads"]
    tau_thread = 2.0 * load / (math.pi * dr * s["nut_w"] * p * nt)     # shear of the screw thread
    sigma_bearing = 2.0 * load / (math.pi * dm * p * nt)          # thread bearing pressure
    sigma_vm = math.sqrt(sigma_axial ** 2 + 3 * tau_torsion ** 2)
    return dict(
        area_mm2=area,
        sigma_axial_MPa=sigma_axial,
        tau_torsion_MPa=tau_torsion,
        tau_thread_MPa=tau_thread,
        sigma_bearing_MPa=sigma_bearing,
        sigma_vonMises_MPa=sigma_vm,
        safety_factor_yield=s["Sy"] / sigma_vm,
    )


def buckling(load: float, screw: dict | None = None, end_condition: float = 1.0) -> dict:
    """Euler / Johnson column check of the screw between its bearings."""
    s = geometry(screw)
    dr = s["dr"]
    area = math.pi * dr ** 2 / 4.0
    inertia = math.pi * dr ** 4 / 64.0
    k = math.sqrt(inertia / area)                 # radius of gyration = dr / 4
    slenderness = end_condition * s["length"] / k
    slenderness_limit = math.sqrt(2 * math.pi ** 2 * s["E"] / s["Sy"])
    p_euler = math.pi ** 2 * s["E"] * inertia / (end_condition * s["length"]) ** 2
    p_johnson = area * (s["Sy"] - (s["Sy"] * slenderness / (2 * math.pi)) ** 2 / s["E"])
    critical = p_euler if slenderness > slenderness_limit else p_johnson
    return dict(
        I_mm4=inertia, area_mm2=area, radius_gyration_mm=k,
        slenderness=slenderness, slenderness_limit=slenderness_limit,
        P_euler_N=p_euler, P_johnson_N=p_johnson,
        P_critical_N=critical, safety_factor=critical / load,
        column_regime="Euler" if slenderness > slenderness_limit else "Johnson",
    )


def report(load: float, screw: dict | None = None) -> dict:
    t = torque(load, screw)
    st = stresses(load, t["T_raise_Nmm"], screw)
    bk = buckling(load, screw)
    return dict(torque=t, stresses=st, buckling=bk)
