"""Linear guides of the R-pair carriages.

Each actuated axis is built from a lead screw plus two parallel hardened rods.
The nut takes the axial load (sheet 3); the rods and their linear bushings take
everything else: the two transverse force components and the three moment
components of the prismatic joint.

The loads used here are the prismatic-joint reactions produced by the static
analysis of ``statics.solve``, so this sheet consumes verified results instead
of re-deriving them.

Geometry is taken from the delivery 2 drawings: the bearing-support sheet shows
a 12 H7 bore for the rod, and the carriage sheet gives the 60 mm spacing of the
two bushings and the 115 mm distance between the two rods.
"""

from __future__ import annotations

import math

import numpy as np

from . import parameters as P

GUIDE = dict(
    rod_diameter=12.0,      # hardened chromed rod [mm]          (bearing support drawing, 12 H7)
    rod_span=500.0,         # clear span between end supports [mm]
    bushing_spacing=60.0,   # distance between the two bushings of one carriage [mm]
    rod_spacing=115.0,      # track width, distance between the two rods [mm]
    n_rods=2,
    n_bushings=4,           # two per rod
    E=207000.0,             # [MPa]
    Sy=350.0,               # hardened shaft [MPa]
    bushing_C0=540.0,       # static load rating of an LM12UU bushing [N]   (catalogue)
    bushing_C=390.0,        # dynamic load rating of an LM12UU bushing [N]  (catalogue)
)


def section() -> dict:
    """Second moment of area and section modulus of one guide rod."""
    d = GUIDE["rod_diameter"]
    inertia = math.pi * d ** 4 / 64.0
    return dict(d_mm=d, I_mm4=inertia, W_mm3=inertia / (d / 2.0), area_mm2=math.pi * d ** 2 / 4.0)


def _split(vec: np.ndarray, axis: np.ndarray) -> tuple[float, np.ndarray]:
    """Split a vector into its component along ``axis`` and the rest."""
    along = float(np.dot(vec, axis))
    perp = vec - along * axis
    return along, perp


def leg_loads(reactions: dict, leg: int, guide: dict | None = None) -> dict:
    """Bushing and rod loads of one carriage, from the prismatic joint reaction.

    The prismatic joint transmits a force ``F`` (its component along the travel
    axis is the actuator force, which the nut takes) and a moment ``M``.  The
    rods and bushings react:

    * the transverse force ``F_t``, shared by the four bushings;
    * the roll moment ``M . u`` about the travel axis, as a couple between the
      two rods, lever arm = the rod spacing ``s``;
    * the pitch/yaw moment perpendicular to ``u``, as a couple between the two
      bushings of each rod, lever arm = the bushing spacing ``b``.
    """
    g = guide or GUIDE
    u = P.U[leg]
    force = np.array([reactions[f"guide{leg}_F{c}"] for c in "xyz"])
    moment = np.array([reactions[f"guide{leg}_M{c}"] for c in "xyz"])

    actuator, f_transverse = _split(force, u)
    roll, pitch_yaw = _split(moment, u)

    ft = float(np.linalg.norm(f_transverse))
    m_roll = abs(roll)
    m_pitch = float(np.linalg.norm(pitch_yaw))

    s, b = g["rod_spacing"], g["bushing_spacing"]
    f_roll = m_roll / s          # per rod, shared by its two bushings
    f_pitch = m_pitch / b        # per bushing pair of one rod

    # Worst bushing: transverse force shared by four, plus both couples.
    bushing_load = ft / g["n_bushings"] + f_roll / 2.0 + f_pitch / 2.0
    # Worst rod: half the transverse force plus the roll couple.
    rod_load = ft / g["n_rods"] + f_roll

    sec = section()
    span = g["rod_span"]
    deflection = rod_load * span ** 3 / (48.0 * g["E"] * sec["I_mm4"])
    bending = (rod_load * span / 4.0) / sec["W_mm3"]

    return dict(
        leg=leg,
        actuator_force_N=actuator,
        transverse_force_N=ft,
        roll_moment_Nmm=m_roll,
        pitch_moment_Nmm=m_pitch,
        bushing_load_N=bushing_load,
        bushing_static_sf=g["bushing_C0"] / bushing_load if bushing_load else math.inf,
        rod_load_N=rod_load,
        rod_deflection_mm=deflection,
        rod_bending_MPa=bending,
        rod_bending_sf=g["Sy"] / bending if bending else math.inf,
    )


def spacing_sensitivity(reactions: dict, leg: int,
                        spacings=(40.0, 60.0, 80.0, 100.0, 120.0),
                        guide: dict | None = None) -> list[dict]:
    """Worst bushing load of one carriage as a function of the bushing spacing."""
    g = dict(guide or GUIDE)
    out = []
    for b in spacings:
        g["bushing_spacing"] = b
        res = leg_loads(reactions, leg, g)
        out.append(dict(bushing_spacing_mm=b,
                        bushing_load_N=res["bushing_load_N"],
                        static_sf=res["bushing_static_sf"]))
    return out


def report(reactions: dict, guide: dict | None = None) -> dict:
    """Guide check for the three carriages of one pose."""
    legs = [leg_loads(reactions, i, guide) for i in (1, 2, 3)]
    worst = max(legs, key=lambda r: r["bushing_load_N"])
    return dict(
        legs=legs,
        section=section(),
        worst_leg=worst["leg"],
        worst_bushing_load_N=worst["bushing_load_N"],
        worst_bushing_static_sf=worst["bushing_static_sf"],
        worst_rod_deflection_mm=max(r["rod_deflection_mm"] for r in legs),
        worst_rod_bending_MPa=max(r["rod_bending_MPa"] for r in legs),
        sensitivity=spacing_sensitivity(reactions, worst["leg"], guide=guide),
    )
