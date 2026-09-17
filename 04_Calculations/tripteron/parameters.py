"""Machine data for the 3-DOF CNC router (Tripteron architecture).

All lengths are in millimetres, forces in newtons, angles in radians unless
another unit is stated in the name.  Every value is traceable either to the
deliverables of the course project or to the manufacturer data sheet of the
selected component; the source is given in the comment.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# 1. Kinematic dimensions
# ---------------------------------------------------------------------------
LINK_C = 240.0          # first link of every leg [mm]   (drawing package, delivery 2)
LINK_D = 240.0          # second link of every leg [mm]
STROKE = 200.0          # usable travel of each lead screw [mm]
B_HOME = 0.5 * STROKE   # actuator coordinate at the home pose [mm]

# Offsets from the tool centre point (TCP) to the revolute joint of each leg,
# measured on the end effector (delivery 2, effector drawing).
E = {
    1: np.array([62.5, 0.0, 62.5]),
    2: np.array([62.5, 0.0, 0.0]),
    3: np.array([0.0, 0.0, 62.5]),
}

# Direction of the prismatic actuator of every leg.
U = {
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([0.0, 1.0, 0.0]),
    3: np.array([0.0, 0.0, 1.0]),
}

# In-plane axes of every leg: the two links of leg i rotate about an axis
# parallel to U[i], so they move in the plane spanned by PLANE[i].
PLANE = {
    1: (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # y, z
    2: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # x, z
    3: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),   # x, y
}

# Elbow angles used to define the home pose (symmetric "elbow out" posture).
THETA_C_HOME = math.radians(60.0)
THETA_D_HOME = math.radians(-60.0)

TCP_HOME = np.array([0.0, 0.0, 0.0])    # the TCP at mid stroke defines the origin


def base_points() -> dict[int, np.ndarray]:
    """Position of the origin of each linear guide (vector a_i of the loop equations).

    The guides are placed so that at mid stroke the TCP sits at the origin with
    the symmetric posture defined by THETA_C_HOME / THETA_D_HOME.
    """
    a = {}
    for i in (1, 2, 3):
        p, q = PLANE[i]
        chain = (LINK_C * (math.cos(THETA_C_HOME) * p + math.sin(THETA_C_HOME) * q)
                 + LINK_D * (math.cos(THETA_D_HOME) * p + math.sin(THETA_D_HOME) * q))
        a[i] = TCP_HOME - E[i] - chain - B_HOME * U[i]
    return a


A = base_points()

# ---------------------------------------------------------------------------
# 2. Masses and loads
# ---------------------------------------------------------------------------
G = 9.81                     # gravity [m/s^2]
M_MOTOR = 5.216              # NEMA 34 86HB250 closed-loop stepper [kg]  (SolidWorks mass properties)
M_SCREW = 1.120              # one lead screw [kg]
M_GUIDE = 2.530              # one linear guide [kg]
M_ARM = 0.322                # one link (arm) [kg]
M_EFFECTOR = 0.574           # end effector with the spindle mount [kg]
M_CARRIAGE = 15.49           # moving carriage driven by one screw [kg] (delivery 3: P = 151.9 N)

# Cutting load used for the design checks. The routing requirement of the PDS
# is a 1 mm deep pass in wood / soft metal at 10 mm/s.
F_CUT = np.array([80.0, 80.0, -120.0])  # worst case cutting force components [N] (Z pushes the tool down)

# Axial design load of one lead screw, kept from delivery 3 so that the
# corrected numbers can be compared with the original ones.
F_SCREW_AXIAL = 151.9        # [N]

# ---------------------------------------------------------------------------
# 3. Lead screw (trapezoidal Tr8x8, 4 starts, P = 2 mm)
# ---------------------------------------------------------------------------
SCREW = dict(
    d_nominal=8.0,           # major diameter [mm]
    pitch=2.0,               # thread pitch [mm]
    starts=4,                # number of thread starts
    alpha_deg=15.0,          # half of the included thread angle (30 deg metric trapezoidal)
    mu=0.15,                 # thread friction coefficient (steel on bronze, dry)
    mu_c=0.02,               # collar (thrust bearing) friction coefficient
    d_c=6.0,                 # mean collar diameter [mm]
    length=500.0,            # unsupported length between bearings [mm]
    E=207000.0,              # Young's modulus, carbon steel [MPa]
    Sy=370.0,                # yield strength, C45 lead screw [MPa]
    nut_engaged_threads=6,   # threads in contact inside the nut
    nut_w=0.38,              # fraction of the pitch carrying the thread load (trapezoidal)
)

# ---------------------------------------------------------------------------
# 4. Arm cross-section (square steel tube) and material
# ---------------------------------------------------------------------------
ARM = dict(side=25.0, wall=2.0, E=207000.0, Sy=250.0)   # [mm], [MPa] structural steel
CHASSIS_MATERIAL = dict(name="Alloy steel", Sy=620.0, E=210000.0)   # SolidWorks library

# ---------------------------------------------------------------------------
# 5. Drive
# ---------------------------------------------------------------------------
MOTOR = dict(name="NEMA 34 86HB250 closed loop", holding_torque=4.5)   # [N*m] catalogue value
