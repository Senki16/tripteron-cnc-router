"""Run every calculation of the Tripteron CNC router and write the results.

    python run_all.py            (or press "Run" in VS Code)

Outputs
-------
results/   CSV tables and summary.json with every number quoted in the report
figures/   PNG figures used in the calculation report

If numpy, pandas, matplotlib or scipy are missing, the first run creates a local
environment (.venv in this folder), installs them and re-launches itself.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / ".venv"
REQUIRED = ["numpy", "pandas", "matplotlib"]


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _install(python: str, packages: list[str]) -> None:
    uv = shutil.which("uv")
    if uv:
        subprocess.check_call([uv, "pip", "install", "--python", python, *packages])
    else:
        subprocess.check_call([python, "-m", "pip", "install", *packages])


def _ensure_dependencies() -> None:
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if not missing:
        return
    if Path(sys.prefix).resolve() == VENV_DIR.resolve() or os.environ.get("TRIPTERON_BOOTSTRAPPED"):
        print(f"Installing missing libraries: {', '.join(missing)} ...")
        _install(sys.executable, missing)
        importlib.invalidate_caches()
        return
    py = _venv_python()
    if not py.exists():
        if VENV_DIR.exists():
            print(f"Removing unusable environment in {VENV_DIR} ...")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        print(f"Creating a local Python environment in {VENV_DIR} ...")
        uv = shutil.which("uv")
        if uv:
            subprocess.check_call([uv, "venv", str(VENV_DIR), "--python", sys.executable])
        else:
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    if subprocess.run([str(py), "-c", "import " + ", ".join(REQUIRED)], capture_output=True).returncode:
        print(f"Installing {', '.join(REQUIRED)} (first run only) ...")
        _install(str(py), REQUIRED)
    env = dict(os.environ, TRIPTERON_BOOTSTRAPPED="1")
    raise SystemExit(subprocess.call([str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


_ensure_dependencies()

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

sys.path.insert(0, str(HERE))
from tripteron import fea_postprocess as FEA          # noqa: E402
from tripteron import kinematics as K                 # noqa: E402
from tripteron import parameters as P                 # noqa: E402
from tripteron import power_screw as PS               # noqa: E402
from tripteron import statics as ST                   # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#d9d9d9"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 10,
})


# ---------------------------------------------------------------------------
def kinematics_case(traj, tag: str) -> dict:
    sol = K.solve_trajectory(traj, 401)
    t, q, qd, qdd = sol["t"], sol["q"], sol["qd"], sol["qdd"]

    df = pd.DataFrame({
        "t_s": t,
        "Px_mm": q[:, 9], "Py_mm": q[:, 10], "Pz_mm": q[:, 11],
        "b1_mm": q[:, 0], "b2_mm": q[:, 1], "b3_mm": q[:, 2],
        "b1_mm_s": qd[:, 0], "b2_mm_s": qd[:, 1], "b3_mm_s": qd[:, 2],
        "b1_mm_s2": qdd[:, 0], "b2_mm_s2": qdd[:, 1], "b3_mm_s2": qdd[:, 2],
        "thetaC1_deg": np.degrees(q[:, 3]), "thetaD1_deg": np.degrees(q[:, 4]),
        "thetaC2_deg": np.degrees(q[:, 5]), "thetaD2_deg": np.degrees(q[:, 6]),
        "thetaC3_deg": np.degrees(q[:, 7]), "thetaD3_deg": np.degrees(q[:, 8]),
    })
    df.to_csv(RESULTS / f"kinematics_{tag}.csv", index=False)

    fig, ax = plt.subplots(3, 1, figsize=(6.6, 6.2), sharex=True)
    for k, (col, color, lbl) in enumerate(zip(("b1", "b2", "b3"), (BLUE, ORANGE, GREEN),
                                              ("actuator 1 (X)", "actuator 2 (Y)", "actuator 3 (Z)"))):
        ax[0].plot(t, q[:, k], color=color, lw=1.8, label=lbl)
        ax[1].plot(t, qd[:, k], color=color, lw=1.8, label=lbl)
        ax[2].plot(t, qdd[:, k], color=color, lw=1.8, label=lbl)
    ax[0].set_ylabel("stroke b [mm]")
    ax[1].set_ylabel("speed [mm/s]")
    ax[2].set_ylabel("acceleration [mm/s²]")
    ax[2].set_xlabel("time [s]")
    ax[0].set_title(traj.name)
    ax[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    fig.tight_layout()
    fig.savefig(FIGURES / f"kinematics_{tag}.png", bbox_inches="tight")
    plt.close(fig)

    # verification against the closed-form inverse kinematics
    ik = np.array([K.inverse_kinematics(traj.position(tk)) for tk in t])
    dt = t[1] - t[0]
    err_v = np.abs(np.gradient(q, dt, axis=0)[3:-3] - qd[3:-3]).max()
    err_a = np.abs(np.gradient(qd, dt, axis=0)[3:-3] - qdd[3:-3]).max()
    return dict(
        case=traj.name, duration_s=float(traj.duration),
        newton_iterations_max=int(sol["iterations"].max()),
        constraint_residual_max=float(sol["residual"].max()),
        error_vs_closed_form_mm=float(np.abs(ik - q).max()),
        finite_difference_velocity_error=float(err_v),
        finite_difference_acceleration_error=float(err_a),
        stroke_min_mm=[float(q[:, k].min()) for k in range(3)],
        stroke_max_mm=[float(q[:, k].max()) for k in range(3)],
        speed_max_mm_s=[float(np.abs(qd[:, k]).max()) for k in range(3)],
        accel_max_mm_s2=[float(np.abs(qdd[:, k]).max()) for k in range(3)],
    )


def workspace_figure() -> dict:
    ws = K.workspace(step=5.0)
    pts = ws["points"]
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.2))
    planes = [(0, 1, "X [mm]", "Y [mm]", 2), (0, 2, "X [mm]", "Z [mm]", 1), (1, 2, "Y [mm]", "Z [mm]", 0)]
    for a, (i, j, xl, yl, k) in zip(ax, planes):
        mask = np.abs(pts[:, k]) < 2.5
        a.scatter(pts[mask, i], pts[mask, j], s=6, color=BLUE)
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_aspect("equal")
        a.set_title(f"slice {'XYZ'[k]} = 0")
    fig.suptitle("Reachable workspace of the TCP", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "workspace.png", bbox_inches="tight")
    plt.close(fig)
    return dict(bounds_mm={k: [float(v[0]), float(v[1])] for k, v in ws["bounds"].items()},
                volume_dm3=ws["volume_mm3"] / 1e6, samples=int(len(pts)))


def statics_study() -> dict:
    poses = {
        "home (0, 0, 0)": np.zeros(3),
        "corner (+90, +90, +90)": np.array([90.0, 90.0, 90.0]),
        "corner (-90, -90, -90)": np.array([-90.0, -90.0, -90.0]),
        "corner (+90, -90, +90)": np.array([90.0, -90.0, 90.0]),
    }
    rows, joint_rows = [], []
    for name, p in poses.items():
        q = K.inverse_kinematics(p)
        f_vw = ST.actuator_forces(q, P.F_CUT)
        full = ST.solve(q, P.F_CUT)
        worst_link = max(full["links"].items(), key=lambda kv: kv[1]["moment_Nmm"])
        rows.append(dict(
            pose=name,
            Fx_actuator_N=f_vw[1], Fy_actuator_N=f_vw[2], Fz_actuator_N=f_vw[3],
            worst_link=worst_link[0],
            worst_link_moment_Nmm=worst_link[1]["moment_Nmm"],
            worst_link_bending_MPa=ST.bending_stress(worst_link[1]["moment_Nmm"]),
            equilibrium_residual_N=full["residual"],
        ))
        for jname, value in full["reactions"].items():
            joint_rows.append(dict(pose=name, reaction=jname, value=value))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "statics_poses.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(RESULTS / "statics_joint_reactions.csv", index=False)

    # gravity only, to show what each screw has to hold without cutting
    q0 = K.inverse_kinematics(np.zeros(3))
    gravity_only = ST.actuator_forces(q0, np.zeros(3))
    full0 = ST.solve(q0, P.F_CUT)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(rows))
    width = 0.26
    for k, (key, color, lbl) in enumerate(zip(("Fx_actuator_N", "Fy_actuator_N", "Fz_actuator_N"),
                                              (BLUE, ORANGE, GREEN),
                                              ("screw X", "screw Y", "screw Z"))):
        ax.bar(x + (k - 1) * width, np.abs(df[key]), width, color=color, label=lbl)
    ax.set_xticks(x, [r["pose"] for r in rows], rotation=12, ha="right")
    ax.set_ylabel("axial force on the screw [N]")
    ax.set_title("Actuator forces with the design cutting load")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "actuator_forces.png", bbox_inches="tight")
    plt.close(fig)

    design_load = float(np.abs(df[["Fx_actuator_N", "Fy_actuator_N", "Fz_actuator_N"]].values).max())
    return dict(poses=rows, design_axial_load_N=design_load,
                gravity_only_N={str(k): v for k, v in gravity_only.items()},
                equations=full0["equations"], unknowns=full0["unknowns"], rank=full0["rank"],
                redundant_constraints=full0["unknowns"] - full0["rank"],
                arm_section=ST.section_properties())


def power_screw_study(design_load: float) -> dict:
    out = {}
    for label, load in (("design load from the static analysis", design_load),
                        ("load used in delivery 3", P.F_SCREW_AXIAL)):
        out[label] = PS.report(load)

    loads = np.linspace(50, 400, 60)
    torques = [PS.torque(f)["T_raise_Nmm"] for f in loads]
    mus = np.linspace(0.05, 0.40, 60)
    eff = [PS.torque(design_load, {**P.SCREW, "mu": m})["efficiency"] for m in mus]
    lengths = np.linspace(200, 700, 60)
    sf = [PS.buckling(design_load, {**P.SCREW, "length": L})["safety_factor"] for L in lengths]

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.1))
    ax[0].plot(loads, torques, color=BLUE, lw=1.8)
    ax[0].axvline(design_load, color=INK, ls="--", lw=1)
    ax[0].set_xlabel("axial load [N]"); ax[0].set_ylabel("raising torque [N·mm]")
    ax[0].set_title("Torque")
    ax[1].plot(mus, eff, color=BLUE, lw=1.8)
    ax[1].axvline(P.SCREW["mu"], color=INK, ls="--", lw=1)
    ax[1].axvline(PS.torque(design_load)["mu_required_for_self_locking"], color=ORANGE, ls=":", lw=1.4)
    ax[1].set_xlabel("friction coefficient µ"); ax[1].set_ylabel("efficiency")
    ax[1].set_title("Efficiency (orange: self-locking limit)")
    ax[2].plot(lengths, sf, color=BLUE, lw=1.8)
    ax[2].axhline(1.0, color=ORANGE, ls=":", lw=1.4)
    ax[2].axvline(P.SCREW["length"], color=INK, ls="--", lw=1)
    ax[2].set_xlabel("unsupported length [mm]"); ax[2].set_ylabel("buckling safety factor")
    ax[2].set_title("Column stability")
    fig.tight_layout()
    fig.savefig(FIGURES / "power_screw.png", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for label, rep in out.items():
        rows.append(dict(case=label, **{k: v for k, v in rep["torque"].items() if isinstance(v, (int, float))},
                         **{k: v for k, v in rep["stresses"].items()},
                         **{f"buckling_{k}": v for k, v in rep["buckling"].items() if isinstance(v, (int, float))}))
    pd.DataFrame(rows).to_csv(RESULTS / "power_screw.csv", index=False)
    return out


def fea_study() -> dict:
    rows = FEA.summary()
    pd.DataFrame(rows).to_csv(RESULTS / "fea_summary.csv", index=False)
    sf = FEA.safety_factors(rows, P.CHASSIS_MATERIAL["Sy"])

    stress_rows = [r for r in rows if "MPa" in r["quantity"]]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    labels = [r["quantity"].replace(" (MPa)", "") for r in stress_rows]
    ax[0].barh(labels, [r["abs_max"] for r in stress_rows], color=BLUE)
    ax[0].set_xlabel("largest magnitude [MPa]")
    ax[0].set_title("Chassis stresses (FEA)")
    ax[0].grid(axis="y", visible=False)
    disp = [r for r in rows if "Displacement" in r["quantity"]][0]
    buck = [r for r in rows if "Buckling" in r["quantity"]]
    ax[1].barh(["displacement"] + [b["quantity"].replace("Buckling mode 1, ", "mode 1 ") for b in buck],
               [disp["abs_max"]] + [b["abs_max"] for b in buck], color=ORANGE)
    ax[1].set_xlabel("amplitude [mm]")
    ax[1].set_title("Displacement and first buckling mode")
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fea_summary.png", bbox_inches="tight")
    plt.close(fig)
    return dict(rows=rows, safety=sf)


def main() -> None:
    print("=" * 74)
    print("TRIPTERON CNC ROUTER - CALCULATION SUITE")
    print("=" * 74)

    kin = [kinematics_case(K.straight_line(), "case1_straight"),
           kinematics_case(K.accelerated_move(), "case2_accelerated"),
           kinematics_case(K.helical_path(), "case3_helical")]
    ws = workspace_figure()
    st = statics_study()
    ps = power_screw_study(st["design_axial_load_N"])
    fea = fea_study()

    summary = dict(kinematics=kin, workspace=ws, statics=st,
                   power_screw={k: v for k, v in ps.items()}, fea=fea,
                   parameters=dict(link_c_mm=P.LINK_C, link_d_mm=P.LINK_D, stroke_mm=P.STROKE,
                                   cutting_force_N=list(P.F_CUT), screw=PS.geometry(),
                                   motor=P.MOTOR))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print(f"\nWorkspace: {ws['bounds_mm']}  ->  {ws['volume_dm3']:.2f} dm³")
    for case in kin:
        print(f"\n{case['case']}")
        print(f"   Newton-Raphson: max {case['newton_iterations_max']} iterations, "
              f"residual {case['constraint_residual_max']:.1e} mm")
        print(f"   agreement with the closed-form solution: {case['error_vs_closed_form_mm']:.1e} mm")
        print(f"   max |speed| per axis: {[round(v, 2) for v in case['speed_max_mm_s']]} mm/s")
    print(f"\nStatics: {st['equations']} equations, {st['unknowns']} unknowns, "
          f"{st['redundant_constraints']} redundant reactions")
    print(f"   design axial load per screw: {st['design_axial_load_N']:.1f} N "
          f"(delivery 3 used {P.F_SCREW_AXIAL} N)")
    worst = max(st["poses"], key=lambda r: r["worst_link_bending_MPa"])
    print(f"   worst arm bending: {worst['worst_link_bending_MPa']:.1f} MPa at {worst['pose']}")
    key = "design load from the static analysis"
    t, s, b = ps[key]["torque"], ps[key]["stresses"], ps[key]["buckling"]
    print(f"\nLead screw (Tr8x8, 4 starts): T_raise = {t['T_raise_Nmm']:.1f} N·mm, "
          f"efficiency = {t['efficiency'] * 100:.1f} %")
    print(f"   self-locking: {t['self_locking']} (needs µ ≥ {t['mu_required_for_self_locking']:.3f})")
    print(f"   von Mises {s['sigma_vonMises_MPa']:.1f} MPa -> SF {s['safety_factor_yield']:.1f}; "
          f"buckling SF {b['safety_factor']:.2f} ({b['column_regime']})")
    print(f"\nChassis FEA: worst stress {fea['safety']['worst_stress_MPa']} MPa "
          f"-> SF {fea['safety']['static_safety_factor']:.0f}, "
          f"buckling load factor {fea['safety']['buckling_load_factor']}")
    print(f"\nTables written to {RESULTS}")
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
