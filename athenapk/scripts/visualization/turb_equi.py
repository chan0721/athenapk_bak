#!/usr/bin/env python3
import yt
import numpy as np
import matplotlib.pyplot as plt
import argparse, glob, os

def main():
    ap = argparse.ArgumentParser(description="Plot kinetic energy evolution from Parthenon/AthenaPK outputs.")
    ap.add_argument("--pattern", default="parthenon.prim.*.phdf",
                    help="Glob for snapshot files (e.g. 'parthenon.prim.*.phdf' or 'prim.*.phdf')")
    ap.add_argument("--time-unit", default="Myr", help="Time unit for x-axis (e.g. s, kyr, Myr)")
    ap.add_argument("--out", default="kinetic_energy_evolution.png", help="Output figure filename")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    times = []
    Ekin = []     # erg
    Vrms = []     # km/s

    for fn in files:
        ds = yt.load(fn)
        ad = ds.all_data()

        # fields
        rho = ad[("gas","density")]                      # g/cm^3
        vx  = ad[("gas","velocity_x")]                   # cm/s
        vy  = ad[("gas","velocity_y")]
        vz  = ad[("gas","velocity_z")]
        dV  = ad[("index","cell_volume")]                # cm^3

        v2 = vx*vx + vy*vy + vz*vz
        ke_density = 0.5 * rho * v2                      # erg/cm^3
        E = (ke_density * dV).sum().to("erg").value      # erg
        M = (rho * dV).sum().to("g").value               # g
        vrms = np.sqrt(2.0 * E / M) / 1e5                # km/s

        times.append(ds.current_time.to(args.time_unit).value)
        Ekin.append(E)
        Vrms.append(vrms)

        print(f"{os.path.basename(fn)}: t={times[-1]:.4g} {args.time_unit}, "
              f"Ekin={E:.3e} erg, v_rms={vrms:.2f} km/s")

    # sort by time in case filenames are not ordered
    order = np.argsort(times)
    times = np.array(times)[order]
    Ekin  = np.array(Ekin)[order]
    Vrms  = np.array(Vrms)[order]

    fig, ax1 = plt.subplots(figsize=(7,4.5))
    l1, = ax1.plot(times, Ekin, color="tab:blue", lw=2, label="Total kinetic energy")
    ax1.set_xlabel(f"Time ({args.time_unit})")
    ax1.set_ylabel("E_kin (erg)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    l2, = ax2.plot(times, Vrms, color="tab:orange", lw=1.8, ls="--", label="v_rms")
    ax2.set_ylabel("v_rms (km/s)", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    # Legend combining both axes
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()