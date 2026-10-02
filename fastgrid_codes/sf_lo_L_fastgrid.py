# Fast grid for the Leading Order (LO) inclusive structure function F_L in the dipole framework.
# Code includes all factors except transverse profile integral.
#
# Changing variables to u = log(r) and doing the z-integral first,
#     F_L = int du N(r, Y) G_L(r),   G_L(r) = sum_f prefactorL_f * r^2 * int_0^1/2 dz 2 z^2 (1-z)^2 K_0(eps_f r)^2
# G_L depends only on Q (not on x or the dipole), so it is tabulated once on a uniform log(r) grid.
# Any dipole table can then be combined with it via chained Simpson's rule (combine_with_dipole).
import os
import subprocess
import json
import math
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad, IntegrationWarning
from scipy.special import kv
import argparse

# At large r the z-integrand is exponentially small; quad's roundoff warning there is
# expected and non-fatal (the returned error estimate already reflects it).
warnings.filterwarnings("ignore", category=IntegrationWarning)


def GetGridParameters(path_to_file):
    with open(path_to_file) as f:
        content = f.read().split("###")

    content = content[1:]
    content = [i.split() for i in content]

    NrY_data = []
    pars = []

    for i in content:
        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

    NrY_data = np.array(NrY_data)

    Y_values = NrY_data[:, 0]

    ymin, ymax, yinc = min(Y_values), max(Y_values), Y_values[2]-Y_values[1]

    pars = np.array(pars).flatten()

    minr, mult, n = pars[0], pars[1], int(pars[2])
    return minr, mult, n, ymin, ymax, yinc


def GetYRgrid(path_to_file):
    # Returns a 2D numpy grid of dipole values with increasing r(Y) on the  x(y) axis.
    with open(path_to_file) as f:
        content = f.read().split("###")

    content = content[1:]
    content = [i.split() for i in content]

    NrY_data = []
    pars = []

    for i in content:
        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

    NrY_data = np.array(NrY_data)
    return NrY_data[:,1:]


def setup_interpolator(dipolegrid, x_ref_min, x_ref_max):
    # Linear (Y, log r) interpolator, boundary value held flat outside the grid
    # (matches Julia's linear_interpolation(..., extrapolation_bc=Flat())). Accepts arrays.
    logrmin, Ymin = x_ref_min
    logrmax, Ymax = x_ref_max
    nY, nR = dipolegrid.shape

    logr_range = np.linspace(logrmin, logrmax, nR)
    Y_range = np.linspace(Ymin, Ymax, nY)

    itp = RegularGridInterpolator(
        (Y_range, logr_range), dipolegrid,
        method="linear", bounds_error=False, fill_value=None
    )

    def interpolator(Y, logr):
        Yc, logrc = np.broadcast_arrays(np.clip(Y, Ymin, Ymax), np.clip(logr, logrmin, logrmax))
        return itp(np.stack([Yc, logrc], axis=-1))

    return interpolator


def epsilon_LO(z, Q, mf):
    return math.sqrt(z * (1.0 - z) * Q**2 + mf**2)


# z-integral at fixed dipole size r; [0, 0.5] multiplied by 2 due to z <-> 1-z symmetry
def z_integral_L(r, Q, mf, rtol):
    def integrand(z):
        return 2.0 * z**2 * (1.0 - z)**2 * kv(0, epsilon_LO(z, Q, mf) * r)**2

    return quad(integrand, 0.0, 0.5, epsabs=0.0, epsrel=rtol, limit=200)


def simpson_weights(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


def combine_with_dipole(fastgrid_path, dipole_path, x):
    # Returns (F_L, F_L charm part) for the given dipole table at Bjorken-x.
    data = np.loadtxt(fastgrid_path)
    logr, G_lq, G_charm = data[:, 0], data[:, 2], data[:, 3]

    grid = GetYRgrid(dipole_path)
    minr, mult, n, ymin, ymax, yinc = GetGridParameters(dipole_path)
    maxr = minr * mult**(n - 1)
    itp = setup_interpolator(grid, (math.log(minr), ymin), (math.log(maxr), ymax))

    x0 = 0.01
    Y = math.log(x0 / x)

    # (1 - S) = N(r, Y); the factor 2 of sigma_dip = 2 * int d^2b N is included in the prefactor
    N = itp(Y, logr)
    w = simpson_weights(len(logr), (logr[-1] - logr[0]) / (len(logr) - 1))

    result_charm = float(np.sum(w * N * G_charm))
    total_result = float(np.sum(w * N * (G_lq + G_charm)))
    return total_result, result_charm


def parse_commandline():
    parser = argparse.ArgumentParser(description="Fast grid (uniform in log r) of the z-integrated LO Inclusive DIS F_L kernel")
    parser.add_argument("--Q", type=float, default=3.1622, help="Q - Photon virtuality")
    parser.add_argument("--x", type=float, default=0.01, help="Bjorken-x (only used when combining with --dipole_path)")
    parser.add_argument("--dipole_path", type=str, default="", help="Path to a BK table; if given, the fast grid is combined with it and F_L is reported")
    parser.add_argument("--rmin", type=float, default=1e-6, help="Smallest r of the fast grid (uniform in log r)")
    parser.add_argument("--rmax", type=float, default=100.0, help="Largest r of the fast grid (upper integration bound for |r|)")
    parser.add_argument("--nr", type=int, default=2001, help="Number of log(r) grid points (odd, for chained Simpson's rule); the spacing should be well below the dipole table's log(r) spacing")
    parser.add_argument("--mc", type=float, default=1.27, help="mc - charm mass")
    parser.add_argument("--no_charm", action="store_true", help="Exclude the charm contribution (u, d, s only, as in arXiv:2311.10491)")
    parser.add_argument("--save_dir", type=str, default="", help="Saves result to specified folder")
    parser.add_argument("--fastgrid", type=str, default="fastgrid.txt", help="Filename of the fast grid (located in save_dir)")
    parser.add_argument("--json", type=str, default="", help="Provide JSON filename to store input and output")
    args = vars(parser.parse_args())

    if args["nr"] < 3 or args["nr"] % 2 == 0:
        parser.error("--nr must be an odd integer >= 3 for chained Simpson's rule")
    if not 0.0 < args["rmin"] < args["rmax"]:
        parser.error("need 0 < rmin < rmax")

    # --- provenance info ---
    args["script_file"] = os.path.basename(__file__)

    def run_cmd(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            return None

    commit = run_cmd("git rev-parse HEAD")
    args["git_commit"] = commit if commit else "N/A"

    if commit:
        # Does the repo have any uncommitted changes? Ignore untracked files checking this.
        repo_status = run_cmd("git status --porcelain")
        # Does the file have any uncommitted changes?
        file_status = run_cmd(f"git status --porcelain -- {os.path.abspath(__file__)}")
        args["git_is_dirty"] = bool(repo_status)
        args["script_is_dirty"] = bool(file_status)
    else:
        args["git_is_dirty"] = "N/A"
        args["script_is_dirty"] = "N/A"

    return args


def main():
    Nc = 3.0
    mq = 0.14  # light quark mass (GeV), as in arXiv:2311.10491
    sum_ef_squared_lq = 2.0 / 3.0  # 4/9 + 1/9 + 1/9 = 2/3
    charm_ef_squared = 4.0 / 9.0
    rtol = 1e-7

    args = parse_commandline()

    # Input parameters
    param_keys = ["Q", "x", "dipole_path", "rmin", "rmax", "nr", "mc", "no_charm"]
    params = {k: args[k] for k in param_keys}
    params["mq"] = mq
    params["rtol"] = rtol

    # Metadata (where output files are stored etc)
    meta_keys = ["save_dir", "fastgrid", "json"]
    meta = {k: args[k] for k in meta_keys}

    # Provenance info
    provenance_keys = ["script_file", "git_commit", "git_is_dirty", "script_is_dirty"]
    provenance = {k: args[k] for k in provenance_keys}

    x = args["x"]
    Q = args["Q"]
    rmin = args["rmin"]
    rmax = args["rmax"]
    nr = args["nr"]
    mc = args["mc"]
    no_charm = args["no_charm"]
    dipole_path = args["dipole_path"]
    save_dir = args["save_dir"]
    fastgrid_filename = args["fastgrid"]
    json_filename = args["json"]

    print("Description: Fast grid (uniform in log r) of the z-integrated LO Inclusive DIS F_L kernel")

    # F_L = Q^2/(4 pi^2 alpha_em) sigma_L with Eqs. (4) and (7) of arXiv:2311.10491, d^2r = 2 pi r dr
    prefactorL_lq = 2.0 * Nc * Q**4 * sum_ef_squared_lq / (math.pi**3)
    prefactorL_charm = 2.0 * Nc * Q**4 * charm_ef_squared / (math.pi**3)

    logr = np.linspace(math.log(rmin), math.log(rmax), nr)
    r = np.exp(logr)

    # r^2: one r from d^2r = 2 pi r dr, one from dr = r dlog(r)
    zint_lq = np.array([z_integral_L(ri, Q, mq, rtol) for ri in r])
    G_lq = prefactorL_lq * r**2 * zint_lq[:, 0]
    G_lq_err = prefactorL_lq * r**2 * zint_lq[:, 1]

    if no_charm:
        G_charm = np.zeros(nr)
        G_charm_err = np.zeros(nr)
    else:
        zint_charm = np.array([z_integral_L(ri, Q, mc, rtol) for ri in r])
        G_charm = prefactorL_charm * r**2 * zint_charm[:, 0]
        G_charm_err = prefactorL_charm * r**2 * zint_charm[:, 1]

    target_dir = save_dir if save_dir != "" else "."
    os.makedirs(target_dir, exist_ok=True)
    fastgrid_path = os.path.join(target_dir, fastgrid_filename)
    header = (
        "LO inclusive DIS F_L fast grid: F_L = int dlog(r) N(r, Y) (G_lq + G_charm), uniform log(r) grid for chained Simpson's rule\n"
        f"Q = {Q}, mq = {mq}, mc = {mc}, no_charm = {no_charm}, rmin = {rmin}, rmax = {rmax}, nr = {nr}, rtol = {rtol}\n"
        "columns: log(r)  r  G_lq  G_charm  G_lq_err  G_charm_err"
    )
    np.savetxt(fastgrid_path, np.column_stack([logr, r, G_lq, G_charm, G_lq_err, G_charm_err]), fmt="%.16e", header=header)
    print(f"Saved fast grid to {fastgrid_path}")

    metrics = {}
    if dipole_path != "":
        total_result, result_charm = combine_with_dipole(fastgrid_path, dipole_path, x)
        metrics = {"result": total_result, "result_charm": result_charm}
        print(f"Result F_L: {total_result}")

    if json_filename != "":
        payload = {
            "parameters": params,
            "metrics": metrics,
            "provenance": provenance,
            "meta": meta,
        }
        path = os.path.join(target_dir, json_filename)
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"Saved JSON results to {path}")


if __name__ == "__main__":
    main()
