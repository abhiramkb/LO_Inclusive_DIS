# Leading Order (LO) inclusive structure function in the dipole framework. Code includes all factors except transverse profile integral.
#
import time
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

# The radial integrand decays to near-machine-precision magnitudes at large r;
# quad's roundoff warning there is expected and non-fatal (the returned error
# estimate already reflects it), matching HCubature's silent behavior in Julia.
warnings.filterwarnings("ignore", category=IntegrationWarning)


def ReadBKDipole(path_to_file):
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
    N_values = NrY_data[:, 1:]

    pars = np.array(pars).flatten()
    minr, mult, n = pars[0], pars[1], int(pars[2])

    r_values = np.array([minr * mult**i for i in range(n)])
    #logr_values = np.array([np.log(r) for r in r_values])

    #print(r_values)

    # N_values should have shape (len(Y), len(r))
    interpolator = RegularGridInterpolator(
        (Y_values, r_values),
        N_values
    )

    return interpolator

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
    # Builds a (Y, log r) linear interpolator over the dipole grid, with the
    # boundary value held flat outside the grid range (matches Julia's
    # linear_interpolation(..., extrapolation_bc=Flat())).
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
        Yc = min(max(Y, Ymin), Ymax)
        logrc = min(max(logr, logrmin), logrmax)
        return itp([[Yc, logrc]])[0]

    return interpolator


def S(dipole_interpolator, Y, r):
    return 1.0 - dipole_interpolator(Y, math.log(r))


def epsilon_LO(z, Q, mf):
    return math.sqrt(z * (1.0 - z) * Q**2 + mf**2)


# Radial integral over dipole size r at fixed z
def r_integral_L(dipole_interpolator, Y, xmax, z, Q, mf):
    rtol = 1e-5
    eps = epsilon_LO(z, Q, mf)

    # (1 - S) = N(r, Y); the factor 2 of sigma_dip = 2 * int d^2b N is included in the prefactor
    def integrand(r):
        return r * kv(0, eps * r)**2 * (1.0 - S(dipole_interpolator, Y, r))

    value, _ = quad(integrand, 0.0, xmax, epsabs=1e-15, epsrel=rtol, limit=200)
    return value


def parse_commandline():
    parser = argparse.ArgumentParser(description="Quadrature based integration for LO Inclusive DIS F_L")
    parser.add_argument("--Q", type=float, default=3.1622, help="Q - Photon virtuality")
    parser.add_argument("--x", type=float, default=0.01, help="Bjorken-x")
    parser.add_argument("--dipole_path", type=str, required=True, help="Path to the BK table")
    parser.add_argument("--xmax", type=float, default=20.0, help="xmax (upper integration bound for |r|)")
    parser.add_argument("--mc", type=float, default=1.27, help="mc - charm mass")
    parser.add_argument("--no_charm", action="store_true", help="Exclude the charm contribution (u, d, s only, as in arXiv:2311.10491)")
    parser.add_argument("--save_dir", type=str, default="", help="Saves result to specified folder")
    parser.add_argument("--json", type=str, default="", help="Provide JSON filename to store input and output")
    args = vars(parser.parse_args())

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

    args = parse_commandline()

    # Input parameters
    param_keys = ["Q", "x", "dipole_path", "xmax", "mc", "no_charm"]
    params = {k: args[k] for k in param_keys}
    params["mq"] = mq

    # Metadata (where output files are stored etc)
    meta_keys = ["save_dir", "json"]
    meta = {k: args[k] for k in meta_keys}

    # Provenance info
    provenance_keys = ["script_file", "git_commit", "git_is_dirty", "script_is_dirty"]
    provenance = {k: args[k] for k in provenance_keys}

    x = args["x"]
    xmax = args["xmax"]
    Q = args["Q"]
    mc = args["mc"]
    no_charm = args["no_charm"]
    dipole_path = args["dipole_path"]
    save_dir = args["save_dir"]
    json_filename = args["json"]

    print("Description: Quadrature based integration for LO Inclusive DIS F_L")

    grid = GetYRgrid(dipole_path)
    minr, mult, n, ymin, ymax, yinc = GetGridParameters(dipole_path)
    maxr = minr * mult**(n - 1)

    x_ref_min = (math.log(minr), ymin)
    x_ref_max = (math.log(maxr), ymax)

    itp = setup_interpolator(grid, x_ref_min, x_ref_max)

    x0 = 0.01
    Y = math.log(x0 / x)
    rtol = 1e-5

    # Integration over z in [0, 0.5] multiplied by 2 due to z <-> 1-z symmetry
    integral_lq, error_lq = quad(
        lambda z: 2.0 * z**2 * (1.0 - z)**2 * r_integral_L(itp, Y, xmax, z, Q, mq),
        0.0, 0.5, epsabs=0.0, epsrel=rtol, limit=200
    )

    if no_charm:
        integral_charm, error_charm_raw = 0.0, 0.0
    else:
        integral_charm, error_charm_raw = quad(
            lambda z: 2.0 * z**2 * (1.0 - z)**2 * r_integral_L(itp, Y, xmax, z, Q, mc),
            0.0, 0.5, epsabs=0.0, epsrel=rtol, limit=200
        )

    # F_L = Q^2/(4 pi^2 alpha_em) sigma_L with Eqs. (4) and (7) of arXiv:2311.10491, d^2r = 2 pi r dr
    prefactorL_lq = 2.0 * Nc * Q**4 * sum_ef_squared_lq / (math.pi**3)
    prefactorL_charm = 2.0 * Nc * Q**4 * charm_ef_squared / (math.pi**3)

    result_charm = prefactorL_charm * integral_charm
    error_charm = prefactorL_charm * error_charm_raw

    total_result = prefactorL_lq * integral_lq + prefactorL_charm * integral_charm
    total_error = math.sqrt((prefactorL_lq * error_lq)**2 + (prefactorL_charm * error_charm_raw)**2)
    res = (total_result, total_error)

    print(f"Result F_L: {res}")

    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, f"result_FL_x_{x}_Q_{Q}.txt")
        with open(result_path, "w") as f:
            f.write(f"({res[0]},{res[1]})")

    if json_filename != "":
        payload = {
            "parameters": params,
            "metrics": {
                "result": res[0],
                "error": res[1],
                "result_charm": result_charm,
                "error_charm": error_charm,
            },
            "provenance": provenance,
            "meta": meta,
        }
        path = os.path.join(save_dir if save_dir != "" else ".", json_filename)
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"Saved JSON results to {path}")


if __name__ == "__main__":
    main()
