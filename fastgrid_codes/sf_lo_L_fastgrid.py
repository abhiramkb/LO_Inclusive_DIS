# Fast grid for the Leading Order (LO) inclusive structure function F_L in the dipole framework.
# Code includes all factors except transverse profile integral.
#
# Changing variables to u = log(r) and doing the z-integral first,
#     F_L = int du N(r, Y) G_L(r),   G_L(r) = sum_f prefactorL_f * r^2 * int_0^1/2 dz 2 z^2 (1-z)^2 K_0(eps_f r)^2
# G_L depends only on Q (not on x or the dipole). With N linear in u between the dipole grid's nodes u_n,
#     F_L = sum_n w_n N(r_n, Y),   w_n = int du G_L(u) phi_n(u),
# where phi_n are the hat functions of that linear interpolation. The node weights w_n are tabulated once on
# the r grid of the dipoles that will be used (r_n = rmin * mult**n) and are then exact for any dipole on
# that grid (combine_with_dipole). Choosing rmin, mult and nr to match the dipole format is up to the user.
#
# Two modes:
#     make: compute the fast grid for one Q and save it (no dipole needed)
#     use:  combine an existing fast grid with a dipole table at Bjorken-x (no integration)
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


def hat_quadrature(logr_nodes, logr_max, ngauss):
    # Gauss-Legendre points u (weights du) on [logr_nodes[0], logr_max], one interval per node spacing,
    # with the left node k of each point and its fractional position frac in [u_k, u_k+1].
    # Beyond the last node N is held at its last value (k = last node, frac = 0); below the first node N = 0.
    n = len(logr_nodes)
    h = logr_nodes[1] - logr_nodes[0]
    breaks = logr_nodes[logr_nodes < logr_max]
    if logr_max > logr_nodes[-1]:
        breaks = np.append(breaks, np.arange(logr_nodes[-1] + h, logr_max, h))
    breaks = np.append(breaks, logr_max)

    x_gl, w_gl = np.polynomial.legendre.leggauss(ngauss)
    mid = 0.5 * (breaks[1:] + breaks[:-1])
    half = 0.5 * (breaks[1:] - breaks[:-1])
    u = (mid[:, None] + half[:, None] * x_gl).ravel()
    du = (half[:, None] * w_gl).ravel()

    k = np.minimum(np.floor((u - logr_nodes[0]) / h).astype(int), n - 1)
    frac = np.where(k < n - 1, (u - logr_nodes[k]) / h, 0.0)
    return u, du, k, frac


def to_node_weights(f, du, k, frac, n):
    # w_n = sum over quadrature points of f(u) du phi_n(u), with phi_n the hat function of node n
    w = np.zeros(n)
    np.add.at(w, k, f * du * (1.0 - frac))
    inside = k < n - 1
    np.add.at(w, k[inside] + 1, (f * du * frac)[inside])
    return w


def combine_with_dipole(fastgrid_path, dipole_path, x):
    # Returns (F_L, F_L charm part) for the given dipole table at Bjorken-x.
    # The node weights are only valid for a dipole table on exactly the fast grid's r grid.
    settings = read_fastgrid_settings(fastgrid_path)
    minr, mult, n, ymin, ymax, yinc = GetGridParameters(dipole_path)
    minr, mult = float(minr), float(mult)  # plain floats, so the message below prints copyable values
    if (n != settings["nr"] or not math.isclose(minr, settings["rmin"], rel_tol=1e-9)
            or not math.isclose(mult, settings["mult"], rel_tol=1e-9)):
        raise SystemExit(
            f"The r grid of {dipole_path} (rmin = {minr!r}, mult = {mult!r}, nr = {n}) differs from the fast grid's "
            f"(rmin = {settings['rmin']!r}, mult = {settings['mult']!r}, nr = {settings['nr']}); "
            f"make a fast grid with --rmin {minr!r} --mult {mult!r} --nr {n}")

    data = np.loadtxt(fastgrid_path)
    logr, w_lq, w_charm = data[:, 0], data[:, 2], data[:, 3]

    grid = GetYRgrid(dipole_path)
    maxr = minr * mult**(n - 1)
    itp = setup_interpolator(grid, (math.log(minr), ymin), (math.log(maxr), ymax))

    x0 = 0.01
    Y = math.log(x0 / x)

    # (1 - S) = N(r, Y) at the dipole's own nodes; the factor 2 of sigma_dip = 2 * int d^2b N is included in the prefactor
    N = itp(Y, logr)

    result_charm = float(np.sum(w_charm * N))
    total_result = float(np.sum((w_lq + w_charm) * N))
    return total_result, result_charm


def read_fastgrid_settings(fastgrid_path):
    # Settings stored on the second header line by the make mode, e.g. "# Q = 2.0, mq = 0.14, ..., no_charm = True"
    with open(fastgrid_path) as f:
        title = f.readline()
        settings = f.readline()

    if "F_L fast grid" not in title:
        raise SystemExit(f"{fastgrid_path} is not an F_L fast grid (first line: {title.strip()})")
    if "(node weights)" not in title:
        raise SystemExit(f"{fastgrid_path} is an old-format fast grid on a uniform log(r) grid; remake it with the make mode")

    def parse_value(value):
        if value in ("True", "False"):
            return value == "True"
        return int(value) if value.lstrip("-").isdigit() else float(value)

    return {key: parse_value(value) for key, value in
            (item.split(" = ") for item in settings.lstrip("#").strip().split(", "))}


def parse_commandline():
    parser = argparse.ArgumentParser(description="Fast grid (node weights on the dipole's log(r) grid) of the z-integrated LO Inclusive DIS F_L kernel")
    modes = parser.add_subparsers(dest="mode", required=True, metavar="{make,use}")

    make = modes.add_parser("make", help="Compute the fast grid for one Q and save it (no dipole needed)")
    make.add_argument("--Q", type=float, default=3.1622, help="Q - Photon virtuality")
    make.add_argument("--rmin", type=float, default=1e-6, help="First r node of the dipoles that will be used, r_n = rmin * mult**n (default: BK_setup in BK/BK_evolve_tf.py)")
    make.add_argument("--mult", type=float, default=1.04725, help="Ratio of neighbouring r nodes of the dipoles that will be used")
    make.add_argument("--nr", type=int, default=400, help="Number of r nodes of the dipoles that will be used")
    make.add_argument("--rmax", type=float, default=100.0, help="Upper integration bound for |r|; beyond the last node N is held at its last value")
    make.add_argument("--mc", type=float, default=1.27, help="mc - charm mass")
    make.add_argument("--no_charm", action="store_true", help="Exclude the charm contribution (u, d, s only, as in arXiv:2311.10491)")
    make.add_argument("--save_dir", type=str, default="", help="Folder for the fast grid and the JSON file")
    make.add_argument("--fastgrid", type=str, default="fastgrid.txt", help="Filename of the fast grid (written to save_dir)")
    make.add_argument("--json", type=str, default="", help="Provide JSON filename to store input and output")

    use = modes.add_parser("use", help="Combine an existing fast grid with a dipole table (no integration)")
    use.add_argument("--fastgrid", type=str, required=True, help="Path to a fast grid written by the make mode")
    use.add_argument("--dipole_path", type=str, required=True, help="Path to the BK table")
    use.add_argument("--x", type=float, default=0.01, help="Bjorken-x")
    use.add_argument("--save_dir", type=str, default="", help="Folder for the JSON file")
    use.add_argument("--json", type=str, default="", help="Provide JSON filename to store input and output")

    args = vars(parser.parse_args())

    if args["mode"] == "make":
        if args["nr"] < 2 or args["mult"] <= 1.0:
            make.error("need nr >= 2 and mult > 1")
        if not 0.0 < args["rmin"] < args["rmax"]:
            make.error("need 0 < rmin < rmax")

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


def make_grid(args):
    # Computes the fast grid and saves it to save_dir/fastgrid. Returns (params, metrics).
    Nc = 3.0
    mq = 0.14  # light quark mass (GeV), as in arXiv:2311.10491
    sum_ef_squared_lq = 2.0 / 3.0  # 4/9 + 1/9 + 1/9 = 2/3
    charm_ef_squared = 4.0 / 9.0
    rtol = 1e-7
    ngauss = 6  # Gauss-Legendre points per node interval for the node weights

    # Input parameters
    param_keys = ["Q", "rmin", "mult", "nr", "rmax", "mc", "no_charm"]
    params = {k: args[k] for k in param_keys}
    params["mq"] = mq
    params["rtol"] = rtol
    params["ngauss"] = ngauss

    Q = args["Q"]
    rmin = args["rmin"]
    mult = args["mult"]
    nr = args["nr"]
    rmax = args["rmax"]
    mc = args["mc"]
    no_charm = args["no_charm"]

    print("Description: make mode - computing node weights of the z-integrated LO Inclusive DIS F_L kernel on the dipole's log(r) grid")

    # F_L = Q^2/(4 pi^2 alpha_em) sigma_L with Eqs. (4) and (7) of arXiv:2311.10491, d^2r = 2 pi r dr
    prefactorL_lq = 2.0 * Nc * Q**4 * sum_ef_squared_lq / (math.pi**3)
    prefactorL_charm = 2.0 * Nc * Q**4 * charm_ef_squared / (math.pi**3)

    # Nodes exactly as in the dipoles that will be used, r_n = rmin * mult**n
    logr = math.log(rmin) + math.log(mult) * np.arange(nr)
    r = np.exp(logr)

    u, du, k, frac = hat_quadrature(logr, math.log(rmax), ngauss)
    ru = np.exp(u)

    # r^2: one r from d^2r = 2 pi r dr, one from dr = r dlog(r)
    zint_lq = np.array([z_integral_L(ri, Q, mq, rtol) for ri in ru])
    w_lq = to_node_weights(prefactorL_lq * ru**2 * zint_lq[:, 0], du, k, frac, nr)
    w_lq_err = to_node_weights(prefactorL_lq * ru**2 * zint_lq[:, 1], du, k, frac, nr)

    if no_charm:
        w_charm = np.zeros(nr)
        w_charm_err = np.zeros(nr)
    else:
        zint_charm = np.array([z_integral_L(ri, Q, mc, rtol) for ri in ru])
        w_charm = to_node_weights(prefactorL_charm * ru**2 * zint_charm[:, 0], du, k, frac, nr)
        w_charm_err = to_node_weights(prefactorL_charm * ru**2 * zint_charm[:, 1], du, k, frac, nr)

    target_dir = args["save_dir"] if args["save_dir"] != "" else "."
    os.makedirs(target_dir, exist_ok=True)
    fastgrid_path = os.path.join(target_dir, args["fastgrid"])
    header = (
        "LO inclusive DIS F_L fast grid (node weights): F_L = sum_n (w_lq + w_charm)_n N(r_n, Y), N linear in log(r) between the nodes r_n = rmin * mult**n\n"
        f"Q = {Q}, mq = {mq}, mc = {mc}, no_charm = {no_charm}, rmin = {rmin!r}, mult = {mult!r}, nr = {nr}, rmax = {rmax}, rtol = {rtol}, ngauss = {ngauss}\n"
        "columns: log(r)  r  w_lq  w_charm  w_lq_err  w_charm_err"
    )
    np.savetxt(fastgrid_path, np.column_stack([logr, r, w_lq, w_charm, w_lq_err, w_charm_err]), fmt="%.16e", header=header)
    print(f"Saved fast grid to {fastgrid_path}")

    return params, {}


def use_grid(args):
    # Combines an existing fast grid with a dipole table; no integrals are computed. Returns (params, metrics).
    fastgrid_path = args["fastgrid"]
    if not os.path.isfile(fastgrid_path):
        raise SystemExit(f"Fast grid not found: {fastgrid_path} (create it with the make mode)")

    # Grid settings come from the grid file itself, so they cannot disagree with the grid
    params = read_fastgrid_settings(fastgrid_path)
    params["x"] = args["x"]
    params["dipole_path"] = args["dipole_path"]

    print(f"Description: use mode - combining the LO Inclusive DIS F_L fast grid {fastgrid_path} with a dipole table")

    total_result, result_charm = combine_with_dipole(fastgrid_path, args["dipole_path"], args["x"])
    print(f"Result F_L: {total_result}")

    return params, {"result": total_result, "result_charm": result_charm}


def main():
    args = parse_commandline()

    # Metadata (where output files are stored etc)
    meta_keys = ["mode", "save_dir", "fastgrid", "json"]
    meta = {k: args[k] for k in meta_keys}

    # Provenance info
    provenance_keys = ["script_file", "git_commit", "git_is_dirty", "script_is_dirty"]
    provenance = {k: args[k] for k in provenance_keys}

    if args["mode"] == "make":
        params, metrics = make_grid(args)
    else:
        params, metrics = use_grid(args)

    json_filename = args["json"]
    if json_filename != "":
        payload = {
            "parameters": params,
            "metrics": metrics,
            "provenance": provenance,
            "meta": meta,
        }
        target_dir = args["save_dir"] if args["save_dir"] != "" else "."
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, json_filename)
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"Saved JSON results to {path}")


if __name__ == "__main__":
    main()
