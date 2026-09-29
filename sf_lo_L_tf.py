# Leading Order (LO) inclusive structure function in the dipole framework. Code includes all factors except transverse profile integral.
# TensorFlow version of sf_lo_L.py: the (z, r) integral is done with a vectorized Gauss-Legendre rule (torchquad).
#
import time
import os
import subprocess
import json
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.experimental import numpy as tnp #Use tnp instead of numpy

# Set precision to float64 to match Julia's numerical accuracy
tf.keras.backend.set_floatx('float64')
DTYPE = tf.float64

import torchquad
from torchquad import MonteCarlo, set_up_backend, Gaussian
# Enable Tensorflow's NumPy behaviour and set the floating point precision
set_up_backend("tensorflow", data_type="float64")
# torchquad logs every integration step at DEBUG level
torchquad.set_log_level("WARNING")

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


@tf.function(jit_compile=True)
def S(tfgrid, x_ref_min, x_ref_max, Y, r):
    # x_ref_min and x_ref_max contain the lower and upper bounds respectively
    # of the Y-log(r) grid of dipole value. I am using the same notation as in the
    # documentation for the interpolation function.
    # x is the array of inputs for the batch interpolation [[Y,log(x20)], [Y,log(x21)], [Y,log(x10)]]
    # 'constant_extension' holds the boundary value flat outside the grid range
    # (matches Julia's linear_interpolation(..., extrapolation_bc=Flat())).

    # Correct stacking:
    # Use axis=-1 to ensure the shape is (batch_size, 2) for each dipole
    #x = tf.stack([tf.stack([Y, tf.math.log(r)], axis=-1)], axis=0)

    # AFTER (much faster in TF)
    x = tf.stack([Y, tf.math.log(r)], axis=-1)
    x = tf.expand_dims(x, axis=0)

    Nval = tfp.math.batch_interp_regular_nd_grid(x, x_ref_min, x_ref_max, tfgrid, axis=-2, fill_value='constant_extension')

    Sval = 1.0 - Nval

    return Sval[0]

def epsilon_LO(z, Q, mf):
    return tf.sqrt(z * (1.0 - z) * Q**2 + mf**2)


# Integrand of the combined (z, r) integral, vectorized over the points (z_i, r_i)
def integrand_L(tfgrid, x_ref_min, x_ref_max, Y, z, r, Q, mf):
    eps = epsilon_LO(z, Q, mf)
    Ytemp = tf.broadcast_to(Y, tf.shape(r))

    # 2 z^2 (1-z)^2: z in [0, 0.5], multiplied by 2 due to z <-> 1-z symmetry
    # (1 - S) = N(r, Y); the factor 2 of sigma_dip = 2 * int d^2b N is included in the prefactor
    return 2.0 * z**2 * (1.0 - z)**2 * r * tf.math.special.bessel_k0(eps * r)**2 * (1.0 - S(tfgrid, x_ref_min, x_ref_max, Ytemp, r))


@tf.function
# Integral over z in [0, 0.5] and dipole size r in [0, xmax], with n_points Gauss-Legendre nodes per dimension.
# n_points must be a Python int (torchquad requires it); each distinct value triggers one retrace.
def zr_integral_L(tfgrid, x_ref_min, x_ref_max, Y, xmax, Q, mf, n_points):
    gauss = torchquad.GaussLegendre()

    zero = tf.constant(0.0, dtype=DTYPE)
    integration_domain = tf.stack([tf.stack([zero, tf.constant(0.5, dtype=DTYPE)]), tf.stack([zero, xmax])])

    def integrand(x):
        return integrand_L(tfgrid, x_ref_min, x_ref_max, Y, x[:, 0], x[:, 1], Q, mf)

    value = gauss.integrate(integrand, dim=2, N=n_points**2, integration_domain=integration_domain,
    backend="tensorflow",)

    return value


def integral_with_error(tfgrid, x_ref_min, x_ref_max, Y, xmax, Q, mf, n_points):
    # Gauss-Legendre gives no error estimate; use the difference to a rule with half as many
    # nodes per dimension. This is a conservative estimate of the error of the finer result.
    fine = zr_integral_L(tfgrid, x_ref_min, x_ref_max, Y, xmax, Q, mf, n_points)
    coarse = zr_integral_L(tfgrid, x_ref_min, x_ref_max, Y, xmax, Q, mf, n_points // 2)
    return float(fine), float(abs(fine - coarse))


def parse_commandline():
    parser = argparse.ArgumentParser(description="Gauss-Legendre (TensorFlow) integration for LO Inclusive DIS F_L")
    parser.add_argument("--Q", type=float, default=3.1622, help="Q - Photon virtuality")
    parser.add_argument("--x", type=float, default=0.01, help="Bjorken-x")
    parser.add_argument("--dipole_path", type=str, required=True, help="Path to the BK table")
    parser.add_argument("--xmax", type=float, default=20.0, help="xmax (upper integration bound for |r|)")
    parser.add_argument("--mc", type=float, default=1.27, help="mc - charm mass")
    parser.add_argument("--no_charm", action="store_true", help="Exclude the charm contribution (u, d, s only, as in arXiv:2311.10491)")
    parser.add_argument("--n_points", type=int, default=400, help="Gauss-Legendre nodes per dimension (z and r)")
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
    param_keys = ["Q", "x", "dipole_path", "xmax", "mc", "no_charm", "n_points"]
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
    n_points = args["n_points"]
    dipole_path = args["dipole_path"]
    save_dir = args["save_dir"]
    json_filename = args["json"]

    print("Description: Gauss-Legendre (TensorFlow) integration for LO Inclusive DIS F_L")

    grid = GetYRgrid(dipole_path)
    minr, mult, n, ymin, ymax, yinc = GetGridParameters(dipole_path)
    maxr = minr * mult**(n - 1)

    # Grid axes are (Y, log r), in the same order as the grid's (row, column)
    tfgrid = tf.constant(grid, dtype=DTYPE)
    x_ref_min = tf.constant([ymin, math.log(minr)], dtype=DTYPE)
    x_ref_max = tf.constant([ymax, math.log(maxr)], dtype=DTYPE)

    x0 = 0.01
    Y = tf.constant(math.log(x0 / x), dtype=DTYPE)
    xmax_tf = tf.constant(xmax, dtype=DTYPE)
    Q_tf = tf.constant(Q, dtype=DTYPE)

    integral_lq, error_lq = integral_with_error(tfgrid, x_ref_min, x_ref_max, Y, xmax_tf, Q_tf,
                                                tf.constant(mq, dtype=DTYPE), n_points)

    if no_charm:
        integral_charm, error_charm_raw = 0.0, 0.0
    else:
        integral_charm, error_charm_raw = integral_with_error(tfgrid, x_ref_min, x_ref_max, Y, xmax_tf, Q_tf,
                                                              tf.constant(mc, dtype=DTYPE), n_points)

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
