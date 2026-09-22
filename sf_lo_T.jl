# Leading Order (LO) inclusive structure function in the dipole framework. Code includes all factors except transverse profile integral.
#
using StaticArrays
using JSON
using HCubature
using SpecialFunctions
using LinearAlgebra
using ArgParse
using Base.Filesystem: basename
using Interpolations

function get_YR_grid(path_to_file::String)
    # Read the file and split by "###"
    raw_content = read(path_to_file, String)
    content = split(raw_content, "###")

    content = content[2:end]

    nr_y_data = Vector{Vector{Float64}}()
    pars = Vector{Vector{Float64}}()

    for item in content
        # Split whitespace and parse to Float64
        x = parse.(Float64, split(item))
        
        if length(x) == 1
            push!(pars, x)
        else
            push!(nr_y_data, x)
        end
    end

    # Convert the Vector of Vectors into a 2D Matrix
    # stack(..., dims=1) turns the vectors into rows of a matrix (Julia 1.9+)
    grid = stack(nr_y_data, dims=1)

    # Return the grid from the 2nd column onwards
    # Python [:, 1:] skips index 0. Julia starts at 1, so we skip to 2.
    return grid[:, 2:end]
end

function get_grid_parameters(path_to_file::String)
    # 1. Read and split the file
    raw_content = read(path_to_file, String)
    content = split(raw_content, "###")[2:end]

    nr_y_data_vecs = Vector{Vector{Float64}}()
    pars_vecs = Vector{Vector{Float64}}()

    # 2. Parse and separate data
    for item in content
        x = parse.(Float64, split(item))
        if isempty(x)
            continue
        elseif length(x) == 1
            push!(pars_vecs, x)
        else
            push!(nr_y_data_vecs, x)
        end
    end

    # 3. Extract Y information
    # stack() creates a matrix where each inner vector is a row
    nr_y_data = stack(nr_y_data_vecs, dims=1)
    y_values = nr_y_data[:, 1]

    ymin = minimum(y_values)
    ymax = maximum(y_values)
    
    # Python: Y_values[2] - Y_values[1] (3rd element minus 2nd element)
    # Julia: indices are 1-based, so 3rd - 2nd is [3] - [2]
    yinc = y_values[3] - y_values[2]
    
    # 4. Extract Parameters
    pars = vcat(pars_vecs...)
    minr = pars[1]
    mult = pars[2]
    n    = Int(pars[3])

    return minr, mult, n, ymin, ymax, yinc
end

function setup_interpolator(dipolegrid, x_ref_min, x_ref_max)
    logrmin, Ymin = x_ref_min
    logrmax, Ymax = x_ref_max
    nY, nR = size(dipolegrid)

    logr_range = range(logrmin, logrmax, length=nR)
    Y_range = range(Ymin, Ymax, length=nY)

    # Create the interpolator
    itp = linear_interpolation((Y_range, logr_range), dipolegrid, extrapolation_bc=Flat())
    
    return (Y, logr)->itp(Y, logr)
end

function S(dipole_interpolator, Y, r)
    return (1.0 - dipole_interpolator(Y, log(r)))
end

function epsilon_LO(z::Float64, Q::Float64, mf::Float64)
    return sqrt(z*(1.0-z)*Q^2 + mf^2);
end

function r_integral_T(dipole_interpolator, Y::Number, xmax::Number, z::Float64, Q::Float64, mf::Float64)
    rtol = 1e-5
    eps = epsilon_LO(z, Q, mf)
    # (1 - S) = N(r, Y); the factor 2 of sigma_dip = 2 * int d^2b N is included in the prefactor
    integrand(r) = r * ((z^2 + (1.0 - z)^2) * eps^2 * (besselk(1, eps * r))^2 + mf^2 * (besselk(0, eps * r))^2) * (1.0 - S(dipole_interpolator, Y, r))
    integral = hquadrature(integrand, 0.0, xmax; rtol = rtol)
    return integral[1]
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--Q"
            help = "Q - Photon virtuality"
            arg_type = Float64
            default = 3.1622
        "--x"
            help = "Bjorken-x"
            arg_type = Float64
            default = 0.01
        "--dipole_path"
            help = "Path to the BK table"
            arg_type = String
            required = true
        "--xmax"
            help = "xmax (upper integration bound for |r|)"
            arg_type = Float64
            default = 20.0
        "--mc"
            help = "mc - charm mass"
            arg_type = Float64
            default = 1.27
        "--no_charm"
            help = "Exclude the charm contribution (u, d, s only, as in arXiv:2311.10491)"
            action = :store_true
        "--save_dir"
            help = "Saves result to specified folder"
            default = ""
        "--json"
            help = "Provide JSON filename to store input and output"
            default = ""
    end

    args = parse_args(s)

    # --- provenance info ---
    args["script_file"] = basename(@__FILE__)

    try
        args["git_commit"] = readchomp(`git rev-parse HEAD`)
    catch
        args["git_commit"] = "N/A"
    end

    try
    	# Does the repo have any uncommitted changes? Ignore untracked files checking this.
	repo_dirty = !isempty(readchomp(`git status --porcelain`))
    	
    	# Does the file have any uncommitted changes?
    	file_dirty = !isempty(readchomp(`git status --porcelain -- $(abspath(@__FILE__))`))
    	args["git_is_dirty"] = repo_dirty
    	args["script_is_dirty"] = file_dirty
    catch
    	args["git_is_dirty"] = "N/A"
    	args["script_is_dirty"] = "N/A"
    end

    return args
end

function main()
    
    Nc = 3.0;
    mq = 0.14; # light quark mass (GeV), as in arXiv:2311.10491
    sum_ef_squared_lq = 2.0/3.0; # 4/9 + 1/9 + 1/9 = 2/3
    charm_ef_squared = 4.0/9.0;

    parsed_args = parse_commandline()

    # Input parameters
    param_keys = ["Q", "x", "dipole_path", "xmax", "mc", "no_charm"]
    params = Dict(k => parsed_args[k] for k in param_keys)
    params["mq"] = mq

    # Metadata (where output files are stored etc)
    meta_keys = ["save_dir", "json"]
    meta = Dict(k => parsed_args[k] for k in meta_keys)

    # Provenance info
    provenance_keys = ["script_file", "git_commit", "git_is_dirty", "script_is_dirty"]
    provenance = Dict(k => parsed_args[k] for k in provenance_keys)


	x = parsed_args["x"]
    xmax = parsed_args["xmax"]
    Q = parsed_args["Q"]
    mc = parsed_args["mc"]
    no_charm = parsed_args["no_charm"]
    dipole_path = parsed_args["dipole_path"]
    save_dir = parsed_args["save_dir"]
    json = parsed_args["json"]

    
    println("Description: Quadrature based integration for LO Inclusive DIS F_T");
    
    grid = get_YR_grid(dipole_path);
    minr, mult, n, ymin, ymax, yinc = get_grid_parameters(dipole_path);
    maxr = minr*mult^(n-1);
    
    x_ref_min = (log(minr), ymin)
    x_ref_max = (log(maxr), ymax)
    
    itp = setup_interpolator(grid, x_ref_min, x_ref_max);

    x0 = 0.01;
    Y = log(x0/x);
    rtol = 1e-5;
    
    # Integration over z in [0, 0.5] multiplied by 2 due to z <-> 1-z symmetry
    integral_lq = hquadrature(z -> 2.0 * r_integral_T(itp, Y, xmax, z, Q, mq), 0.0, 0.5; rtol=rtol)
    integral_charm = no_charm ? (0.0, 0.0) : hquadrature(z -> 2.0 * r_integral_T(itp, Y, xmax, z, Q, mc), 0.0, 0.5; rtol=rtol)

    # F_T = Q^2/(4 pi^2 alpha_em) sigma_T with Eqs. (4) and (6) of arXiv:2311.10491, d^2r = 2 pi r dr
    prefactorT_lq = (Nc * Q^2 * sum_ef_squared_lq / (2.0 * pi^3))
    prefactorT_charm = (Nc * Q^2 * charm_ef_squared / (2.0 * pi^3))

    result_charm = prefactorT_charm * integral_charm[1]
    error_charm = prefactorT_charm * integral_charm[2]

    total_result = prefactorT_lq * integral_lq[1] + prefactorT_charm * integral_charm[1]
    total_error = sqrt((prefactorT_lq * integral_lq[2])^2 + (prefactorT_charm * integral_charm[2])^2)
    res = (total_result, total_error)

    println("Result F_T: ", res)

    if save_dir != ""
        mkpath(save_dir)
        open(joinpath(save_dir, "result_FT_x_$(x)_Q_$(Q).txt"), "w") do file
            write(file, "($(res[1]),$(res[2]))")
        end
    end

    if json != ""
        payload = Dict(
            "parameters" => params,
            "metrics" => Dict("result" => res[1], "error" => res[2], "result_charm" => result_charm, "error_charm" => error_charm),
            "provenance" => provenance,
            "meta" => meta
        )
        path = joinpath(save_dir == "" ? "." : save_dir, json)
        open(path, "w") do io
            JSON.print(io, payload) 
        end
        println("Saved JSON results to $path")
    end
end

main()
