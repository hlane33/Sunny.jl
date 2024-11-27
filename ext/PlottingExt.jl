module PlottingExt

using Sunny
import Sunny: Mat3, Vec3, orig_crystal, natoms
using LinearAlgebra
import Statistics
import Makie


let warned = false
    global warn_wglmakie() = begin
        if !warned && string(Makie.current_backend()) == "WGLMakie"
            @info """
            Using the WGLMakie graphics backend. If you encounter graphics problems,
            try restarting the Julia session and load GLMakie instead of WGLMakie.
            Issue tracker: https://github.com/SunnySuite/Sunny.jl/issues/211.
            """
        end
        warned = true
    end
end


getindex_cyclic(a, i) = a[mod1(i, length(a))] 

const seaborn_bright = [
    Makie.RGBf(0.00784313725490196,0.24313725490196078,1.0),
    Makie.RGBf(1.0,0.48627450980392156,0.0),
    Makie.RGBf(0.10196078431372549,0.788235294117647,0.2196078431372549),
    Makie.RGBf(0.9098039215686274,0.0,0.043137254901960784),
    Makie.RGBf(0.5450980392156862,0.16862745098039217,0.8862745098039215),
    Makie.RGBf(0.6235294117647059,0.2823529411764706,0.0),
    Makie.RGBf(0.9450980392156862,0.2980392156862745,0.7568627450980392),
    Makie.RGBf(0.6392156862745098,0.6392156862745098,0.6392156862745098),
    Makie.RGBf(1.0,0.7686274509803922,0.0),
    Makie.RGBf(0.0,0.8431372549019608,1.0),
]

const seaborn_muted = [
    Makie.RGBf(0.2823529411764706,0.47058823529411764,0.8156862745098039),
    Makie.RGBf(0.9333333333333333,0.5215686274509804,0.2901960784313726),
    Makie.RGBf(0.41568627450980394,0.8,0.39215686274509803),
    Makie.RGBf(0.8392156862745098,0.37254901960784315,0.37254901960784315),
    Makie.RGBf(0.5843137254901961,0.4235294117647059,0.7058823529411765),
    Makie.RGBf(0.5490196078431373,0.3803921568627451,0.23529411764705882),
    Makie.RGBf(0.8627450980392157,0.49411764705882355,0.7529411764705882),
    Makie.RGBf(0.4745098039215686,0.4745098039215686,0.4745098039215686),
    Makie.RGBf(0.8352941176470589,0.7333333333333333,0.403921568627451),
    Makie.RGBf(0.5098039215686274,0.7764705882352941,0.8862745098039215),
]

# Colors from Jmol table, https://jmol.sourceforge.net/jscolors/
atom_colors = let
    pairs = [
        "H" =>"#FFFFFF", "He"=>"#D9FFFF", "Li"=>"#CC80FF", "Be"=>"#C2FF00",
        "B" =>"#FFB5B5", "C"=> "#909090", "N" =>"#3050F8", "O" =>"#FF0D0D",
        "F" =>"#90E050", "Ne"=>"#B3E3F5", "Na"=>"#AB5CF2", "Mg"=>"#8AFF00",
        "Al"=>"#BFA6A6", "Si"=>"#F0C8A0", "P" =>"#FF8000", "S"=>"#FFFF30",
        "Cl"=>"#1FF01F", "Ar"=>"#80D1E3", "K" =>"#8F40D4", "Ca"=>"#3DFF00",
        "Sc"=>"#E6E6E6", "Ti"=>"#BFC2C7", "V" =>"#A6A6AB", "Cr"=>"#8A99C7",
        "Mn"=>"#9C7AC7", "Fe"=>"#E06633", "Co"=>"#F090A0", "Ni"=>"#50D050",
        "Cu"=>"#C88033", "Zn"=>"#7D80B0", "Ga"=>"#C28F8F", "Ge"=>"#668F8F",
        "As"=>"#BD80E3", "Se"=>"#FFA100", "Br"=>"#A62929", "Kr"=>"#5CB8D1",
        "Rb"=>"#702EB0", "Sr"=>"#00FF00", "Y"=>"#94FFFF",  "Zr"=>"#94E0E0",
        "Nb"=>"#73C2C9", "Mo"=>"#54B5B5", "Tc"=>"#3B9E9E", "Ru"=>"#248F8F",
        "Rh"=>"#0A7D8C", "Pd"=>"#006985", "Ag"=>"#C0C0C0", "Cd"=>"#FFD98F",
        "In"=>"#A67573", "Sn"=>"#668080", "Sb"=>"#9E63B5", "Te"=>"#D47A00",
        "I" =>"#940094", "Xe"=>"#429EB0", "Cs"=>"#57178F", "Ba"=>"#00C900",
        "La"=>"#70D4FF", "Ce"=>"#FFFFC7", "Pr"=>"#D9FFC7", "Nd"=>"#C7FFC7",
        "Pm"=>"#A3FFC7", "Sm"=>"#8FFFC7", "Eu"=>"#61FFC7", "Gd"=>"#45FFC7",
        "Tb"=>"#30FFC7", "Dy"=>"#1FFFC7", "Ho"=>"#00FF9C", "Er"=>"#00E675",
        "Tm"=>"#00D452", "Yb"=>"#00BF38", "Lu"=>"#00AB24", "Hf"=>"#4DC2FF",
        "Ta"=>"#4DA6FF", "W" =>"#2194D6", "Re"=>"#267DAB", "Os"=>"#266696",
        "Ir"=>"#175487", "Pt"=>"#D0D0E0", "Au"=>"#FFD123", "Hg"=>"#B8B8D0",
        "Tl"=>"#A6544D", "Pb"=>"#575961", "Bi"=>"#9E4FB5", "Po"=>"#AB5C00",
        "At"=>"#754F45", "Rn"=>"#428296", "Fr"=>"#420066", "Ra"=>"#007D00",
        "Ac"=>"#70ABFA", "Th"=>"#00BAFF", "Pa"=>"#00A1FF", "U"=>"#008FFF",
        "Np"=>"#0080FF", "Pu"=>"#006BFF", "Am"=>"#545CF2", "Cm"=>"#785CE3",
        "Bk"=>"#8A4FE3", "Cf"=>"#A136D4", "Es"=>"#B31FD4", "Fm"=>"#B31FBA",
        "Md"=>"#B30DA6", "No"=>"#BD0D87", "Lr"=>"#C70066", "Rf"=>"#CC0059",
        "Db"=>"#D1004F", "Sg"=>"#D90045", "Bh"=>"#E00038", "Hs"=>"#E6002E",
        "Mt"=>"#EB0026",
    ]
    # Workaround for absence of `mapvalues` in Julia
    Dict(map(pairs) do (n, c)
        (lowercase(n), Makie.color(c))
    end)
end

function type_to_color(t::String)
    letters = vcat('a':'z', 'A':'Z')
    idx = findfirst(c -> !in(c, letters), t)
    elem = isnothing(idx) ? t : t[begin:idx-1]
    return get(atom_colors, lowercase(elem), nothing)
end

function build_class_colors(cryst::Crystal)
    cnt = 0
    ret = Dict{Int, Makie.RGB}()
    root = @something cryst.root cryst
    for (type, class) in zip(root.types, root.classes)
        if !haskey(ret, class)
            color = type_to_color(type)
            if isnothing(color)
                cnt += 1
                color = getindex_cyclic(seaborn_muted, cnt)
            end
            push!(ret, class => color)
        end
    end
    return ret
end

# Analogous to internal Makie function `numbers_to_colors`
function numbers_to_colors!(out::AbstractArray{Makie.RGBAf}, in::AbstractArray{<: Number}, colormap, colorrange)
    @assert length(out) == length(in)
    if isnothing(colorrange) || colorrange[1] >= colorrange[2] - 1e-8
        fill!(out, first(colormap))
    else
        cmin, cmax = colorrange
        len = length(colormap)
        map!(out, in) do c
            # If `cmin ≤ in[i] ≤ cmax` then `0.5 ≤ x ≤ len+0.5`
            x = (c - cmin) / (cmax - cmin) * len + 0.5
            # Round to integer and clip to range [1, len]
            colormap[max(min(round(Int, x), len), 1)]
        end
    end
    return nothing
end

# Alternatively: Makie.RGBAf(Makie.RGBf(c), alpha)
set_alpha(c, alpha) = Makie.coloralpha(c, alpha)


function cell_center(ndims)
    if ndims == 3
        return [1, 1, 1] / 2
    elseif ndims == 2
        return [1, 1, 0] / 2
    else
        error("Unsupported `ndims=$ndims`.")
    end
end

function cell_diameter(latvecs, ndims)
    (a1, a2, a3) = eachcol(latvecs)
    if ndims == 3
        return max(norm(a1+a2+a3), norm(a1+a2-a3), norm(a1-a2+a3), norm(a1-a2-a3))
    elseif ndims == 2
        return max(norm(a1+a2), norm(a1-a2))
    else
        error("Unsupported `ndims=$ndims`.")
    end
end


function orient_camera!(ax, latvecs; ghost_radius, ℓ0, orthographic, ndims)
    a1, a2, a3 = eachcol(latvecs)

    if ndims == 3
        lookat = (a1 + a2 + a3)/2
        camshiftdir = normalize(a1 + a2)
        upvector = normalize(a1 × a2)
    elseif ndims == 2
        lookat = (a1 + a2) / 2
        camshiftdir = -normalize(a1 × a2)
        upvector = normalize((a1 × a2) × a1)
    else
        error("Unsupported dimension: $ndims")
    end

    # The extra shift ℓ0 is approximately the nearest-neighbor distance
    camdist = max(cell_diameter(latvecs, ndims)/2 + 0.8ℓ0, ghost_radius)
    if orthographic
        eyeposition = lookat - camdist * camshiftdir
        projectiontype = Makie.Orthographic
    else
        eyeposition = lookat - 2.5 * camdist * camshiftdir
        projectiontype = Makie.Perspective
    end

    # Disable the key that would reset camera
    reset = false
    # Do not automatically "recenter" when adding objects
    center = false
    # No rotations on zoom
    zoom_shift_lookat = false
    # Mouse-drag rotations are SO(3) symmetric
    fixed_axis = false

    Makie.cam3d!(ax.scene; lookat, eyeposition, upvector, projectiontype, reset, center, fixed_axis,
                 zoom_shift_lookat, clipping_mode=:view_relative, near=0.01, far=100)
end

function register_compass_callbacks(axcompass, lscene)
    refcam = lscene.scene.camera_controls
    Makie.onany(refcam.eyeposition, refcam.lookat, refcam.upvector; update=true) do cam_eye, cam_lookat, cam_upvector
        eye = 4normalize(cam_eye - cam_lookat)
        lookat = Makie.Point3f0(0, 0, 0)
        Makie.update_cam!(axcompass.scene, eye, lookat, cam_upvector)
    end
end

function add_cartesian_compass(fig, lscene; left=0, right=150, bottom=0, top=150)
    axcompass = Makie.LScene(fig, bbox=Makie.BBox(left, right, bottom, top), show_axis=false)

    # Draw arrows at origin
    pts = [Makie.Point3f0(0, 0, 0), Makie.Point3f0(0, 0, 0), Makie.Point3f0(0, 0, 0)]
    vecs = [Makie.Point3f0(1, 0, 0), Makie.Point3f0(0, 1, 0), Makie.Point3f0(0, 0, 1)]
    Makie.arrows!(axcompass, pts, 0.8*vecs; color=[:red, :orange, :yellow], arrowsize=0.3, inspectable=false)

    # Draw labels
    for (pos, text) in zip(1.2vecs, ["x", "y", "z"])
        Makie.text!(axcompass, pos; text, color=:black, fontsize=16, font=:bold, glowwidth=4.0,
            glowcolor=(:white, 0.6), align=(:center, :center), depth_shift=-1f0)
    end
    
    # The intention is that the parent scene fully controls the camera, and
    # ideally the compass "inset" wouldn't receive any events at all. However,
    # there is a GLMakie bug where events do go to the inset when the figure is
    # first created, and the window in the background. As a workaround, set all
    # speeds to zero to disable rotation, translation, and zooming of compass.
    # TODO: File bug using the example set-up code at
    # https://github.com/SunnySuite/Sunny.jl/issues/147#issuecomment-1866608609
    Makie.cam3d!(axcompass.scene; center=false, mouse_rotationspeed=0, mouse_translationspeed=0, mouse_zoomspeed=0)

    # Update compass on any changes to `lscene` camera
    register_compass_callbacks(axcompass, lscene)

    return axcompass
end

function cell_wireframe(latvecs, ndims)
    vecs = Makie.Point3f0.(eachcol(latvecs))
    ret = Tuple{Makie.Point3f0, Makie.Point3f0}[]

    origin = zero(Makie.Point3f0)

    if ndims == 3
        for j in 0:1, k in 0:1
            shift = j*vecs[2]+k*vecs[3]
            push!(ret, (origin+shift, vecs[1]+shift))
        end
        for i in 0:1, k in 0:1
            shift = i*vecs[1]+k*vecs[3]
            push!(ret, (origin+shift, vecs[2]+shift))
        end
        for i in 0:1, j in 0:1
            shift = i*vecs[1]+j*vecs[2]
            push!(ret, (origin+shift, vecs[3]+shift))
        end
    elseif ndims == 2
        for j in 0:1
            shift = j*vecs[2]
            push!(ret, (origin+shift, vecs[1]+shift))
        end
        for i in 0:1
            shift = i*vecs[1]
            push!(ret, (origin+shift, vecs[2]+shift))
        end
    end
    return ret
end


function characteristic_length_between_atoms(cryst::Crystal)
    # Detect if atom displacements are on a submanifold (aligned line or plane)
    ps = cryst.positions[1:end-1] .- Ref(cryst.positions[end])
    any_nonzero = map(1:3) do i
        any(p -> !iszero(p[i]), ps)
    end
    vecs = eachcol(cryst.latvecs)[findall(any_nonzero)]

    # Take nth root of appropriate hypervolume per atom
    if length(vecs) == 0
        ℓ = Inf                            # For a single atom, use ℓ0 below
    elseif length(vecs) == 1
        ℓ = norm(vecs[1]) / natoms(cryst)  # Atoms aligned with single lattice vector
    elseif length(vecs) == 2
        ℓ = sqrt(norm(vecs[1] × vecs[2]) / natoms(cryst))
    elseif length(vecs) == 3
        ℓ = cbrt(abs(det(cryst.latvecs)) / natoms(cryst))
    else
        error("Internal error")
    end

    # An upper bound is the norm of the smallest lattice vector.
    ℓ0 = minimum(norm.(eachcol(cryst.latvecs)))

    return min(ℓ0, ℓ)
end

# Like `reference_bonds` but supply a number of bonds
function reference_bonds_upto(cryst, nbonds, ndims)
    # Calculate heuristic maximum distance
    min_a = minimum(norm.(eachcol(cryst.latvecs)))
    nclasses = length(unique(cryst.classes))
    max_dist = 2 * min_a * (nbonds / (nclasses*natoms(cryst)))^(1/ndims)

    # Find bonds up to distance, without self-bonds
    refbonds = filter(reference_bonds(cryst, max_dist)) do b
        return !(b.i == b.j && iszero(b.n))
    end
    
    # Verify max_dist heuristic
    if length(refbonds) > 10nbonds
        @warn "Found $(length(refbonds)) bonds using max_dist of $max_dist"
    end

    return first(refbonds, nbonds)
end

function propagate_reference_bond_for_cell(cryst, b_ref)
    symops = Sunny.canonical_group_order(cryst.sg.symops)

    found = map(_ -> Bond[], cryst.positions)
    for s in symops
        b = Sunny.transform(cryst, s, b_ref)
        # If this bond hasn't been found, add it to the list
        if !(b in found[b.i]) && !(reverse(b) in found[b.j])
            push!(found[b.i], b)
        end
    end

    return reduce(vcat, found)
end


# Get the 3×3 exchange matrix for bond `b`
function exchange_on_bond(interactions, b)
    isnothing(interactions) && return zero(Sunny.Mat3)
    pairs = interactions[b.i].pair
    indices = findall(pc -> pc.bond == b, pairs)
    isempty(indices) && return zero(Sunny.Mat3)
    return pairs[only(indices)].bilin * Mat3(I)
end

# Get largest exchange interaction scale. For symmetric part, this is the
# largest eigenvalue. For antisymmetric part, this is an empirical rescaling of
# the norm of the DM vector. (Note that a typical system has small DM vector
# magnitude relative to the symmetric exchange, and the heuristics for visual
# size are taking this into account.)
function exchange_magnitude(interactions)
    ret = -Inf
    for int in interactions, pc in int.pair
        J = pc.bilin * Mat3(I)
        sym = maximum(abs.(eigvals(Hermitian(J+J')/2)))
        dm = norm(Sunny.extract_dmvec(J))
        ret = max(ret, sym + 2dm)
    end
    return ret
end

# Return an axis scaling and quaternion rotation corresponding to J
function exchange_decomposition(J)
    # Absolute value of eigenvalues control scaling of ellipsoidal axis, with
    # ellipsoid volume depicting interaction strength.
    vals, vecs = eigen(Hermitian(J+J')/2)

    # If vecs includes a reflection, then permute columns
    if det(vecs) < 0
        vals = [vals[2], vals[1], vals[3]]
        vecs = hcat(vecs[:,2], vecs[:,1], vecs[:,3])
    end

    # Now vecs is a pure rotation
    @assert vecs'*vecs ≈ I && det(vecs) ≈ 1
    
    # Quaternion that rotates Cartesian coordinates into principle axes of J.
    axis, angle = Sunny.matrix_to_axis_angle(Mat3(vecs))
    q = iszero(axis) ? Makie.Quaternionf(0,0,0,1) : Makie.qrotation(axis, angle)

    return (vals, q)
end

function draw_exchange_geometries(; ax, obs, ionradius, pts, scaled_exchanges)

    ### Ellipsoids for symmetric exchanges

    # Dimensionless scalings and rotations associated with principle axes
    decomps = exchange_decomposition.(scaled_exchanges)            
    scalings = map(x -> x[1], decomps)
    rotation = map(x -> x[2], decomps)

    # Enlarge scalings so that the maximum scaling _cubed_ denotes magnitude
    scalings = map(scalings) do scal
        szmax = maximum(abs.(scal))
        cbrt(szmax) * (scal/szmax)
    end

    markersize = map(scalings) do scal
        # Make sure ellipsoids don't get flattened to zero
        szmax = maximum(abs.(scal))
        ionradius * Makie.Vec3f([max(abs(x), szmax/4) for x in scal])
    end

    # Draw ellipsoidal bounding box
    color = map(scalings) do x
        y = sum(x) / sum(abs.(x)) # -1 ≤ y ≤ 1
        c = 0.8
        d = c+(1-c)*abs(y) # c ≤ d ≤ 1
        y > 0 ? Makie.RGBf(c, c, d) : Makie.RGBf(d, c, c)
    end
    o = Makie.meshscatter!(ax, pts; color, markersize, rotation, specular=0, diffuse=1.5, inspectable=false)
    Makie.connect!(o.visible, obs)

    # Draw dots using cylinders
    cylinders = map(eachcol(Sunny.Mat3(I))) do x
        p = Makie.GeometryBasics.Point(x...)
        Makie.GeometryBasics.Cylinder(-p, p, 0.3)
    end
    for dim in 1:3
        color = map(scalings) do x
            x[dim] < 0 ? :red : :blue
        end

        # Apply some additional scaling so that all the dots on a given
        # ellipsoid have a roughly constant linear size
        rescalings = map(scalings) do x
            c = sqrt(abs(x[dim]) / maximum(abs.(x)))
            [dim == 1 ? 1 : c,
             dim == 2 ? 1 : c,
             dim == 3 ? 1 : c]
        end
        markersize2 = [ms .* rs for (ms, rs) in zip(markersize, rescalings)]
    
        o = Makie.meshscatter!(ax, pts; color, markersize=markersize2, rotation, marker=cylinders[dim], inspectable=false)
        Makie.connect!(o.visible, obs)            
    end

    ### Cones for DM vectors. Because they tend to be weaker in magnitude,
    ### we apply some heuristic amplification to the arrow size.

    dmvecs = Sunny.extract_dmvec.(scaled_exchanges)
    dirs = @. Makie.Vec3f0(normalize(dmvecs))
    # The largest possible ellipsoid occurs in the case of `scalings ==
    # [1,1,1]`, yielding a sphere with size `ionradius`.
    ellipsoid_radii = @. ionradius * norm(scalings) / √3
    arrowsize = @. 2ionradius * cbrt(norm(dmvecs)) # size of arrow head
    dm_pts = @. pts + 1.1ellipsoid_radii * dirs
    o = Makie.arrows!(ax, dm_pts, dirs; lengthscale=0, arrowsize, diffuse=1.15, color=:magenta, specular=0.0, inspectable=false) 
    Makie.connect!(o.visible, obs)
end

function draw_bonds(; ax, obs, ionradius, exchange_mag, cryst, interactions, bonds, refbonds, color)
    
    # Map each bond to line segments in global coordinates
    segments = map(bonds) do b
        (; ri, rj) = Sunny.BondPos(cryst, b)
        Makie.Point3f0.(Ref(cryst.latvecs) .* (ri, rj))
    end

    # If the bonds are distinct from the refbonds, then add periodic "ghost" images
    if bonds !== refbonds
        # Indices for the bonds which most be repeated
        ghosts = findall(b -> !iszero(b.n), bonds)

        # Concatenate ghosts to the end of arrays
        bonds = vcat(bonds, bonds[ghosts])
        refbonds = vcat(refbonds, refbonds[ghosts])
        color = vcat(color, color[ghosts])

        # Ghost bonds are offset by -n multiples of lattice vectors
        segments = vcat(segments, map(ghosts) do i
            offset = - cryst.latvecs * bonds[i].n
            segments[i] .+ Ref(offset)
        end)
    end

    # String for each bond b′. Like print_bond(b′), but shorter.
    bond_labels = map(zip(bonds, refbonds)) do (b, b_ref)
        dist = Sunny.global_distance(cryst, b)
        dist_str = Sunny.number_to_simple_string(dist; digits=4, atol=1e-12)

        if isnothing(interactions)
            basis = Sunny.basis_for_symmetry_allowed_couplings(cryst, b; b_ref)
            basis_strs = Sunny.coupling_basis_strings(zip('A':'Z', basis); digits=4, atol=1e-12)
            J_matrix_str = Sunny.formatted_matrix(basis_strs; prefix="J:  ")
            antisym_basis_idxs = findall(J -> J ≈ -J', basis)
            if !isempty(antisym_basis_idxs)
                antisym_basis_strs = Sunny.coupling_basis_strings(collect(zip('A':'Z', basis))[antisym_basis_idxs]; digits=4, atol=1e-12)
                dmvecstr = join([antisym_basis_strs[2,3], antisym_basis_strs[3,1], antisym_basis_strs[1,2]], ", ")
                J_matrix_str *= "\nDM: [$dmvecstr]"
            end
        else
            J = exchange_on_bond(interactions, b)
            basis_strs = Sunny.number_to_simple_string.(J; digits=3)
            J_matrix_str = Sunny.formatted_matrix(basis_strs; prefix="J:  ")
            if J ≉ J'
                dmvec = Sunny.extract_dmvec(J)
                dmvecstr = join(Sunny.number_to_simple_string.(dmvec; digits=3), ", ")
                J_matrix_str *= "\nDM: [$dmvecstr]"
            end
        end

        return """
            $b
            Distance $dist_str
            $J_matrix_str
            """
    end
    inspector_label(_plot, index, _position) = bond_labels[index]

    # A bond has an arrowhead if it allows DM interactions
    hasarrowhead = map(bonds) do b
        basis = Sunny.basis_for_symmetry_allowed_couplings(cryst, b)
        any(J -> J ≈ -J', basis)
    end

    # Draw cylinders or arrows for each bond
    linewidth = 0.25ionradius
    arrowwidth = 1.8linewidth
    arrowlength = 2.2arrowwidth
    disps = [rj-ri for (ri, rj) in segments]
    dirs = normalize.(disps)
    pts = @. getindex.(segments, 1) + ionradius*dirs
    arrowsize = hasarrowhead .* Ref(Makie.Vec3f(arrowwidth, arrowwidth, arrowlength))
    lengthscale = @. norm(disps) - 2ionradius - hasarrowhead*arrowlength
    o = Makie.arrows!(ax, pts, dirs; arrowsize, lengthscale, linewidth, color, diffuse=3,
                      transparency=true, inspectable=true, inspector_label)
    Makie.connect!(o.visible, obs)

    # Draw exchange interactions if data is available
    if exchange_mag > 0
        pts = [(ri+rj)/2 for (ri, rj) in segments]
        exchanges = exchange_on_bond.(Ref(interactions), bonds)
        draw_exchange_geometries(; ax, obs, ionradius, pts, scaled_exchanges=exchanges/exchange_mag)
    end

    return
end


# Return true if `type` doesn't uniquely identify the site equivalence class
function is_type_degenerate(cryst, i)
    typ = cryst.types[i]
    same_typ = findall(==(typ), cryst.types)
    return !allequal(cryst.classes[same_typ])
end

# Construct atom labels for use in DataInspector
function label_atoms(cryst; ismagnetic)
    return map(1:natoms(cryst)) do i
        typ = cryst.types[i]
        rstr = Sunny.fractional_vec3_to_string(cryst.positions[i])
        ret = []

        if ismagnetic && is_type_degenerate(cryst, i)
            c = cryst.classes[i]
            push!(ret, isempty(typ) ? "Class $c at $rstr" : "'$typ' (class $c) at $rstr")
        else
            push!(ret, isempty(typ) ? "Position $rstr" : "'$typ' at $rstr")
        end
        if ismagnetic
            # TODO: Show onsite couplings?
        end
        join(ret, "\n")
    end
end

function draw_atoms_or_dipoles(; ax, full_crystal_toggle, dipole_menu, cryst, sys, class_colors, ionradius, ndims, ghost_radius)
    selection = isnothing(dipole_menu) ? Makie.Observable("No dipoles") : dipole_menu.selection
    show_spin_dipoles = Makie.lift(==("Spin dipoles"), selection)
    show_magn_dipoles = Makie.lift(==("Magnetic dipoles"), selection)
    show_atom_spheres = Makie.lift(==("No dipoles"), selection)

    # Draw magnetic and non-magnetic ions
    for ismagnetic in (false, true)
        if ismagnetic
            xtal = cryst
            relevant_classes = cryst.classes
        else
            isnothing(cryst.root) && continue
            # Relevant classes are those present in cryst.root but not cryst
            xtal = cryst.root
            relevant_classes = setdiff(xtal.classes, cryst.classes)
        end

        for isghost in (false, true)
            if isghost
                (idxs, offsets) = Sunny.all_offsets_within_distance(xtal.latvecs, xtal.positions, cell_center(ndims); max_dist=ghost_radius, nonzeropart=true)
                alpha = 0.08
            else
                idxs = eachindex(xtal.positions)
                offsets = [zero(Vec3) for _ in idxs]
                alpha = 1.0
            end

            # Reduce idxs and offsets to include only atom indices according to
            # `relevant_classes`, as set by `ismagnetic`
            downselect = findall(in(relevant_classes), xtal.classes[idxs])
            isempty(downselect) && continue
            idxs = idxs[downselect]
            offsets = offsets[downselect]

            # Information for drawing atoms in xtal labeled by idxs
            color = [(class_colors[c], alpha) for c in xtal.classes[idxs]]
            rs = xtal.positions[idxs] .+ offsets
            pts = [xtal.latvecs * r for r in rs]

            # Labels for non-ghost atoms
            inspector_label = nothing
            if !isghost
                labels = label_atoms(xtal; ismagnetic)[idxs]
                inspector_label = (_plot, index, _position) -> labels[index]
            end
            
            # Show dipoles. Mostly consistent with code in plot_spins.
            if !isnothing(sys) && ismagnetic
                sites = Sunny.position_to_site.(Ref(sys), rs)
                g0 = norm(sys.gs) / sqrt(length(sys.gs) * 3)
                N0 = norm(sys.Ns) / sqrt(length(sys.Ns))
                s0 = (N0 - 1) / 2
                spin_dipoles = sys.dipoles[sites] / s0
                magn_dipoles = magnetic_moment.(Ref(sys), sites) / (s0*g0)
                for (dipoles, obs) in [(spin_dipoles, show_spin_dipoles), (magn_dipoles, show_magn_dipoles)]
                    a0 = 5ionradius
                    arrowsize = 0.4a0
                    linewidth = 0.12a0
                    lengthscale = 0.6a0
                    markersize = 0.9ionradius
                    arrow_fractional_shift = 0.6

                    vecs = lengthscale * dipoles
                    pts_shifted = pts - arrow_fractional_shift * vecs

                    # Draw arrows
                    linecolor = (:white, alpha)
                    arrowcolor = (:gray, alpha)
                    o = Makie.arrows!(ax, Makie.Point3f.(pts_shifted), Makie.Vec3f.(vecs); arrowsize, linewidth, linecolor, arrowcolor, diffuse=1.15, transparency=isghost, inspectable=false)
                    Makie.connect!(o.visible, obs)

                    # Small sphere inside arrow to mark atom position
                    o = Makie.meshscatter!(ax, pts; markersize, color, diffuse=1.15, transparency=isghost, inspectable=!isghost, inspector_label)
                    Makie.connect!(o.visible, obs)
                end
            end

            # Show atoms as spheres
            markersize = ionradius * (ismagnetic ? 1 : 1/2)
            o = Makie.meshscatter!(ax, pts; markersize, color, diffuse=1.15, transparency=isghost, inspectable=!isghost, inspector_label)
            Makie.connect!(o.visible, ismagnetic ? show_atom_spheres : full_crystal_toggle.active)
        
            # White numbers for real, magnetic ions
            if !isghost && ismagnetic
                text = repr.(eachindex(pts))
                o = Makie.text!(ax, pts; text, color=:white, fontsize=16, align=(:center, :center), depth_shift=-1f0)
                !ismagnetic && Makie.connect!(o.visible, full_crystal_toggle.active)
            end
        end
    end
end


function Sunny.view_crystal(cryst::Crystal, max_dist::Number)
    @warn "view_crystal(cryst, max_dist) is deprecated! Use `view_crystal(cryst)` instead. See also optional `ghost_radius` argument."
    Sunny.view_crystal(cryst; ghost_radius=max_dist)
end

"""
    view_crystal(crystal::Crystal; refbonds=10, orthographic=false, ghost_radius=nothing, ndims=3, compass=true)
    view_crystal(sys::System; ...)

Launches a graphical user interface to visualize the [`Crystal`](@ref) unit
cell. If a [`System`](@ref) is provided, then the 3×3 exchange matrices for each
bond will be depicted graphically.

 - `refbonds`: By default, calculate up to 10 reference bonds using the
   `reference_bonds` function. An explicit list of reference bonds may also be
   provided.
 - `orthographic`: Use orthographic camera perspective.
 - `ghost_radius`: Show periodic images up to a given distance. Defaults to the
   cell size.
 - `ndims`: Spatial dimensions of system (1, 2, or 3).
 - `compass`: If true, draw Cartesian axes in bottom left.
"""
function Sunny.view_crystal(cryst::Crystal; refbonds=10, orthographic=false, ghost_radius=nothing, ndims=3, compass=true, size=(768, 512), dims=nothing)
    isnothing(dims) || error("Use notation `ndims=$dims` instead of `dims=$dims`")
    view_crystal_aux(cryst, nothing; refbonds, orthographic, ghost_radius, ndims, compass, size)
end

function Sunny.view_crystal(sys::System; refbonds=10, orthographic=false, ghost_radius=nothing, ndims=3, compass=true, size=(768, 512), dims=nothing)
    isnothing(dims) || error("Use notation `ndims=$dims` instead of `dims=$dims`")
    Sunny.is_homogeneous(sys) || error("Cannot plot interactions for inhomogeneous system.")
    view_crystal_aux(orig_crystal(sys), sys;
                     refbonds, orthographic, ghost_radius, ndims, compass, size)
end

function view_crystal_aux(cryst, sys; refbonds, orthographic, ghost_radius, ndims, compass, size)
    warn_wglmakie()

    interactions = isnothing(sys) ? nothing : Sunny.interactions_homog(something(sys.origin, sys))

    # Dict that maps atom class to color
    class_colors = build_class_colors(cryst)

    # Distance to show periodic images
    if isnothing(ghost_radius)
        ghost_radius = cell_diameter(cryst.latvecs, ndims)/2
    end

    # Use provided reference bonds or find from symmetry analysis
    if refbonds isa Number
        @assert isinteger(refbonds)
        custombonds = false
        refbonds = reference_bonds_upto(cryst, Int(refbonds), ndims)
    elseif refbonds isa AbstractArray{Bond}
        custombonds = true
    else
        error("Parameter `refbonds` must be an integer or a `Bond` list.")
    end

    refbonds_dists = [Sunny.global_distance(cryst, b) for b in refbonds]

    # Radius of the magnetic ions. Sets a length scale for other objects too.
    ionradius = let
        # The root crystal may contain non-magnetic ions. If present, these
        # should reduce the characteristic length scale.
        ℓ0 = characteristic_length_between_atoms(something(cryst.root, cryst))
        # If there exists a very short bond distance, then appropriately reduce the
        # length scale
        ℓ0 = min(ℓ0, 0.8minimum(refbonds_dists))
        # Small enough to fit everything
        0.2ℓ0
    end

    fig = Makie.Figure(; size)

    # Main scene
    ax = Makie.LScene(fig[1, 1], show_axis=false)

    # Show Cartesian axes, with link to main camera
    if compass
        axcompass = add_cartesian_compass(fig, ax)
    end

    # Set of widgets
    widget_list = Makie.GridLayout(; tellheight=false, valign=:top)
    fig[1, 2] = widget_list
    widget_cnt = 0
    fontsize = 16

    # Controls for camera perspective
    menu = Makie.Menu(fig; options=["Perspective", "Orthographic"], default=(orthographic ? "Orthographic" : "Perspective"), fontsize)
    button = Makie.Button(fig; label="Reset", fontsize)
    Makie.onany(button.clicks, menu.selection; update=true) do _, mselect
        orthographic = mselect == "Orthographic"
        # Zoom out a little bit extra according to nn bond distance
        ℓ0=minimum(refbonds_dists)
        orient_camera!(ax, cryst.latvecs; ghost_radius, orthographic, ndims, ℓ0)
        compass && register_compass_callbacks(axcompass, ax)
    end
    widget_list[widget_cnt+=1, 1] = Makie.hgrid!(menu, button)

    # Controls for dipoles
    if !isnothing(sys)
        dipole_menu = Makie.Menu(fig; options=["No dipoles", "Spin dipoles", "Magnetic dipoles"], fontsize)
        widget_list[widget_cnt+=1, 1] = dipole_menu
    else
        dipole_menu = nothing
    end

    # Set up grid of toggles
    toggle_grid = widget_list[widget_cnt+=1,1] = Makie.GridLayout()
    toggle_cnt = 0
    buttoncolor = Makie.RGB(0.2, 0.2, 0.2)
    framecolor_active = Makie.RGB(0.7, 0.7, 0.7)
    framecolor_inactive = Makie.RGB(0.9, 0.9, 0.9)

    # Toggle for non-magnetic ions
    if !isnothing(cryst.root)
        full_crystal_toggle = Makie.Toggle(fig; active=true, buttoncolor, framecolor_inactive, framecolor_active)
        toggle_grid[toggle_cnt+=1, 1:2] = [full_crystal_toggle, Makie.Label(fig, "Full crystal"; fontsize, halign=:left)]
    else
        full_crystal_toggle = nothing
    end
    draw_atoms_or_dipoles(; ax, full_crystal_toggle, dipole_menu, cryst, sys, class_colors, ionradius, ndims, ghost_radius)

    exchange_mag = isnothing(interactions) ? 0.0 : exchange_magnitude(interactions)

    # Toggle on/off atom reference bonds
    bond_colors = [getindex_cyclic(seaborn_bright, i) for i in eachindex(refbonds)]
    active = custombonds
    toggle = Makie.Toggle(fig; active, buttoncolor, framecolor_inactive, framecolor_active)
    color = set_alpha.(bond_colors, 0.25)
    draw_bonds(; ax, obs=toggle.active, ionradius, exchange_mag, cryst, interactions, bonds=refbonds, refbonds, color)
    toggle_grid[toggle_cnt+=1, 1:2] = [toggle, Makie.Label(fig, "Reference bonds"; fontsize, halign=:left)]
    
    # Toggle on/off bonds within each class
    for (i, (b, bond_color)) in enumerate(zip(refbonds, bond_colors))
        active = (i == 1)
        framecolor_active = set_alpha(bond_color, 0.7)
        framecolor_inactive = set_alpha(bond_color, 0.15)
        toggle = Makie.Toggle(fig; active, buttoncolor, framecolor_inactive, framecolor_active)
        bonds = propagate_reference_bond_for_cell(cryst, b)
        refbonds = fill(b, length(bonds))
        color = fill(set_alpha(bond_color, 0.25), length(bonds))
        draw_bonds(; ax, obs=toggle.active, ionradius, exchange_mag, cryst, interactions, bonds, refbonds, color)
        bondstr = "Bond($(b.i), $(b.j), $(b.n))"
        toggle_grid[toggle_cnt+=1, 1:2] = [toggle, Makie.Label(fig, bondstr; fontsize, halign=:left)]
    end

    # Show cell volume
    Makie.linesegments!(ax, cell_wireframe(cryst.latvecs, ndims); color=:teal, linewidth=1.5, inspectable=false)
    pos = [(3/4)*Makie.Point3f0(p) for p in eachcol(cryst.latvecs)[1:ndims]]
    text = [Makie.rich("a", Makie.subscript(repr(i))) for i in 1:ndims]
    Makie.text!(ax, pos; text, color=:black, fontsize=20, font=:bold, glowwidth=4.0,
                glowcolor=(:white, 0.6), align=(:center, :center), depth_shift=-1f0)

    # Add inspector for pop-up information. Use a monospaced font provided
    # available in Makie.jl/assets/fonts/. The depth needs to be almost exactly
    # 1e4, but not quite, otherwise only a white background will be shown.
    Makie.DataInspector(ax; indicator_color=:gray, fontsize, font="Deja Vu Sans Mono", depth=(1e4 - 1))

    return fig
end



# Wrapper over `FigureLike` to support both `show` and `notify`.
struct NotifiableFigure
    notifier :: Makie.Observable{Nothing}
    figure :: Makie.FigureLike
end
Base.showable(mime::MIME, fig::NotifiableFigure) = showable(mime, fig.figure)
Base.show(io::IO, ::MIME"text/plain", fig::NotifiableFigure) = print(io, "(Notifiable) " * repr(fig.figure))
Base.show(io::IO, m::MIME, fig::NotifiableFigure) = show(io, m, fig.figure)
Base.display(fig::NotifiableFigure; kwargs...) = display(fig.figure; kwargs...)
Base.notify(fig::NotifiableFigure) = notify(fig.notifier)
Makie.record(func, nf::NotifiableFigure, path, iter; kwargs...) = Makie.record(func, nf.figure, path, iter; kwargs...)

"""
    plot_spins(sys::System; arrowscale=1.0, color=:red, colorfn=nothing,
               colormap=:viridis, colorrange=nothing, show_cell=true, orthographic=false,
               ghost_radius=0, ndims=3, compass=true)

Plot the spin configuration defined by `sys`. Optional parameters are:

  - `arrowscale`: Scale all arrows by dimensionless factor.
  - `color`: Arrow colors. May be symbolic or numeric. If scalar, will be shared
    among all sites.
  - `colorfn`: Function that dynamically maps from a site index to a numeric
    color value. Useful for animations.
  - `colormap`, `colorrange`: Used to populate colors from numbers following
    Makie conventions.
  - `show_cell`: Show original crystallographic unit cell.
  - `orthographic`: Use orthographic camera perspective.
  - `ghost_radius`: Show periodic images up to a given distance (length units).
  - `ndims`: Spatial dimensions of system (1, 2, or 3).
  - `compass`: If true, draw Cartesian axes in bottom left.

Calling `notify` on the return value will animate the figure.
"""
function Sunny.plot_spins(sys::System; size=(768, 512), compass=true, kwargs...)
    fig = Makie.Figure(; size)
    ax = Makie.LScene(fig[1, 1]; show_axis=false)
    notifier = Makie.Observable(nothing)
    Sunny.plot_spins!(ax, sys; notifier, kwargs...)
    compass && add_cartesian_compass(fig, ax)
    return NotifiableFigure(notifier, fig)
end

"""
    plot_spins!(ax, sys::System; opts...)

Mutating variant of [`plot_spins`](@ref) that allows drawing into a single panel
of a Makie figure.

# Example

```julia
fig = Figure()
plot_spins!(fig[1, 1], sys1)
plot_spins!(fig[2, 1], sys2)
display(fig)
```
"""
function Sunny.plot_spins!(ax, sys::System; notifier=Makie.Observable(nothing), arrowscale=1.0, stemcolor=:lightgray, color=:red,
                     colorfn=nothing, colormap=:viridis, colorrange=nothing, show_cell=true, orthographic=false,
                     ghost_radius=0, ndims=3, dims=nothing)
    isnothing(dims) || error("Use notation `ndims=$dims` instead of `dims=$dims`")
    warn_wglmakie()
    
    if ndims == 2
        sys.dims[3] == 1 || error("System not two-dimensional in (a₁, a₂)")
    elseif ndims == 1
        sys.dims[[2,3]] == [1,1] || error("System not one-dimensional in (a₁)")
    end

    # Show bounding box of magnetic supercell in gray (this needs to come first
    # to set a scale for the scene in case there is only one atom).
    supervecs = sys.crystal.latvecs * diagm(Vec3(sys.dims))
    Makie.linesegments!(ax, cell_wireframe(supervecs, ndims); color=:gray, linewidth=1.5)

    # Infer characteristic length scale between sites
    ℓ0 = characteristic_length_between_atoms(orig_crystal(sys))

    # Quantum spin-s, averaged over all sites. Will be used to normalize
    # dipoles.
    s0 = (sum(sys.Ns)/length(sys.Ns) - 1) / 2

    # Parameters defining arrow shape
    a0 = arrowscale * ℓ0
    arrowsize = 0.4a0
    linewidth = 0.12a0
    lengthscale = 0.6a0
    markersize = 0.8linewidth
    arrow_fractional_shift = 0.6

    # Positions in fractional coordinates of supercell vectors
    rs = [supervecs \ global_position(sys, site) for site in eachsite(sys)]

    for isghost in (false, true)
        if isghost
            alpha = 0.08
            (idxs, offsets) = Sunny.all_offsets_within_distance(supervecs, rs, cell_center(ndims); max_dist=ghost_radius, nonzeropart=true)
        else
            alpha = 1.0
            idxs = eachindex(rs)
            offsets = [zero(Vec3) for _ in idxs]
        end

        # Every call to RGBf constructor allocates, so pre-calculate color
        # arrays to speed animations
        cmap_with_alpha = set_alpha.(Makie.to_colormap(colormap), Ref(alpha))
        numeric_colors = zeros(size(sys.dipoles))
        rgba_colors = zeros(Makie.RGBAf, size(sys.dipoles))

        if isnothing(colorfn)
            # In this case, we can precompute the fixed `rgba_colors` array
            # according to `color`
            if color isa AbstractArray
                @assert length(color) == length(sys.dipoles)
                if eltype(color) <: Number
                    dyncolorrange = @something colorrange extrema(color)
                    numbers_to_colors!(rgba_colors, color, cmap_with_alpha, dyncolorrange)
                else
                    map!(rgba_colors, color) do c
                        set_alpha(Makie.to_color(c), alpha)
                    end
                end
            else
                c = set_alpha(Makie.to_color(color), alpha)
                fill!(rgba_colors, c)
            end
        end

        # These observables will be reanimated upon calling `notify(notifier)`.
        vecs = Makie.Observable(Makie.Vec3f0[])
        pts = Makie.Observable(Makie.Point3f0[])
        pts_shifted = Makie.Observable(Makie.Point3f0[])
        arrowcolor = Makie.Observable(Makie.RGBAf[])

        Makie.on(notifier, update=true) do _
            empty!.((vecs[], pts[], pts_shifted[], arrowcolor[]))

            # Dynamically adapt `rgba_colors` according to `colorfn`
            if !isnothing(colorfn)
                numeric_colors .= colorfn.(CartesianIndices(sys.dipoles))
                dyncolorrange = @something colorrange extrema(numeric_colors)
                numbers_to_colors!(rgba_colors, numeric_colors, cmap_with_alpha, dyncolorrange)
            end
            
            for (site, n) in zip(idxs, offsets)
                v = (lengthscale / s0) * vec(sys.dipoles[site])
                pt = supervecs * (rs[site] + n)
                pt_shifted = pt - arrow_fractional_shift * v
                push!(vecs[], Makie.Vec3f0(v))
                push!(pts[], Makie.Point3f0(pt))
                push!(pts_shifted[], Makie.Point3f0(pt_shifted))
                push!(arrowcolor[], rgba_colors[site])
            end
            # Trigger Makie redraw
            notify.((vecs, pts, pts_shifted, arrowcolor))
            # isnothing(color) || notify(arrowcolor)
        end

        # Draw arrows
        linecolor = (stemcolor, alpha)
        Makie.arrows!(ax, pts_shifted, vecs; arrowsize, linewidth, linecolor, arrowcolor, diffuse=1.15, transparency=isghost)

        # Small sphere inside arrow to mark atom position
        Makie.meshscatter!(ax, pts; markersize=2, color=arrowcolor, diffuse=1.15, transparency=isghost,shading=Makie.NoShading)
    end

    # Bounding box of original crystal unit cell in teal
    if show_cell
        Makie.linesegments!(ax, cell_wireframe(orig_crystal(sys).latvecs, ndims); color=:teal, linewidth=1.5)
        pos = [(3/4)*Makie.Point3f0(p) for p in eachcol(orig_crystal(sys).latvecs)[1:ndims]]
        text = [Makie.rich("a", Makie.subscript(repr(i))) for i in 1:ndims]
        Makie.text!(ax, pos; text, color=:black, fontsize=20, font=:bold, glowwidth=4.0,
                    glowcolor=(:white, 0.6), align=(:center, :center), depth_shift=-1f0)
    end

    orient_camera!(ax, supervecs; ghost_radius, ℓ0, orthographic, ndims)

    return ax
end

function get_unit_energy(units, into)
    if isnothing(units)
        isnothing(into) || error("`into` parameter requires `units` parameter")
        return (1.0, "Energy")
    else
        sym = @something into units.energy
        str = Sunny.unit_strs[sym]
        return (getproperty(units, sym), "Energy ($str)")
    end
end

function colorrange_from_data(; data, saturation, sensitivity, allpositive)
    cmax = Statistics.quantile(filter(!isnan, vec(maximum(data; dims=1))), saturation)
    cmin = Statistics.quantile(filter(!isnan, vec(minimum(data; dims=1))), 1 - saturation)

    # The returned pair (lo, hi) should be strictly ordered, lo < hi, for use in
    # Makie.Colorbar
    if allpositive
        # Intensities are supposed to be non-negative in this branch, but might
        # not be due to any of: User error, very small round-off around negative
        # zero, or Gibbs ringing in a KPM calculation.
        @assert 0 <= sensitivity < 1
        if cmax <= 0
            return (0, 1e-15)
        else
            (sensitivity, 1) .* cmax
        end
    else
        cscale = max(abs(cmax), abs(cmin))
        if iszero(cscale)
            return (-1e-15, 1e-15)
        else
            return (-cscale, cscale)
        end
    end
end

"""
    plot_intensities!(panel, res::AbstractIntensities; opts...)

Mutating variant of [`plot_intensities`](@ref) that allows drawing into a single
panel of a Makie figure.

# Example

```julia
fig = Figure()
plot_intensities!(fig[1, 1], res1; title="Panel 1")
plot_intensities!(fig[2, 1], res2; title="Panel 2")
display(fig)
```
"""
function Sunny.plot_intensities!(panel, res::Sunny.BandIntensities{Float64}; colormap=nothing, colorrange=nothing,
                                 saturation=0.9, sensitivity=0.0025, allpositive=true, units=nothing, into=nothing,
                                 fwhm=nothing, ylims=nothing, title="", axisopts=NamedTuple())
    unit_energy, ylabel = get_unit_energy(units, into)
    axisopts = (; title, axisopts...)
 
    if res.qpts isa Sunny.QPath 
        mindisp, maxdisp = extrema(res.disp)
        ylims = @something ylims (min(0, mindisp), 1.2*maxdisp)
        energies = range(ylims[1], ylims[2], 512)
        fwhm = @something fwhm 0.02*(ylims[2]-ylims[1])
        σ = fwhm/2√(2log(2))
        kernel = Sunny.Broadening(x -> exp(-x^2/2σ^2)) # Gaussian without normalization
        broadened = Sunny.broaden(res; energies, kernel)

        colorrange_suggest = colorrange_from_data(; broadened.data, saturation, sensitivity, allpositive)
        colormap = @something colormap (allpositive ? Makie.Reverse(:thermal) : :bwr)
        colorrange = @something colorrange colorrange_suggest

        xticklabelrotation = maximum(length.(res.qpts.xticks[2])) > 3 ? π/6 : 0.0
        ax = Makie.Axis(panel; xlabel="Momentum (r.l.u.)", ylabel, limits=(nothing, ylims ./ unit_energy), res.qpts.xticks, xticklabelrotation, axisopts...)
        Makie.heatmap!(ax, axes(res.data, 2), collect(energies / unit_energy), broadened.data'; colorrange, colormap, lowclip=:white)
        for i in axes(res.disp, 1)
            Makie.lines!(ax, res.disp[i,:] / unit_energy; color=:lightskyblue3)
        end
        return ax
    else
        error("Cannot plot type $(typeof(res.qpts))")
    end
end

function grid_aspect_ratio(cryst::Crystal, grid::Sunny.QGrid{2})
    # Aspect ratio for global distances
    Δq_global = cryst.recipvecs * (grid.qs[end] - grid.qs[begin])
    e1, e2 = normalize.(Ref(cryst.recipvecs) .* grid.axes)
    abs(dot(e1, e2)) < 1e-12 || error("Cannot yet plot non-orthogonal grid")
    return (Δq_global ⋅ e1) / (Δq_global ⋅ e2)
end

function suggest_labels_for_grid(grid::Sunny.QGrid{N}) where N
    (; axes, offset) = grid

    varidxs = [findmax(abs.(a))[2] for a in axes]
    if varidxs[2] == varidxs[1]
        varidxs[2] = mod1(varidxs[2] + 1, 3)
    end
    varstrs = ("H", "K", "L")

    labels = map(axes, varstrs[varidxs]) do axis, c
        elems = map(axis) do x
            if abs(x) < 1e-12
                "0"
            else
                Sunny.coefficient_to_math_string(x)*c
            end
        end
        return "[" * join(elems, ", ") * "]"
    end

    if norm(offset) > 1e-12
        label1 = labels[begin] * " + " * Sunny.fractional_vec3_to_string(offset)
        labels = (label1, labels[2:end]...)
    end

    return labels
end


function Sunny.plot_intensities!(panel, res::Sunny.Intensities{Float64}; colormap=nothing, colorrange=nothing, saturation=0.9, allpositive=true, units=nothing, into=nothing, title="", axisopts=NamedTuple())
    axisopts = (; title, axisopts...)
    (; crystal, qpts, data, energies) = res

    colorrange_suggest = colorrange_from_data(; data, saturation, sensitivity=0, allpositive)
    colormap = @something colormap (allpositive ? :gnuplot2 : :bwr)
    colorrange = @something colorrange colorrange_suggest
    
    if qpts isa Sunny.QPath
        unit_energy, ylabel = get_unit_energy(units, into)
        xticklabelrotation = maximum(length.(qpts.xticks[2])) > 3 ? π/6 : 0.0
        ax = Makie.Axis(panel[1, 1]; xlabel="Momentum (r.l.u.)", ylabel, qpts.xticks, xticklabelrotation, axisopts...)
        hm = Makie.heatmap!(ax, axes(data, 2), collect(energies/unit_energy), data'; colormap, colorrange)
        Makie.Colorbar(panel[1, 2], hm)
        return ax
    elseif qpts isa Sunny.QGrid{2}
        if isone(length(energies))
            aspect = grid_aspect_ratio(crystal, qpts)
            xlabel, ylabel = suggest_labels_for_grid(qpts)
            (xs, ys) = map(range, qpts.coefs_lo, qpts.coefs_hi, size(qpts.qs))
            ax = Makie.Axis(panel[1, 1]; xlabel, ylabel, aspect, axisopts...)
            hm = Makie.heatmap!(ax, xs, ys, dropdims(data; dims=1); colormap, colorrange)
            Makie.Colorbar(panel[1, 2], hm)
            return ax
        else
            error("Cannot yet plot $(typeof(res))")
        end
    else
        error("Cannot yet plot $(typeof(res))")
    end
end

function Sunny.plot_intensities!(panel, res::Sunny.StaticIntensities{Float64}; colormap=nothing, colorrange=nothing, saturation=0.9, allpositive=true, units=nothing, into=nothing, title="", axisopts=NamedTuple())
    axisopts = (; title, axisopts...)
    (; crystal, qpts, data) = res

    colorrange_suggest = colorrange_from_data(; data=reshape(data, 1, size(data)...), saturation, sensitivity=0, allpositive)
    colormap = @something colormap (allpositive ? :gnuplot2 : :bwr)
    colorrange = @something colorrange colorrange_suggest

    if qpts isa Sunny.QGrid{2}
        aspect = grid_aspect_ratio(crystal, qpts)
        xlabel, ylabel = suggest_labels_for_grid(qpts)
        (xs, ys) = map(range, qpts.coefs_lo, qpts.coefs_hi, size(qpts.qs))
        ax = Makie.Axis(panel[1, 1]; xlabel, ylabel, aspect, axisopts...)
        hm = Makie.heatmap!(ax, xs, ys, data; colormap, colorrange)
        Makie.Colorbar(panel[1, 2], hm)
        return ax
    elseif qpts isa Sunny.QPath
        xticklabelrotation = maximum(length.(qpts.xticks[2])) > 3 ? π/6 : 0.0
        ax = Makie.Axis(panel; xlabel="Momentum (r.l.u.)", ylabel="Intensity", qpts.xticks, xticklabelrotation, axisopts...)
        Makie.lines!(ax, data)
        Makie.ylims!(ax, colorrange)
        return ax
    else
        error("Cannot yet plot $(typeof(res))")
    end
end

function Sunny.plot_intensities!(panel, res::Sunny.PowderIntensities{Float64}; colormap=nothing, colorrange=nothing, saturation=0.9, allpositive=true, units=nothing, into=nothing, title="", axisopts=NamedTuple())
    axisopts = (; title, axisopts...)
    unit_energy, ylabel = get_unit_energy(units, into)
    xlabel = isnothing(units) ? "Momentum " : "Momentum ($(Sunny.unit_strs[units.length])⁻¹)" 
 
    colorrange_suggest = colorrange_from_data(; res.data, saturation, sensitivity=0, allpositive)
    colormap = @something colormap (allpositive ? :gnuplot2 : :bwr)
    colorrange = @something colorrange colorrange_suggest

    ax = Makie.Axis(panel[1, 1]; xlabel, ylabel, axisopts...)
    hm = Makie.heatmap!(ax, res.radii, collect(res.energies/unit_energy), res.data'; colormap, colorrange)
    Makie.Colorbar(panel[1, 2], hm)
    return ax
end

# Axes will currently be labeled as a linear combination of crystal lattice
# vectors. See https://github.com/SunnySuite/Sunny.jl/pull/310 for details.
function Sunny.plot_intensities!(panel, res::Sunny.BinnedIntensities{Float64}; colormap=nothing, colorrange=nothing, saturation=0.9, allpositive=true, units=nothing, into=nothing, title="", axisopts=NamedTuple(), divide_counts=true)
    axisopts = (; title, axisopts...)
    unit_energy, elabel = get_unit_energy(units, into)

    data = divide_counts ? (res.data ./ res.counts) : res.data

    colorrange_suggest = colorrange_from_data(; data, saturation, sensitivity=0, allpositive)
    colormap = @something colormap (allpositive ? :gnuplot2 : :bwr)
    colorrange = @something colorrange colorrange_suggest

    # Low-dimension cases
    n_dims_resolved = count(res.params.numbins .!= 1)

    if n_dims_resolved == 0
        # No resolved data: Just display the one value!
        ax = Makie.Axis(panel[1,1]; axisopts...)
        text = Sunny.number_to_simple_string(data[1]; digits=4)
        Makie.text!(ax, 0, 0; text)
        return ax
    elseif n_dims_resolved == 1
        # Only resolved on one axis!
        x_axis = findfirst(res.params.numbins .!= 1)
        xlabel = (x_axis == 4) ? elabel : Sunny.covector_name(res.params.covectors[x_axis, :])
        ax = Makie.Axis(panel[1,1]; xlabel, ylabel="Integrated Intensity", axisopts...)
        bcs = Sunny.axes_bincenters(res.params)
        bcs[4] /= unit_energy
        Makie.barplot!(ax, bcs[x_axis], data[:]; colormap, colorrange)
        return ax
    elseif n_dims_resolved == 2
        x_axis = findfirst(res.params.numbins .!= 1)
        y_axis = x_axis + findfirst(res.params.numbins[x_axis+1:end] .!= 1)
        xlabel = Sunny.covector_name(res.params.covectors[x_axis, :])
        ylabel = (y_axis == 4) ? elabel : Sunny.covector_name(res.params.covectors[y_axis, :])
        ax = Makie.Axis(panel[1,1]; xlabel, ylabel, axisopts...)
        bcs = Sunny.axes_bincenters(res.params)
        bcs[4] /= unit_energy
        data = reshape(data, size(data, x_axis), size(data, y_axis))
        hm = Makie.heatmap!(ax, bcs[x_axis], bcs[y_axis], data; colormap, colorrange)
        Makie.Colorbar(panel[1, 2], hm)
        return ax
    elseif n_dims_resolved > 2
        error("Higher-dimensional binned data visualization not yet implemented!")
    end
end

#=
  * `axisopts`: An additional collection of named arguments that will be passed
    to the `Makie.Axis` constructor. This allows to override the axis `xlabel`,
    `ylabel`, `xticks`, etc. See [Makie
    documentation](https://docs.makie.org/release/reference/blocks/axis#attributes)
    for details.
=#
"""
    plot_intensities(res::BandIntensities; colormap=nothing, colorrange=nothing, allpositive=true,
                     saturation=0.9, sensitivity=0.0025, fwhm=nothing, ylims=nothing, units=nothing,
                     into=nothing, title="")
    plot_intensities(res::Intensities; colormap=nothing, colorrange=nothing, allpositive=true, 
                     saturation=0.9, units=nothing, into=nothing, title="")
    plot_intensities(res::StaticIntensities; colormap=nothing, colorrange=nothing, allpositive=true, 
                     saturation=0.9, units=nothing, into=nothing, title="")
    plot_intensities(res::PowderIntensities; colormap=nothing, colorrange=nothing, allpositive=true, 
                     saturation=0.9, units=nothing, into=nothing, title="")

Keyword arguments:

  * `colormap`: Color palette for plotting broadened intensities. See Makie docs
    for allowed values.
  * `colorrange`: A lower and upper bound on intensities. For heatmaps, these
    bounds define the intensity values that saturate the colormap.
  * `allpositive`: Should intensities be all positive, apart from numerical
    error? If true, the default colors will clip below zero intensity. If false,
    the default colors will be symmetric about zero intensity.
  * `saturation`: If `colorrange` is not explicitly set, this dimensionless
    parameter defines the upper saturated intensity value as a quantile of
    maximum intensities taken over wavevectors.
  * `sensitivity`: When plotting `BandIntensities`, this defines a lower bound
    on the visible intensities as a fraction of the upper saturated intensity
    value.
  * `fwhm`: When plotting `BandIntensities`, this overrides the full-width at
    half-maximum value used for Gaussian broadening.
  * `ylims`: Limits of the y-axis.
  * `units`: A [`Units`](@ref) instance for labeling axes and performing
    conversions.
  * `into`: A symbol for conversion into a new base energy unit (e.g. `:meV`,
    `:K`, etc.)
  * `title`: An optional title for the plot.
"""
function Sunny.plot_intensities(res::Sunny.AbstractIntensities; opts...)
    fig = Makie.Figure()
    Sunny.plot_intensities!(fig[1,1], res; opts...)
    return fig
end


function Sunny.viz_qqq_path(params::Sunny.BinningParameters; kwargs...)
    f = Makie.Figure()
    ax = Makie.LScene(f[1,1]; show_axis=false)
    Makie.cam3d!(ax.scene; projectiontype=Makie.Orthographic)
    viz_qqq_path!(ax, params; kwargs...)
  
    aabb_lo, aabb_hi = Sunny.binning_parameters_aabb(params)
    lo = min.(0, floor.(Int64, aabb_lo))
    hi = max.(0, ceil.(Int64, aabb_hi))
    Makie.scatter!(ax, map(x -> Makie.Point3f(lo .+ x.I .- 1), CartesianIndices(ntuple(i -> 1 + hi[i] - lo[i], 3)))[:], color=:black)
    global_axes = [(Makie.Point3f(-1,0,0), Makie.Point3f(1,0,0)),
                   (Makie.Point3f(0,-1,0), Makie.Point3f(0,1,0)),
                   (Makie.Point3f(0,0,-1), Makie.Point3f(0,0,1))]
    Makie.linesegments!(ax, global_axes, color=:black)
    Makie.text!(1.1, 0, 0; text="𝐛₁ [R.L.U.]")
    Makie.text!(0, 1.1, 0; text="𝐛₂ [R.L.U.]")
    Makie.text!(0, 0, 1.1; text="𝐛₃ [R.L.U.]")
    display(f)
    ax
end

function viz_qqq_path!(ax, params::Sunny.BinningParameters; line_alpha=0.3, color=nothing, colorrange=nothing, bin_colors=[:red,:blue,:green], bin_line_width=0.5)
    @assert iszero(params.covectors[1:3, 4]) && iszero(params.covectors[4, 1:3])
    bes = Sunny.axes_binedges(params)
    M = inv(params.covectors[1:3, 1:3])
    for dir in 1:3
        ix = [2, 3, 1][dir]
        iy = [3, 1, 2][dir]
        iz = dir

        # The grid of q points making up the lowest side of the histogram
        # along the iz direction
        grid = Vector{Float64}[]
        grid_sparse = Vector{Float64}[]
        for i in 1:length(bes[ix]), j in 1:length(bes[iy])
            is_i_edge = (i == 1 || i == length(bes[ix]))
            is_j_edge = (j == 1 || j == length(bes[iy]))
            grid_point = [bes[ix][i], bes[iy][j], bes[iz][1]][invperm([ix, iy, iz])]
            if is_i_edge && is_j_edge # Corner case; render special for outline
                push!(grid_sparse, grid_point)
                continue
            end
            push!(grid,grid_point)
        end
        offset = [0, 0, bes[iz][end] - bes[iz][1]][invperm([ix, iy, iz])]

        if !isempty(grid)
            segs = map(x -> (Makie.Point3f(M * x), Makie.Point3f(M * (x .+ offset))), grid[:])
            Makie.linesegments!(ax, segs, color=bin_colors[dir], linewidth=bin_line_width, alpha=line_alpha)
        end

        segs = map(x -> (Makie.Point3f(M * x), Makie.Point3f(M * (x .+ offset))), grid_sparse[:])
        Makie.linesegments!(ax, segs; color=isnothing(color) ? :black : color, linewidth=2.5, colorrange)
    end
end

end
