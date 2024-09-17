# Luttinger-Tisza
"""
The following three functions are common to standard LT and extended LT. Firstly we simply build a matrix, J which gives the FT 
of the exchange interactions. Then we find the Q that minimizes the minimum eigenvalue.
"""
function Jmatrix(sys::System, k̃ :: Vector{Float64})
    Nm= length(sys.dipoles)
    Jij_heisen=zeros(ComplexF64,Nm,Nm)
    Jmat_exchange=zeros(ComplexF64,3*Nm,3*Nm)
    # DANGER - this can give wrong answers if dim of the original sys is not (1,1,1)
    # SWT throws away this information so we need a way to know this
    # commented out the check on k - I don't think this is needed.
    #for k̃ᵢ in k̃
    #    (k̃ᵢ < 0.0 || k̃ᵢ ≥ 1.0) &&  throw("k̃ outside [0, 1) range") 
    #end
    # pairexchange interactions
    for matom = 1:Nm
        ints = sys.interactions_union[matom]
        # Heisenberg exchange
        for (; isculled, bond, J) in ints.heisen
            #isculled && break
            sub_i, sub_j, ΔRδ = bond.i, bond.j, bond.n
            phase  = exp(2im * π * dot(k̃, ΔRδ))
            cphase = conj(phase)
            Jij_heisen[sub_i, sub_j] += J * phase 
        end

        for (; isculled, bond, J) in ints.exchange
            #isculled && break
            sub_i, sub_j, ΔRδ = bond.i, bond.j, bond.n
            phase  = exp(2im * π * dot(k̃, ΔRδ))
            cphase = conj(phase)
            start_i = 3*(sub_i-1)+1
            end_i = start_i+2
            start_j = 3*(sub_j-1)+1
            end_j = start_j+2
            Jmat_exchange[start_i:end_i, start_j:end_j] +=  J * phase
        end

    end
    Jmat_heisen= kron(Jij_heisen, I(3))
    Jmat = Jmat_exchange + Jmat_heisen
    Jmat = 0.5 * (Jmat + Jmat')
end

function lambda_min(sys::System,Q :: Vector{Float64})
    Jmat=Jmatrix(sys, Q)
    val,vec = eigen(Jmat)
    val=real(val)
    sorted_indices = sortperm(val)
    sorted_eigenvalues, = val[sorted_indices]
    sorted_eigenvectors = vec[:, sorted_indices]
    return sorted_eigenvalues[1], sorted_eigenvectors[:,1]
end

function Min_Q(sys::System,no_seeds) 
    lower_bound = [0.0, 0.0, 0.0]
    upper_bound = [1.0, 1.0, 1.0]
    optimal_qs = []
    energies = []
    eigenvectors = []
    objective(q) = real(lambda_min(sys, q)[1])
    for i in 1:no_seeds
        initial_q = rand(3)
        result = optimize(objective, lower_bound, upper_bound, initial_q, Fminbox(BFGS()))
        precision = 6 # how many digits to resolve Q
        optimal_q = round.(result.minimizer,digits=precision)
        if optimal_q ∉ optimal_qs
            energy, vector = lambda_min(sys, optimal_q)
            push!(optimal_qs, optimal_q)
            push!(energies, energy)
            push!(eigenvectors, vector)
        end
    end
    return optimal_qs, energies, eigenvectors
end

function suggest_initial_parameters(sys::System,no_seeds)
    wavevectors, energies, eigenvectors = Sunny.Min_Q(sys,no_seeds)
    energies = real(energies) # real deals with +0.0im
    sorted_indices = sortperm(energies)
    energies_sorted = energies[sorted_indices]
    wavevectors_sorted = wavevectors[sorted_indices]
    eigenvectors_sorted = eigenvectors[sorted_indices]
    degeneracy = count_closest_values(energies_sorted,1e-6)
    kept_wavevectors= wavevectors_sorted[1:degeneracy]
    return kept_wavevectors, eigenvectors_sorted
end


"""
These are specific to extended LT. The BuildTMatrix fuction builds a discrete grid of the Fourier components in momentum space
given n primary Q vectors and L harmonics. We start with some initial guesses for the primary Q vector weights. In practice
these will usually be the standard LT eigenvectors.
"""

function build_empty_T_matrix(Primary_Qs,L::Int64)
    L̃ = 2*L+1
    dims = tuple([L̃ for _ in 1:length(Primary_Qs)]...)  # Tuple of dimensions (size along x, y, z dimensions)
    dims = tuple(dims...,3)
    offsets = tuple([-(L+1) for _ in 1:length(Primary_Qs)]...)
    offsets = tuple(offsets...,0)
    T = OffsetArray(fill(0.0im, dims), offsets)
    return  T
end

function AandB(T,Primary_Qs,L)
    Tn=fft_vectors(collect(T),Primary_Qs,L)
    Tn_squared = Tn .* conj(Tn)
    L̃ =2 * L +1
    D = length(Primary_Qs)
    Ñ = L̃^D
    A = (1/Ñ)*sum(sum(Tn_squared, dims=length(size(Tn_squared))))
    B = (1/Ñ)*sum(sum(Tn_squared, dims=length(size(Tn_squared))).^2)
    return A, B
end

function k_array(Primary_Qs, L)
    L̃ = 2 * L + 1
    dims = tuple(fill(L̃, length(Primary_Qs))...) 
    offsets = tuple(fill(-(L+1), length(Primary_Qs))...)
    array = OffsetArray(fill([0.0,0.0,0.0], dims), offsets)
    for (indices, value) in zip(CartesianIndices(array), array)
        index_tuple = Tuple(indices)
        sum_value = sum(index_tuple[i] * Primary_Qs[i] for i in 1:length(Primary_Qs))
        array[indices] = sum_value
    end
    return array
end

function J_array(sys, k_array,L,Primary_Qs)
    dims = size(k_array)
    offsets = tuple([-(L+1) for _ in 1:length(Primary_Qs)]...)
    result_array = OffsetArray(fill(zeros(ComplexF64, 3, 3), dims...), offsets)
    result_array = [Jmatrix(sys, k_array[Tuple(i)...]) for i in CartesianIndices(k_array)]
    return unfold_dimensions(result_array,L,Primary_Qs)
end

function classical_energy_fourier(T,Primary_Qs,L,J_input)
    # approach is to reshape to an L^D x 3 x 3 array times an L^D x 3 array
    D=length(Primary_Qs)
    L̃=2*L+1
    S=reshape(collect(T),(L̃^D,3))
    S_prime = conj(S)
    J_reshape = reshape(collect(J_input),(L̃^D,3,3))
    C=zeros(ComplexF64,size(S))
    F=0.0
    for i = 1:size(J_reshape, 1), j = 1:size(J_reshape, 2), k = 1:size(J_reshape,3)
        C[i,k] += J_reshape[i, j,k] * S_prime[i,j]
    end
    for i = 1:size(S, 1), j = 1:size(S, 2)
        F += S[i, j] * C[i,j]
    end
    return F # times by 1/2?
end

function update_T_matrix!(T::OffsetArray,NewVectors) # change to a function of vectors not dict?
    for (key, vec) in NewVectors
        T[key...,:] = vec/sqrt(2)
        T[-1 .*key...,:] = conj(vec)/sqrt(2)
    end
    return T
end

function ℒ(sys::System,Primary_Qs,L::Int,T,α::Float64,Js)
    A, B = AandB(T,Primary_Qs,L)
    out = classical_energy_fourier(T,Primary_Qs,L,Js)+abs(α*(sum(sys.κs.^4)-2*sum(sys.κs.^2)*A+B)) # check for multi-site!
    imag(out) > 1e-5  && throw("Imaginary Energy")
    out=real(out)
    return out
end


function optimize_fourier_components(sys::System,Primary_Qs,L::Int64,initial_guess,α;time_limit=60.0 )
    T=Sunny.build_empty_T_matrix(Primary_Qs,L)
    unique_els = Sunny.get_unique_elements(T,L)
    empty_list = fill(zeros(ComplexF64,length(initial_guess[1])),length(unique_els))
    init_dict = Dict(zip(unique_els,empty_list))
    dims = Tuple([length(unique_els)])
    ks = k_array(Primary_Qs,L)
    Js = J_array(sys, ks,L,Primary_Qs)
    for q=1:length(Primary_Qs)
        address= zeros(Int64,length(Primary_Qs))
        address[q]=1
        init_dict[Tuple(address)]=initial_guess[q]
    end
    for i in eachindex(unique_els)
        if haskey(init_dict, unique_els[i])
            empty_list[i] = init_dict[unique_els[i]]
        end
    end
    function objective_function(fourier_components)
        fourier_components=Sunny.refold_vector_array(fourier_components,dims)
        dict = Dict(zip(unique_els,fourier_components))
        Sunny.update_T_matrix!(T, dict)
        cost = Sunny.ℒ(sys,Primary_Qs,L,T,α,Js)
        return cost
        end
    x_tol = 1e-4
    f_tol = 1e-4
    initial_guess_unfolded = Sunny.unfold_vector_array(empty_list)
    result = optimize(objective_function, initial_guess_unfolded, BFGS(),Options(time_limit=time_limit, x_tol = x_tol, f_tol = f_tol))
    vector_out=refold_vector_array(minimizer(result),dims)
    out = Array{Any, 2}(undef, length(unique_els), 3)
    for i in 1:length(unique_els)
        out[i, 1] = unique_els[i]
        out[i, 2] = vector_out[i]/sqrt(2)
        out[i, 3] = norm(vector_out[i])
    end
    out=sortslices(out, dims = 1, by=x->x[3],rev=true)
    Aout, Bout = AandB(T,Primary_Qs,L)
    return result, out, minimum(result), Aout, Bout, T
end


###################################################
# helper functions
function unfold_vector_array(matrix)
    reshaped_matrix=reshape(matrix,(1,length(matrix)))
    return vcat(reshaped_matrix...)
end

function refold_vector_array(list,dims)
    num_groups = length(list) ÷ 3
    vector_list = Vector{Vector{ComplexF64}}(undef, num_groups)
    for i in 1:num_groups
        vector_list[i] = list[(i - 1) * 3 + 1 : i * 3]
    end
    return reshape(vector_list,dims)
end

function more_negatives_than_positives(tpl::Tuple) #helper
    count_negatives = sum(x -> x < 0, tpl)
    count_positives = sum(x -> x > 0, tpl)
    return count_negatives > count_positives
end

function add_tuple_and_negated!(tuple_set::Set{Tuple}, tuple::Tuple) #helper
    negated_tuple = tuple .* -1
    push!(tuple_set, tuple)
    push!(tuple_set, negated_tuple)
end

function get_unique_elements(array::OffsetArray,L) #helper
    output = OffsetArray(fill(tuple(zeros(Int,length(size(array))-1)...),size(array)), array.offsets)
    for (indices, value) in zip(CartesianIndices(array), array)
        index_tuple = Tuple(indices)
        output[indices] = index_tuple[1:end-1]
    end
    complete_list=reshape(output,(length(output)))
    negs_removed=filter(tpl -> !more_negatives_than_positives(tpl),complete_list) # still need to deal with (1,0,-1) vs (-1,0,1) etc
    tuple_set = Set{Tuple}()
    filtered_tuples = Tuple[]
    for tuple in negs_removed
        if !(tuple in tuple_set)
            push!(filtered_tuples, tuple)
            add_tuple_and_negated!(tuple_set, tuple)
        end
    end
    filtered_tuples_no_pad = filter_tuples_by_magnitude(filtered_tuples, L)
    return filtered_tuples_no_pad
end

function filter_tuples_by_magnitude(tuples_list, L)
    return filter(tuple -> all(abs.(x) <= L/2 for x in tuple), tuples_list)
end

function unfold_dimensions(A,L,Primary_Qs) #helper
    dimens= size(A)
    element_size=size(A[1])
    new_dims = tuple(dimens...,element_size...)
    offsets = tuple([-(L+1) for _ in 1:length(Primary_Qs)]...)
    new_offsets = tuple(offsets...,0,0)
    new_array = OffsetArray(zeros(ComplexF64,new_dims),new_offsets)
    B=collect(A)
    for j in 1:length(A)
        new_array[indices_jth_element(A,j)...,1,1] = B[j][1]
        new_array[indices_jth_element(A,j)...,2,1] = B[j][2]
        new_array[indices_jth_element(A,j)...,3,1] = B[j][3]
        new_array[indices_jth_element(A,j)...,1,2] = B[j][4]
        new_array[indices_jth_element(A,j)...,2,2] = B[j][5]
        new_array[indices_jth_element(A,j)...,3,2] = B[j][6]
        new_array[indices_jth_element(A,j)...,1,3] = B[j][7]
        new_array[indices_jth_element(A,j)...,2,3] = B[j][8]
        new_array[indices_jth_element(A,j)...,3,3] = B[j][9]
    end
    return new_array
end

function indices_jth_element(A, j) #helper
    # if j < 1 || j > length(A)
        # throw(ArgumentError("Invalid value of j. j must be within the range of the array."))
    # end
    all_indices = CartesianIndices(A)
    indices_tuple = tuple(all_indices...)[j]
    return indices_tuple.I
end

function fft_vectors(arr,Primary_Qs,L) #helper
    L̃ = 2*L +1
    arr = collect(arr)
    dims = tuple(fill(:,length(Primary_Qs))...)
    out = zeros(ComplexF64,size(arr))
    for d in 1:3
        out[dims...,d]=FFTW.fft(arr[dims...,d])
    end
    return out
end

function get_elements(arr,list_of_indices)
    elements_from_list = []
    for indices in list_of_indices
        element = arr[indices]
        push!(elements_from_list, element)
    end
end

function count_closest_values(values,tol)
    smallest_value = minimum(values)
    tolerance = tol
    count_closest = count(x -> abs(x - smallest_value) < tolerance, values)
    return count_closest
end

function compare_vectors(key1, key2, dict)
    norm1 = norm(dict[key1])
    norm2 = norm(dict[key2])
    if norm1 < norm2
        return -1
    elseif norm1 > norm2
        return 1
    else
        return 0
    end
end


#########################################################
# old versions

function optimize_fourier_components_legacy(sys::System,Primary_Qs,L::Int64,initial_guess,α)
    T=Sunny.build_empty_T_matrix(Primary_Qs,L)
    unique_els = Sunny.get_unique_elements(T,L)
    initial_guess_matrix = fill(zeros(ComplexF64,length(initial_guess[1])),length(unique_els))
    dims = size(initial_guess_matrix)
    for q in 1:length(Primary_Qs)
        index=zeros(length(Primary_Qs))
        index[q]=1
        place_in_list = findall(x -> x == tuple(index...),unique_els)
        initial_guess_matrix[place_in_list[1]] = initial_guess[q]
    end
    function objective_function(fourier_components)
        #fourier_components=Sunny.refold_vector_array(fourier_components,dims)
        dict = Dict(zip(unique_els,fourier_components))
        Sunny.update_T_matrix!(T, dict)
        cost = Sunny.ℒ(sys,Primary_Qs,L,T,α)
        return cost
        end
    initial_guess_unfolded = Sunny.unfold_vector_array(initial_guess_matrix)
    result = optimize(objective_function, initial_guess_unfolded, BFGS())
    return refold_vector_array(minimizer(result),dims), minimum(result)
end


function ℒ_legacy(sys::System,Primary_Qs,L::Int,T,α::Float64)
    ks = k_array(Primary_Qs,L)
    Js = J_array(sys, ks,L,Primary_Qs)
    out = classical_energy_fourier(T,Primary_Qs,L,Js)+abs(α*(B(T,Primary_Qs,L)-sum(sys.κs.^4))) # check for multi-site!
    imag(out) > 1e-7  && throw("Imaginary Energy")
    out=real(out)
    return out
end


function optimize_fourier_components_legacy2(sys::System,Primary_Qs,L::Int64,initial_guess,α;iteration_limit=5, obj_call_limit=50 )
    T=Sunny.build_empty_T_matrix(Primary_Qs,L)
    unique_els = Sunny.get_unique_elements(T,L)
    empty_list = fill(zeros(ComplexF64,length(initial_guess[1])),length(unique_els))
    init_dict = Dict(zip(unique_els,empty_list))
    dims = Tuple([length(unique_els)])
    for q=1:length(Primary_Qs)
        address= zeros(Int64,length(Primary_Qs))
        address[q]=1
        init_dict[Tuple(address)]=initial_guess[q]
    end
    for i in eachindex(unique_els)
        if haskey(init_dict, unique_els[i])
            empty_list[i] = init_dict[unique_els[i]]
        end
    end
    function objective_function(fourier_components)
        fourier_components=Sunny.refold_vector_array(fourier_components,dims)
        dict = Dict(zip(unique_els,fourier_components))
        Sunny.update_T_matrix!(T, dict)
        cost = Sunny.ℒ(sys,Primary_Qs,L,T,α)
        return cost
        end
    x_tol = 1e-5
    f_tol = 1e-5
    initial_guess_unfolded = Sunny.unfold_vector_array(empty_list)
    result = optimize(objective_function, initial_guess_unfolded, BFGS(),Options(iterations = iteration_limit, f_calls_limit = obj_call_limit, x_tol = x_tol, f_tol = f_tol))
    vector_out=refold_vector_array(minimizer(result),dims)
    dict_out=Dict(zip(unique_els,vector_out))
    return result, dict_out, minimum(result)
end

function suggest_initial_parameters_legacy(sys::System)
    wavevectors, energies, eigenvectors = Sunny.Min_Q(sys)
    energies = real(energies) # real deals with +0.0im
    sorted_indices = sortperm(energies)
    energies_sorted = energies[sorted_indices]
    wavevectors_sorted = wavevectors[sorted_indices]
    eigenvectors_sorted = eigenvectors[sorted_indices]
    degeneracy = count_closest_values(energies_sorted,1e-6)
    kept_wavevectors= wavevectors_sorted[1:degeneracy]
    return kept_wavevectors, eigenvectors_sorted
end

#########################################
#########################################
## Has a problem if multiple minima ####
function lambda_min_legacy(sys::System,Q :: Vector{Float64},mode)
    Jmat=Jmatrix(sys, Q)
    val,vec = eigen(Jmat)
    value=val[mode]
    vector=vec[:,mode]
    return value, vector
end

function Min_Q_legacy(sys::System) 
    lower_bound = [0.0, 0.0, 0.0]
    upper_bound = [1.0, 1.0, 1.0]
    optimal_qs=[]
    energies = []
    eigenvectors = []
    for mode = 1:length(sys.κs)*3
        objective(q) = real(lambda_min(sys,q,mode)[1])
        initial_q = rand(3)
        result = optimize(objective, lower_bound, upper_bound, initial_q, Fminbox(BFGS()))
        optimal_q = result.minimizer
        energy, vector= lambda_min(sys,optimal_q,mode)
        push!(optimal_qs,optimal_q)
        push!(energies,energy)
        push!(eigenvectors,vector)
    end
    return optimal_qs, energies, eigenvectors
end
