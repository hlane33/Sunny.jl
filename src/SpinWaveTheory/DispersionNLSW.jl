function excitations_nlsw!(T, tmp1, tmp2, ptt::PerturbativeTheory, q)
    (; swt) = ptt
    L = nbands(swt)
    size(T) == size(tmp1) == size(tmp2) == (2L, 2L) || error("Arguments T and tmp must be $(2L)×$(2L) matrices")

    q_reshaped = to_reshaped_rlu(swt.sys, q)
    dynamical_matrix!(tmp1, swt, q_reshaped)
    swt_hamiltonian_dipole_nlsw!(tmp2, ptt, q_reshaped)
    @. tmp1 += tmp2
    try
        return bogoliubov!(T, tmp1)
    catch _
        error("Instability at wavevector q = $q")
    end
end

function excitations_nlsw(ptt::PerturbativeTheory, q)
    @assert ptt.swt.sys.mode == :dipole ":SUN mode not supported yet"
    L = nbands(ptt.swt)
    T = zeros(ComplexF64, 2L, 2L)
    H_lsw  = zeros(ComplexF64, 2L, 2L)
    H_nlsw = zeros(ComplexF64, 2L, 2L)
    energies = excitations_nlsw!(T, copy(H_lsw), copy(H_nlsw), ptt, q)
    return (energies, T)
end

function dispersion_nlsw(ptt::PerturbativeTheory, qpts)
    L = nbands(ptt.swt)
    qpts = convert(AbstractQPoints, qpts)
    disp = zeros(L, length(qpts.qs))
    for (iq, q) in enumerate(qpts.qs)
        view(disp, :, iq) .= view(excitations_nlsw(ptt, q)[1], 1:L)
    end
    return reshape(disp, L, size(qpts.qs)...)
end
