"""
Cyclic Variational Quantum Eigensolver (CVQE)
=======================================================
Implementations of CVQE algorithms for molecular electronic structure calculations using PennyLane.
Author: Hao Zhang (hao.zhang.quantum@gmail.com)
Date: 202508
"""

from time import time
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import pickle
from CyclicAdamax import CyclicAdamaxUpdater

def gs_uccsd(symbols, geometry, electrons, orbitals, charge, shots=None, max_iter=5, stepsize=2, layer=1):
    # Build the electronic Hamiltonian
    # STO-3G basis set
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, 
                                                active_electrons=electrons, active_orbitals=orbitals)
    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    # Map excitations to the wires the UCCSD circuit will act on
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    wires=range(qubits)
    # depth = 1
    params = np.zeros((layer,len(singles) + len(doubles)))
    # Define the device https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html
    dev_exact = qml.device("lightning.qubit", wires=qubits, shots=None)
    dev = qml.device("lightning.qubit", wires=qubits, shots=shots)


    @qml.qnode(dev_exact, interface="autograd", diff_method="adjoint")
    def circuit_exact(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state,n_repeats=layer)
        return qml.expval(H)

    @qml.qnode(dev, interface="autograd")
    def circuit_shots(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state,n_repeats=layer)
        return qml.expval(H)

    optimizer = qml.GradientDescentOptimizer(stepsize=stepsize)
    energies = [] # record exact energies
    for n in range(max_iter):
        t1 = time()
        if shots==None:
            params, energy = optimizer.step_and_cost(circuit_exact, params, wires=wires,
                                                 s_wires=s_wires, d_wires=d_wires,
                                                 hf_state=hf_state)
        else:
            params, energy = optimizer.step_and_cost(circuit_shots, params, wires=wires,
                                                 s_wires=s_wires, d_wires=d_wires,
                                                 hf_state=hf_state)
            
            energy = circuit_exact(params, wires=range(qubits),
                                s_wires=s_wires, d_wires=d_wires,
                                hf_state=hf_state)
        energies.append(energy)
        t2 = time()
        print("Iteration", n, "Energy", energy, "Time", round(float(t2 - t1),2), "s")

    return params, energies


def probs_to_coeffs_bases(probs, tol=1e-10):
    n = int(np.log2(len(probs)))                    # number of qubits
    nz_idx = np.where(probs > tol)[0]              # non-zero probability indices
    coeffs = np.sqrt(probs[nz_idx])                # amplitude = √probability
    bases = ((nz_idx[:, None] & (1 << np.arange(n)[::-1])) > 0).astype(int)
    coeffs = coeffs / np.linalg.norm(coeffs)
    sorted_indices = np.argsort(-coeffs)  # negative sign means descending order!
    coeffs = coeffs[sorted_indices]
    bases = bases[sorted_indices]
    return coeffs, bases

def params_to_state(params, coeffs, bases, shots=None, layer=1, electrons=None):
    qubits = len(bases[0])
    if electrons is None:
        electrons = qubits // 2
    dev = qml.device("lightning.qubit", wires=qubits+1, shots=shots)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    @qml.qnode(dev, interface="autograd", diff_method="adjoint" if dev.shots is None else "best")
    def circuit_state(params, coeffs, bases):
        qml.Superposition(coeffs=coeffs, bases=bases, wires=range(qubits), work_wire=qubits)
        for l in range(layer):
            for i, (w1, w2) in enumerate(d_wires):
                qml.FermionicDoubleExcitation(params[l*(len(s_wires) + len(d_wires)) + len(s_wires) + i], wires1=w1, wires2=w2)
            for j, s_wires_ in enumerate(s_wires):
                qml.FermionicSingleExcitation(params[l*(len(s_wires) + len(d_wires)) + j], wires=s_wires_)
        return qml.probs(wires=range(qubits)), qml.state()
    probs, state = circuit_state(params, coeffs, bases)
    return probs_to_coeffs_bases(probs, tol=1e-10), state

def print_bases_diff(set_curr, set_new):
    added = sorted(set_new - set_curr)
    removed = sorted(set_curr - set_new)

    def convert_to_numpy_matrix(tensor_set):
        return onp.array([[int(bit.item()) for bit in base] for base in tensor_set])

    if added:
        print("🆕 Bases Added:")
        print(convert_to_numpy_matrix(added))
    else:
        print("✅ No bases added")

    if removed:
        print("🗑️ Bases Removed:")
        print(convert_to_numpy_matrix(removed))
    else:
        print("✅ No bases removed")

def gs_cvqe(symbols, geometry, electrons, charge, orbitals=None,
            method="dhf",
            shots=None, max_iter=200,
            layer=1,
            optimizer = qml.GradientDescentOptimizer(stepsize=2.0),
            lr = 0.3, #learning rate for coeffs
            cad=True,
            cad_lr=0.02,
            cad_restart_period=200,
            dynamic_add_bases=True,
            threshold_add_bases=5,
            tol=1e-10,
            initial_state={},
            initial_params=None,
            initial_coeffs_strength=2,
            random_init=0.1,
            num_bases=None,
            threshold_replace=0.001
            ):

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols, geometry, charge=charge,
        active_electrons=electrons, active_orbitals=orbitals,
        method=method)
    H = qml.simplify(H)
    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    wires = range(qubits)
    if initial_params is None:
        if random_init:
            n_params = (len(singles) + len(doubles))*layer
            params = np.random.uniform(low=-1/np.sqrt(n_params)*random_init, high=1/np.sqrt(n_params)*random_init, size=n_params)
        else:
            params = np.zeros((len(singles) + len(doubles))*layer)
    else:
        params = initial_params

    dev_exact = qml.device("lightning.qubit", wires=qubits+1, shots=None)
    dev = qml.device("lightning.qubit", wires=qubits+1, shots=shots)

    def make_circuit(dev):
        @qml.qnode(dev, interface="autograd", diff_method="best" if dev.shots is None else "best")
        def circuit(params, coeffs, bases):
            qml.Superposition(coeffs=coeffs, bases=bases, wires=wires, work_wire=qubits)
            for l in range(layer):
                for i, (w1, w2) in enumerate(d_wires):
                    qml.FermionicDoubleExcitation(params[l*(len(s_wires) + len(d_wires)) + len(s_wires) + i], wires1=w1, wires2=w2)
                for j, s_wires_ in enumerate(s_wires):
                    qml.FermionicSingleExcitation(params[l*(len(s_wires) + len(d_wires)) + j], wires=s_wires_)
            return qml.expval(H)
        return circuit
    
    @qml.qnode(dev, interface="autograd", diff_method="best" if dev.shots is None else "best")
    def circuit_state(params, coeffs, bases):
        qml.Superposition(coeffs=coeffs, bases=bases, wires=wires, work_wire=qubits)
        for l in range(layer):
            for i, (w1, w2) in enumerate(d_wires):
                qml.FermionicDoubleExcitation(params[l*(len(s_wires) + len(d_wires)) + len(s_wires) + i], wires1=w1, wires2=w2)
            for j, s_wires_ in enumerate(s_wires):
                qml.FermionicSingleExcitation(params[l*(len(s_wires) + len(d_wires)) + j], wires=s_wires_)
        return qml.probs(wires=range(qubits))

    # 🎯 Create exact & shots circuits
    circuit_exact = make_circuit(dev_exact)
    circuit = make_circuit(dev)

    energies = []  # record energies
    history = []  # record history
    grads = []  # record gradients
    coeffs = initial_state.get('coeffs', np.sqrt(np.array([1])))
    bases = initial_state.get('bases', np.array([hf_state]))
    best_energy = np.inf
    best_params = params.copy()

    print(circuit_exact(params, coeffs=coeffs, bases=bases))
    print(f"initial coeffs and bases, {coeffs}, {bases}")

    grad_coeffs_fn = qml.grad(circuit, argnum=1)
    coeffs_updater = CyclicAdamaxUpdater(lr=cad_lr, restart_period=cad_restart_period)
    max_bases = num_bases if num_bases is not None else float('inf')
    removed_bases = set()

    for n in range(max_iter):
        t1 = time()

        # add noise (future version)

        params = optimizer.step(circuit, best_params, coeffs=coeffs, bases=bases)
        magnitude = np.linalg.norm(params-best_params)
        
        grad_coeffs = grad_coeffs_fn(params, coeffs+1e-13, bases)

        if cad:
            coeffs, magnitude_coeffs = coeffs_updater.step(coeffs, grad_coeffs)
        else:
            grad_coeffs = np.clip(np.nan_to_num(grad_coeffs,nan=0),-0.1,0.1)
            coeffs = coeffs - lr*np.nan_to_num(grad_coeffs,nan=0)
            coeffs = coeffs / np.linalg.norm(coeffs)
            magnitude_coeffs = np.linalg.norm(grad_coeffs)

        print(f"magnitude_coeffs = {magnitude_coeffs}")
        print(f"grad_coeffs = {grad_coeffs}")


        eval_exact_every = 5
        if n % eval_exact_every == 0 or n == max_iter - 1:
            energy = circuit_exact(params, coeffs=coeffs, bases=bases)

        if energy < best_energy:
            best_energy = energy

        coeffs_candidate, bases_candidate = probs_to_coeffs_bases(circuit_state(params, coeffs=coeffs, bases=bases), tol=tol)
        if dynamic_add_bases:
            # Convert original bases to tuples for comparison
            # —— (A) Prepare current basis set —— 
            new_bases = []
            new_coeffs = []

            # O(1) # for fast lookup of existing bases
            existing = set()

            for b, c in zip(bases, coeffs):
                t = tuple(b)
                new_bases.append(t)
                new_coeffs.append(c)
                existing.add(t)

            # ——— (B) Only add "new and important" bases, set coefficients to 0 ———
            for c_cand, b_cand in zip(coeffs_candidate, bases_candidate):
                b_t = tuple(b_cand)

                if b_t in existing or b_t in removed_bases:
                    continue

                if len(new_bases) < max_bases:
                    # quota not full, add directly
                    if abs(c_cand)*threshold_add_bases > magnitude:
                        new_bases.append(b_t)
                        new_coeffs.append(
                            (np.random.uniform(low=-1, high=1) * initial_coeffs_strength * magnitude_coeffs)
                            if initial_coeffs_strength > 0 else 0.0
                        )
                        existing.add(b_t)

                elif abs(c_cand) > threshold_replace:
                    # quota full, try to replace the least contributing basis
                    low_idx = [i for i, coeff in enumerate(new_coeffs) if abs(coeff) < threshold_replace]
                    if low_idx:
                        # Find the index with the smallest coefficient among these small ones
                        idx = min(low_idx, key=lambda i: abs(new_coeffs[i]))
                        # Remove the old one from existing and record it in removed_bases
                        old_base_tuple = new_bases[idx]
                        existing.remove(old_base_tuple)
                        removed_bases.add(old_base_tuple)
                        print(f"Replace {old_base_tuple} with {b_t}")

                        # Execute replacement
                        new_bases[idx]  = b_t
                        new_coeffs[idx] = (
                            (np.random.uniform(low=-1, high=1) * initial_coeffs_strength * magnitude_coeffs)
                            if initial_coeffs_strength > 0 else 0.0
                        )
                        # Add the new one to existing
                        existing.add(b_t)
                    # If none are smaller than threshold_replace, do nothing
                
            bases = np.array(new_bases)
            coeffs = np.array(new_coeffs)
            coeffs = coeffs / np.linalg.norm(coeffs)
        else:
            # Indicate logic error
            ValueError("dynamic_add_bases must be True, otherwise bases and coeffs will not be updated.")

        best_params = params.copy()
        energies.append(energy)
        history.append((coeffs, bases, params.copy()))
        grads.append((magnitude,magnitude_coeffs))
        t2 = time()

        print(f"Iteration {n}, current_energy = {energy:.8f}, best_energy = {best_energy:.8f} Ha, Time = {round(float(t2 - t1), 2)} s")
        print(f"params change magnitude: {magnitude}")
        print(f"coeffs = {coeffs}")
        print(f"bases = {bases}")
        print("================================================")

    return params, energies, coeffs, bases, history, grads
