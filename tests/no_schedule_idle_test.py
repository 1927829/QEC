
import numpy as np
import stim
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csc_matrix
from bposd.css import css_code

print("Importing checks ...")

import matrix_checks as matrix_checks

X_checks = matrix_checks.X_checks
Z_checks = matrix_checks.Z_checks

print("Done.")

p = 0.0068 
round_nums = 12

n = X_checks.shape[1]
num_X = X_checks.shape[0]
num_Z = Z_checks.shape[0]
checks_per_round = num_X+num_Z

print("Initializing Circuit ...")

code = css_code(hx=X_checks,hz=Z_checks)

L_X = np.asarray(csc_matrix(code.lx).toarray(),dtype=np.uint8)
L_Z = np.asarray(csc_matrix(code.lz).toarray(),dtype=np.uint8)

#logical for logical matrix, L* residual = 0 if correct logical class

#NOTE: L_x L_z do not necessarily satisfy pauli commutivity

k = L_X.shape[0] #number of logical qubits
assert k > 0
assert L_Z.shape[0]==k

circuit = stim.Circuit()
circuit.append("R",list(range(n+n+num_X+num_Z)))
#we add ANOTHER n simply because we want to track the final propagated pauli error
#so we construct L_x, L_z

for qubit in range(n):
    circuit.append("H",[qubit])
    circuit.append("CX",[qubit,n+num_X+num_Z+qubit])
#we create bell pairs between data and residual
reference_start = n+num_X+num_Z #distance between residual and data



for r in range(round_nums+1):
    # first round, last round is noiseless
    # first round is intialization
    noisy = 0 < r < round_nums-1
    error_rate = p if noisy else 0

    for i in range(num_X):
        ancilla = n+i
        circuit.append("H",[ancilla])
        if noisy:
            circuit.append("DEPOLARIZE1",[ancilla],p/10)
        for qubit in np.flatnonzero(X_checks[i]):
            qubit = int(qubit)
            circuit.append("CX",[ancilla,qubit])
            if noisy:
                circuit.append("DEPOLARIZE2",[ancilla,qubit],p)

        circuit.append("H",[ancilla])
        if noisy:
            circuit.append("DEPOLARIZE1",[ancilla],p/10)

        circuit.append("MR",[ancilla],error_rate)

    for i in range(num_Z):
        ancilla = n+num_X+i

        for qubit in np.flatnonzero(Z_checks[i]):
            qubit = int(qubit)
            circuit.append("CX",[qubit,ancilla])

            if noisy:
                circuit.append("DEPOLARIZE2",[qubit,ancilla],p)

        circuit.append("MR",[ancilla],error_rate)

    if r > 0:
        for i in range(checks_per_round):
            current = stim.target_rec(-checks_per_round+i)
            previous = stim.target_rec(-2*checks_per_round+i)
            circuit.append("DETECTOR",[current,previous])

for sector, logicals in enumerate((L_X,L_Z)):
    for i in range(k):
        targets = []

        for qubit in np.flatnonzero(logicals[i]):
            qubit = int(qubit)

            for q in (qubit,reference_start+qubit):
                if len(targets) > 0:
                    targets.append(stim.target_combiner())

                if sector==0:
                    targets.append(stim.target_x(q))
                else:
                    targets.append(stim.target_z(q))

        circuit.append("MPP",targets)
        circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-1)],sector*k+i)


print("Done.")
print("Creating Tanner graph and logical matrix ...")

dem = circuit.detector_error_model(
    decompose_errors=False
).flattened()

# error(p) detector x, detector y, detector z, ... flipped

rows = [] #what detector was flipped from error x
cols = [] #which error is it
error_probs = [] #probability for this fault to occur
#ith index for rows and cols gives error(cols) and flipped (row) and prob flipped (errors_prob)

logical_rows = [] #same thing
logical_cols = []

for instruction in dem:
    if not isinstance(instruction,stim.DemInstruction):
        continue

    if instruction.type=="error":
        column = len(error_probs)
        error_probs.append(instruction.args_copy()[0])

        for target in instruction.targets_copy():
            if not isinstance(target,stim.DemTarget):
                continue

            if target.is_relative_detector_id():
                rows.append(target.val)
                cols.append(column)

            elif target.is_logical_observable_id():
                logical_rows.append(target.val)
                logical_cols.append(column)

spacetime_graph = csc_matrix(
    (
        np.ones(len(rows),dtype=np.uint8),
        (rows,cols)
    ),
    shape=(dem.num_detectors,len(error_probs))
)

logical_matrix = csc_matrix(
    (
        np.ones(len(logical_rows),dtype=np.uint8),
        (logical_rows,logical_cols)
    ),
    shape=(2*k,len(error_probs))
)

error_probs = np.asarray(error_probs)

print("Tanner graph:",spacetime_graph.shape)
print("Logical matrix:",logical_matrix.shape)

print("Saving ...")

save_npz("spacetime_graph.npz",spacetime_graph)
save_npz("logical_matrix.npz",logical_matrix)
np.save("error_probs.npy",error_probs)