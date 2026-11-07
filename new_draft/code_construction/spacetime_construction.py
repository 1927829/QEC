
import numpy as np
import stim
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csc_matrix

print("Importing checks ...")

import matrix_checks

X_checks = matrix_checks.X_checks
Z_checks = matrix_checks.Z_checks

print("Done.")

p = 0.001
round_nums = 10

n = X_checks.shape[1]
num_X = X_checks.shape[0]
num_Z = Z_checks.shape[0]
checks_per_round = num_X+num_Z

print("Initializing Circuit ...")

circuit = stim.Circuit()
circuit.append("R",list(range(n+num_X+num_Z)))


for r in range(round_nums+1):
    # first round, last round is noiseless
    # first round is intialization
    noisy = 0 < r < round_nums
    error_rate = p if noisy else 0

    for i in range(num_X):
        ancilla = n+i
        circuit.append("H",[ancilla])
        if noisy:
            circuit.append("DEPOLARIZE1",[ancilla],p)
        for qubit in np.flatnonzero(X_checks[i]):
            qubit = int(qubit)
            circuit.append("CX",[ancilla,qubit])
            if noisy:
                circuit.append("DEPOLARIZE2",[ancilla,qubit],p)

        circuit.append("H",[ancilla])
        if noisy:
            circuit.append("DEPOLARIZE1",[ancilla],p)

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

print("Done.")

print("Creating tanner graph ...")

dem = circuit.detector_error_model(decompose_errors=False).flattened()
# error(p) detector x, detector y, detector z, ... flipped

rows = [] #what detector was flipped from error x
cols = [] #which error is it
error_probs = [] #probability for this fault to occur
#ith index for rows and cols gives error(cols) and flipped (row) and prob flipped (errors_prob)

for instruction in dem:
    if not isinstance(instruction,stim.DemInstruction):
        print("This should not happen: instruction")
        continue
    if instruction.type=="error":
        column = len(error_probs)
        error_probs.append(instruction.args_copy()[0])

        for target in instruction.targets_copy():
            if isinstance(target,stim.DemTarget):
                if target.is_relative_detector_id():
                    rows.append(target.val)
                    cols.append(column)
            else:
                print("This should not happen: target")

#len rows -> how many flipped in total

spacetime_graph = csc_matrix((np.ones(len(rows),dtype=np.uint8),(rows,cols)),
    shape=(dem.num_detectors,len(error_probs))
)

error_probs = np.asarray(error_probs)

print("Done.")
print("Saving ...")

save_npz('QEC/spacetime_graph.npz', spacetime_graph)
np.save('QEC/error_prob.npy', error_probs)

print("Done.")