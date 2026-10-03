import class_init
import basic_phase_rotations
import logical_basis
import state_cultivation

q = class_init.LDPC()
q.h_gate(0)
q.cx_gate(0,1)
q.logical_ppms([["Z",0],["Z",1]])
results = q.get_shots(10000)

xxxx_results = [shot[-1] for shot in results]
xxxx_expectation = sum(1 if x == 0 else -1 for x in xxxx_results) / len(results)

print("<> =", xxxx_expectation)