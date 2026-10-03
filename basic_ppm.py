import stim
import state_cultivation
import logical_basis
import history_tracking

def measure_ppm_error_propogation(circuit: stim.Circuit, history: list, ppms: list[list] , measurement_num: int, ancilla: list[int],ancilla_in_y_state=False):
    '''ppms: list of first PAULI, then which PHYSICAL bit (no list). Only supports X and Z paulis.
    Ancilla should be one bit, list.
    Measurement_num should increase either before or after running
    Method: Basic, allows propogation
    '''
    if ancilla_in_y_state:
        state_cultivation.basic_y(circuit,history,ancilla)

    for measures in ppms:
        if measures[0] == "X":
            history_tracking.add_history(circuit, history, "physical_gate", ["H",[measures[1]]])

        history_tracking.add_history(circuit, history, "physical_gate", ["CX",[measures[1],ancilla[0]]])
        if measures[0] == "X":
            history_tracking.add_history(circuit, history, "physical_gate", ["H",[measures[1]]])
        
    history_tracking.add_history(circuit,history,"measure+reset",["MR",ancilla,measurement_num])
