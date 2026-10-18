import stim
import logical_basis
import basic_ppm
import history_tracking
def basic_phase_rotation(circuit: stim.Circuit, history: list, ppms: list[list], ancilla: list[int], measurement_num: int, angle=True):
    '''angle: true = pi/2, false = -pi/2
    
    Repeated only once

    Add measuremen_num after completed
    ppms: list, each eement is [pauli, physical_bit]
    
    '''
    basic_ppm.measure_ppm_error_propogation(circuit,history,ppms,measurement_num,ancilla,True)
    if angle:
        history_tracking.add_history(circuit,history,"controlled_pauli",[ppms,[measurement_num],Indicator_true])
    else:
        history_tracking.add_history(circuit,history,"controlled_pauli",[ppms,[measurement_num],Indicator_false])
def Indicator_true(measurement: int):
    if measurement == 1:
        return True
    return False
def Indicator_false(measurement: int):
    if measurement==1:
        return False
    return True