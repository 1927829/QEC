import stim
import history_tracking
def basic_y(circuit: stim.Circuit, history: list, ancilla: list[int]):
    '''ancilla is clear in the beginning'''
    # move to history tracking
    history_tracking.add_history(circuit,history,"physical_gate",["H", ancilla])
    history_tracking.add_history(circuit,history,"physical_gate",["S", ancilla])
def basic_cat_binary(circuit: stim.Circuit, history: list, ancilla: list[int],error=False):
    '''length of ancilla is length of cat. Ancilla is list of bits'''
    length = len(ancilla)
    history_tracking.add_history(circuit, history,"physical_gate",["H", [ancilla[0]]],error)
    for i in range(0,length):
        if((1+i*2)< length):
            history_tracking.add_history(circuit,history,"physical_gate",["CX",[ancilla[i],ancilla[2*i+1]]],error)
        if ((2+i*2)<length):
            history_tracking.add_history(circuit,history,"physical_gate",["CX",[ancilla[i],ancilla[2*i+2]]],error)
def basic_cat(circuit: stim.Circuit, history: list, ancilla: list[int]):
    '''length of ancilla is length of cat. Ancilla is list of bits'''
    length = len(ancilla)
    history_tracking.add_history(circuit, history,"physical_gate",["H", [ancilla[0]]])
    for i in range(0,length-1):
        history_tracking.add_history(circuit,history,"physical_gate",["CX",[ancilla[i],ancilla[i+1]]])
