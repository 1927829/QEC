import stim
import logical_basis
import history_tracking
import history_decoder
import basic_ppm
import basic_phase_rotations
class LDPC:
    def __init__(self):
        '''Gross code'''
        self.num_qubits = 144
        self.basis_x = logical_basis.basis_x
        self.basis_z = logical_basis.basis_z
        self.measurement_ancilla = logical_basis.measurement_ancilla
        self.circuit = stim.Circuit()
        self.pauli_frame = ["I"]*(self.num_qubits + 144)
        self.measurement_num = 0 # used for pauli frame tracking, increased per measurement
        self.history = [] # tracking pauli frames

    def _apply_pauli_gate(self, gate,basis):
        history_tracking.add_history(self.circuit,self.history, "pauli", [gate, basis])
    def x_gate(self, logical_bit: int):
        '''logical basis'''
        self._apply_pauli_gate("X", logical_basis.basis_x[logical_bit])
    def z_gate(self, logical_bit: int):
        '''logical basis'''
        self._apply_pauli_gate("Z", logical_basis.basis_z[logical_bit])
    def y_gate(self, logical_bit: int):
        '''logical basis'''
        self.x_gate(logical_bit)
        self.z_gate(logical_bit)
    def _basic_ppm(self, ppms: list[list], use_y=False):
        '''no list'''
        basic_ppm.measure_ppm_error_propogation(self.circuit, self.history, ppms,self.measurement_num,self.measurement_ancilla,use_y)
        self.measurement_num+=1;

    def get_shots(self, shot_num: int):
        self.sampler = self.circuit.compile_sampler()
        answers= self.sampler.sample(shots = shot_num)
        final_answers=[]
        for measures in answers:
            final_answers.append(history_decoder.history_decode(self.history,self.pauli_frame,measures))
            self.pauli_frame = ["I"]*(self.num_qubits + 144)
        return final_answers
    def s_gate(self, logical_bit: int,angle=True):
        ppms=[]
        for things in self.basis_z[logical_bit]:
            ppms.append(["Z",things])
        basic_phase_rotations.basic_phase_rotation(self.circuit,self.history,ppms,self.measurement_ancilla,self.measurement_num,angle)
        self.measurement_num+=1;
    def sqx_gate(self, logical_bit: int,angle=True):
        ppms=[]
        for things in self.basis_x[logical_bit]:
            ppms.append(["X",things])
        basic_phase_rotations.basic_phase_rotation(self.circuit,self.history,ppms,self.measurement_ancilla,self.measurement_num,angle)
        self.measurement_num+=1;
    def h_gate(self, logical_bit: int):
        self.s_gate(logical_bit)
        self.sqx_gate(logical_bit)
        self.s_gate(logical_bit)
    def cx_gate(self, logical_control,logical_target):
        self.s_gate(logical_control,False)
        self.sqx_gate(logical_target,False)
        ppms=[]
        for things in self.basis_z[logical_control]:
            ppms.append(["Z",things])
        for things in self.basis_x[logical_target]:
            ppms.append(["X",things])
        basic_phase_rotations.basic_phase_rotation(self.circuit,self.history,ppms,self.measurement_ancilla,self.measurement_num,True)
        self.measurement_num+=1
    def logical_ppms(self, logical_bits: list[list]):
        '''EX: [["Z", 1], ["X",2]]'''
        ppms=[]
        for bit in logical_bits:
            if(bit[0]=="Z"):
                for physical in self.basis_z[bit[1]]:
                    ppms.append(["Z",physical])
            if(bit[0]=="X"):
                for physical in self.basis_x[bit[1]]:
                    ppms.append(["X",physical])
        self._basic_ppm(ppms)