import numpy as np

class Measure_Circuit:
    def __init__(self,num_qubits: int):
        self.qubits = num_qubits
        self.index=0
        self.measurement_num=0
        self.error_locations=[] #pauli
        self.measurement_locations=[]
        self.circuit=[] #tuple of length 3 (gate, [qubit(s)], noise) EXCEPT for noise, where a fourth argument on the exact noise is added
        #noise is applied afterward, gate = "mr" with noise = True denotes measurement faults only. Qubits is [qubit, measurement_num]
    def noise(self, qubit: int, type: int,pauli: int):
        '''assuming independent noise:
        pauli=0: X, pauli = 1: Z, pauli = 2: Y
        type = 0: idle, type = 1: single qubit fault, type = 2: entangling qubit fault (not actually used in this code)'''
        self.circuit.append(["noise",[qubit],True, [type, pauli]])
        self.error_locations.append(self.index)
        self.index+=1
    def h(self, qubit: int):
        self.circuit.append(("h", [qubit],False))
        self.index+=1
    def cx(self, control: int, target: int):
        self.circuit.append(("cx",[control,target],False))
        self.index+=1
    def mr(self, qubit: int, noise = True):
        #assuming measurement is part of syndrome extraction
        self.circuit.append(("mr",[qubit, self.measurement_num],noise))
        self.measurement_locations.append(self.index)
        self.measurement_num+=1
        self.index+=1
    def propagate_single_error(self, error_location: int, qubit: int, pauli: int):
        '''pauli = 0: X, pauli = 1: Z, pauli = 2: Y'''
        x = np.zeros(self.qubits)
        z = np.zeros(self.qubits)
        flipped = np.zeros(self.measurement_num)
        if pauli%2==0: #fancy method, really nothing special
            x[qubit]=1
        if pauli > 0:
            z[qubit]=1
        for i in range(error_location+1, self.index):
            # do appropriate stuff
            gate = self.circuit[i][0]
            qubits = self.circuit[i][1]
            if gate == "h":
                x[qubits[0]], z[qubits[0]] = z[qubits[0]], x[qubits[0]]
            if gate == "mr":
                if(x[qubits[0]]==1):
                    flipped[qubits[1]]=1
                x[qubits[0]]=0
                z[qubits[0]]=0
            if gate == "cx":
                x[qubits[1]]= (x[qubits[0]] + x[qubits[1]]) % 2
                z[qubits[0]] = (z[qubits[0]]+z[qubits[1]]) % 2
            #error, continue
        return flipped
    def generate_tanner_graph(self):
        graph = []
        for i in range(0,len(self.circuit)):
            gate=self.circuit[i][0]
            qubits = self.circuit[i][1]
            noise = self.circuit[i][2]
            if noise:
                if gate == "mr":
                    er = np.zeros(self.measurement_num)
                    er[qubits[1]]=1
                    graph.append(er)
                if gate == "noise":
                    graph.append(self.propagate_single_error(i,qubits[0],self.circuit[i][3][1]))
        return np.asarray(graph).T
    

