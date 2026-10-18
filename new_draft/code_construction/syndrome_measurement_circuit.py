import numpy as np

class Measure_Circuit:
    def __init__(self,num_qubits: int):
        self.qubits = num_qubits
        self.index=0
        self.measurement_num=0
        self.error_locations=[] #pauli
        self.measurement_locations=[]
        self.circuit=[] #tuple of length 3 (gate, [qubit(s)], noise)
        #noise is applied afterward, gate = "mr" with noise = True denotes measurement faults only. Qubits is [qubit, measurement_num]
    def h(self, qubit: int, noise=False):
        self.circuit.append(("h", [qubit], noise))
        if noise:
            self.error_locations.append(self.index)
        self.index+=1
    def cx(self, control: int, target: int, noise = True):
        self.circuit.append(("cx",[control,target],noise))
        if noise:
            self.error_locations.append(self.index)
        self.index+=1
    def mr(self, qubit: int, noise = True):
        #assuming measurement is part of syndrome extraction
        self.circuit.append(("mr",[qubit, self.measurement_num],noise))
        self.measurement_locations.append(self.index)
        self.measurement_num+=1
        self.index+=1
    def propagate_x_error(self, error_location: int, qubit: int):
        x = np.zeros(self.qubits)
        z = np.zeros(self.qubits)
        flipped = np.zeros(self.measurement_num)
        x[qubit]=1
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
        return flipped
    def propagate_z_error(self, error_location: int, qubit: int):
            x = np.zeros(self.qubits)
            z = np.zeros(self.qubits)
            flipped = np.zeros(self.measurement_num)
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
            return flipped
    def generate_x_tanner_graph(self):
        graph = []
        for i in range(0,len(self.circuit)):
            gate=self.circuit[i][0]
            qubits = self.circuit[i][1]
            noise = self.circuit[i][2]
            if noise:
                if gate == "h":
                    graph.append(self.propagate_x_error(i,qubits[0]))
                if gate == "cx":
                    graph.append(self.propagate_x_error(i,qubits[0]))
                    graph.append(self.propagate_x_error(i,qubits[1]))
                if gate == "mr":
                    er = np.zeros(self.measurement_num)
                    er[qubits[1]]=1
                    graph.append(er)
        return np.asarray(graph)
    def generate_z_tanner_graph(self):
            graph = []
            for i in range(0,len(self.circuit)):
                gate=self.circuit[i][0]
                qubits = self.circuit[i][1]
                noise = self.circuit[i][2]
                if noise:
                    if gate == "h":
                        graph.append(self.propagate_z_error(i,qubits[0]))
                    if gate == "cx":
                        graph.append(self.propagate_z_error(i,qubits[0]))
                        graph.append(self.propagate_z_error(i,qubits[1]))
                    if gate == "mr":
                        er = np.zeros(self.measurement_num)
                        er[qubits[1]]=1
                        graph.append(er)
            return np.asarray(graph)


