import matrix_checks

#f me sort of construction

base = matrix_checks.n*3 #this is the starting point if you add bell
#u_0 to u_11
u = [
    (5, 0, 1),
    (6, 0, 0),
    (4, 3, 0),
    (4, 5, 1),
    (5, 5, 0),
    (6, 2, 0),
    (5, 2, 1),
    (5, 1, 0),
    (4, 1, 0),
    (4, 2, 1),
    (4, 1, 1),
    (4, 0, 1),
]

d = len(u)

data_qubits_u=[]
for i in range(0,d):
    data_qubits_u.append(matrix_checks.get_qubit_tuple(u[i]))


edges_1=[(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11)]

edges_2 = [(0,1),(0,4),(0,8),(1,11),(2,6),(3,11),(5,9),(7,10)]

edges = sorted(edges_1+edges_2)

reverse_dictionary = {} #edge to what qubit
edge_adjacency_matrix =  [[] for _ in range(d)] #index i with qubit data_qubits[i] has what edges

for i in range(0,len(edges)):
    reverse_dictionary[edges[i]]=(i+base)
    reverse_dictionary[(edges[i][1],edges[i][0])]=i+base
    edge_adjacency_matrix[edges[i][0]].append(edges[i])
    edge_adjacency_matrix[edges[i][1]].append(edges[i])


bridge_X_checks = [[] for _ in range(d)]

for i in range(0,d):
    qubit = data_qubits_u[i]
    bridge_X_checks[i].append(qubit)
    for edge in edge_adjacency_matrix[i]:
        bridge_X_checks[i].append(reverse_dictionary[edge])


Z_pre = [] #cycle checks

Z_pre.append([8,9,10,7,8]) #loops back onto itself
Z_pre.append([3,11,1,2,3])
Z_pre.append([3,11,1,0,4,3])
Z_pre.append([8,9,5,4,0,8])
Z_pre.append([10,9,5,6,7,10])

cycle_Z_checks = []

for i in range(0,len(Z_pre)):
    cycle_Z_checks.append([])
    cycle = Z_pre[i]
    length = len(cycle)
    for j in range(0,length-1):
        cycle_Z_checks[i].append(reverse_dictionary[(cycle[j],cycle[(j+1)])])

qubit_X_checks = [check.copy() for check in matrix_checks.sparse_X_checks]
qubit_Z_checks = [check.copy() for check in matrix_checks.sparse_Z_checks]

#adjust

for i in range(0,len(qubit_Z_checks)):
    check = qubit_Z_checks[i]
    involved=[]
    for qubits in check:
        if qubits in data_qubits_u:
            involved.append(data_qubits_u.index(qubits))
    if len(involved)>2:
        print(i,involved)
        raise AssertionError("Idk more overlap")
    if len(involved)==0:
        continue
    else:
        check.append(reverse_dictionary[(involved[0],involved[1])])