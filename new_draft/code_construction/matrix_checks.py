import numpy as np
from bposd.css import css_code

#[[144,12,12]]
n=144 #144 data qubits
k=12 #12 logical qubits
d=12 #distance 12 code
order_x=12 #x^12=1
order_y=6 #y^6=1
A_polynomial = [(3,0),(0,1),(0,2)] #(a,b) -> x^ay^b
B_polynomial = [(0,3),(1,0),(2,0)]

x_cyclic_shift = []
y_cyclic_shift=[]
for i in range(0,order_x):
    matrix = np.identity(order_x)
    x_cyclic_shift.append(np.roll(matrix,i,axis=1)) 
    #axis = 1 because we want to shift HORIZONTALLY since they represent the checks
for i in range(0,order_y):
    matrix=np.identity(order_y)
    y_cyclic_shift.append(np.roll(matrix,i,axis=1))

#A=x^3+y+y^2, y^3+x^2+x, x and y represent the cyclic shift matrices
#x^ay^b = x_cyclic_shift[a] TENSOR y_cyclic_shift[b]
A_poly = np.zeros((order_x*order_y, order_x*order_y))
B_poly = np.zeros((order_x*order_y, order_x*order_y))

for a,b in A_polynomial:
    A_poly += np.kron(x_cyclic_shift[a], y_cyclic_shift[b])
for a,b in B_polynomial:
    B_poly += np.kron(x_cyclic_shift[a],y_cyclic_shift[b])
A_poly %=2; B_poly %=2;

X_checks = np.hstack((A_poly,B_poly))
Z_checks = np.hstack((np.transpose(B_poly),np.transpose(A_poly)))

#performing stean syndrome extraction

#data qubts 0-143

#ancilla qubits 144-431

