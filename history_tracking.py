import stim 
def add_history(circuit: stim.Circuit, history: list, function: str, arguments: list,include_noise=True):
    '''
    PLEASE NOTE MEAUSREMENT IDENTIFIERS MUST BE 0 INDEXED AND INCREASED BY 1
    Appends to history and adds gate if necessary.
    function = "pauli". Adds a pauli gate, arguments[0] is pauli gate, arguments[1] is physical bits (list). Tracks pauli frame, does not physically apply

    function = "physical_gate". Adds a physical gate. Can be pauli, hadamard, s-gates, CX.
    arguments[0] is gate, arguments[1] is physical bits (list). if arguments[0] is a CNOT then arguments[1] must only contain two bits.


    function = "measurement". Measures a qubit. arguments[0] = "M" (required), arguments[1] which single qubit (list), arguments[2] is measurement_number (identifier)

    function = "measure+reset". Measures a qubit, then sets it to |0>. arguments[0] = "MR" (required), arguments[1] which single qubit (list), arguments[2] is measurement_number (identifier)

    
    function = "controlled_pauli". Adds a pauli frame, 
    arguments[0] is list, [[pauli, physical_bit],etc], 
    arguments[1] is requested measurements as arguments for the function, 
    arguments[2] is a function, pauli is applied when measurements returns true upon applying the function

    function = "variable_pauli". Adds a pauli frame
    arguments[0] is requested measurements as arguments for the function
    arguments[1] is a function, returns changes in pauli [["X", 0], ["Z",1], ...]
    '''
    if function == "pauli":
        history.append(["pauli", arguments[0], arguments[1]]) # pauli, gate, bits
    elif function == "physical_gate":
        history.append(["physical_gate", arguments[0], arguments[1]])

        circuit.append(arguments[0], arguments[1])
        if include_noise:
            circuit.append("DEPOLARIZE1",arguments[1], 0.001)


        #add error in the future
    elif function == "measurement":
        if arguments[0] != "M":
            raise TypeError("Measurement did not use value 'M'")
        history.append(["measurement", "M", arguments[1], arguments[2]])
        if include_noise: 
            circuit.append("M", arguments[1], 0.001)
        else:
            circuit.append("M", arguments[1])

    elif function == "measure+reset":
        if arguments[0] != "MR":
            raise TypeError("Measurement+reset did not use value 'MR'")
        history.append(["measure+reset", "MR", arguments[1], arguments[2]])
        if include_noise:
            circuit.append("MR", arguments[1], 0.001)
        else:
            circuit.append("MR", arguments[1])
    elif function == "controlled_pauli":
        history.append(["controlled_pauli",arguments[0],arguments[1],arguments[2]])
    elif function == "variable_pauli":
        history.append(["variable_pauli",arguments[0],arguments[1]])
