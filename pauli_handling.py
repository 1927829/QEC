def update(bit, pauli):
    if pauli == "X":
        if bit == "I":
            return "X"
        if bit == "X":
            return "I"
        if bit == "Y":
            return "Z"
        if bit == "Z":
            return "Y"

    if pauli == "Y":
        if bit == "I":
            return "Y"
        if bit == "X":
            return "Z"
        if bit == "Y":
            return "I"
        if bit == "Z":
            return "X"

    if pauli == "Z":
        if bit == "I":
            return "Z"
        if bit == "X":
            return "Y"
        if bit == "Y":
            return "X"
        if bit == "Z":
            return "I"
    if pauli == "I":
        return bit
def conjugate_pauli(bit, gate):
    if gate == "H":
        if bit == "I":
            return "I"
        if bit == "X":
            return "Z"
        if bit == "Y":
            return "Y"   
        if bit == "Z":
            return "X"
    if gate == "S":
        if bit == "I":
            return "I"
        if bit == "X":
            return "Y"
        if bit == "Y":
            return "X"   
        if bit == "Z":
            return "Z"
def conjugate_cnot(control, target):
    """
    control frame, target frame
    """
    pauli_to_bits = {
        "I": (0, 0),
        "X": (1, 0),
        "Y": (1, 1),
        "Z": (0, 1),
    }
    bits_to_pauli = {
        (0, 0): "I",
        (1, 0): "X",
        (1, 1): "Y",
        (0, 1): "Z",
    }
    xc, zc = pauli_to_bits[control]
    xt, zt = pauli_to_bits[target]

    new_xc = xc
    new_zc = zc ^ zt
    new_xt = xt ^ xc
    new_zt = zt

    return (
        bits_to_pauli[(new_xc, new_zc)],
        bits_to_pauli[(new_xt, new_zt)]
    )