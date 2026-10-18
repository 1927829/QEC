import pauli_handling
def history_decode(history: list,pauli_frame: list, measure_results):
    actual_measurements = []
    for items in history:
        if (items[0]=="pauli"):
            for things in items[2]:
                pauli_frame[things]=pauli_handling.update(pauli_frame[things],items[1])
        if(items[0]=="physical_gate"):
            if items[1] == "X" or items[1]=="Z" or items[1] == "Y":
                continue
            if items[1]=="H" or items[1] == "S":
                for things in items[2]:
                    pauli_frame[things]=pauli_handling.conjugate_pauli(pauli_frame[things],items[1])
            if items[1] == "CX":
                #items[2] contains both bits
                new_frame = pauli_handling.conjugate_cnot(pauli_frame[items[2][0]],pauli_frame[items[2][1]])
                pauli_frame[items[2][0]]=new_frame[0]
                pauli_frame[items[2][1]]=new_frame[1]
        if(items[0]=="measurement" or items[0] == "measure+reset"):
            bit = items[2][0]
            if pauli_frame[bit]=="X" or pauli_frame[bit]=="Y":
                actual_measurements.append(not measure_results[items[3]])
            else:
                actual_measurements.append(measure_results[items[3]])
            if items[0] == "measure+reset":
                pauli_frame[bit]="I"
        if(items[0]=="controlled_pauli"):
            args = []
            for indexes in items[2]:
                args.append(actual_measurements[indexes])
            if items[3](*args):
                for things in items[1]:
                    pauli_frame[things[1]]=pauli_handling.update(pauli_frame[things[1]],things[0])
        if(items[0]=="variable_pauli"):
            args = []
            for indexes in items[1]:
                args.append(actual_measurements[indexes])
            updates = items[2](*args)
            for things in updates:
                pauli_frame[things]=pauli_handling.update(pauli_frame[things],updates)
    return actual_measurements
        #TODO: add rest