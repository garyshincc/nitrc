import numpy as np
import pyedflib

for i in range(1, 15):
    read_path = f"raw_data/ibib_pan/dataverse_files/h{str(i).zfill(2)}.edf"
    write_path = f"data/ibib_pan/h{str(i).zfill(2)}.csv"
    print(read_path, write_path)

    f = pyedflib.EdfReader(read_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    np.savetxt(write_path, sigbufs, delimiter=",")

    f.close()

for i in range(1, 15):
    read_path = f"raw_data/ibib_pan/dataverse_files/s{str(i).zfill(2)}.edf"
    write_path = f"data/ibib_pan/s{str(i).zfill(2)}.csv"
    print(read_path, write_path)

    f = pyedflib.EdfReader(read_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    np.savetxt(write_path, sigbufs, delimiter=",")

    f.close()
