# research/config.py
FS = 500  # Sampling frequency in Hz

N_EPOCHS = 100  # Number of training epochs
LR = 1e-4  # Learning rate
GRAD_CLIP = 10.0  # Gradient clipping threshold

NOTCH_MIN = 59.5  # Notch filter lower bound (Hz)
NOTCH_MAX = 60.5  # Notch filter upper bound (Hz)
BP_MIN = 1.0  # Bandpass filter lower bound (Hz)
BP_MAX = 70.0  # Bandpass filter upper bound (Hz)
