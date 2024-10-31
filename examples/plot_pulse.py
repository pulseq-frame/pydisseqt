# %%
import pydisseqt
import matplotlib.pyplot as plt
import numpy as np

seq = pydisseqt.load_dsv("AA_loc_RFP/SimulationProtocol")

# %% Sample and plot the pulse
time = np.linspace(1.50, 1.55, 10000)
# time = np.linspace(0, seq.duration(), 10000)
pulse_amp = []
gx_amp = []
gy_amp = []
gz_amp = []

for t in time:
    sample = seq.sample_one(t)
    pulse_amp.append(sample.pulse.amplitude * np.cos(sample.pulse.phase))
    gx_amp.append(sample.gradient.x / 1000)
    gy_amp.append(sample.gradient.y / 1000)
    gz_amp.append(sample.gradient.z / 1000)

plt.figure(figsize=(7, 7))
plt.subplot(211)
plt.plot(time, pulse_amp)
plt.grid()
plt.ylabel("RF Pulse Amplitude [Hz]")
plt.subplot(212, sharex=plt.gca())
plt.plot(time, gx_amp, label="gx")
plt.plot(time, gy_amp, label="gy")
plt.plot(time, gz_amp, label="gz")
plt.grid()
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Gradient Amplitude [kHz/m]")
plt.show()


# %%
