# %%
import pydisseqt
import matplotlib.pyplot as plt
import numpy as np

seq = pydisseqt.load_pulseq("gre.seq")
t_start, t_end = seq.next_block(0.0, "rf-pulse")

# %% Sample and plot the pulse
time = np.linspace(0, 5e-3, 1000)
pulse_amp = []
gx_amp = []
gy_amp = []
gz_amp = []

for t in time:
    pulse, grad, _ = seq.sample(t)
    pulse_amp.append(pulse[0] * np.cos(pulse[1]))
    gx_amp.append(grad[0] / 1000)
    gy_amp.append(grad[1] / 1000)
    gz_amp.append(grad[2] / 1000)

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


# %% Count pulse samples
t = t_start
sample_count = 0

while True:
    t_sample = seq.next_poi(t, "rf-pulse")
    if not t_sample or t_sample > t_end:
        break

    t = t_sample + 1e-7
    sample_count += 1

print(f"First pulse: [{t_start}..{t_end}] s, {sample_count} samples")

# %%
