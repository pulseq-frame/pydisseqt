# %%
import pydisseqt
import matplotlib.pyplot as plt
import numpy as np

seq = pydisseqt.load_pulseq("gre.seq")
t_start, t_end = seq.next_block(0.0, "rf-pulse")

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

# %% Sample and plot the pulse
plot_width = 50
plot_height = 30

time = np.linspace(t_start, t_end, 200)
amp = []

for t in time:
    pulse, _, _ = seq.sample(t)
    amp.append(pulse[0] * np.cos(pulse[1]))


plt.figure()
plt.plot(time, amp)
plt.grid()
plt.show()

# %%
