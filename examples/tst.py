import pydisseqt
import matplotlib.pyplot as plt
import numpy as np

parser = pydisseqt.load_pulseq(r"C:\Users\endresjn\Documents\MRzero\MRzero-Core\documentation\playground_mr0\gre_pypulseq.seq")

gx_t = parser.events("grad x")
gx = parser.sample(gx_t).gradient.x
gy_t = parser.events("grad x")
gy = parser.sample(gy_t).gradient.y
gz_t = parser.events("grad x")
gz = parser.sample(gz_t).gradient.z
rf_t = parser.events("rf")
rf = parser.sample(rf_t).pulse
adc_t = parser.events("adc")

# %matplotlib qt

plt.figure()
plt.subplot(211)
plt.plot(gx_t, gx)
plt.plot(gy_t, gy)
plt.plot(gz_t, gz)
plt.plot(adc_t, np.zeros(len(adc_t)), "rx")
plt.grid()
plt.subplot(212, sharex=plt.gca())
plt.plot(rf_t, rf.amplitude)
plt.plot(rf_t, rf.phase)
plt.grid()
plt.show()