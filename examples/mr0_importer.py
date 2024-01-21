# %%
import pydisseqt
import MRzeroCore as mr0
import numpy as np

# NOTE: This importer is not diffusion-save.
# For this, add more events at appropriate positions (maybe gated by a flag)


# %%
def import_pulseq(path: str) -> mr0.Sequence:
    parser = pydisseqt.load_pulseq(path)
    seq = mr0.Sequence()
    t = 0

    def pulse_usage(angle: float) -> mr0.PulseUsage:
        if abs(angle) < 100 * np.pi / 180:
            return mr0.PulseUsage.EXCIT
        else:
            return mr0.PulseUsage.REFOC

    fov = parser.fov()
    if fov is None:
        fov = (1, 1, 1)

    while parser.next_block(t, "rf-pulse") is not None:
        pulse_start, pulse_end = parser.next_block(t, "rf-pulse")
        rep_start = (pulse_start + pulse_end) / 2

        # Calculate end of repetition
        next_pulse = parser.next_block(pulse_end, "rf-pulse")
        if next_pulse is not None:
            rep_end = (next_pulse[0] + next_pulse[1]) / 2
        else:
            rep_end = parser.duration()

        # Get all ADC sample times
        adc_times = parser.pois("adc", rep_start, rep_end)
        if len(adc_times) > 0:
            t = adc_times[-1]

        # Now build the mr0 repetition

        rep = seq.new_rep(len(adc_times) + 1)
        (angle, phase), _ = parser.integrate(pulse_start, pulse_end)
        rep.pulse.angle = angle
        rep.pulse.phase = phase
        rep.pulse.usage = pulse_usage(angle)

        abs_times = [rep_start] + adc_times + [rep_end]

        moments = parser.integrate_n(abs_times)
        samples = parser.sample_n(adc_times)

        for i in range(len(abs_times) - 1):
            rep.event_time[i] = abs_times[i + 1] - abs_times[i]

            _, (gx, gy, gz) = moments[i]
            rep.gradm[i, 0] = gx * fov[0]
            rep.gradm[i, 1] = gy * fov[1]
            rep.gradm[i, 2] = gz * fov[2]

            # There is no ADC at the end of the last sample
            if i < len(adc_times):
                rep.adc_usage[i] = 1
                # ADC is at the end of sample, we skip [rep_start]
                _, _, (_, phase, _) = samples[i]
                rep.adc_phase[i] = np.pi / 2 - phase

    return seq


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from time import time

    start = time()
    seq = import_pulseq("gre.seq")
    # mr0.Sequence.from_seq_file("gre.seq")
    print(f"Importing took {time() - start} seconds")
    phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz")
    data = phantom.interpolate(65, 65, 32).slices([16]).build()

    graph = mr0.compute_graph(seq, data)
    signal = mr0.execute_graph(graph, seq, data)
    reco = reco = torch.fft.fftshift(torch.fft.ifft2(
        torch.fft.ifftshift(signal.view(256, 256))))

    plt.figure()
    plt.imshow(reco.abs(), origin='lower', vmin=0)
    plt.colorbar()
    plt.show()

# %%
import cProfile
cProfile.run('import_pulseq("gre.seq")', sort="cumtime")

# %%
