# %%
import cProfile
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

    while parser.encounter("rf", t) is not None:
        pulse_start, pulse_end = parser.encounter("rf", t)
        rep_start = (pulse_start + pulse_end) / 2

        # Calculate end of repetition
        next_pulse = parser.encounter("rf", pulse_end)
        if next_pulse is not None:
            rep_end = (next_pulse[0] + next_pulse[1]) / 2
        else:
            rep_end = parser.duration()

        # Get all ADC sample times
        adc_times = parser.events("adc", rep_start, rep_end, 100000)
        if len(adc_times) > 0:
            t = adc_times[-1]

        # Now build the mr0 repetition

        # We simulate always up to the next ADC sample, except for the last
        # event where we simulate up to the next pulse (could skip in last rep)
        event_count = len(adc_times) + 1

        rep = seq.new_rep(event_count)
        pulse = parser.integrate_one(pulse_start, pulse_end).pulse
        rep.pulse.angle = pulse.angle
        rep.pulse.phase = pulse.phase
        rep.pulse.usage = pulse_usage(pulse.angle)

        abs_times = [rep_start] + adc_times + [rep_end]
        samples = parser.sample(adc_times)
        moments = parser.integrate(abs_times)

        rep.event_time[:] = torch.as_tensor(np.diff(abs_times))

        # This is how it should look like:

        rep.gradm[:, 0] = torch.as_tensor(moments.gradient.x) * fov[0]
        rep.gradm[:, 1] = torch.as_tensor(moments.gradient.y) * fov[1]
        rep.gradm[:, 2] = torch.as_tensor(moments.gradient.z) * fov[2]

        rep.adc_usage[:-1] = 1
        rep.adc_phase[:-1] = np.pi / 2 - torch.as_tensor(samples.adc.phase)

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

    cProfile.run('import_pulseq("gre.seq")', sort="cumtime")

# %%
