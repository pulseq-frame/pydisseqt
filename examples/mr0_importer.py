# %%
import pydisseqt
import MRzeroCore as mr0
import numpy as np

# NOTE: This importer is not diffusion-save.
# For this, add more events at appropriate positions (maybe gated by a flag)


# %%
def import_file(file_name: str,
                default_fov: tuple[float, float, float] = (1, 1, 1),
                overwrite_fov: bool = False
                ) -> mr0.Sequence:
    """Import a pulseq .seq file.
    
    Parameters
    ----------
    file_name : str
        The path to the file that is imported
    default_fov : (float, float, float)
        If no FOV is provided by the file, use this default value
    overwrite_fov : bool
        If true, use `default_fov` even if the file provides an FOV value
    
    Returns
    -------
    mr0.Sequence
        The imported file as mr0 Sequence
    
    Note
    ----
    This function itself is not specific to pulseq, but supports whatever
    pydisseqt supports. In the future, other sequence formats might be added.
    """
    parser = pydisseqt.load_pulseq(file_name)
    seq = mr0.Sequence()

    fov = parser.fov()
    if fov is None or overwrite_fov:
        fov = default_fov

    # We should do at least _some_ guess for the pulse usage
    def pulse_usage(angle: float) -> mr0.PulseUsage:
        if abs(angle) < 100 * np.pi / 180:
            return mr0.PulseUsage.EXCIT
        else:
            return mr0.PulseUsage.REFOC
    
    # Get time points of all pulses
    pulses = [] # Contains pairs of (pulse_start, pulse_end)
    tmp = parser.encounter("rf", 0.0)
    while tmp is not None:
        pulses.append(tmp)
        tmp = parser.encounter("rf", tmp[1]) # pulse_end

    # Iterate over all repetitions (we ignore stuff before the first pulse)
    for i in range(len(pulses)):
        # Calculate repetition start and end time based on pulse centers
        rep_start = (pulses[i][0] + pulses[i][1]) / 2
        if i + 1 < len(pulses):
            rep_end = (pulses[i + 1][0] + pulses[i + 1][1]) / 2
        else:
            rep_end = parser.duration()

        # Fetch additional data needed for building the mr0 sequence
        pulse = parser.integrate_one(pulses[i][0], pulses[i][1]).pulse
        adc_times = parser.events("adc", rep_start, rep_end)
        abs_times = [rep_start] + adc_times + [rep_end]
        event_count = len(adc_times) + 1

        samples = parser.sample(adc_times)
        moments = parser.integrate(abs_times)

        # -- Now we build the mr0 Sequence repetition --

        rep = seq.new_rep(event_count)
        rep.pulse.angle = pulse.angle
        rep.pulse.phase = pulse.phase
        rep.pulse.usage = pulse_usage(pulse.angle)

        rep.event_time[:] = torch.as_tensor(np.diff(abs_times))

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
    import torchkbnufft as tkbn

    start = time()
    # seq = import_pulseq("../../test-seqs/pypulseq/1.4.0/haste.seq")
    seq = import_file("../../test-seqs/spiral-TSE/ssTSE.seq", (0.24, 0.24, 1))
    print(f"Importing took {time() - start} seconds")
    seq.plot_kspace_trajectory((15, 15), "xy", False)

    phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz")
    data = phantom.interpolate(128, 128, 32).slices([16]).build()

    graph = mr0.compute_graph(seq, data)
    signal = mr0.execute_graph(graph, seq, data)

    # NUFFT Reconstruction
    res = [256, 256]
    kspace = seq.get_kspace()[:, :2] / 30
    dcomp = tkbn.calc_density_compensation_function(kspace.T, res)
    nufft_adj = tkbn.KbNufftAdjoint(res, [res[0]*2, res[1]*2])
    reco = nufft_adj(signal[None, None, :, 0] * dcomp, kspace.T)[0, 0, ...]

    plt.figure(figsize=(7, 7), dpi=120)
    plt.imshow(reco.abs().T, origin='lower', vmin=0)
    plt.axis("off")
    plt.show()

# %%
