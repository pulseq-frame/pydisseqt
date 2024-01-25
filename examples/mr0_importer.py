# %%
import pydisseqt
import MRzeroCore as mr0
import numpy as np
from time import time

# NOTE: This importer is not diffusion-save.
# For this, add more events at appropriate positions (maybe gated by a flag)


# %%
def import_file(file_name: str,
                exact_trajectories: bool = True,
                print_stats: bool = False
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
    exact_trajectories : bool
        If true, the gradients before and after the ADC blocks are imported
        exactly. If false, they are summed into a single event. Depending on
        the sequence, simulation might be faster if set to false, but the
        simulated diffusion changes with simplified trajectoreis.
    print_stats : bool
        If set to true, additional information is printed during import

    Returns
    -------
    mr0.Sequence
        The imported file as mr0 Sequence
    """
    start = time()
    parser = pydisseqt.load_pulseq(file_name)
    if print_stats:
        print(f"Importing the .seq file took {time() - start} s")
    start = time()
    seq = mr0.Sequence()

    # We should do at least _some_ guess for the pulse usage
    def pulse_usage(angle: float) -> mr0.PulseUsage:
        if abs(angle) < 100 * np.pi / 180:
            return mr0.PulseUsage.EXCIT
        else:
            return mr0.PulseUsage.REFOC

    # Get time points of all pulses
    pulses = []  # Contains pairs of (pulse_start, pulse_end)
    tmp = parser.encounter("rf", 0.0)
    while tmp is not None:
        pulses.append(tmp)
        tmp = parser.encounter("rf", tmp[1])  # pulse_end

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

        # To simulate diffusion, we want to more exactly simulate gradient
        # trajectories between pulses and the ADC block
        if exact_trajectories:
            # First and last timepoint in repetition with a gradient sample
            first = pulses[i][1]
            last = (pulses[i + 1][0] if i + 1 < len(pulses) else rep_end)
            eps = 1e-6  # Move a bit past start / end of repetition
            # Gradient samples can be duplicated (x, y, z with identical times)
            # They are deduplicated after rounding to `precision` digits
            precision = 6

            if len(adc_times) > 0:
                grad_before = sorted(set([round(t, precision) for t in (
                    parser.events("grad x", first + eps, adc_times[0] - eps) +
                    parser.events("grad y", first + eps, adc_times[0] - eps) +
                    parser.events("grad z", first + eps, adc_times[0] - eps)
                )]))
                grad_after = sorted(set([round(t, precision) for t in (
                    parser.events("grad x", adc_times[-1] + eps, last - eps) +
                    parser.events("grad y", adc_times[-1] + eps, last - eps) +
                    parser.events("grad z", adc_times[-1] + eps, last - eps)
                )]))
                # Last repetition: there is no pulse, ignore [last, rep_end]
                if i == len(pulses) - 1:
                    abs_times = [rep_start, first] + grad_before + adc_times
                else:
                    abs_times = ([rep_start, first] + grad_before + adc_times +
                                 grad_after + [last, rep_end])
                # Index of first ADC: -1 because we count spans between indices
                adc_start = 2 + len(grad_before) - 1
            else:
                grad = sorted(set([round(t, precision) for t in (
                    parser.events("grad x", first + eps, last - eps) +
                    parser.events("grad y", first + eps, last - eps) +
                    parser.events("grad z", first + eps, last - eps)
                )]))
                # Last repetition: there is no pulse, ignore [last, rep_end]
                if i == len(pulses) - 1:
                    abs_times = [rep_start, first] + grad
                else:
                    abs_times = ([rep_start, first] + grad + [last, rep_end])
                adc_start = None
        else:
            # No gradient samples, only adc and one final to the next pulse
            abs_times = [rep_start] + adc_times + [rep_end]
            adc_start = 0

        event_count = len(abs_times) - 1
        samples = parser.sample(adc_times)
        moments = parser.integrate(abs_times)

        if print_stats:
            print(
                f"Rep. {i + 1}: {event_count} samples, of which "
                f"{len(adc_times)} are ADC (starting at {adc_start})"
            )

        # -- Now we build the mr0 Sequence repetition --

        rep = seq.new_rep(event_count)
        rep.pulse.angle = pulse.angle
        rep.pulse.phase = pulse.phase
        rep.pulse.usage = pulse_usage(pulse.angle)

        rep.event_time[:] = torch.as_tensor(np.diff(abs_times))

        rep.gradm[:, 0] = torch.as_tensor(moments.gradient.x)
        rep.gradm[:, 1] = torch.as_tensor(moments.gradient.y)
        rep.gradm[:, 2] = torch.as_tensor(moments.gradient.z)

        if adc_start is not None:
            phases = np.pi / 2 - torch.as_tensor(samples.adc.phase)
            rep.adc_usage[adc_start:adc_start + len(adc_times)] = 1
            rep.adc_phase[adc_start:adc_start + len(adc_times)] = phases

    if print_stats:
        print(f"Converting the sequence to mr0 took {time() - start} s")
    return seq


# %%
def current_fig_as_img(dpi: float = 180) -> np.ndarray:
    import cv2
    import io
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import torchkbnufft as tkbn
    import imageio

    # seq = import_pulseq("../../test-seqs/pypulseq/1.4.0/haste.seq")
    seq = import_file("../../test-seqs/spiral-TSE/ssTSE.seq")
    seq.plot_kspace_trajectory((7, 7), "xy", False)

    phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz")
    data = phantom.interpolate(128, 128, 32).slices([16]).build(use_SI_FoV=True)
    B0 = data.B0.clone()

    gif = []
    B0_factors = [1]  # np.linspace(-5, 5, 150)
    for i, dB0 in enumerate(B0_factors):
        print(f"{i + 1} / {len(B0_factors)}")

        data.B0 = dB0 * B0
        graph = mr0.compute_graph(seq, data)
        signal = mr0.execute_graph(graph, seq, data)

        # NUFFT Reconstruction
        res = [256, 256]
        kspace = seq.get_kspace()[:, :2] / 150
        dcomp = tkbn.calc_density_compensation_function(kspace.T, res)
        nufft_adj = tkbn.KbNufftAdjoint(res, [res[0]*2, res[1]*2])
        reco = nufft_adj(signal[None, None, :, 0] * dcomp, kspace.T)[0, 0, ...]

        # Quick and dirty FWHM: our synthetic B0 map is not really normal dist.
        fwhm = 2 * np.sqrt(2 * np.log(2)) * data.B0.std()

        plt.figure(figsize=(9, 5), dpi=80)
        plt.suptitle(f"$FWHM(B_0) = {fwhm:.0f}\\,$Hz")
        plt.subplot(121)
        plt.title("Magnitude")
        plt.axis("off")
        plt.imshow(reco.abs().T, origin='lower', vmin=0)
        plt.subplot(122)
        plt.title("Phase")
        plt.imshow(reco.angle().T, origin='lower',
                   vmin=-np.pi, vmax=np.pi, cmap="twilight")
        plt.axis("off")
        plt.subplots_adjust(wspace=0.02)
        if len(B0_factors) == 1:
            plt.show()
        else:
            gif.append(current_fig_as_img(80))
            plt.close()

    if len(gif) > 0:
        imageio.mimsave("B0 Spiral.gif", gif, fps=1, loop=0)

# %%
