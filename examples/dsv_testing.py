# %%
import pydisseqt
import matplotlib.pyplot as plt
import torch
import numpy as np
import MRzeroCore as mr0


# A very simple guess for the pulse usage - can't really do better from some unkown sequence
def pulse_usage(angle: float) -> mr0.PulseUsage:
    if abs(angle) < 100 * np.pi / 180:
        return mr0.PulseUsage.EXCIT
    else:
        return mr0.PulseUsage.REFOC


def import_dsv(file):
    seq = pydisseqt.load_dsv(file)

    # First, load the position of pulses and calculate the starting points
    # of the repetitions from it
    pulses = []
    tmp = seq.encounter("rf", 0.0)
    while tmp is not None:
        pulses.append(tmp)
        tmp = seq.encounter("rf", tmp[1])

    # Then, crate the sequence by iterating over repetitions
    mr0_seq = mr0.Sequence(normalized_grads=False)

    for i in range(len(pulses)):
        # First calculate imporant time points
        pulse_start = pulses[i][0]
        pulse_end = pulses[i][1]
        if i + 1 < len(pulses):
            next_pulse_start = pulses[i + 1][0]
            next_pulse_end = pulses[i + 1][1]
        else:
            next_pulse_start = seq.duration()
            next_pulse_end = seq.duration()
        rep_start = (pulse_start + pulse_end) / 2
        rep_end = (next_pulse_start + next_pulse_end) / 2

        t_adc = seq.events("adc", rep_start, rep_end)
        t_event = np.asarray([rep_start, pulse_end, *t_adc, next_pulse_start, rep_end])

        # Create the repetition and fill it with data
        rep = mr0_seq.new_rep(len(t_event) - 1)

        pulse = seq.integrate_one(pulse_start, pulse_end).pulse
        rep.pulse.angle = pulse.angle
        rep.pulse.phase = pulse.phase
        rep.pulse.usage = pulse_usage(pulse.angle)

        events = seq.integrate(t_event)
        adcs = seq.sample(t_adc).adc
        rep.event_time[:] = torch.as_tensor(np.diff(t_event))
        rep.gradm[:, 0] = torch.as_tensor(events.gradient.x)
        rep.gradm[:, 1] = torch.as_tensor(events.gradient.y)
        rep.gradm[:, 2] = torch.as_tensor(events.gradient.z)
        rep.adc_phase[1:-2] = np.pi / 2 - torch.as_tensor(adcs.phase)
        rep.adc_usage[1:-2] = torch.as_tensor(adcs.active)

    return mr0_seq


# %%

# seq = import_dsv("3DSnapshotGRE_Comparision_E_0_64_64_8_alternating_fully_sampled/SimulationProtocol")
seq = import_dsv("AA_loc_RFP/SimulationProtocol")

plt.plot([rep.pulse.angle * 180/np.pi for rep in seq])

# %%
