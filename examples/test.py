from pydisseqt import SeqParser

with open("examples/gre.seq") as f:
    parser = SeqParser(f.read())


# Example: Instantaneous pulse
t0 = parser.next(0, "pulse_start")
t1 = parser.next(0, "pulse_end")

_, g1 = parser.integrate(t0, (t0 + t1) / 2)
pulse, _ = parser.integrate(t0, t1)
_, g2 = parser.integrate((t0 + t1) / 2, t1)


# Example: ADC readout
t = parser.next(0, "adc_start")
t_end = parser.next(0, "adc_end")

while t < t_end:
    t_next = parser.next(t, "adc_sample")
    _, grad = parser.integrate(t, t_next)
    t = t_next
    _, _, (adc_phase, adc_freq) = parser.sample(t)
    # Simulate gradient moment and then do an ADC sample

_, grad = parser.integrate(t, t_end)
# Simulate gradient moment after last sample


# Example application: Plot seq file
t_start, t_end = parser.time_range()

# Plot pulses
t = t_start
while t < t_end:
    t = parser.next(t, "pulse_start")
    if not t:
        break
    t_pulse_end = parser.next(t, "pulse_end")

    (_, _, pulse_freq), _, _ = parser.sample((t_start + t_end) / 2)

    time_shape = []
    amp_shape = []
    phase_shape = []
    while t < t_pulse_end:
        t = parser.next(t, "pulse_sample")
        (amp, phase, _), _, _ = parser.sample(t)
        time_shape.append(t)
        amp_shape.append(amp)
        phase_shape.append(phase)
    # Now plot the pulse

# Plot ADCs
t = t_start
while t < t_end:
    t = parser.next(t, "adc_start")
    if not t:
        break
    t_adc_end = parser.next(t, "adc_end")

    _, _, (adc_phase, adc_freq) = parser.sample((t_start + t_end) / 2)

    samples = []
    while t < t_adc_end:
        t = parser.next(t, "adc_sample")
        samples.append(t)
    # Now plot the ADC

# Plot gradients
t = t_start
while t < t_end:
    t = parser.next(t, "gradient_start")
    if not t:
        break
    t_grad_end = parser.next(t, "gradient_end")

    time_shape = []
    gx_shape = []
    gy_shape = []
    gz_shape = []
    while t < t_grad_end:
        t = parser.next(t, "gradient_sample")
        _, (gx, gy, gz), _ = parser.sample(t)
        time_shape.append(t)
        gx_shape.append(gx)
        gy_shape.append(gy)
        gz_shape.append(gz)
    # Now plot the gradient
