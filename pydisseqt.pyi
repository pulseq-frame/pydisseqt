from typing import Literal
# TODO: These are really basic descriptions, replace by properly formatted and detailed documentation


class SeqParser:
    """The seq parser is a class that takes the .seq source and parses it,
    in order to provede query functions that simulations can use."""

    def __init__(self, source: str) -> None:
        """Construct a SeqParser from the .seq source code.
        Can raise an exception if parsing failed.
        """

    def time_range() -> tuple[float, float]:
        """Return the time range in which sequence lies. These are upper
        bounds, the actual sequence might be shorter."""

    def next(self, t_start: float, poi: Literal[
             'pulse_start', 'pulse_sample', 'pulse_end',
             'gradient_start', 'gradient_sample', 'gradient_end',
             'adc_start', 'adc_sample', 'adc_end']
             ) -> float | None:
        """Return the time point of the next point of interest (POI), beginning
        with the provided starting time. Can return None if no POI of the given
        type was found."""

    def integrate(self, t0: float, t1: float
                  ) -> tuple[tuple[float, float], tuple[float, float, float]]:
        """Returns the integration result over the time [t0, t1].

        Returns (flip_angle, phase), (gx, gy, gz)
        """

    def sample(self, t: float) -> tuple[tuple[float, float, float],
                                        tuple[float, float, float],
                                        tuple[float, float] | tuple[None, None]
                                        ]:
        """Returns the current rf and gradient amplitudes as well as adc phase
        (if there is an adc block active) for the given time point.

        Returns (pulse_amplitude, pulse_phase, pulse_frequency),
        (gx, gy, gz), (adc_phase, adc_frequency)
        """
