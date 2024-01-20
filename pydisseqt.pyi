from enum import Enum
# TODO: These are really basic descriptions, replace by properly formatted and detailed documentation


def load_pulseq(file_path: str) -> Sequence:
    """Load sequence from a pulseq .seq file."""


class EventType(Enum):
    RfPulse = 0
    Adc = 1
    Gradient = 2


class Sequence:
    """Some sequence, wraps a Rust disseqt::Sequence trait object."""

    def fov(self) -> tuple[float, float, float] | None:
        pass

    def duration(self) -> float:
        """ Returns the next time range of the next block of the given type.
        If `t_start` is inside of a block, this block is not returned: only
        blocks *starting* after `t_start` are considered."""

    def next_block(self, t_start: float, ty: str):
        pass

    def next_poi(self, t_start: float, ty: str):
        pass

    def integrate(self, t_start: float, t_end: float
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
