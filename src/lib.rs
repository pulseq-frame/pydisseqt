use pyo3::{create_exception, prelude::*};

mod types;
pub use types::*;

create_exception!(pydisseqt, ParseError, pyo3::exceptions::PyException);

#[pyfunction]
fn load_pulseq(path: &str) -> PyResult<Sequence> {
    match disseqt::load_pulseq(path) {
        Ok(seq) => Ok(Sequence(seq)),
        Err(err) => Err(ParseError::new_err(err.to_string())),
    }
}

#[pyclass]
struct Sequence(disseqt::Sequence);

// TODO: provide pyO3 signatures with default values (if not in conflict with .pyi interface def)
// https://pyo3.rs/v0.20.2/function/signature#:~:text=Like%20Python%2C%20by%20default%20PyO3,signature%20%3D%20(...))%5D

#[pymethods]
impl Sequence {
    fn fov(&self) -> Option<(f32, f32, f32)> {
        self.0.fov()
    }

    fn duration(&self) -> f32 {
        self.0.duration()
    }

    fn encounter(&self, t_start: f32, ty: &str) -> PyResult<Option<(f32, f32)>> {
        let ty = str_to_event_type(ty)?;
        Ok(self.0.encounter(t_start, ty))
    }

    fn events(&self, ty: &str, t_start: f32, t_end: f32, max_count: usize) -> PyResult<Vec<f32>> {
        let ty = str_to_event_type(ty)?;
        Ok(self.0.events(ty, t_start, t_end, max_count))
    }

    fn next_event(&self, ty: &str, t_start: f32) -> PyResult<Option<f32>> {
        let ty = str_to_event_type(ty)?;
        Ok(self.0.next_event(t_start, ty))
    }

    fn integrate(&self, time: Vec<f32>) -> MomentVec {
        let tmp = self.0.integrate(&time);
        MomentVec {
            pulse: RfPulseMomentVec {
                angle: tmp.pulse.angle,
                phase: tmp.pulse.phase,
            },
            gradient: GradientMomentVec {
                x: tmp.gradient.x,
                y: tmp.gradient.y,
                z: tmp.gradient.z,
            },
        }
    }

    fn integrate_one(&self, t_start: f32, t_end: f32) -> Moment {
        let tmp = self.0.integrate_one(t_start, t_end);
        Moment {
            pulse: RfPulseMoment {
                angle: tmp.pulse.angle,
                phase: tmp.pulse.phase,
            },
            gradient: GradientMoment {
                x: tmp.gradient.x,
                y: tmp.gradient.y,
                z: tmp.gradient.z,
            },
        }
    }

    fn sample(&self, time: Vec<f32>) -> SampleVec {
        let tmp = self.0.sample(&time);
        SampleVec {
            pulse: RfPulseSampleVec {
                amplitude: tmp.pulse.amplitude,
                phase: tmp.pulse.phase,
                frequency: tmp.pulse.frequency,
            },
            gradient: GradientSampleVec {
                x: tmp.gradient.x,
                y: tmp.gradient.y,
                z: tmp.gradient.z,
            },
            adc: AdcBlockSampleVec {
                active: tmp.adc.active,
                phase: tmp.adc.phase,
                frequency: tmp.adc.frequency,
            },
        }
    }

    fn sample_one(&self, t: f32) -> Sample {
        let tmp = self.0.sample_one(t);
        Sample {
            pulse: RfPulseSample {
                amplitude: tmp.pulse.amplitude,
                phase: tmp.pulse.phase,
                frequency: tmp.pulse.frequency,
            },
            gradient: GradientSample {
                x: tmp.gradient.x,
                y: tmp.gradient.y,
                z: tmp.gradient.z,
            },
            adc: AdcBlockSample {
                active: tmp.adc.active,
                phase: tmp.adc.phase,
                frequency: tmp.adc.frequency,
            },
        }
    }
}

#[pymodule]
fn pydisseqt(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add_function(wrap_pyfunction!(load_pulseq, m)?)?;
    m.add_class::<Sequence>()?;
    Ok(())
}

// Simple helpers not directly exposed to python

fn str_to_event_type(ty: &str) -> PyResult<disseqt::EventType> {
    Ok(match ty {
        "rf" => disseqt::EventType::RfPulse,
        "adc" => disseqt::EventType::Adc,
        "grad x" => disseqt::EventType::Gradient(disseqt::GradientChannel::X),
        "grad y" => disseqt::EventType::Gradient(disseqt::GradientChannel::Y),
        "grad z" => disseqt::EventType::Gradient(disseqt::GradientChannel::Z),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Illegal event type",
            ))
        }
    })
}
