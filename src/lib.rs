use pyo3::{create_exception, prelude::*};

create_exception!(pydisseqt, ParseError, pyo3::exceptions::PyException);

#[pyclass]
struct Sequence(Box<dyn disseqt::Sequence>);

#[pyfunction]
fn load_pulseq(path: &str) -> PyResult<Sequence> {
    match disseqt::load_pulseq(path) {
        Ok(seq) => Ok(Sequence(seq)),
        Err(err) => Err(ParseError::new_err(err.to_string())),
    }
}

// TODO: The return values should be typed - wrap PulseSample, GradientSample etc.

#[pymethods]
impl Sequence {
    fn fov(&self) -> Option<(f32, f32, f32)> {
        self.0.fov()
    }

    fn duration(&self) -> f32 {
        self.0.duration()
    }

    fn next_block(&self, t_start: f32, ty: &str) -> PyResult<Option<(f32, f32)>> {
        let ty = str_to_event_type(ty)?;
        Ok(self.0.next_block(t_start, ty))
    }

    fn next_poi(&self, t_start: f32, ty: &str) -> PyResult<Option<f32>> {
        let ty = str_to_event_type(ty)?;
        Ok(self.0.next_poi(t_start, ty))
    }

    fn pois(&self, ty: &str, t_start: Option<f32>, t_end: Option<f32>) -> PyResult<Vec<f32>> {
        let ty = str_to_event_type(ty)?;
        let t_start = t_start.unwrap_or(f32::NEG_INFINITY);
        let t_end = t_end.unwrap_or(f32::INFINITY);
        Ok(self.0.pois(&(t_start..=t_end), ty))
    }

    fn integrate(&self, t_start: f32, t_end: f32) -> ((f32, f32), (f32, f32, f32)) {
        let (pulse, gradient) = self.0.integrate(t_start, t_end);
        (
            (pulse.angle, pulse.phase),
            (gradient.gx, gradient.gy, gradient.gz),
        )
    }

    fn integrate_n(&self, time: Vec<f32>) -> Vec<((f32, f32), (f32, f32, f32))> {
        self.0.integrate_n(&time).into_iter().map(|moment| {
            (
                (moment.pulse.angle, moment.pulse.phase),
                (moment.gradient.gx, moment.gradient.gy, moment.gradient.gz),
            )
        }).collect()
    }

    fn sample(&self, t: f32) -> ((f32, f32, f32), (f32, f32, f32), (bool, f32, f32)) {
        let (pulse, gradient, adc) = self.0.sample(t);
        (
            (pulse.amplitude, pulse.phase, pulse.frequency),
            (gradient.x, gradient.y, gradient.z),
            (adc.active, adc.phase, adc.frequency)
        )
    }

    fn sample_n(&self, time: Vec<f32>) -> Vec<((f32, f32, f32), (f32, f32, f32), (bool, f32, f32))> {
        self.0.sample_n(&time).into_iter().map(|sample| {
            (
                (sample.pulse.amplitude, sample.pulse.phase, sample.pulse.frequency),
                (sample.gradient.x, sample.gradient.y, sample.gradient.z),
                (sample.adc.active, sample.adc.phase, sample.adc.frequency)
            )
        }).collect()
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

// TODO: rename rf-pulse to just rf
fn str_to_event_type(ty: &str) -> PyResult<disseqt::EventType> {
    Ok(match ty {
        "rf-pulse" => disseqt::EventType::RfPulse,
        "adc" => disseqt::EventType::Adc,
        "gradient-x" => disseqt::EventType::Gradient(disseqt::GradientChannel::X),
        "gradient-y" => disseqt::EventType::Gradient(disseqt::GradientChannel::Y),
        "gradient-z" => disseqt::EventType::Gradient(disseqt::GradientChannel::Z),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Illegal event type",
            ))
        }
    })
}
