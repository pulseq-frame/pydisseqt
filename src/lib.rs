use pyo3::{create_exception, exceptions::PyValueError, prelude::*};

create_exception!(pydisseqt, ParseError, PyValueError);

#[pyclass]
struct SeqParser(disseqt::SeqParser);

#[pymethods]
impl SeqParser {
    #[new]
    fn new(source: &str) -> PyResult<Self> {
        match disseqt::SeqParser::new(source) {
            Ok(parser) => Ok(Self(parser)),
            // TODO: Can we wrap this error better instead of converting it to a sting?
            // On the other hand, python can't really do much with rust errors anyways.
            Err(err) => Err(ParseError::new_err(err.to_string())),
        }
    }

    fn time_range(&self) -> (f32, f32) {
        self.0.time_range()
    }

    fn next(&self, t_start: f32, poi: &str) -> PyResult<Option<f32>> {
        use disseqt::Poi;
        let poi = match poi {
            "pulse_start" => Poi::PulseStart,
            "pulse_sample" => Poi::PulseSample,
            "pulse_end" => Poi::PulseEnd,
            "gradient_start" => Poi::GradientStart,
            "gradient_sample" => Poi::GradientSample,
            "gradient_end" => Poi::GradientEnd,
            "adc_start" => Poi::AdcStart,
            "adc_sample" => Poi::AdcSample,
            "adc_end" => Poi::AdcEnd,
            _ => return Err(PyValueError::new_err("Illegal POI name")),
        };
        Ok(self.0.next(t_start, poi))
    }

    fn integrate(&self, t0: f32, t1: f32) -> ((f32, f32), (f32, f32, f32)) {
        let (pulse, gradient) = self.0.integrate(t0, t1);
        (
            (pulse.angle, pulse.angle),
            (gradient.x, gradient.y, gradient.z),
        )
    }

    fn sample(&self, t: f32) -> ((f32, f32, f32), (f32, f32, f32), (Option<f32>, Option<f32>)) {
        let (pulse, gradient, adc) = self.0.sample(t);
        (
            (pulse.amplitude, pulse.phase, pulse.frequency),
            (gradient.x, gradient.y, gradient.z),
            match adc {
                disseqt::Adc::Inactive => (None, None),
                disseqt::Adc::Active { phase, frequency } => (Some(phase), Some(frequency)),
            },
        )
    }
}

#[pymodule]
fn pydisseqt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SeqParser>()?;
    Ok(())
}
