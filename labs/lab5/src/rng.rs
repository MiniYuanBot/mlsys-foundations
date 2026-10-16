//! A tiny seeded RNG so every run is reproducible. Provided.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Sample an index from an (unnormalised, non-negative) weight vector.
    pub fn sample(&mut self, weights: &[f32]) -> usize {
        let total: f64 = weights.iter().map(|&w| w as f64).sum();
        let mut u = self.uniform() as f64 * total;
        for (i, &w) in weights.iter().enumerate() {
            u -= w as f64;
            if u < 0.0 {
                return i;
            }
        }
        // rounding: return the last index with positive weight
        weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
    }
}
