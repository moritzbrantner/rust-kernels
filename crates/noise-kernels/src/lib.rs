//! Deterministic, dependency-light procedural noise kernels.

mod perlin;
mod permutation;

pub use perlin::{perlin2, perlin3};
pub use permutation::Permutation;
