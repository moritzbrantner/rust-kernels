//! Deterministic, dependency-light procedural noise kernels.
//!
//! Base fields ([`perlin2`], [`simplex2`], [`worley2`], ...) borrow a seeded
//! [`Permutation`]. Combinators ([`Fractal`], [`DomainWarp`], [`Curl`]) take a
//! caller-supplied [`Sampler2`] / [`Sampler3`] through generics and validate
//! their configuration once, returning [`CombinatorError`] for invalid input.

mod curl;
mod fractal;
mod perlin;
mod permutation;
mod sampler;
mod simplex;
mod warp;
mod worley;

pub use curl::Curl;
pub use fractal::Fractal;
pub use perlin::{perlin2, perlin3};
pub use permutation::Permutation;
pub use sampler::{CombinatorError, Sampler2, Sampler3};
pub use simplex::{simplex2, simplex3, simplex4};
pub use warp::DomainWarp;
pub use worley::{
    WorleySample, WorleySample2, WorleySample3, worley2, worley2_feature, worley3, worley3_feature,
};
