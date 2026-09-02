//! Exact similarity and sequence-distance kernels.
//!
//! These kernels intentionally operate on caller-provided sequences and sorted
//! sets. Text normalization, Unicode segmentation, tokenization, and feature
//! extraction belong in consumers.

mod edit_distance;
mod jaccard;
mod sorted_set;

pub use edit_distance::levenshtein;
pub use jaccard::{jaccard_distance, jaccard_similarity};
pub use sorted_set::{
    sorted_difference, sorted_intersection, sorted_intersection_count,
    sorted_symmetric_difference, sorted_union, sorted_union_count,
};
