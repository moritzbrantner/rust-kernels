//! Reusable graph algorithms that do not impose a graph storage model.

mod search;
mod topology;
mod traversal;

pub use search::{Path, astar, dijkstra};
pub use topology::{CycleDetected, strongly_connected_components, topological_sort};
pub use traversal::{breadth_first, depth_first};
