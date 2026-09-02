//! Reusable graph algorithms that do not impose a graph storage model.

mod search;

pub use search::{Path, astar, dijkstra};
