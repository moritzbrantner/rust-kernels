//! Reusable graph algorithms that do not impose a graph storage model.

mod assignment;
mod flow;
mod matching;
mod minimum_spanning;
mod search;
mod topology;
mod traversal;

pub use assignment::{Assignment, AssignmentError, AssignmentWork, hungarian_assignment};
pub use flow::{CapacityEdge, FlowError, FlowWork, MaxFlow, dinic_max_flow};
pub use matching::{BipartiteMatching, MatchingError, MatchingWork, hopcroft_karp};
pub use minimum_spanning::{SpanningForest, WeightedEdge, kruskal_minimum_spanning_forest};
pub use search::{Path, astar, dijkstra};
pub use topology::{
    CycleDetected, PageRank, PageRankConfig, PageRankError, page_rank,
    strongly_connected_components, topological_sort,
};
pub use traversal::{breadth_first, depth_first};
