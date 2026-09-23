//! Execute potentially aborting stack regressions in an isolated subprocess.
use graph_kernels::strongly_connected_components;

#[test]
fn boundary_scc_deep_graph_fits_small_thread_stack() {
    const MARKER: &str = "RUST_KERNELS_DEEP_SCC_CHILD";
    if std::env::var_os(MARKER).is_some() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                const N: usize = 20_000;
                let chain =
                    strongly_connected_components([0_usize], |&v| (v + 1 < N).then_some(v + 1));
                assert_eq!(chain.len(), N);
                for (v, component) in chain.iter().enumerate() {
                    assert_eq!(component, &[v]);
                }
                let cycle = strongly_connected_components([0_usize], |&v| [(v + 1) % N]);
                assert_eq!(cycle, vec![(0..N).collect::<Vec<_>>()]);
            })
            .unwrap()
            .join()
            .unwrap();
        println!("DEEP_SCC_OK nodes=20000 chain_and_cycle");
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "boundary_scc_deep_graph_fits_small_thread_stack",
            "--nocapture",
        ])
        .env(MARKER, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "deep SCC child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("DEEP_SCC_OK nodes=20000 chain_and_cycle")
    );
}

#[test]
fn boundary_scc_ordinary_control() {
    let actual = strongly_connected_components([0_usize], |&v| match v {
        0 => vec![1],
        1 => vec![0, 2],
        _ => vec![],
    });
    assert_eq!(actual, vec![vec![0, 1], vec![2]]);
}

#[test]
fn scc_matches_reachability_oracle_for_all_four_vertex_graphs() {
    const N: usize = 4;
    // All 4096 directed graphs without self-edges; a separate case covers
    // self-loops and parallel edges. Floyd closure is independent of Tarjan.
    for mask in 0_u32..(1 << 12) {
        let mut graph = vec![Vec::new(); N];
        let mut bit = 0;
        for (a, edges) in graph.iter_mut().enumerate() {
            for b in 0..N {
                if a != b {
                    if mask & (1 << bit) != 0 {
                        edges.push(b);
                    }
                    bit += 1;
                }
            }
        }
        let mut reach = [[false; N]; N];
        for (a, edges) in graph.iter().enumerate() {
            reach[a][a] = true;
            for &b in edges {
                reach[a][b] = true;
            }
        }
        for via in 0..N {
            for a in 0..N {
                for b in 0..N {
                    reach[a][b] |= reach[a][via] && reach[via][b];
                }
            }
        }
        let mut assigned = [false; N];
        let mut expected = Vec::new();
        for a in 0..N {
            if assigned[a] {
                continue;
            }
            let component = (0..N)
                .filter(|&b| reach[a][b] && reach[b][a])
                .collect::<Vec<_>>();
            for &b in &component {
                assigned[b] = true;
            }
            expected.push(component);
        }
        assert_eq!(
            strongly_connected_components(0..N, |&v| graph[v].iter().copied()),
            expected,
            "graph={graph:?}"
        );
    }
}

#[test]
fn scc_keeps_seed_order_parallel_edges_and_finished_component_boundaries() {
    let graph = [vec![0, 1, 1], vec![0, 2], vec![2], vec![0, 4], vec![3]];
    // Visiting a previously completed SCC must not merge it with a later SCC.
    let actual = strongly_connected_components([2, 3, 0, 3], |&v| graph[v].iter().copied());
    assert_eq!(actual, vec![vec![2], vec![3, 4], vec![0, 1]]);
    let empty = strongly_connected_components(std::iter::empty::<usize>(), |_| std::iter::empty());
    assert!(empty.is_empty());
}
