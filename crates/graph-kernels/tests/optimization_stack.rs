use graph_kernels::{CapacityEdge, dinic_max_flow, hopcroft_karp};

#[test]
fn optimization_handles_twenty_thousand_vertex_paths_on_a_small_stack() {
    const MARKER: &str = "RUST_KERNELS_OPTIMIZATION_STACK_CHILD";
    if std::env::var_os(MARKER).is_some() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                let n = 20_000;
                let edges: Vec<_> = (0..n - 1)
                    .map(|i| CapacityEdge {
                        from: i,
                        to: i + 1,
                        capacity: 1,
                    })
                    .collect();
                let result = dinic_max_flow(n, &edges, 0, n - 1).unwrap();
                assert_eq!(result.value, 1);
                assert_eq!(result.work.phases, 1);
                assert!(result.edge_flows.iter().all(|&f| f == 1));
                let mut edges: Vec<_> = (0..n - 1).flat_map(|i| [(i, i + 1), (i, i)]).collect();
                edges.push((n - 1, n - 1));
                let result = hopcroft_karp(n, n, &edges).unwrap();
                assert_eq!(result.size, n);
                assert_eq!(result.work.phases, 2);
            })
            .unwrap()
            .join()
            .unwrap();
        println!("OPTIMIZATION_STACK_OK");
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "optimization_handles_twenty_thousand_vertex_paths_on_a_small_stack",
            "--nocapture",
        ])
        .env(MARKER, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("OPTIMIZATION_STACK_OK"));
}
