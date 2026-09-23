use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssignmentError {
    RaggedMatrix {
        row: usize,
        expected: usize,
        actual: usize,
    },
    InsufficientColumns {
        rows: usize,
        columns: usize,
    },
}
impl fmt::Display for AssignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid assignment matrix: {self:?}")
    }
}
impl std::error::Error for AssignmentError {}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssignmentWork {
    pub augmentations: usize,
    pub column_scans: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Assignment {
    /// One distinct column for each input row, in row order.
    pub row_to_column: Vec<usize>,
    pub total_cost: i128,
    /// Dual certificate: `row_potentials[i] + column_potentials[j] <= cost[i][j]`,
    /// with equality for assigned edges. Unused columns have zero potential.
    pub row_potentials: Vec<i128>,
    pub column_potentials: Vec<i128>,
    pub work: AssignmentWork,
}

/// Minimum-cost assignment of every row to a distinct column using the
/// rectangular Hungarian shortest-augmenting-path algorithm.
///
/// Rows must have equal length and rows <= columns. Empty input is valid.
/// Negative costs and all i64 values (including extremes) are ordinary costs;
/// there is no forbidden-edge sentinel. Arithmetic widens to i128 for potentials
/// and total cost. Ascending row/column scans break ties deterministically;
/// lexicographically minimal assignment is not promised. Inputs are not copied.
/// O(rows² * columns) time and O(rows + columns) scratch/output memory.
///
/// ```
/// use graph_kernels::hungarian_assignment;
/// let result = hungarian_assignment(&[[1, 2], [2, 100]]).unwrap();
/// assert_eq!(result.row_to_column, [1, 0]);
/// assert_eq!(result.total_cost, 4);
/// ```
pub fn hungarian_assignment<Row: AsRef<[i64]>>(
    costs: &[Row],
) -> Result<Assignment, AssignmentError> {
    let rows = costs.len();
    let columns = costs.first().map_or(0, |r| r.as_ref().len());
    for (row, values) in costs.iter().enumerate() {
        if values.as_ref().len() != columns {
            return Err(AssignmentError::RaggedMatrix {
                row,
                expected: columns,
                actual: values.as_ref().len(),
            });
        }
    }
    if rows > columns {
        return Err(AssignmentError::InsufficientColumns { rows, columns });
    }
    let mut u = vec![0_i128; rows + 1];
    let mut v = vec![0_i128; columns + 1];
    let mut matched_row = vec![0; columns + 1];
    let mut predecessor = vec![0; columns + 1];
    let mut slack = vec![i128::MAX; columns + 1];
    let mut used = vec![false; columns + 1];
    let mut work = AssignmentWork::default();
    for row in 1..=rows {
        matched_row[0] = row;
        slack.fill(i128::MAX);
        used.fill(false);
        let mut column = 0;
        loop {
            used[column] = true;
            let current_row = matched_row[column];
            let mut delta = i128::MAX;
            let mut next_column = 0;
            for j in 1..=columns {
                if used[j] {
                    continue;
                }
                work.column_scans += 1;
                let reduced =
                    i128::from(costs[current_row - 1].as_ref()[j - 1]) - u[current_row] - v[j];
                if reduced < slack[j] {
                    slack[j] = reduced;
                    predecessor[j] = column;
                }
                if slack[j] < delta {
                    delta = slack[j];
                    next_column = j;
                }
            }
            for j in 0..=columns {
                if used[j] {
                    u[matched_row[j]] += delta;
                    v[j] -= delta;
                } else {
                    slack[j] -= delta;
                }
            }
            column = next_column;
            if matched_row[column] == 0 {
                break;
            }
        }
        while column != 0 {
            let previous = predecessor[column];
            matched_row[column] = matched_row[previous];
            column = previous;
        }
        work.augmentations += 1;
    }
    let mut row_to_column = vec![0; rows];
    for (column, &row) in matched_row.iter().enumerate().skip(1) {
        if row != 0 {
            row_to_column[row - 1] = column - 1;
        }
    }
    let total_cost = row_to_column
        .iter()
        .enumerate()
        .map(|(i, &j)| i128::from(costs[i].as_ref()[j]))
        .sum();
    Ok(Assignment {
        row_to_column,
        total_cost,
        row_potentials: u[1..].to_vec(),
        column_potentials: v[1..].to_vec(),
        work,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn brute(costs: &[Vec<i64>], row: usize, used: usize) -> i128 {
        if row == costs.len() {
            return 0;
        }
        (0..costs[row].len())
            .filter(|&j| used & (1 << j) == 0)
            .map(|j| i128::from(costs[row][j]) + brute(costs, row + 1, used | (1 << j)))
            .min()
            .unwrap()
    }
    fn certify(costs: &[Vec<i64>], result: &Assignment) {
        assert_eq!(result.row_to_column.len(), costs.len());
        let mut columns = result.row_to_column.clone();
        columns.sort_unstable();
        columns.dedup();
        assert_eq!(columns.len(), costs.len());
        let mut total = 0;
        for (i, row) in costs.iter().enumerate() {
            total += i128::from(row[result.row_to_column[i]]);
            for (j, &cost) in row.iter().enumerate() {
                assert!(result.row_potentials[i] + result.column_potentials[j] <= i128::from(cost));
            }
            let j = result.row_to_column[i];
            assert_eq!(
                result.row_potentials[i] + result.column_potentials[j],
                i128::from(row[j])
            );
        }
        for (j, &potential) in result.column_potentials.iter().enumerate() {
            assert!(potential <= 0);
            if !columns.contains(&j) {
                assert_eq!(potential, 0);
            }
        }
        assert_eq!(total, result.total_cost);
        assert_eq!(
            result.row_potentials.iter().sum::<i128>()
                + result.column_potentials.iter().sum::<i128>(),
            total
        );
    }
    #[test]
    fn all_small_signed_matrices_match_enumeration_and_dual_certificate() {
        for mut code in 0..3_usize.pow(6) {
            let mut costs = vec![vec![0; 3]; 2];
            for row in &mut costs {
                for cost in row {
                    *cost = (code % 3) as i64 - 1;
                    code /= 3;
                }
            }
            let result = hungarian_assignment(&costs).unwrap();
            assert_eq!(result.total_cost, brute(&costs, 0, 0));
            certify(&costs, &result);
        }
        let mut state = 125_u32;
        for rows in 1..=6 {
            for columns in rows..=7 {
                for _ in 0..16 {
                    let costs: Vec<Vec<_>> = (0..rows)
                        .map(|_| {
                            (0..columns)
                                .map(|_| {
                                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                                    (state % 41) as i64 - 20
                                })
                                .collect()
                        })
                        .collect();
                    let result = hungarian_assignment(&costs).unwrap();
                    assert_eq!(result.total_cost, brute(&costs, 0, 0));
                    certify(&costs, &result);
                }
            }
        }
    }
    #[test]
    fn greedy_counterexample_wide_totals_negative_costs_and_ties() {
        let costs = vec![vec![1, 2], vec![2, 100]];
        let result = hungarian_assignment(&costs).unwrap();
        assert_eq!(result.total_cost, 4);
        certify(&costs, &result);
        for cost in [i64::MIN, i64::MAX, 0] {
            let costs = vec![vec![cost; 4]; 3];
            let result = hungarian_assignment(&costs).unwrap();
            assert_eq!(result.total_cost, 3 * i128::from(cost));
            assert_eq!(result.row_to_column, vec![0, 1, 2]);
            certify(&costs, &result);
            assert_eq!(result, hungarian_assignment(&costs).unwrap());
        }
        let costs = vec![vec![i64::MAX, i64::MIN, 0], vec![i64::MIN, i64::MAX, -1]];
        let result = hungarian_assignment(&costs).unwrap();
        assert_eq!(result.total_cost, 2 * i128::from(i64::MIN));
        certify(&costs, &result);
    }
    #[test]
    fn empty_and_invalid_shapes_are_explicit() {
        assert!(
            hungarian_assignment::<Vec<i64>>(&[])
                .unwrap()
                .row_to_column
                .is_empty()
        );
        assert!(matches!(
            hungarian_assignment(&[vec![1], vec![]]),
            Err(AssignmentError::RaggedMatrix { row: 1, .. })
        ));
        assert!(matches!(
            hungarian_assignment(&[[1], [2]]),
            Err(AssignmentError::InsufficientColumns { .. })
        ));
        assert!(matches!(
            hungarian_assignment(&[[] as [i64; 0]]),
            Err(AssignmentError::InsufficientColumns { .. })
        ));
    }
    #[test]
    fn dense_work_is_bounded_and_large_extremes_do_not_overflow() {
        for n in [8, 32, 128] {
            let costs = vec![vec![i64::MIN; n + 1]; n];
            let result = hungarian_assignment(&costs).unwrap();
            assert_eq!(result.total_cost, (n as i128) * i128::from(i64::MIN));
            assert_eq!(result.work.augmentations, n);
            assert!(result.work.column_scans <= n * n * (n + 1));
            certify(&costs, &result);
        }
    }
}
