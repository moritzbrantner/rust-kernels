/// Computes the Levenshtein edit distance between two sequences.
///
/// Insertions, deletions, and substitutions each cost one. The algorithm runs
/// in `O(a.len() * b.len())` time and uses `O(min(a.len(), b.len()))`
/// auxiliary memory.
///
/// The sequence element type only needs equality. Callers therefore choose the
/// semantic unit: bytes, Unicode scalar values, grapheme IDs, words, phonemes,
/// or another token type.
pub fn levenshtein<T: Eq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut row: Vec<usize> = (0..=shorter.len()).collect();

    for (long_index, long_item) in longer.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = long_index + 1;

        for (short_index, short_item) in shorter.iter().enumerate() {
            let column = short_index + 1;
            let above = row[column];
            let insertion = row[column - 1] + 1;
            let deletion = above + 1;
            let substitution = diagonal + if long_item == short_item { 0 } else { 1 };
            row[column] = insertion.min(deletion).min(substitution);
            diagonal = above;
        }
    }

    row[shorter.len()]
}

#[cfg(test)]
mod tests {
    use super::levenshtein;

    fn reference<T: Eq>(a: &[T], b: &[T]) -> usize {
        let width = b.len() + 1;
        let mut matrix = vec![0usize; (a.len() + 1) * width];

        for i in 0..=a.len() {
            matrix[i * width] = i;
        }
        for (j, cell) in matrix.iter_mut().take(width).enumerate() {
            *cell = j;
        }

        for i in 1..=a.len() {
            for j in 1..=b.len() {
                let deletion = matrix[(i - 1) * width + j] + 1;
                let insertion = matrix[i * width + j - 1] + 1;
                let substitution = matrix[(i - 1) * width + j - 1]
                    + if a[i - 1] == b[j - 1] { 0 } else { 1 };
                matrix[i * width + j] = deletion.min(insertion).min(substitution);
            }
        }

        matrix[a.len() * width + b.len()]
    }

    #[test]
    fn handles_empty_sequences() {
        assert_eq!(levenshtein::<u8>(&[], &[]), 0);
        assert_eq!(levenshtein(b"abc", b""), 3);
        assert_eq!(levenshtein(b"", b"abc"), 3);
    }

    #[test]
    fn matches_classic_examples() {
        assert_eq!(levenshtein(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein(b"flaw", b"lawn"), 2);
        assert_eq!(levenshtein(b"gumbo", b"gambol"), 2);
    }

    #[test]
    fn works_for_non_text_tokens() {
        let a = [10u16, 20, 30, 40];
        let b = [10u16, 25, 30, 50, 40];
        assert_eq!(levenshtein(&a, &b), 2);
    }

    #[test]
    fn is_symmetric() {
        let a = b"synchronization";
        let b = b"synchronisation";
        assert_eq!(levenshtein(a, b), levenshtein(b, a));
    }

    #[test]
    fn matches_quadratic_reference_on_generated_sequences() {
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8 % 7
        };

        for left_len in 0..16 {
            for right_len in 0..16 {
                let left: Vec<_> = (0..left_len).map(|_| next()).collect();
                let right: Vec<_> = (0..right_len).map(|_| next()).collect();
                assert_eq!(levenshtein(&left, &right), reference(&left, &right));
            }
        }
    }
}
