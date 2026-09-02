/// Fixed-capacity first-in, first-out ring buffer.
///
/// The buffer owns its storage and never reallocates after construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RingBuffer<T> {
    slots: Vec<Option<T>>,
    head: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            head: 0,
            len: 0,
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    /// Pushes to the back of the buffer, returning the value unchanged when
    /// the fixed capacity is already full.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }

        let capacity = self.capacity();
        debug_assert!(capacity > 0);
        let tail = (self.head + self.len) % capacity;
        debug_assert!(self.slots[tail].is_none());
        self.slots[tail] = Some(value);
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let capacity = self.capacity();
        let value = self.slots[self.head].take();
        debug_assert!(value.is_some());
        self.len -= 1;
        if self.len == 0 {
            self.head = 0;
        } else {
            self.head = (self.head + 1) % capacity;
        }
        value
    }

    #[must_use]
    pub fn front(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.slots[self.head].as_ref()
        }
    }

    #[must_use]
    pub fn back(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let index = (self.head + self.len - 1) % self.capacity();
        self.slots[index].as_ref()
    }

    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let capacity = self.capacity();
        (0..self.len).filter_map(move |offset| {
            let index = (self.head + offset) % capacity;
            self.slots[index].as_ref()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::RingBuffer;

    fn assert_matches_model(buffer: &RingBuffer<usize>, model: &VecDeque<usize>, capacity: usize) {
        assert_eq!(buffer.capacity(), capacity);
        assert_eq!(buffer.len(), model.len());
        assert_eq!(buffer.is_empty(), model.is_empty());
        assert_eq!(buffer.is_full(), model.len() == capacity);
        assert_eq!(buffer.front(), model.front());
        assert_eq!(buffer.back(), model.back());
        assert_eq!(
            buffer.iter().copied().collect::<Vec<_>>(),
            model.iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn preserves_fifo_order_across_wraparound() {
        let mut buffer = RingBuffer::new(3);
        assert_eq!(buffer.push(10), Ok(()));
        assert_eq!(buffer.push(20), Ok(()));
        assert_eq!(buffer.pop(), Some(10));
        assert_eq!(buffer.push(30), Ok(()));
        assert_eq!(buffer.push(40), Ok(()));

        assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), vec![20, 30, 40]);
        assert_eq!(buffer.front(), Some(&20));
        assert_eq!(buffer.back(), Some(&40));
        assert_eq!(buffer.pop(), Some(20));
        assert_eq!(buffer.pop(), Some(30));
        assert_eq!(buffer.pop(), Some(40));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn full_buffer_returns_rejected_value_without_mutating_contents() {
        let mut buffer = RingBuffer::new(2);
        assert_eq!(buffer.push("a"), Ok(()));
        assert_eq!(buffer.push("b"), Ok(()));
        assert!(buffer.is_full());
        assert_eq!(buffer.push("c"), Err("c"));
        assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), vec!["a", "b"]);
    }

    #[test]
    fn exhaustive_short_sequences_match_vec_deque_model() {
        for capacity in 0_usize..=4 {
            for case in 0_usize..3_usize.pow(7) {
                let mut encoded = case;
                let mut buffer = RingBuffer::new(capacity);
                let mut model = VecDeque::new();

                for step in 0_usize..7 {
                    match encoded % 3 {
                        0 => assert_eq!(buffer.pop(), model.pop_front()),
                        1 => {
                            let value = case * 8 + step;
                            let expected = if model.len() == capacity {
                                Err(value)
                            } else {
                                model.push_back(value);
                                Ok(())
                            };
                            assert_eq!(buffer.push(value), expected);
                        }
                        2 => {
                            buffer.clear();
                            model.clear();
                        }
                        _ => unreachable!(),
                    }
                    encoded /= 3;
                    assert_matches_model(&buffer, &model, capacity);
                }
            }
        }
    }

    #[test]
    fn zero_capacity_buffer_is_valid_and_always_full() {
        let mut buffer = RingBuffer::new(0);
        assert!(buffer.is_empty());
        assert!(buffer.is_full());
        assert_eq!(buffer.push(1), Err(1));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn clear_drops_contents_and_resets_reuse_position() {
        let mut buffer = RingBuffer::new(2);
        assert_eq!(buffer.push(1), Ok(()));
        assert_eq!(buffer.push(2), Ok(()));
        assert_eq!(buffer.pop(), Some(1));
        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.push(3), Ok(()));
        assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), vec![3]);
    }
}
