use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_queue_id() -> u64 {
    NEXT_QUEUE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("exhausted addressable priority queue identities")
}

/// Opaque identity for an entry in an [`AddressablePriorityQueue`].
///
/// Handles are stable while the entry is present and are specific to the queue
/// that created them. Once the entry is removed, the handle is permanently
/// stale even if its internal storage slot is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PriorityQueueHandle {
    queue_id: u64,
    slot: usize,
    generation: u64,
}

/// Returned when an operation receives a stale or foreign queue handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidPriorityQueueHandle;

impl Display for InvalidPriorityQueueHandle {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("priority queue handle is invalid or stale")
    }
}

impl Error for InvalidPriorityQueueHandle {}

#[derive(Debug)]
struct Entry<P, V> {
    priority: P,
    value: V,
    slot: usize,
    insertion_order: u128,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    heap_index: Option<usize>,
    generation: u64,
}

/// Addressable min-priority queue backed by an indexed binary heap.
///
/// In addition to ordinary minimum lookup/removal, callers can update or remove
/// an existing entry in `O(log n)` through its opaque [`PriorityQueueHandle`].
/// Equal priorities are ordered by insertion time and priority updates preserve
/// that tie-breaking order.
#[derive(Debug)]
pub struct AddressablePriorityQueue<P, V> {
    queue_id: u64,
    heap: Vec<Entry<P, V>>,
    slots: Vec<Slot>,
    free_slots: Vec<usize>,
    next_insertion_order: u128,
}

impl<P, V> Default for AddressablePriorityQueue<P, V> {
    fn default() -> Self {
        Self {
            queue_id: allocate_queue_id(),
            heap: Vec::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            next_insertion_order: 0,
        }
    }
}

impl<P: Ord, V> AddressablePriorityQueue<P, V> {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of active entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Inserts an entry and returns its stable handle.
    pub fn insert(&mut self, priority: P, value: V) -> PriorityQueueHandle {
        let heap_index = self.heap.len();
        let handle = self.allocate_handle(heap_index);
        let insertion_order = self.next_insertion_order;
        self.next_insertion_order = self.next_insertion_order.saturating_add(1);

        self.heap.push(Entry {
            priority,
            value,
            slot: handle.slot,
            insertion_order,
        });
        self.sift_up(heap_index);
        handle
    }

    /// Returns references to the minimum priority and value.
    #[must_use]
    pub fn peek_min(&self) -> Option<(&P, &V)> {
        self.heap
            .first()
            .map(|entry| (&entry.priority, &entry.value))
    }

    /// Removes and returns the minimum entry.
    pub fn pop_min(&mut self) -> Option<(P, V)> {
        if self.heap.is_empty() {
            return None;
        }
        let entry = self.remove_at(0);
        Some((entry.priority, entry.value))
    }

    /// Changes the priority associated with `handle` in `O(log n)`.
    ///
    /// The entry retains its original insertion-order tie breaker.
    pub fn update_priority(
        &mut self,
        handle: PriorityQueueHandle,
        priority: P,
    ) -> Result<(), InvalidPriorityQueueHandle> {
        let index = self.resolve(handle)?;
        self.heap[index].priority = priority;
        self.repair_at(index);
        Ok(())
    }

    /// Removes the entry associated with `handle` in `O(log n)`.
    pub fn remove(
        &mut self,
        handle: PriorityQueueHandle,
    ) -> Result<(P, V), InvalidPriorityQueueHandle> {
        let index = self.resolve(handle)?;
        let entry = self.remove_at(index);
        Ok((entry.priority, entry.value))
    }

    fn allocate_handle(&mut self, heap_index: usize) -> PriorityQueueHandle {
        if let Some(slot) = self.free_slots.pop() {
            let generation = self.slots[slot].generation;
            self.slots[slot].heap_index = Some(heap_index);
            PriorityQueueHandle {
                queue_id: self.queue_id,
                slot,
                generation,
            }
        } else {
            let slot = self.slots.len();
            self.slots.push(Slot {
                heap_index: Some(heap_index),
                generation: 0,
            });
            PriorityQueueHandle {
                queue_id: self.queue_id,
                slot,
                generation: 0,
            }
        }
    }

    fn resolve(&self, handle: PriorityQueueHandle) -> Result<usize, InvalidPriorityQueueHandle> {
        if handle.queue_id != self.queue_id {
            return Err(InvalidPriorityQueueHandle);
        }

        self.slots
            .get(handle.slot)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.heap_index)
            .ok_or(InvalidPriorityQueueHandle)
    }

    fn precedes(&self, left: usize, right: usize) -> bool {
        let left = &self.heap[left];
        let right = &self.heap[right];
        left.priority < right.priority
            || (left.priority == right.priority && left.insertion_order < right.insertion_order)
    }

    fn swap_entries(&mut self, left: usize, right: usize) {
        self.heap.swap(left, right);
        self.slots[self.heap[left].slot].heap_index = Some(left);
        self.slots[self.heap[right].slot].heap_index = Some(right);
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if !self.precedes(index, parent) {
                break;
            }
            self.swap_entries(index, parent);
            index = parent;
        }
    }

    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = index * 2 + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let next = if right < self.heap.len() && self.precedes(right, left) {
                right
            } else {
                left
            };
            if !self.precedes(next, index) {
                break;
            }
            self.swap_entries(index, next);
            index = next;
        }
    }

    fn repair_at(&mut self, index: usize) {
        if index > 0 {
            let parent = (index - 1) / 2;
            if self.precedes(index, parent) {
                self.sift_up(index);
                return;
            }
        }
        self.sift_down(index);
    }

    fn remove_at(&mut self, index: usize) -> Entry<P, V> {
        let removed = self.heap.swap_remove(index);

        if index < self.heap.len() {
            self.slots[self.heap[index].slot].heap_index = Some(index);
            self.repair_at(index);
        }

        let slot = &mut self.slots[removed.slot];
        slot.heap_index = None;
        if slot.generation < u64::MAX {
            slot.generation += 1;
            self.free_slots.push(removed.slot);
        }

        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug, Clone)]
    struct ModelEntry {
        handle: PriorityQueueHandle,
        priority: i16,
        value: u16,
        insertion_order: u64,
    }

    fn model_min_index(model: &[ModelEntry]) -> Option<usize> {
        model
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| (entry.priority, entry.insertion_order))
            .map(|(index, _)| index)
    }

    fn assert_same_min(queue: &AddressablePriorityQueue<i16, u16>, model: &[ModelEntry]) {
        let expected = model_min_index(model).map(|index| {
            let entry = &model[index];
            (entry.priority, entry.value)
        });
        let actual = queue
            .peek_min()
            .map(|(priority, value)| (*priority, *value));
        assert_eq!(actual, expected);
        assert_eq!(queue.len(), model.len());
        assert_eq!(queue.is_empty(), model.is_empty());
    }

    #[test]
    fn equal_priorities_follow_insertion_order() {
        let mut queue = AddressablePriorityQueue::new();
        let first = queue.insert(5, "first");
        let second = queue.insert(5, "second");
        queue.insert(5, "third");

        queue.update_priority(second, 4).expect("active handle");
        queue.update_priority(second, 5).expect("active handle");

        assert_eq!(queue.pop_min(), Some((5, "first")));
        assert_eq!(queue.pop_min(), Some((5, "second")));
        assert_eq!(queue.pop_min(), Some((5, "third")));
        assert_eq!(
            queue.update_priority(first, 0),
            Err(InvalidPriorityQueueHandle)
        );
    }

    #[test]
    fn removed_handles_do_not_alias_reused_slots() {
        let mut queue = AddressablePriorityQueue::new();
        let stale = queue.insert(3, "old");
        assert_eq!(queue.remove(stale), Ok((3, "old")));

        let replacement = queue.insert(2, "new");
        assert_ne!(stale, replacement);
        assert_eq!(
            queue.update_priority(stale, 1),
            Err(InvalidPriorityQueueHandle)
        );
        assert_eq!(queue.remove(stale), Err(InvalidPriorityQueueHandle));
        assert_eq!(queue.pop_min(), Some((2, "new")));
    }

    #[test]
    fn handles_from_other_queues_are_rejected() {
        let mut first = AddressablePriorityQueue::new();
        let foreign = first.insert(1, "first");

        let mut second = AddressablePriorityQueue::new();
        second.insert(2, "second");

        assert_eq!(
            second.update_priority(foreign, 0),
            Err(InvalidPriorityQueueHandle)
        );
        assert_eq!(second.remove(foreign), Err(InvalidPriorityQueueHandle));
        assert_eq!(second.pop_min(), Some((2, "second")));
    }

    proptest! {
        #[test]
        fn operation_sequences_match_scan_reference(
            operations in prop::collection::vec((0u8..5, any::<i16>(), any::<u16>(), any::<u8>()), 0..300)
        ) {
            let mut queue = AddressablePriorityQueue::new();
            let mut model = Vec::<ModelEntry>::new();
            let mut handles = Vec::<PriorityQueueHandle>::new();
            let mut insertion_order = 0u64;

            for (kind, priority, value, selector) in operations {
                match kind {
                    0 => {
                        let handle = queue.insert(priority, value);
                        handles.push(handle);
                        model.push(ModelEntry {
                            handle,
                            priority,
                            value,
                            insertion_order,
                        });
                        insertion_order += 1;
                    }
                    1 if !handles.is_empty() => {
                        let handle = handles[selector as usize % handles.len()];
                        let expected = model.iter_mut().find(|entry| entry.handle == handle);
                        let expected_ok = expected.is_some();
                        if let Some(entry) = expected {
                            entry.priority = priority;
                        }
                        prop_assert_eq!(queue.update_priority(handle, priority).is_ok(), expected_ok);
                    }
                    2 if !handles.is_empty() => {
                        let handle = handles[selector as usize % handles.len()];
                        let expected = model
                            .iter()
                            .position(|entry| entry.handle == handle)
                            .map(|index| {
                                let entry = model.remove(index);
                                (entry.priority, entry.value)
                            });
                        prop_assert_eq!(queue.remove(handle).ok(), expected);
                    }
                    3 => {
                        let expected = model_min_index(&model).map(|index| {
                            let entry = model.remove(index);
                            (entry.priority, entry.value)
                        });
                        prop_assert_eq!(queue.pop_min(), expected);
                    }
                    _ => {}
                }

                let expected = model_min_index(&model).map(|index| {
                    let entry = &model[index];
                    (entry.priority, entry.value)
                });
                let actual = queue.peek_min().map(|(priority, value)| (*priority, *value));
                prop_assert_eq!(actual, expected);
                prop_assert_eq!(queue.len(), model.len());
                prop_assert_eq!(queue.is_empty(), model.is_empty());
            }
        }
    }

    #[test]
    fn reference_helper_matches_queue_for_basic_sequence() {
        let mut queue = AddressablePriorityQueue::new();
        let first = queue.insert(7, 1);
        let second = queue.insert(3, 2);
        let model = vec![
            ModelEntry {
                handle: first,
                priority: 7,
                value: 1,
                insertion_order: 0,
            },
            ModelEntry {
                handle: second,
                priority: 3,
                value: 2,
                insertion_order: 1,
            },
        ];
        assert_same_min(&queue, &model);
    }
}
