use std::mem;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArenaKey {
    index: usize,
    generation: u64,
}

impl ArenaKey {
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Debug)]
enum Slot<T> {
    Occupied {
        generation: u64,
        value: T,
    },
    Vacant {
        generation: u64,
        next_free: Option<usize>,
    },
    Retired,
}

/// Arena with stable generational handles and O(1) insertion/removal.
///
/// Removed slots are reused with an incremented generation. A slot whose
/// generation reaches `u64::MAX` is retired instead of wrapping, so a stale key
/// can never become valid again through generation overflow.
#[derive(Clone, Debug, Default)]
pub struct GenerationalArena<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<usize>,
    len: usize,
}

impl<T> GenerationalArena<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn insert(&mut self, value: T) -> ArenaKey {
        let key = if let Some(index) = self.free_head {
            let vacant = mem::replace(&mut self.slots[index], Slot::Retired);
            let Slot::Vacant {
                generation,
                next_free,
            } = vacant
            else {
                panic!("arena free list points to a non-vacant slot");
            };
            self.free_head = next_free;
            self.slots[index] = Slot::Occupied { generation, value };
            ArenaKey { index, generation }
        } else {
            let index = self.slots.len();
            let generation = 0;
            self.slots.push(Slot::Occupied { generation, value });
            ArenaKey { index, generation }
        };

        self.len += 1;
        key
    }

    #[must_use]
    pub fn contains(&self, key: ArenaKey) -> bool {
        self.get(key).is_some()
    }

    #[must_use]
    pub fn get(&self, key: ArenaKey) -> Option<&T> {
        match self.slots.get(key.index)? {
            Slot::Occupied { generation, value } if *generation == key.generation => Some(value),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: ArenaKey) -> Option<&mut T> {
        match self.slots.get_mut(key.index)? {
            Slot::Occupied { generation, value } if *generation == key.generation => Some(value),
            _ => None,
        }
    }

    pub fn remove(&mut self, key: ArenaKey) -> Option<T> {
        let generation = match self.slots.get(key.index)? {
            Slot::Occupied { generation, .. } if *generation == key.generation => *generation,
            _ => return None,
        };

        let occupied = mem::replace(&mut self.slots[key.index], Slot::Retired);
        let Slot::Occupied { value, .. } = occupied else {
            unreachable!("validated occupied slot changed during removal");
        };

        if generation == u64::MAX {
            self.slots[key.index] = Slot::Retired;
        } else {
            self.slots[key.index] = Slot::Vacant {
                generation: generation + 1,
                next_free: self.free_head,
            };
            self.free_head = Some(key.index);
        }
        self.len -= 1;
        Some(value)
    }

    pub fn clear(&mut self) {
        self.free_head = None;
        self.len = 0;

        for index in (0..self.slots.len()).rev() {
            let replacement = match &self.slots[index] {
                Slot::Occupied { generation, .. } if *generation < u64::MAX => Slot::Vacant {
                    generation: generation + 1,
                    next_free: self.free_head,
                },
                Slot::Occupied { .. } | Slot::Retired => Slot::Retired,
                Slot::Vacant { generation, .. } => Slot::Vacant {
                    generation: *generation,
                    next_free: self.free_head,
                },
            };

            if matches!(replacement, Slot::Vacant { .. }) {
                self.free_head = Some(index);
            }
            self.slots[index] = replacement;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (ArenaKey, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            if let Slot::Occupied { generation, value } = slot {
                Some((
                    ArenaKey {
                        index,
                        generation: *generation,
                    },
                    value,
                ))
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ArenaKey, GenerationalArena};

    #[derive(Clone, Copy)]
    struct Tracked {
        key: ArenaKey,
        value: i32,
        active: bool,
    }

    fn assert_matches_model(arena: &GenerationalArena<i32>, tracked: &[Tracked]) {
        let expected_len = tracked.iter().filter(|entry| entry.active).count();
        assert_eq!(arena.len(), expected_len);
        assert_eq!(arena.is_empty(), expected_len == 0);

        for entry in tracked {
            assert_eq!(arena.contains(entry.key), entry.active);
            assert_eq!(
                arena.get(entry.key).copied(),
                entry.active.then_some(entry.value)
            );
        }

        let mut expected = tracked
            .iter()
            .filter(|entry| entry.active)
            .map(|entry| (entry.key.index(), entry.key.generation(), entry.value))
            .collect::<Vec<_>>();
        expected.sort_by_key(|entry| entry.0);
        let actual = arena
            .iter()
            .map(|(key, value)| (key.index(), key.generation(), *value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn removed_slots_are_reused_without_reviving_stale_keys() {
        let mut arena = GenerationalArena::new();
        let first = arena.insert("first");
        let second = arena.insert("second");

        assert_eq!(arena.remove(first), Some("first"));
        let replacement = arena.insert("replacement");

        assert_eq!(replacement.index(), first.index());
        assert_eq!(replacement.generation(), first.generation() + 1);
        assert!(!arena.contains(first));
        assert_eq!(arena.get(replacement), Some(&"replacement"));
        assert_eq!(arena.get(second), Some(&"second"));
    }

    #[test]
    fn mutable_access_requires_the_current_generation() {
        let mut arena = GenerationalArena::new();
        let key = arena.insert(10);
        *arena.get_mut(key).expect("inserted key must resolve") = 42;
        assert_eq!(arena.get(key), Some(&42));
    }

    #[test]
    fn iteration_is_deterministic_by_slot_index() {
        let mut arena = GenerationalArena::new();
        let first = arena.insert('a');
        let middle = arena.insert('b');
        let _last = arena.insert('c');
        assert_eq!(arena.remove(middle), Some('b'));
        let reused = arena.insert('d');
        assert_eq!(reused.index(), middle.index());

        assert_eq!(
            arena
                .iter()
                .map(|(key, value)| (key.index(), *value))
                .collect::<Vec<_>>(),
            vec![(first.index(), 'a'), (middle.index(), 'd'), (2, 'c')]
        );
    }

    #[test]
    fn exhaustive_short_sequences_keep_stale_handles_invalid() {
        for case in 0_usize..4_usize.pow(6) {
            let mut arena = GenerationalArena::new();
            let mut tracked = Vec::new();
            let mut encoded = case;

            for step in 0_usize..6 {
                match encoded % 4 {
                    0 => {
                        let value = (case * 8 + step) as i32;
                        let key = arena.insert(value);
                        tracked.push(Tracked {
                            key,
                            value,
                            active: true,
                        });
                    }
                    1 => {
                        if let Some(entry) = tracked.first_mut() {
                            let expected = entry.active.then_some(entry.value);
                            assert_eq!(arena.remove(entry.key), expected);
                            entry.active = false;
                        }
                    }
                    2 => {
                        if let Some(entry) = tracked.last_mut() {
                            let expected = entry.active.then_some(entry.value);
                            assert_eq!(arena.remove(entry.key), expected);
                            entry.active = false;
                        }
                    }
                    3 => {
                        arena.clear();
                        for entry in &mut tracked {
                            entry.active = false;
                        }
                    }
                    _ => unreachable!(),
                }
                encoded /= 4;
                assert_matches_model(&arena, &tracked);
            }
        }
    }

    #[test]
    fn clear_invalidates_live_keys_and_reuses_storage() {
        let mut arena = GenerationalArena::with_capacity(4);
        let a = arena.insert(1);
        let b = arena.insert(2);
        arena.clear();

        assert!(arena.is_empty());
        assert!(!arena.contains(a));
        assert!(!arena.contains(b));

        let c = arena.insert(3);
        assert_eq!(c.index(), a.index());
        assert_eq!(arena.get(c), Some(&3));
    }
}
