use std::hash::{Hash, Hasher};

use fnv::FnvHasher;

pub(crate) fn hash_key(key: &str) -> u64 {
    let mut hasher = FnvHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}
