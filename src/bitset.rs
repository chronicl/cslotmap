#[derive(Default)]
pub struct Bitset {
    data: Vec<u64>,
}

impl Bitset {
    fn word_and_offset(bit: u32) -> (usize, u32) {
        ((bit / 64) as usize, bit % 64)
    }

    pub fn get(&self, bit: u32) -> bool {
        let (word, offset) = Self::word_and_offset(bit);

        if word >= self.data.len() {
            return false;
        }

        (self.data[word] & (1 << offset)) != 0
    }

    /// Returns true if the bit was NOT previously already set.
    pub fn set(&mut self, bit: u32) -> bool {
        let (word, offset) = Self::word_and_offset(bit);

        if word >= self.data.len() {
            self.data.resize(word + 1, 0);
        }

        let before = self.data[word];
        self.data[word] |= 1 << offset;
        before != self.data[word]
    }

    pub fn unset(&mut self, bit: u32) -> bool {
        let (word, offset) = Self::word_and_offset(bit);

        if word >= self.data.len() {
            return false;
        }

        let before = self.data[word];
        self.data[word] &= !(1 << offset);
        before != self.data[word]
    }
}
