#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RadixFactor {
    Factor2 = 2,
    Factor3 = 3,
    Factor4 = 4,
    Factor5 = 5,
    Factor6 = 6,
    Factor7 = 7,
}
impl RadixFactor {
    pub const fn radix(&self) -> usize {
        *self as usize
    }
}
