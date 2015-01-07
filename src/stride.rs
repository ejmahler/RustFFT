use std::ops::{Index, IndexMut};
use std::cmp::min;

#[derive(Clone)]
pub struct Stride<'a, I: Index<uint> + 'a> {
    items: I,
    base_len: uint,
    current_idx: uint,
    stride: uint,
}

impl<'a, I: Index<uint> + Clone + 'a> Stride<'a, I> {
    pub fn stride_trivial(xs: I, len: uint) -> Self {
        Stride {
            items: xs,
            base_len: len,
            current_idx: 0,
            stride: 1,
        }
    }

    pub fn stride(&self, stride: uint) -> Stride<'a, I> {
        Stride {
            items: self.items.clone(),
            base_len: self.base_len,
            current_idx: self.current_idx,
            stride: self.stride * stride,
        }
    }

    pub fn skip_some(&self, len: uint) -> Stride<'a, I> {
        Stride {
            items: self.items.clone(),
            base_len: self.base_len,
            current_idx: self.current_idx + len * self.stride,
            stride: self.stride,
        }
    }

    pub fn take_some(&self, num: uint) -> Stride<'a, I> {
        Stride {
            items: self.items.clone(),
            base_len: min(self.base_len, num * self.stride),
            current_idx: self.current_idx,
            stride: self.stride,
        }
    }
}

impl<'a, I> Iterator for Stride<'a, I> where I: Index<uint> {
    type Item = &'a <I as Index<uint>>::Output;

    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if (self.current_idx >= self.base_len) {
            None
        } else {
            let idx = self.current_idx;
            self.current_idx += self.stride;
            Some(self.items.index(&idx))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let len = (self.base_len - self.current_idx) / self.stride + 
            if (self.base_len - self.current_idx) % self.stride > 0 { 1 } else { 0 };
        (len, Some(len))
    }
}

impl<'a, I> DoubleEndedIterator for Stride<'a, I> where I: Index<uint> {
    fn next_back(&mut self) -> Option<&'a <I as Index<uint>>::Output> {
        panic!("Not implemented");
        None
    }
}

impl<'a, I> ExactSizeIterator for Stride<'a, I> where I: Index<uint> { }

#[derive(Clone)]
pub struct StrideMut<'a, I: IndexMut<uint> + 'a> {
    items: I,
    base_len: uint,
    current_idx: uint,
    stride: uint,
}

impl<'a, I: IndexMut<uint> + Clone + 'a> StrideMut<'a, I> {
    pub fn stride_trivial(xs: I, len: uint) -> Self {
        StrideMut {
            items: xs,
            base_len: len,
            current_idx: 0,
            stride: 1,
        }
    }

    pub fn stride(&self, stride: uint) -> StrideMut<'a, I> {
        StrideMut {
            items: self.items.clone(),
            base_len: self.base_len,
            current_idx: self.current_idx,
            stride: self.stride * stride,
        }
    }

    pub fn skip_some(&self, len: uint) -> StrideMut<'a, I> {
        StrideMut {
            items: self.items.clone(),
            base_len: self.base_len,
            current_idx: self.current_idx + len * self.stride,
            stride: self.stride,
        }
    }

    pub fn take_some(&self, num: uint) -> StrideMut<'a, I> {
        StrideMut {
            items: self.items.clone(),
            base_len: min(self.base_len, num * self.stride),
            current_idx: self.current_idx,
            stride: self.stride,
        }
    }
}

impl<'a, I> Iterator for StrideMut<'a, I> where I: IndexMut<uint> {
    type Item = &'a mut <I as IndexMut<uint>>::Output;

    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.current_idx >= self.base_len {
            None
        } else {
            let idx = self.current_idx;
            self.current_idx += self.stride;
            Some(self.items.index_mut(&idx))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let len = (self.base_len - self.current_idx) / self.stride + 
            if (self.base_len - self.current_idx) % self.stride > 0 { 1 } else { 0 };
        (len, Some(len))
    }
}

impl<'a, I> DoubleEndedIterator for StrideMut<'a, I> where I: IndexMut<uint> {
    fn next_back(&mut self) -> Option<&'a mut <I as IndexMut<uint>>::Output> {
        panic!("Not implemented");
        None
    }
}

impl<'a, I> ExactSizeIterator for StrideMut<'a, I> where I: IndexMut<uint> { }
