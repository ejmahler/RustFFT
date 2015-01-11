use std::mem;

#[derive(Clone)]
pub struct Stride<'a, A: 'a> {
    items: &'a [A],
    current_idx: usize,
    stride: usize,
}

impl<'a, A> Stride<'a, A> {
    pub fn from_slice(xs: &'a [A]) -> Self {
        Stride {
            items: xs,
            current_idx: 0,
            stride: 1,
        }
    }

    pub fn stride(&self, stride: usize) -> Stride<'a, A> {
        Stride {
            items: self.items,
            current_idx: self.current_idx,
            stride: self.stride * stride,
        }
    }

    pub fn skip_some(&self, len: usize) -> Stride<'a, A> {
        Stride {
            items: self.items,
            current_idx: self.current_idx + len * self.stride,
            stride: self.stride,
        }
    }

    pub fn take_some(&self, num: usize) -> Stride<'a, A> {
        Stride {
            items: self.items.slice_to((num + 1) * self.stride - 1),
            current_idx: self.current_idx,
            stride: self.stride,
        }
    }
}

impl<'a, A> Iterator for Stride<'a, A> where A: 'a {
    type Item = &'a A;

    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.current_idx >= self.items.len() {
            None
        } else {
            let idx = self.current_idx;
            self.current_idx += self.stride;
            unsafe {
                Some(self.items.get_unchecked(idx))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.items.len() - self.current_idx) / self.stride + 
            if (self.items.len() - self.current_idx) % self.stride > 0 { 1 } else { 0 };
        (len, Some(len))
    }
}

impl<'a, A> DoubleEndedIterator for Stride<'a, A> where A: 'a {
    fn next_back(&mut self) -> Option<&'a A> {
        panic!("Not implemented");
        None
    }
}

impl<'a, A> ExactSizeIterator for Stride<'a, A> where A: 'a { }

pub struct StrideMut<'a, A: 'a> {
    items: &'a mut [A],
    current_idx: usize,
    stride: usize,
}

impl<'a, A> Clone for StrideMut<'a, A> {
    fn clone(&self) -> Self {
        let items_copy = unsafe{ mem::transmute_copy(&self.items) };
        StrideMut {
            items: items_copy,
            current_idx: self.current_idx,
            stride: self.stride,
        }
    }
}

impl<'a, A> StrideMut<'a, A> {
    pub fn from_slice(xs: &'a mut [A]) -> Self {
        StrideMut {
            items: xs,
            current_idx: 0,
            stride: 1,
        }
    }

    pub fn stride(&self, stride: usize) -> StrideMut<'a, A> {
        let items_copy = unsafe{ mem::transmute_copy(&self.items) };
        StrideMut {
            items: items_copy,
            current_idx: self.current_idx,
            stride: self.stride * stride,
        }
    }

    pub fn skip_some(&self, len: usize) -> StrideMut<'a, A> {
        let items_copy = unsafe{ mem::transmute_copy(&self.items) };
        StrideMut {
            items: items_copy,
            current_idx: self.current_idx + len * self.stride,
            stride: self.stride,
        }
    }

    pub fn take_some(&self, num: usize) -> StrideMut<'a, A> {
        let items_copy: &'a mut [A] = unsafe{ mem::transmute_copy(&self.items) };
        StrideMut {
            items: items_copy.slice_to_mut((num + 1) * self.stride - 1),
            current_idx: self.current_idx,
            stride: self.stride,
        }
    }
}

impl<'a, A> Iterator for StrideMut<'a, A> where A: 'a {
    type Item = &'a mut A;

    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.current_idx >= self.items.len() {
            None
        } else {
            let idx = self.current_idx;
            self.current_idx += self.stride;
            unsafe {
                let elt_ref = mem::transmute(self.items.get_unchecked_mut(idx));
                Some(elt_ref)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.items.len() - self.current_idx) / self.stride + 
            if (self.items.len() - self.current_idx) % self.stride > 0 { 1 } else { 0 };
        (len, Some(len))
    }
}

impl<'a, A> DoubleEndedIterator for StrideMut<'a, A> where A: 'a {
    fn next_back(&mut self) -> Option<&'a mut A> {
        panic!("Not implemented");
        None
    }
}

impl<'a, A> ExactSizeIterator for StrideMut<'a, A> where A: 'a { }
