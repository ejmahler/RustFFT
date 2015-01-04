#[derive(Clone)]
pub struct Stride<A, I> {
    iter: I,
    stride: uint,
}

impl<A, I: Iterator<A>> Iterator<A> for Stride<A, I> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let ret = self.iter.next();
        if self.stride > 1 {
            self.iter.nth(self.stride - 2);
        }
        ret
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.stride > 0 {
            match self.iter.size_hint() {
                (lower, None) => (lower / self.stride, None),
                (lower, Some(upper)) => (lower / self.stride, Some(upper / self.stride))
            }
        } else {
            self.iter.size_hint()
        }
    }
}

impl <A, I: ExactSizeIterator<A>> DoubleEndedIterator<A> for Stride<A, I> {
    fn next_back(&mut self) -> Option<A> {
        panic!("Not implemented");
        None
    }
}

impl <A, I: ExactSizeIterator<A>> ExactSizeIterator<A> for Stride<A, I> { }

pub fn stride<A, I: Iterator<A>>(iter: I, stride: uint) -> Stride<A, I> {
    Stride { iter: iter, stride: stride }
}
