use std::num::Int;

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

/// An iterator that skips over `n` elements of `iter`.
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Skip<I> {
    iter: I,
    n: uint
}

#[unstable = "trait is unstable"]
impl<A, I> Iterator<A> for Skip<I> where I: Iterator<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut next = self.iter.next();
        if self.n == 0 {
            next
        } else {
            let mut n = self.n;
            while n > 0 {
                n -= 1;
                match next {
                    Some(_) => {
                        next = self.iter.next();
                        continue
                    }
                    None => {
                        self.n = 0;
                        return None
                    }
                }
            }
            self.n = 0;
            next
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = lower.saturating_sub(self.n);

        let upper = match upper {
            Some(x) => Some(x.saturating_sub(self.n)),
            None => None
        };

        (lower, upper)
    }
}

impl <A, I: ExactSizeIterator<A>> DoubleEndedIterator<A> for Skip<I> {
    fn next_back(&mut self) -> Option<A> {
        panic!("Not implemented");
        None
    }
}

impl<A, I: ExactSizeIterator<A>> ExactSizeIterator<A> for Skip<I> { }

/// Creates an iterator that skips the first `n` elements of this iterator,
/// and then yields all further items.
///
/// # Example
///
/// ```rust
/// let a = [1i, 2, 3, 4, 5];
/// let mut it = a.iter().skip(3);
/// assert_eq!(it.next().unwrap(), &4);
/// assert_eq!(it.next().unwrap(), &5);
/// assert!(it.next().is_none());
/// ```
#[inline]
#[stable]
pub fn myskip<A, I: Iterator<A>>(iter: I, n: uint) -> Skip<I> {
    Skip{iter: iter, n: n}
}
