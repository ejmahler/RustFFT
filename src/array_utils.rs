/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
/// benchmarking shows that loop tiling isn't effective for small arrays (in the range of 50x50 or smaller)
pub unsafe fn transpose_small<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

#[allow(unused)]
pub unsafe fn workaround_transmute<T, U>(slice: &[T]) -> &[U] {
    let ptr = slice.as_ptr() as *const U;
    let len = slice.len();
    std::slice::from_raw_parts(ptr, len)
}
#[allow(unused)]
pub unsafe fn workaround_transmute_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    let ptr = slice.as_mut_ptr() as *mut U;
    let len = slice.len();
    std::slice::from_raw_parts_mut(ptr, len)
}

#[derive(Copy, Clone)]
pub struct RawSlice<T> {
    ptr: *const T,
    slice_len: usize,
}
impl<T> RawSlice<T> {
    #[inline(always)]
    pub fn new(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub unsafe fn new_transmuted<U>(slice: &[U]) -> Self {
        Self {
            ptr: slice.as_ptr() as *const T,
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice_len
    }
}
impl<T: Copy> RawSlice<T> {
    #[inline(always)]
    pub unsafe fn load(&self, index: usize) -> T {
        debug_assert!(index < self.slice_len);
        *self.ptr.add(index)
    }
}

/// A RawSliceMut is a normal mutable slice, but aliasable. Its functionality is severely limited.
#[derive(Copy, Clone)]
pub struct RawSliceMut<T> {
    ptr: *mut T,
    slice_len: usize,
}
impl<T> RawSliceMut<T> {
    #[inline(always)]
    pub fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub unsafe fn new_transmuted<U>(slice: &mut [U]) -> Self {
        Self {
            ptr: slice.as_mut_ptr() as *mut T,
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice_len
    }
    #[inline(always)]
    pub unsafe fn store(&self, value: T, index: usize) {
        debug_assert!(index < self.slice_len);
        *self.ptr.add(index) = value;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::random_signal;
    use num_complex::Complex;
    use num_traits::Zero;

    #[test]
    fn test_transpose() {
        let sizes: Vec<usize> = (1..16).collect();

        for &width in &sizes {
            for &height in &sizes {
                let len = width * height;

                let input: Vec<Complex<f32>> = random_signal(len);
                let mut output = vec![Zero::zero(); len];

                unsafe { transpose_small(width, height, &input, &mut output) };

                for x in 0..width {
                    for y in 0..height {
                        assert_eq!(
                            input[x + y * width],
                            output[y + x * height],
                            "x = {}, y = {}",
                            x,
                            y
                        );
                    }
                }
            }
        }
    }
}

// Loop over exact chunks of the provided buffer. Very similar in semantics to ChunksExactMut, but generates smaller code and requires no modulo operations
// Returns Ok() if every element ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks<T>(
    mut buffer: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T]),
) -> Result<(), ()> {
    // Loop over the buffer, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer.len() >= chunk_size {
        let (head, tail) = buffer.split_at_mut(chunk_size);
        buffer = tail;

        chunk_fn(head);
    }

    // We have a remainder if there's data still in the buffer -- in which case we want to indicate to the caller that there was an unwanted remainder
    if buffer.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

// Loop over exact zipped chunks of the 2 provided buffers. Very similar in semantics to ChunksExactMut.zip(ChunksExactMut), but generates smaller code and requires no modulo operations
// Returns Ok() if every element of both buffers ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks_zipped<T>(
    mut buffer1: &mut [T],
    mut buffer2: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T], &mut [T]),
) -> Result<(), ()> {
    // If the two buffers aren't the same size, record the fact that they're different, then snip them to be the same size
    let uneven = if buffer1.len() > buffer2.len() {
        buffer1 = &mut buffer1[..buffer2.len()];
        true
    } else if buffer2.len() < buffer1.len() {
        buffer2 = &mut buffer2[..buffer1.len()];
        true
    } else {
        false
    };

    // Now that we know the two slices are the same length, loop over each one, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer1.len() >= chunk_size && buffer2.len() >= chunk_size {
        let (head1, tail1) = buffer1.split_at_mut(chunk_size);
        buffer1 = tail1;

        let (head2, tail2) = buffer2.split_at_mut(chunk_size);
        buffer2 = tail2;

        chunk_fn(head1, head2);
    }

    // We have a remainder if the 2 chunks were uneven to start with, or if there's still data in the buffers -- in which case we want to indicate to the caller that there was an unwanted remainder
    if !uneven && buffer1.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}
