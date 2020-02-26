mod mixed_radix_4xn;
mod mixed_radix_4xn_avx;

pub use self::mixed_radix_4xn::MixedRadix4xN;
pub use self::mixed_radix_4xn_avx::MixedRadix4xnAvx;
pub use self::mixed_radix_4xn_avx::MixedRadix4x4Avx;

use std::marker::PhantomData;

#[allow(unused)]
struct ColumnChunksExact3<'a, T> {
	ptr: *const T,
	end: *const T,
	width: usize,
	remainder: usize,
	chunk_size: usize,
	phantom: PhantomData<&'a T>,
}

impl<'a, T> ColumnChunksExact3<'a, T> {
	#[allow(unused)]
	#[inline]
	pub fn new(buffer: &[T], chunk_size: usize) -> Self {
		let width = buffer.len() / 3;
		let chunk_count = width / chunk_size;
		let items_in_chunks = chunk_count * chunk_size;
		Self {
			ptr: buffer.as_ptr(),
			end: unsafe { buffer.as_ptr().add(items_in_chunks) },
			width,
			remainder: width - items_in_chunks,
			chunk_size,
			phantom: PhantomData,
		}
	}
	#[allow(unused)]
	pub fn into_remainder(self) -> [&'a [T]; 3] {
		let row0_ptr = self.ptr;
		unsafe {
			let row1_ptr = row0_ptr.add(self.width);
			let row2_ptr = row0_ptr.add(self.width*2);
		
			let row0_slice = std::slice::from_raw_parts(row0_ptr, self.remainder);
			let row1_slice = std::slice::from_raw_parts(row1_ptr, self.remainder);
			let row2_slice = std::slice::from_raw_parts(row2_ptr, self.remainder);

			[row0_slice, row1_slice, row2_slice]
		}
	}
}

impl<'a, T> Iterator for ColumnChunksExact3<'a, T> {
	type Item = [&'a [T]; 3];

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.ptr != self.end {
			let row0_ptr = self.ptr;
			unsafe { 
				self.ptr = self.ptr.add(self.chunk_size);

				let row1_ptr = row0_ptr.add(self.width);
				let row2_ptr = row0_ptr.add(self.width*2);

				let row0_slice = std::slice::from_raw_parts(row0_ptr, self.chunk_size);
				let row1_slice = std::slice::from_raw_parts(row1_ptr, self.chunk_size);
				let row2_slice = std::slice::from_raw_parts(row2_ptr, self.chunk_size);

				Some([row0_slice, row1_slice, row2_slice])
			}
		} else {
			None
    	}
    }
}

struct ColumnChunksExactMut4<'a, T> {
	ptr: *mut T,
	end: *mut T,
	width: usize,
	remainder: usize,
	chunk_size: usize,
	phantom: PhantomData<&'a mut T>,
}

impl<'a, T> ColumnChunksExactMut4<'a, T> {
	#[inline]
	pub fn new(buffer: &mut [T], chunk_size: usize) -> Self {
		let width = buffer.len() / 4;
		let chunk_count = width / chunk_size;
		let items_in_chunks = chunk_count * chunk_size;
		Self {
			ptr: buffer.as_mut_ptr(),
			end: unsafe { buffer.as_mut_ptr().add(items_in_chunks) },
			width,
			remainder: width - items_in_chunks,
			chunk_size,
			phantom: PhantomData,
		}
	}

	pub fn into_remainder(self) -> [&'a mut [T]; 4] {
		let row0_ptr = self.ptr;
		unsafe {
			let row1_ptr = row0_ptr.add(self.width);
			let row2_ptr = row0_ptr.add(self.width*2);
			let row3_ptr = row0_ptr.add(self.width*3);
		
			let row0_slice = std::slice::from_raw_parts_mut(row0_ptr, self.remainder);
			let row1_slice = std::slice::from_raw_parts_mut(row1_ptr, self.remainder);
			let row2_slice = std::slice::from_raw_parts_mut(row2_ptr, self.remainder);
			let row3_slice = std::slice::from_raw_parts_mut(row3_ptr, self.remainder);

			[row0_slice, row1_slice, row2_slice, row3_slice]
		}
	}
}

impl<'a, T> Iterator for ColumnChunksExactMut4<'a, T> {
	type Item = [&'a mut [T]; 4];

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.ptr != self.end {
			let row0_ptr = self.ptr;
			unsafe { 
				self.ptr = self.ptr.add(self.chunk_size);

				let row1_ptr = row0_ptr.add(self.width);
				let row2_ptr = row0_ptr.add(self.width*2);
				let row3_ptr = row0_ptr.add(self.width*3);

				let row0_slice = std::slice::from_raw_parts_mut(row0_ptr, self.chunk_size);
				let row1_slice = std::slice::from_raw_parts_mut(row1_ptr, self.chunk_size);
				let row2_slice = std::slice::from_raw_parts_mut(row2_ptr, self.chunk_size);
				let row3_slice = std::slice::from_raw_parts_mut(row3_ptr, self.chunk_size);

				Some([row0_slice, row1_slice, row2_slice, row3_slice])
			}
		} else {
			None
    	}
    }
}