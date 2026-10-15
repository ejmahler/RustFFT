use std::any::TypeId;

// Helper function to assert we have the right float type
pub fn assert_f32<T: 'static>() {
    let id_f32 = TypeId::of::<f32>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f32, "Wrong float type, must be f32");
}

// Helper function to assert we have the right float type
pub fn assert_f64<T: 'static>() {
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f64, "Wrong float type, must be f64");
}

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
macro_rules! interleave_complex_f32 {
    ($input:ident, $offset:literal, { $($idx:literal),* }) => {
        [
        $(
            extract_lo_lo_f32($input[$idx], $input[$idx+$offset]),
            extract_hi_hi_f32($input[$idx], $input[$idx+$offset]),
        )*
        ]
    }
}

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
// This statement:
// ```
// let values = separate_interleaved_complex_f32!(input, {0, 2, 4});
// ```
// is equivalent to:
// ```
// let values = [
//    extract_lo_lo_f32(input[0], input[1]),
//    extract_lo_lo_f32(input[2], input[3]),
//    extract_lo_lo_f32(input[4], input[5]),
//    extract_hi_hi_f32(input[0], input[1]),
//    extract_hi_hi_f32(input[2], input[3]),
//    extract_hi_hi_f32(input[4], input[5]),
// ];
macro_rules! separate_interleaved_complex_f32 {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            extract_lo_lo_f32($input[$idx], $input[$idx+1]),
        )*
        $(
            extract_hi_hi_f32($input[$idx], $input[$idx+1]),
        )*
        ]
    }
}
