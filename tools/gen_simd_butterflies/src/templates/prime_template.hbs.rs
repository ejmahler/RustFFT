{{arch.arch_include}}
use std::any::TypeId;
use std::sync::Arc;
use num_complex::Complex;

use crate::{common::FftNum, FftDirection};

use crate::array_utils::DoubleBuf;
use crate::twiddles;
use crate::{Direction, Fft, Length};

use super::{{arch.name_snakecase}}_common::{assert_f32, assert_f64};
use super::{{arch.name_snakecase}}_utils::*;
use super::{{arch.name_snakecase}}_vector::*;

/* 
This file contains autogenerated butterfly algorithms for small prime-sized FFTs using the {{arch.name_display}} instruction set.

NOTE: All of the code in this file was **autogenerated** using the following command in the project root:
    {{command_str}}
    
Do not make changes directly to this file. Instead, update the autogeneration script 
(Located at tools/gen_simd_butterflies/src/main.rs) and then re-run the script to generate the new code.

For these sizes, we use a variant of the naive DFT algorithm. Even though this algorithm is O(n^2),
we're able to factor out some redundant computations, and they up being faster than the fancier algorithms.

To generate more or fewer butterfly sizes, simply add or remove numbers from the command above and re-run.
The code generation script will even handle adding or removing sizes from the planner, all you need to do is run the script.
*/

pub const fn prime_butterfly_lens() -> &'static [usize] {
    &[{{#each lengths }}{{this.len}}, {{/each}}]
}

/// Safety: The current machine must support the {{arch.cpu_feature_name}} target feature
#[target_feature(enable = "{{arch.cpu_feature_name}}")]
pub unsafe fn construct_prime_butterfly<T: FftNum>(len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();
    if id_t == id_f32 {
        match len {
            {{#each lengths}}
            {{this.len}} => Arc::new({{this.struct_name_32}}::new(direction)) as Arc<dyn Fft<T>>,
            {{/each}}
            _ => unimplemented!("Invalid {{arch.name_display}} prime butterfly length: {len}"),
        }
    } else if id_t == id_f64 {
        match len {
            {{#each lengths}}
            {{this.len}} => Arc::new({{this.struct_name_64}}::new(direction)) as Arc<dyn Fft<T>>,
            {{/each}}
            _ => unimplemented!("Invalid {{arch.name_display}} prime butterfly length: {len}"),
        }
    } else {
        unimplemented!("Not f32 or f64");
    }
}

#[inline(always)]
fn make_twiddles<const TW: usize, T: FftNum>(len: usize, direction: FftDirection) -> [Complex<T>; TW] {
    let mut i = 1;
    [(); TW].map(|_| {
        let twiddle = twiddles::compute_twiddle(i, len, direction);
        i += 1;
        twiddle
    })
}

{{#each lengths}}
struct {{this.struct_name_32}}<T> {
    direction: FftDirection,
    twiddles_re: [{{../arch.vector_f32}}; {{this.twiddle_len}}],
    twiddles_im: [{{../arch.vector_f32}}; {{this.twiddle_len}}],
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_{{../arch.name_snakecase}}_f32_butterfly!({{this.struct_name_32}}, {{this.len}}, |this: &{{this.struct_name_32}}<_>| this.direction);
impl<T: FftNum> {{this.struct_name_32}}<T> {
    /// Safety: The current machine must support the {{../arch.cpu_feature_name}} instruction set
    #[target_feature(enable = "{{../arch.cpu_feature_name}}")]
    unsafe fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let twiddles = make_twiddles({{this.len}}, direction);
        Self {
            direction,
            twiddles_re: twiddles.map(|t| {{../arch.vector_trait}}::broadcast_scalar(t.re)),
            twiddles_im: twiddles.map(|t| {{../arch.vector_trait}}::broadcast_scalar(t.im)),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl {{../arch.array_trait}}<f32>) {
        let values = read_partial1_complex_to_array!(buffer, { {{ this.loadstore_indexes }} });

        let out = self.perform_parallel_fft_direct(values);
        
        write_partial_lo_complex_to_array!(out, buffer, { {{ this.loadstore_indexes }} } ); 
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(&self, mut buffer: impl {{../arch.array_trait}}<f32>) {
        let input_packed = read_complex_to_array!(buffer, { {{ this.loadstore_indexes_2x }} });

        let values = [
{{ this.shuffle_in_str }}
        ];

        let out = self.perform_parallel_fft_direct(values);

        let out_packed = [
{{ this.shuffle_out_str }}
        ];

        write_complex_to_array_strided!(out_packed, buffer, 2, { {{ this.loadstore_indexes }} });
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [{{../arch.vector_f32}}; {{this.len}}]) -> [{{../arch.vector_f32}}; {{this.len}}] {
{{this.impl_str}}
    }
}

struct {{this.struct_name_64}}<T> {
    direction: FftDirection,
    twiddles_re: [{{../arch.vector_f64}}; {{this.twiddle_len}}],
    twiddles_im: [{{../arch.vector_f64}}; {{this.twiddle_len}}],
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_{{../arch.name_snakecase}}_f64_butterfly!({{this.struct_name_64}}, {{this.len}}, |this: &{{this.struct_name_64}}<_>| this.direction);
impl<T: FftNum> {{this.struct_name_64}}<T> {
    /// Safety: The current machine must support the {{../arch.cpu_feature_name}} instruction set
    #[target_feature(enable = "{{../arch.cpu_feature_name}}")]
    unsafe fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let twiddles = make_twiddles({{this.len}}, direction);
        unsafe {Self {
            direction,
            twiddles_re: twiddles.map(|t| {{../arch.vector_trait}}::broadcast_scalar(t.re)),
            twiddles_im: twiddles.map(|t| {{../arch.vector_trait}}::broadcast_scalar(t.im)),
            _phantom: std::marker::PhantomData,
        }}
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl {{../arch.array_trait}}<f64>) {
        let values = read_complex_to_array!(buffer, { {{ this.loadstore_indexes }} });

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, { {{ this.loadstore_indexes }} });   
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [{{../arch.vector_f64}}; {{this.len}}]) -> [{{../arch.vector_f64}}; {{this.len}}] {
{{this.impl_str}}
    }
}

{{/each}}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;
    {{#each arch.extra_test_includes }}
    {{this}}
    {{/each}}

    macro_rules! test_butterfly_32_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[{{arch.test_attribute}}]
            fn $test_name() {
                {{#if arch.has_dynamic_cpu_features}}
                assert!({{arch.dynamic_cpu_feature_macro}}!("{{arch.cpu_feature_name}}"));
                {{/if}}

                let fwd = unsafe { $struct_name::new(FftDirection::Forward) };
                check_fft_algorithm::<f32>(&fwd, $size, FftDirection::Forward);

                let inv = unsafe { $struct_name::new(FftDirection::Inverse) }; 
                check_fft_algorithm::<f32>(&inv, $size, FftDirection::Inverse);
            }
        };
    }
    macro_rules! test_butterfly_64_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[{{arch.test_attribute}}]
            fn $test_name() {
                {{#if arch.has_dynamic_cpu_features}}
                assert!({{arch.dynamic_cpu_feature_macro}}!("{{arch.cpu_feature_name}}"));
                {{/if}}

                let fwd = unsafe { $struct_name::new(FftDirection::Forward) };
                check_fft_algorithm::<f64>(&fwd, $size, FftDirection::Forward);

                let inv = unsafe { $struct_name::new(FftDirection::Inverse) };
                check_fft_algorithm::<f64>(&inv, $size, FftDirection::Inverse);
            }
        };
    }
    {{#each lengths }}
    test_butterfly_32_func!(test_{{../arch.name_snakecase}}f32_butterfly{{this.len}}, {{this.struct_name_32}}, {{this.len}});
    {{/each}}
    {{#each lengths }}
    test_butterfly_64_func!(test_{{../arch.name_snakecase}}f64_butterfly{{this.len}}, {{this.struct_name_64}}, {{this.len}});
    {{/each}}
}
