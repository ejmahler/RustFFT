use std::error::Error;

use serde::Serialize;
use tinytemplate::TinyTemplate;

#[derive(Serialize)]
struct FftEntry {
    len: usize,
    twiddle_len: usize,
    loadstore_indexes: String,
    loadstore_indexes_2x: String,
    shuffle_in_str: String,
    shuffle_out_str: String,
    impl_str: String,
}

#[derive(Serialize)]
struct Context {
    command_str: String,
    lengths: Vec<FftEntry>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw_sse_template = include_str!("templates/sse_template.tt.rs");
    let processed_template = preprocess_template(raw_sse_template);

    let mut tt = TinyTemplate::new();
    tt.add_template("simd_template", &processed_template)?;

    let arg_list : Vec<String> = std::env::args().skip(1).collect();
    let lengths : Vec<FftEntry> = arg_list
        .iter()
        .map_while(|arg| arg.parse::<usize>().ok())
        .map(generate_fft_entry)
        .collect();
    let lengths_as_str : Vec<String> = lengths.iter().map(|l| l.len.to_string()).collect();

    let context = Context {
        command_str: format!("cargo run --manifest-path ./tools/gen_simd_butterflies/Cargo.toml -- {}", lengths_as_str.join(" ")),
        lengths,
    };

    let rendered = tt.render("simd_template", &context)?;
    println!("{rendered}");

    Ok(())
}

fn generate_fft_entry(len: usize) -> FftEntry {
    // generate the in-shuffle sequence for f32 parallel FFTs
    let shuffle_in_str = {
        let indent = "            ";
        let halflen = len/2;
        let lenm1 = len - 1;

        let mut shuffle_in_strs = Vec::with_capacity(len);
        for i in 0..halflen {
            shuffle_in_strs.push(format!("{indent}extract_lo_hi_f32(input_packed[{i}], input_packed[{}]),", i + halflen));
            shuffle_in_strs.push(format!("{indent}extract_hi_lo_f32(input_packed[{i}], input_packed[{}]),", i + halflen + 1));
        }
        shuffle_in_strs.push(format!("{indent}extract_lo_hi_f32(input_packed[{halflen}], input_packed[{lenm1}]),"));
        shuffle_in_strs.join("\n")
    };

    // generate the out-shuffle sequence for f32 parallel FFTs
    let shuffle_out_str = {
        let indent = "            ";
        let halflen = len/2;
        let lenm1 = len - 1;

        let mut shuffle_out_strs = Vec::with_capacity(len);
        for i in 0..halflen {
            shuffle_out_strs.push(format!("{indent}extract_lo_lo_f32(out[{}], out[{}]),", 2 * i, 2 * i + 1));
        }
        shuffle_out_strs.push(format!("{indent}extract_lo_hi_f32(out[{lenm1}], out[0]),"));
        for i in 0..halflen {
            shuffle_out_strs.push(format!("{indent}extract_hi_hi_f32(out[{}], out[{}]),", 2 * i + 1, 2 * i + 2));
        }
        shuffle_out_strs.join("\n")
    };

    // generate the impl string. For now, this will be the same for both f32 and f64
    let impl_str = {
        let indent = "        ";
        let halflen = (len + 1)/2;
        let lenm1 = len - 1;

        let mut impl_strs = Vec::with_capacity(len * len);
        impl_strs.push(format!("{indent}let rotate = SseVector::make_rotate90(FftDirection::Inverse);"));
        impl_strs.push(String::new());

        // butterfly2's down the inputs, and rotate the subtraction half of the butterfly 2's
        // todo: when we get FCMA, we can conditionally skip the rotations here!
        impl_strs.push(format!("{indent}let y00 = values[0];"));
        for n in 1..halflen {
            let nrev = len - n;
            impl_strs.push(format!("{indent}let [x{n}p{nrev}, x{n}m{nrev}] =  SseVector::column_butterfly2([values[{n}], values[{nrev}]]);"));
            impl_strs.push(format!("{indent}let x{n}m{nrev} = SseVector::apply_rotate90(rotate, x{n}m{nrev});"));
            impl_strs.push(format!("{indent}let y00 = SseVector::add(y00, x{n}p{nrev});"));
        };
        impl_strs.push(String::new());

        // the meat of the FFT: an O(n^2) pile of FMA's. Even if this instruction set doesn't have FMA,
        // we'll still express it like it does and the ones that don't have it will just internally do separate mul and adds. 
        // That way we don't have to implement it differently for different platforms
        for n in 1..halflen {
            let nrev = len - n;
            let first_twiddle = n - 1;
            let variable_name_a = format!("m{n:02}{nrev:02}a");

            impl_strs.push(format!("{indent}let {variable_name_a} = SseVector::fmadd(values[0], self.twiddles_re[{first_twiddle}], x1p{lenm1});"));
            for m in 2..halflen {
                let mrev = len - m;
                let mn = (m*n)%len;
                let tw_idx = if mn > len/2 { len - mn - 1 } else { mn - 1 };
                impl_strs.push(format!("{indent}let {variable_name_a} = SseVector::fmadd({variable_name_a}, self.twiddles_re[{tw_idx}], x{m}p{mrev});"));
            }

            let variable_name_b = format!("m{n:02}{nrev:02}b");
            impl_strs.push(format!("{indent}let {variable_name_b} = SseVector::mul(self.twiddles_im[{first_twiddle}], x1m{lenm1});"));
            for m in 2..halflen {
                let mrev = len - m;
                let mn = (m*n)%len;
                let tw_idx = if mn > len/2 { len - mn - 1 } else { mn - 1 };
                let func = if mn > len/2 { "nmadd" } else { "fmadd" };

                impl_strs.push(format!("{indent}let {variable_name_b} = SseVector::{func}({variable_name_b}, self.twiddles_im[{tw_idx}], x{m}m{mrev});"));

            }
            impl_strs.push(format!("{indent}let [y{n:02}, y{nrev:02}] = SseVector::column_butterfly2([{variable_name_a}, {variable_name_b}]);"));
            impl_strs.push(String::new());
        }
        impl_strs.push(String::new());

        // last step is to just return an array containing our output variables
        let out_strs = (0..len).map(|i| format!("y{i:02}")).collect::<Vec<_>>();
        impl_strs.push(format!("{indent}[{}]", out_strs.join(", ")));

        impl_strs.join("\n")
    };

    FftEntry {
        len,
        twiddle_len: len / 2,
        loadstore_indexes: (0..len).map(|i| i.to_string()).collect::<Vec<_>>().join(","),
        loadstore_indexes_2x: (0..len).map(|i| (i * 2).to_string()).collect::<Vec<_>>().join(","),
        shuffle_in_str,
        shuffle_out_str,
        impl_str,
    }
}   

// We're using tinytemplate, which has pretty much exactly the feature set we need, with one wrinkle:
// tinytemplate uses { and } as template characters, meaning any { in the original text needs to be escaped
// Since we're outputting rust code, we have { all over the place, which would be a nightmare to escape
// So we're creating our own DSL where { doesn't need to be escaped, and tinytemplate tokens need to be prepended with $
// This function converts that DSL into the format tinytemplate is expecting.
//
// Mechanically speaking, the DSL is drop dead simple: Whenever we see a '$' followed by a sequence of '{', those '{'
// will be passed through and the '$' will be omitted. All other '{' will be escaped, and all other '$' will be passed through. That's it.
fn preprocess_template(text: &str) -> String {
    // we're going to be escaping a bunch of characters, so we may need a little extra memory
    let mut result = String::with_capacity(text.len() * 11 / 10);

    let mut suppression_count : Option<usize> = None;
    for character in text.chars() {
        // Whenever we find a $, we want to suppress escaping for any subsequent curly braces
        if character == '$' {
            if suppression_count.is_some() {
                todo!("Multiple $ characters in a row aren't supported, nor are sequences of alternating between $ and {{");
            }
            suppression_count = Some(0);
        } else if character == '{' {
            // If escaping is currently suppressed, just push the brace as-is so tinytemplate will use it. Otherwise, escape it so tinytemplate passes it through
            if let Some(count) = &mut suppression_count {
                *count += 1;
                result.push('{');
            } else {
                result.push_str("\\{");
            }
        } else {
            // If we get here, it's because we found a character that wasn't a $ and wasn't a {. So if we're suppressing escapes, stop.
            if let Some(count) = suppression_count {
                // If we didn't actually suppress any escapes, then this is a false positive. We didn't putput the $ up above, so do it now.
                if count == 0 {
                    result.push('$');
                }
                suppression_count = None;
            }

            // now push whatever character this was
            result.push(character);
        }
    }

    result
}
