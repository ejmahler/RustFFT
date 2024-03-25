use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use handlebars;
use serde::Serialize;

#[derive(Serialize)]
struct FftEntry {
    len: usize,
    twiddle_len: usize,
    loadstore_indexes: String,
    loadstore_indexes_2x: String,
    shuffle_in_str: String,
    shuffle_out_str: String,
    impl_str: String,
    struct_name_32: String,
    struct_name_64: String,
}

#[derive(Serialize)]
struct Architecture {
    name_snakecase: &'static str,
    name_camelcase: &'static str,
    name_display: &'static str,
    array_trait: &'static str,
    vector_trait: &'static str,
    vector_f32: &'static str,
    vector_f64: &'static str,
    cpu_feature_name: &'static str,
    has_dynamic_cpu_features: bool,
    dynamic_cpu_feature_macro: &'static str,
    arch_include: &'static str,
    test_attribute: &'static str,
    extra_test_includes: Vec<&'static str>,
}

#[derive(Serialize)]
struct Context {
    command_str: String,
    arch: Architecture,
    lengths: Vec<FftEntry>,
}

const USAGE_STR: &'static str =
    "Usage: {executable} sse|wasm_simd|neon lengths [--check <filename>]";

fn main() -> Result<(), Box<dyn Error>> {
    let (arch, lengths, check_filename) = parse_args()?;

    // build the context object for the template
    let lengths_as_str: Vec<String> = lengths.iter().map(|l| l.to_string()).collect();
    let context = Context {
        command_str: format!(
            "cargo run --manifest-path ./tools/gen_simd_butterflies/Cargo.toml -- {} {}",
            arch.name_snakecase,
            lengths_as_str.join(" ")
        ),
        lengths: lengths
            .into_iter()
            .map(|i| generate_fft_entry(i, &arch))
            .collect(),
        arch,
    };

    // load and render the template
    let prime_template = include_str!("templates/prime_template.hbs.rs");

    let mut handlebars = handlebars::Handlebars::new();
    handlebars.register_escape_fn(handlebars::no_escape);
    handlebars.register_template_string("prime_template", prime_template)?;

    let rendered = handlebars.render("prime_template", &context)?;

    // if we're in check mode, verify that the provided file matches the rendered output. otherwise just print the rendered output
    if let Some(filename) = check_filename {
        let mut check_text = String::new();
        File::open(filename)?.read_to_string(&mut check_text)?;

        // check line by line so we don't have to worry about mayching line endings
        for (i, (check_line, rendered_line)) in check_text.lines().zip(rendered.lines()).enumerate()
        {
            if check_line != rendered_line {
                eprintln!(
                    "Mismatch on line {i}

Script Output:  {rendered_line}
Existing File:  {check_line}
"
                );
                return Err(
                    "SIMD prime FFT autogeneration script output does not match existing file contents"
                        .into(),
                );
            }
        }
    } else {
        println!("{rendered}");
    }

    Ok(())
}

// manual argument parsing. we could use a library for this but our requirements are very simple
fn parse_args() -> Result<(Architecture, Vec<usize>, Option<PathBuf>), Box<dyn Error>> {
    let mut arg_iter = std::env::args();

    // skip first arg
    arg_iter.next();

    // next arg is arch
    let arch = parse_architecture(arg_iter.next())?;

    // the next several arguments should be integers. Keep parsing arguments until we run out or find a non-integer
    // todo: can this be an iterator? the hard part is keeping next_arg alive when the loop ends so we can use it afterwards
    let mut next_arg;
    let mut lengths = Vec::new();
    loop {
        next_arg = arg_iter.next();

        let next_arg_str = match next_arg.as_ref() {
            Some(arg) => arg,
            None => break,
        };

        let len = match next_arg_str.parse::<usize>() {
            Ok(len) => len,
            _ => break,
        };

        lengths.push(len);
    }

    // If the next argument is "--check", we're going to operate in checm ode instead of generate mode
    // the argument after that will be the filename to check
    let check_filename = if next_arg.map_or(false, |arg| arg == "--check") {
        let path_str = arg_iter.next().ok_or_else(|| USAGE_STR.to_owned())?;
        Some(path_str.try_into()?)
    } else {
        None
    };

    Ok((arch, lengths, check_filename))
}

fn parse_architecture(arch_str: Option<String>) -> Result<Architecture, String> {
    if let Some(arch_str) = arch_str {
        if arch_str == "sse" {
            return Ok(Architecture {
                name_snakecase: "sse",
                name_camelcase: "Sse",
                name_display: "SSE",
                array_trait: "SseArrayMut",
                vector_trait: "SseVector",
                vector_f32: "__m128",
                vector_f64: "__m128d",
                cpu_feature_name: "sse4.1",
                has_dynamic_cpu_features: true,
                dynamic_cpu_feature_macro: "std::arch::is_x86_feature_detected",
                arch_include: "use core::arch::x86_64::{__m128, __m128d};",
                test_attribute: "test",
                extra_test_includes: Vec::new(),
            });
        } else if arch_str == "wasm_simd" {
            return Ok(Architecture {
                name_snakecase: "wasm_simd",
                name_camelcase: "WasmSimd",
                name_display: "Wasm SIMD",
                array_trait: "WasmSimdArrayMut",
                vector_trait: "WasmVector",
                vector_f32: "WasmVector32",
                vector_f64: "WasmVector64",
                cpu_feature_name: "simd128",
                has_dynamic_cpu_features: false,
                dynamic_cpu_feature_macro: "",
                arch_include: "",
                test_attribute: "wasm_bindgen_test",
                extra_test_includes: vec!["use wasm_bindgen_test::wasm_bindgen_test;"],
            });
        } else if arch_str == "neon" {
            return Ok(Architecture {
                name_snakecase: "neon",
                name_camelcase: "Neon",
                name_display: "NEON",
                array_trait: "NeonArrayMut",
                vector_trait: "NeonVector",
                vector_f32: "float32x4_t",
                vector_f64: "float64x2_t",
                cpu_feature_name: "neon",
                has_dynamic_cpu_features: true,
                dynamic_cpu_feature_macro: "std::arch::is_aarch64_feature_detected",
                arch_include: "use core::arch::aarch64::{float32x4_t, float64x2_t};",
                test_attribute: "test",
                extra_test_includes: vec![],
            });
        }
    }

    Err(USAGE_STR.to_owned())
}

fn generate_fft_entry(len: usize, arch: &Architecture) -> FftEntry {
    let vector_trait = arch.vector_trait;

    // generate the in-shuffle sequence for f32 parallel FFTs
    let shuffle_in_str = {
        let indent = "            ";
        let halflen = len / 2;
        let lenm1 = len - 1;

        let mut shuffle_in_strs = Vec::with_capacity(len);
        for i in 0..halflen {
            shuffle_in_strs.push(format!(
                "{indent}extract_lo_hi_f32(input_packed[{i}], input_packed[{}]),",
                i + halflen
            ));
            shuffle_in_strs.push(format!(
                "{indent}extract_hi_lo_f32(input_packed[{i}], input_packed[{}]),",
                i + halflen + 1
            ));
        }
        shuffle_in_strs.push(format!(
            "{indent}extract_lo_hi_f32(input_packed[{halflen}], input_packed[{lenm1}]),"
        ));
        shuffle_in_strs.join("\n")
    };

    // generate the out-shuffle sequence for f32 parallel FFTs
    let shuffle_out_str = {
        let indent = "            ";
        let halflen = len / 2;
        let lenm1 = len - 1;

        let mut shuffle_out_strs = Vec::with_capacity(len);
        for i in 0..halflen {
            shuffle_out_strs.push(format!(
                "{indent}extract_lo_lo_f32(out[{}], out[{}]),",
                2 * i,
                2 * i + 1
            ));
        }
        shuffle_out_strs.push(format!("{indent}extract_lo_hi_f32(out[{lenm1}], out[0]),"));
        for i in 0..halflen {
            shuffle_out_strs.push(format!(
                "{indent}extract_hi_hi_f32(out[{}], out[{}]),",
                2 * i + 1,
                2 * i + 2
            ));
        }
        shuffle_out_strs.join("\n")
    };

    // generate the impl string. For now, this will be the same for both f32 and f64
    let impl_str = {
        let indent = "        ";
        let halflen = (len + 1) / 2;
        let lenm1 = len - 1;

        let mut impl_strs = Vec::with_capacity(len * len);
        impl_strs.push(format!(
            "{indent}let rotate = {vector_trait}::make_rotate90(FftDirection::Inverse);"
        ));
        impl_strs.push(String::new());

        // butterfly2's down the inputs, and rotate the subtraction half of the butterfly 2's
        // todo: when we get FCMA, we can conditionally skip the rotations here!
        impl_strs.push(format!("{indent}let y00 = values[0];"));
        for n in 1..halflen {
            let nrev = len - n;
            impl_strs.push(format!("{indent}let [x{n}p{nrev}, x{n}m{nrev}] =  {vector_trait}::column_butterfly2([values[{n}], values[{nrev}]]);"));
            impl_strs.push(format!(
                "{indent}let x{n}m{nrev} = {vector_trait}::apply_rotate90(rotate, x{n}m{nrev});"
            ));
            impl_strs.push(format!(
                "{indent}let y00 = {vector_trait}::add(y00, x{n}p{nrev});"
            ));
        }
        impl_strs.push(String::new());

        // the meat of the FFT: an O(n^2) pile of FMA's. Even if this instruction set doesn't have FMA,
        // we'll still express it like it does and the ones that don't have it will just internally do separate mul and adds.
        // That way we don't have to implement it differently for different platforms
        for n in 1..halflen {
            let nrev = len - n;
            let first_twiddle = n - 1;
            let variable_name_a = format!("m{n:02}{nrev:02}a");

            impl_strs.push(format!("{indent}let {variable_name_a} = {vector_trait}::fmadd(values[0], self.twiddles_re[{first_twiddle}], x1p{lenm1});"));
            for m in 2..halflen {
                let mrev = len - m;
                let mn = (m * n) % len;
                let tw_idx = if mn > len / 2 { len - mn - 1 } else { mn - 1 };
                impl_strs.push(format!("{indent}let {variable_name_a} = {vector_trait}::fmadd({variable_name_a}, self.twiddles_re[{tw_idx}], x{m}p{mrev});"));
            }

            let variable_name_b = format!("m{n:02}{nrev:02}b");
            impl_strs.push(format!("{indent}let {variable_name_b} = {vector_trait}::mul(self.twiddles_im[{first_twiddle}], x1m{lenm1});"));
            for m in 2..halflen {
                let mrev = len - m;
                let mn = (m * n) % len;
                let tw_idx = if mn > len / 2 { len - mn - 1 } else { mn - 1 };
                let func = if mn > len / 2 { "nmadd" } else { "fmadd" };

                impl_strs.push(format!("{indent}let {variable_name_b} = {vector_trait}::{func}({variable_name_b}, self.twiddles_im[{tw_idx}], x{m}m{mrev});"));
            }
            impl_strs.push(format!("{indent}let [y{n:02}, y{nrev:02}] = {vector_trait}::column_butterfly2([{variable_name_a}, {variable_name_b}]);"));
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
        loadstore_indexes: (0..len)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
        loadstore_indexes_2x: (0..len)
            .map(|i| (i * 2).to_string())
            .collect::<Vec<_>>()
            .join(","),
        shuffle_in_str,
        shuffle_out_str,
        impl_str,
        struct_name_32: format!("{}F32Butterfly{len}", arch.name_camelcase),
        struct_name_64: format!("{}F64Butterfly{len}", arch.name_camelcase),
    }
}
