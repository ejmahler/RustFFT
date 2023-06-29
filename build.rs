extern crate version_check;

// All platforms except AArch64 with neon support enabled.
static MIN_RUSTC: &str = "1.61.0";
// On AArch64 with neon support enabled.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
static MIN_RUSTC_NEON: &str = "1.61.0";

#[cfg(all(target_arch = "wasm32", feature = "wasm_simd"))]
static MIN_RUSTC_WASM_SIMD: &str = "1.61.0";

#[cfg(not(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "wasm_simd")
)))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_check::is_min_version(MIN_RUSTC) {
        Some(true) => {}
        Some(false) => panic!(
            "\n====\nUnsupported rustc version {}\nRustFFT needs at least {}\n====\n",
            version_check::Version::read().unwrap(),
            MIN_RUSTC
        ),
        None => panic!("Unable to determine rustc version."),
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_check::is_min_version(MIN_RUSTC_NEON) {
        Some(true) => {}
        Some(false) => panic!(
            "\n====\nUnsupported rustc version {}\nRustFFT with neon support needs at least {}\nIf the 'neon' feature flag is disabled, the minimum version is {}\n====\n",
            version_check::Version::read().unwrap(),
            MIN_RUSTC_NEON,
            MIN_RUSTC
        ),
        None => panic!("Unable to determine rustc version."),
    };
}

// Weird bug with wasm-pack, may not work. It may use host information instead, i. e. target_arch = "aarch64" for M1
#[cfg(all(target_arch = "wasm32", feature = "wasm_simd"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_check::is_min_version(MIN_RUSTC_WASM_SIMD) {
        Some(true) => {}
        Some(false) => panic!(
            "\n====\nUnsupported rustc version {}\nRustFFT with WASM SIMD support needs at least {}\nIf the 'neon' feature flag is disabled, the minimum version is {}\n====\n",
            version_check::Version::read().unwrap(),
            MIN_RUSTC_WASM_SIMD,
            MIN_RUSTC
        ),
        None => panic!("Unable to determine rustc version."),
    };
}
