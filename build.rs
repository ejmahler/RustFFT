extern crate version_check;

// All platforms except AArch64 with neon support.
#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
static MIN_RUSTC: &str = "1.37.0";
#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
static RUSTFFT_DESC: &str = "RustFFT";

// On AArch64 with neon support enabled.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
static MIN_RUSTC: &str = "1.61.0";
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
static RUSTFFT_DESC: &str = "RustFFT with neon support";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_check::is_min_version(MIN_RUSTC) {
        Some(true) => {}
        Some(false) => panic!(
            "Unsupported rustc version: {}, {} needs at least: {}",
            version_check::Version::read().unwrap(),
            RUSTFFT_DESC,
            MIN_RUSTC
        ),
        None => panic!("Unable to determine rustc version."),
    };
}
