extern crate version_check;

// All platforms except AArch64 with neon support enabled.
static MIN_RUSTC: &str = "1.37.0";
// On AArch64 with neon support enabled.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
static MIN_RUSTC_NEON: &str = "1.61.0";

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
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
