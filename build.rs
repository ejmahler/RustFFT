extern crate version_check;

#[cfg(feature = "neon")]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_check::is_min_version("1.61.0") {
        Some(true) => {}
        Some(false) => panic!(
            "Unsupported rustc version: {}, RustFFT with 'neon' support needs at least: 1.61.0",
            version_check::Version::read().unwrap()
        ),
        None => panic!("Unable to determine rustc version."),
    };
}

#[cfg(not(feature = "neon"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_check::is_min_version("1.37.0") {
        Some(true) => {}
        Some(false) => panic!(
            "Unsupported rustc version: {}, RustFFT needs at least: 1.37.0",
            version_check::Version::read().unwrap()
        ),
        None => panic!("Unable to determine rustc version."),
    };
}