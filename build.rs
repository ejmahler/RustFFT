extern crate version_check;

static MIN_RUSTC: &str = "1.61.0";

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
