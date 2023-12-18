#![no_main]
#![no_std]

extern crate alloc;

use alloc::vec;
use rustfft::{num_complex::Complex, FftPlanner};

use panic_halt as _;
use stm32f4xx_hal as hal;

use hal::{pac, prelude::*};
use cortex_m_rt::entry;
use embedded_alloc::Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[entry]
fn main() -> ! {
    {
        // Init allocator
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 48 * 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(1234);

    let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; 1234];

    // Set up the LED
    let mut led = {
        let pac = pac::Peripherals::take().unwrap();
        let gpioa = pac.GPIOA.split();
        gpioa.pa5.into_push_pull_output()
    };

    loop {
        led.toggle();
        fft.process(&mut buffer);
    }
}
