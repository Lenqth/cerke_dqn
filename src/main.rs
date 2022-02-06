#![feature(vec_retain_mut)]

use std::time::Instant;

use learn::cerke::agent::CerkeAgent;

pub mod learn;

fn main() {
    let mut cp = CerkeAgent::new();

    let now = Instant::now();
    for i in 0..10000 {
        cp.iteration();
        
        let elapsed_time = now.elapsed();
        println!("{} : {} sec", i + 1, elapsed_time.as_secs_f64());
    }
}

#[test]
fn test_cuda_available() {
    assert!(tch::Cuda::is_available());
}
