#![feature(vec_retain_mut)]

use learn::cerke::agent::CerkeAgent;

pub mod learn;

fn main() {
    let mut cp = CerkeAgent::new();

    for _i in 0..10000 {
        cp.iteration();
    }
}

#[test]
fn test_cuda_available() {
    assert!(tch::Cuda::is_available());
}
