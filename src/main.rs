use std::time::Instant;

use cerke_dqn::learn::cerke::agent::CerkeAgent;
use cerke_dqn::learn::cerke::environment::ParallelCerke;

fn main() {
    let mut cp = CerkeAgent::new();

    let now = Instant::now();
    for i in 0..10000 {
        let mut env = ParallelCerke::new();
        env.iteration(&mut cp);
        
        let elapsed_time = now.elapsed();
        println!("{} : {} sec", i + 1, elapsed_time.as_secs_f64());
    }
}

#[test]
fn test_cuda_available() {
    assert!(tch::Cuda::is_available());
}
