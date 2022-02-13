#![feature(vec_retain_mut)]
pub mod learn;

use std::sync::{Arc, Mutex};

use cetkaik_full_state_transition::{Config, state::Phase};
use lazy_static::lazy_static;
use learn::cerke::environment::Action;

use crate::learn::cerke::agent::CerkeAgent;

lazy_static! {
    static ref agent: Arc<Mutex<CerkeAgent>> = Arc::new(
        Mutex::new(
            CerkeAgent::from_file("ai/".to_string())
        )
    );
}

pub fn bot_action(state: Phase, _confin: Config) -> Action {
    let (action, _) = agent.lock().unwrap().select_action(&state);
    action
}