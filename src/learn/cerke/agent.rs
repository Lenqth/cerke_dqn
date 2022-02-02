use std::error::Error;

use cetkaik_full_state_transition::{
    message::{AfterHalfAcceptance, PureMove},
    state, Config,
};
use rand::{prelude::SliceRandom, thread_rng};
use rand_distr::Distribution;

use super::{brain::QNet, environment::Action};
use crate::learn::{
    cerke::{
        brain::Brain,
        environment::{self, ActionResult, Environment, Phase},
    },
    memory::{Experience, Memory},
    state_to_feature::{
        afterhalf_candidates_to_mask, candidates_to_mask, get_after_half_candidate_by_index,
        get_candidate_by_index, get_tymok_candidate_by_index, state_to_feature, tymok_mask,
        ACTION_SIZE,
    },
};

pub struct CerkeAgent {
    qnet: QNet,
    experience: Memory<Phase, usize>,
    it: i64,
}

impl CerkeAgent {
    pub fn new() -> Self {
        Self {
            qnet: QNet::new(),
            experience: Memory::new(),
            it: 0,
        }
    }

    fn max_q_sction(&self, env: &Phase) -> f32 {
        let vec = state_to_feature(env);
        let res = {
            let mut batch = Vec::new();
            batch.push(vec);
            self.qnet
                .forward(batch.iter().map(|x| x.as_slice()).collect())
                .unwrap()[0]
                .clone()
        };
        let mask = match env {
            Phase::Start(state) => {
                let (hop1zuo1_candidates, candidates) =
                    state.get_candidates(Config::cerke_online_alpha());
                candidates_to_mask(&hop1zuo1_candidates, &candidates)
            }
            Phase::AfterCiurl(state) => {
                let candidates = state.get_candidates(Config::cerke_online_alpha());
                afterhalf_candidates_to_mask(&candidates)
            }
            Phase::Moved(_state) => tymok_mask(),
        };

        res.into_iter()
            .enumerate()
            .filter_map(|(i, x)| if mask[i] > 0 { Some(x) } else { None })
            .reduce(f32::max)
            .unwrap()
    }

    fn select_move(&self, state: &state::A) -> Result<(PureMove, usize), Box<dyn Error>> {
        let (hop1zuo1_candidates, candidates) = state.get_candidates(Config::cerke_online_alpha());
        let mask = candidates_to_mask(&hop1zuo1_candidates, &candidates);
        let state_vec = state_to_feature(&Phase::Start(state.clone()));

        let res = self
            .qnet
            .forward(vec![&state_vec[..]])
            .unwrap()
            .pop()
            .unwrap();

        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0;

        if rand::random::<f32>() < 0.98f32 {
            for (i, v) in res.iter().enumerate() {
                if mask[i] == 1 && max_value < *v {
                    max_index = i;
                    max_value = *v;
                }
            }
        } else {
            let mut candidates = Vec::new();
            for (i, _v) in res.iter().enumerate() {
                if mask[i] == 1 {
                    candidates.push(i);
                }
            }
            max_index = *candidates.choose(&mut thread_rng()).unwrap();
        }

        Ok((
            get_candidate_by_index(max_index, &hop1zuo1_candidates, &candidates),
            max_index,
        ))
    }

    fn select_stepped(
        &self,
        state: &state::C,
    ) -> Result<(AfterHalfAcceptance, usize), Box<dyn Error>> {
        let candidates = state.get_candidates(Config::cerke_online_alpha());
        let mask = afterhalf_candidates_to_mask(&candidates);
        let state_vec = state_to_feature(&Phase::AfterCiurl(state.clone()));

        let res = self
            .qnet
            .forward(vec![&state_vec[..]])
            .unwrap()
            .pop()
            .unwrap();

        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0;

        if rand::random::<f32>() < 0.98f32 {
            for (i, v) in res.iter().enumerate() {
                if mask[i] == 1 && max_value < *v {
                    max_index = i;
                    max_value = *v;
                }
            }
        } else {
            let mut candidates = Vec::new();
            for (i, _v) in res.iter().enumerate() {
                if mask[i] == 1 {
                    candidates.push(i);
                }
            }
            max_index = *candidates.choose(&mut thread_rng()).unwrap();
        }

        Ok((
            get_after_half_candidate_by_index(max_index, &candidates),
            max_index,
        ))
    }

    fn select_tymok(
        &self,
        state: &state::HandNotResolved,
    ) -> Result<(bool, usize), Box<dyn Error>> {
        let mask = tymok_mask();
        let state_vec = state_to_feature(&Phase::Moved(state.clone()));

        let res = self
            .qnet
            .forward(vec![&state_vec[..]])
            .unwrap()
            .pop()
            .unwrap();

        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0;

        if rand::random::<f32>() < 0.98f32 {
            for (i, v) in res.iter().enumerate() {
                if mask[i] == 1 && max_value < *v {
                    max_index = i;
                    max_value = *v;
                }
            }
        } else {
            let mut candidates = Vec::new();
            for (i, _v) in res.iter().enumerate() {
                if mask[i] == 1 {
                    candidates.push(i);
                }
            }
            max_index = *candidates.choose(&mut thread_rng()).unwrap();
        }

        Ok((get_tymok_candidate_by_index(max_index), max_index))
    }

    pub fn select_action(&self, state: &Phase) -> (Action, usize) {
        match state {
            Phase::Start(state) => {
                let (mov, index) = self.select_move(state).unwrap();
                (Action::Pure(mov), index)
            }
            Phase::AfterCiurl(state) => {
                let (mov, index) = self.select_stepped(state).unwrap();
                (Action::AfterHalf(mov), index)
            }
            Phase::Moved(state) => {
                let (mov, index) = self.select_tymok(state).unwrap();
                (Action::IsTymok(mov), index)
            }
        }
    }

    pub fn iteration(&mut self) {
        let mut environment = environment::CerkeEnv::default();

        let mut scores = Vec::new();
        for _i in 0..20 {
            for _turn in 0..100 {
                let env = environment.observe();
                let (act, atc_id) = self.select_action(&env);

                let res = environment.act(act);
                let next_env = environment.observe();
                if let ActionResult::Finish(v) = res {
                    self.experience.put(Experience {
                        current_state: env,
                        next_state: next_env,
                        action: atc_id,
                        value: -1f32,
                    });
                    scores.push(v);
                    break;
                }
                self.experience.put(Experience {
                    current_state: env,
                    next_state: next_env,
                    action: atc_id,
                    value: 0.01f32,
                });
            }
        }
        println!("{} : {}", self.it, scores.into_iter().sum::<f32>());

        let mut update_batch = Vec::new();
        let gamma = 0.998f32;

        for _i in 0..500 {
            let Experience {
                current_state,
                action,
                next_state,
                value,
            } = self.experience.sample();

            let max_q = self.max_q_sction(next_state);
            let new_q = value + gamma * max_q;
            let mut new_q_one_hot = [0f32; ACTION_SIZE];
            let mut mask_one_hot = [0f32; ACTION_SIZE];
            new_q_one_hot[*action] = new_q;
            mask_one_hot[*action] = 1f32;

            update_batch.push((state_to_feature(current_state), new_q_one_hot, mask_one_hot))
        }
        self.qnet
            .train(
                update_batch
                    .iter()
                    .map(|(x, y, m)| (x.as_slice(), y.as_slice(), m.as_slice()))
                    .collect(),
            )
            .expect("Train Failed");

        self.it += 1;
        if self.it % 10 == 9 {
            self.qnet.update_hard();
        }
    }
}
