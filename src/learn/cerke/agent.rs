use std::error::Error;

use cetkaik_full_state_transition::{Config, message::{AfterHalfAcceptance, PureMove}, state::{self, Phase}};
use chrono::Utc;
use rand::{prelude::SliceRandom, thread_rng};
use rand_distr::Distribution;

use super::{brain::QNet, environment::{Action, CerkeEnv}};
use crate::learn::{
    cerke::{
        brain::Brain,
        environment::{self, ActionResult, Environment},
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
    name: String
}

impl CerkeAgent {
    pub fn new() -> Self {
        Self {
            qnet: QNet::new(),
            experience: Memory::new(),
            it: 0,
            name: Utc::now().format("%Y%m%dT%H%M%S").to_string()
        }
    }

    pub fn from_file(path: String) -> Self {
        let mut qnet  = QNet::new();
        qnet.load(&path);
        Self {
            qnet,
            experience: Memory::new(),
            it: 0,
            name: Utc::now().format("%Y%m%dT%H%M%S").to_string()
        }
    }

    fn max_q_sction(&self, env: &Phase, inverted: bool) -> f32 {
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

        if inverted {
            res.into_iter()
                .enumerate()
                .filter_map(|(i, x)| if mask[i] > 0 { Some(-x) } else { None })
                .reduce(f32::min)
                .unwrap()
        } else {
            res.into_iter()
                .enumerate()
                .filter_map(|(i, x)| if mask[i] > 0 { Some(x) } else { None })
                .reduce(f32::max)
                .unwrap()
        }
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

        /*
        // epsilon - greedy 

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
        */
        
        let beta = 2.0f32;
        let items: Vec<f32> = res.iter().enumerate().map(|(i,x)| {
            if mask[i] == 1 {
                f32::exp(beta * x)
            } else {
                0f32
            }
        }).collect();
        let wei = rand::distributions::WeightedIndex::new(items).unwrap();
        let mut rng = thread_rng();
        let max_index = wei.sample(&mut rng);

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

    pub fn parallel_select_action (&self, states: &Vec<Phase> ) -> Vec< (Action, usize) > {
        enum Candidates {
            Start(Vec<PureMove>,Vec<PureMove>),
            AfterCiurl(Vec<AfterHalfAcceptance>),
            Moved
        }

        let mut vecs = Vec::new();
        let mut masks = Vec::new();
        let mut candidates_vec = Vec::new();

        for state in states.iter() {
            vecs.push( state_to_feature(state) );
            masks.push(
                match state {
                Phase::Start(state) => {
                    let (hop1zuo1_candidates, candidates) = state.get_candidates(Config::cerke_online_alpha());
                    let res = candidates_to_mask(&hop1zuo1_candidates, &candidates);
                    candidates_vec.push(Candidates::Start(hop1zuo1_candidates, candidates));
                    res
                },
                Phase::AfterCiurl(state) => {
                    let candidates = state.get_candidates(Config::cerke_online_alpha());
                    let res = afterhalf_candidates_to_mask(&candidates);
                    candidates_vec.push(Candidates::AfterCiurl(candidates));
                    res
                },
                Phase::Moved(_) => {
                    candidates_vec.push(Candidates::Moved);
                    tymok_mask()
                }
            })
        }
        let raw_res =self
            .qnet
            .forward(vecs.iter().map(|x| x.as_slice()).collect::<Vec<&[f32]>>()).unwrap();
        
        let mut result = Vec::new();
        for (i, (res, candidates)) in raw_res.into_iter().zip(candidates_vec).enumerate() {
            let mask = masks[i];

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
            
            result.push( match candidates {
                Candidates::Start(hop1zuo1_candidates, candidates) => { 
                    (
                        Action::Pure(get_candidate_by_index(max_index, &hop1zuo1_candidates, &candidates)),
                        max_index,
                    )
                },
                Candidates::AfterCiurl(candidates) => {
                    (
                        Action::AfterHalf(get_after_half_candidate_by_index(max_index, &candidates)),
                        max_index,
                    )
                },
                Candidates::Moved => {
                    (
                        Action::IsTymok(get_tymok_candidate_by_index(max_index)),
                        max_index
                    )
                },
            })
        } 

        result
    }

    pub fn select_para (&self, environments: Vec<CerkeEnv>) -> Vec<(Action, usize)> {
        let states: Vec<Phase> = environments.iter().map(|environment| environment.observe()).collect();
        self.parallel_select_action(&states)
    }

    pub fn put_memory(&mut self, ex: Experience<Phase, usize>) { 
        self.experience.put(ex)
    }
    pub fn train(&mut self) {         
        let mut update_batch = Vec::new();
        let gamma = 0.99f32;

        for _i in 0..1000 {
            let Experience {
                current_state,
                action,
                next_state,
                value,
            } = self.experience.sample();

            let max_q = self.max_q_sction(next_state, false );
 
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
            let path = format!("./result/{}", self.name);
            self.qnet.save(&path);
        }
    }
}
