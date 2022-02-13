use cetkaik_core::absolute::NonTam2Piece;
use cetkaik_full_state_transition::{Config, IfTaxot, Scores, Victor, message::{AfterHalfAcceptance, PureMove}, state::{self, Phase}};

use rand_distr::StandardNormal;

use crate::learn::memory::Experience;

use super::agent::CerkeAgent;

pub enum ActionResult {
    Finish(f32),
    Continue,
}

pub enum Action {
    Pure(PureMove),
    AfterHalf(AfterHalfAcceptance),
    IsTymok(bool),
}

pub trait Environment {
    type Observable;
    type Action;

    fn observe(&self) -> Self::Observable;
    fn act(&mut self, action: Self::Action) -> ActionResult;
}

#[derive(Debug, Clone)]
pub struct CerkeEnv {
    state: Phase,
}

impl Default for CerkeEnv {
    fn default() -> Self {
        Self::random_new()
    }
}

impl CerkeEnv {
    fn random_new() -> Self {
        let _rng = rand::thread_rng();
        let _sn = StandardNormal;
        Self {
            state: Phase::Start(cetkaik_full_state_transition::initial_state().choose().0),
        }
    }
}

impl Environment for CerkeEnv {
    type Observable = Phase;
    type Action = Action;

    fn observe(&self) -> Phase {
        self.state.clone()
    }

    fn act(&mut self, action: Action) -> ActionResult {
        let config = Config::cerke_online_alpha();

        match &self.state {
            Phase::Start(state) => match action {
                Action::Pure(PureMove::InfAfterStep(action)) => {
                    let res =
                        cetkaik_full_state_transition::apply_inf_after_step(&state, action, config)
                            .unwrap()
                            .choose()
                            .0;
                    self.state = Phase::AfterCiurl(res);
                    ActionResult::Continue
                }
                Action::Pure(PureMove::NormalMove(action)) => {
                    let res =
                        cetkaik_full_state_transition::apply_normal_move(&state, action, config)
                            .unwrap()
                            .choose()
                            .0;
                    
                    let resolved = cetkaik_full_state_transition::resolve(&res, config);
                    self.state = match resolved {
                        state::HandResolved::NeitherTymokNorTaxot(s) => Phase::Start(s),
                        state::HandResolved::GameEndsWithoutTymokTaxot(victor) => {
                            let previous_score = match state.whose_turn {
                                cetkaik_core::absolute::Side::ASide => state.scores.a(),
                                cetkaik_core::absolute::Side::IASide => state.scores.ia(),
                            };
                            let next_score = (match victor {
                                Victor(Some(cetkaik_core::absolute::Side::ASide)) => 20,
                                Victor(Some(cetkaik_core::absolute::Side::IASide)) => -20,
                                Victor(None) => 0,
                            } * match state.whose_turn {
                                cetkaik_core::absolute::Side::ASide => 1,
                                cetkaik_core::absolute::Side::IASide => -1,
                            });
                            return ActionResult::Finish(
                                (next_score - previous_score) as f32,
                            );
                        }
                        _ => Phase::Moved(res),
                    };
                    ActionResult::Continue
                }
                _ => unreachable!(),
            },
            Phase::AfterCiurl(state) => match action {
                Action::AfterHalf(action) => {
                    let res = cetkaik_full_state_transition::apply_after_half_acceptance(
                        &state, action, config,
                    )
                    .unwrap()
                    .choose()
                    .0;

                    let resolved = cetkaik_full_state_transition::resolve(&res, config);
                    self.state = match resolved {
                        state::HandResolved::NeitherTymokNorTaxot(s) => Phase::Start(s),
                        state::HandResolved::GameEndsWithoutTymokTaxot(victor) => {
                            let previous_score = match state.c.whose_turn {
                                cetkaik_core::absolute::Side::ASide => state.c.scores.a(),
                                cetkaik_core::absolute::Side::IASide => state.c.scores.ia(),
                            };
                            let next_score = (match victor {
                                Victor(Some(cetkaik_core::absolute::Side::ASide)) => 20,
                                Victor(Some(cetkaik_core::absolute::Side::IASide)) => -20,
                                Victor(None) => 0,
                            } * match state.c.whose_turn {
                                cetkaik_core::absolute::Side::ASide => 1,
                                cetkaik_core::absolute::Side::IASide => -1,
                            });
                            return ActionResult::Finish(
                                (next_score - previous_score) as f32,
                            );
                        }
                        _ => Phase::Moved(res),
                    };

                    ActionResult::Continue
                }
                _ => unreachable!(),
            },
            Phase::Moved(state) => match action {
                Action::IsTymok(tymok) => {
                    let resolved = cetkaik_full_state_transition::resolve(&state, config);
                    let previous_score = match state.whose_turn {
                        cetkaik_core::absolute::Side::ASide => state.scores.a(),
                        cetkaik_core::absolute::Side::IASide => state.scores.ia(),
                    };

                    match resolved {
                        state::HandResolved::NeitherTymokNorTaxot(_) => unreachable!(),
                        state::HandResolved::HandExists { if_tymok, if_taxot } => {
                            if tymok {
                                self.state = Phase::Start(if_tymok);
                                ActionResult::Continue
                            } else {
                                match if_taxot {
                                    IfTaxot::NextSeason(s) => {
                                        let next_score = match state.whose_turn {
                                            cetkaik_core::absolute::Side::ASide => {
                                                s.choose().0.scores.a()
                                            }
                                            cetkaik_core::absolute::Side::IASide => {
                                                s.choose().0.scores.ia()
                                            }
                                        };
                                        ActionResult::Finish(
                                            (next_score - previous_score) as f32,
                                        )
                                    }
                                    IfTaxot::VictoriousSide(_s) => {
                                        ActionResult::Finish((20 - previous_score) as f32)
                                    }
                                }
                            }
                        }
                        state::HandResolved::GameEndsWithoutTymokTaxot(victor) => {
                            let next_score = (match victor {
                                Victor(Some(cetkaik_core::absolute::Side::ASide)) => 20,
                                Victor(Some(cetkaik_core::absolute::Side::IASide)) => -20,
                                Victor(None) => 0,
                            } * match state.whose_turn {
                                cetkaik_core::absolute::Side::ASide => 1,
                                cetkaik_core::absolute::Side::IASide => -1,
                            });
                            ActionResult::Finish((next_score - previous_score) as f32)
                        }
                    }
                }
                _ => unreachable!(),
            },
        }
    }
}

pub struct ParallelCerke {
    envs: Vec<CerkeEnv>
}

impl ParallelCerke{
    pub fn new() -> Self {
        let mut environments = Vec::with_capacity(100);
        for i in 0..100 {
            environments.push(CerkeEnv::default());
        } 
        Self {
            envs: environments
        }
    }

    fn score_delta(x: &Phase, y: &Phase) -> f32 {
        let score_delta = x.get_score() - y.get_score();
        if x.get_season() != y.get_season() {
            score_delta as f32
        } else {
            score_delta as f32 + ((y.ia_side_hop1zuo1().len() as i32) - (x.ia_side_hop1zuo1().len() as i32) + (x.a_side_hop1zuo1().len() as i32) - (y.a_side_hop1zuo1().len() as i32)) as f32 * 0.001f32
        }
    }

    pub fn iteration(&mut self, agent: &mut CerkeAgent) {
        let mut last_state: (Vec<Option<Phase>>, Vec<Option<Phase>>) = (Vec::new(),Vec::new());
        let mut finished = Vec::new();
        for _i in 0..self.envs.len() {
            last_state.0.push(None);
            last_state.1.push(None);
            finished.push(false);
        }

        for _turn in 0..40 {
            let states: Vec<Phase> = self.envs.iter().map(|environment| environment.observe()).collect();

            let mut actions: Vec<Option<(Action, usize)>> = agent.parallel_select_action(&states).into_iter().map(Some).collect();

            let mut states: Vec<Option<Phase>> = states.into_iter().map(Some).collect();

            for index in 0..self.envs.len() {
                if finished[index] {
                    continue;
                }
                let mut environment = self.envs.get_mut(index).unwrap();
                let prev_env = states.get_mut(index).unwrap().take().unwrap();
                let (act, atc_id) = actions.get_mut(index).unwrap().take().unwrap();

                let last_state = match prev_env.whose_turn() {
                    cetkaik_core::absolute::Side::ASide => &mut last_state.0,
                    cetkaik_core::absolute::Side::IASide => &mut last_state.1,
                }.get_mut(index).unwrap();
                
                let res = environment.act(act);

                if let Some(last_state) = last_state {
                    let v = Self::score_delta(&prev_env,&last_state);
                    let v = match last_state.whose_turn() {
                        cetkaik_core::absolute::Side::ASide => v,
                        cetkaik_core::absolute::Side::IASide => -v,
                    };
                    agent.put_memory(Experience {
                        current_state: last_state.clone(),
                        next_state: prev_env.clone(),
                        action: atc_id,
                        value: v,
                    });
                }
                *last_state = Some(prev_env);

                finished[index] = match res {
                    ActionResult::Finish(v) => {
                        true
                    },
                    ActionResult::Continue => {
                        false
                    },
                };
            }

        }
        agent.train();
    }
}