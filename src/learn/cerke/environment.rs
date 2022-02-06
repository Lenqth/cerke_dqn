use cetkaik_full_state_transition::{
    message::{AfterHalfAcceptance, PureMove},
    state, Config, IfTaxot, Victor,
};

use rand_distr::StandardNormal;

pub enum ActionResult {
    Finish(f32),
    Continue(f32),
}

#[derive(Clone, Debug)]
pub enum Phase {
    Start(state::A),
    AfterCiurl(state::C),
    Moved(state::HandNotResolved),
}

impl Phase {
    pub fn whose_turn (&self) -> cetkaik_core::absolute::Side {
        match self {
            Phase::Start(x) => x.whose_turn,
            Phase::AfterCiurl(x) => x.c.whose_turn,
            Phase::Moved(x ) => x.whose_turn,
        }
    }
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

pub const CARTPOLE_THRESHOLD_X: f32 = 2.4;
pub const CARTPOLE_THRESHOLD_POLE: f32 = 24.0 * 2.0 * std::f32::consts::PI / 360.0;

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
                    ActionResult::Continue(0f32)
                }
                Action::Pure(PureMove::NormalMove(action)) => {
                    let res =
                        cetkaik_full_state_transition::apply_normal_move(&state, action, config)
                            .unwrap()
                            .choose()
                            .0;
                    
                    let piece_point = match state.whose_turn {
                        cetkaik_core::absolute::Side::ASide => {
                            (res.f.a_side_hop1zuo1.len() as i32) - (state.f.a_side_hop1zuo1.len() as i32)
                        },
                        cetkaik_core::absolute::Side::IASide => {
                            (res.f.ia_side_hop1zuo1.len() as i32) - (state.f.ia_side_hop1zuo1.len() as i32)
                        },
                    };
                    
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
                    ActionResult::Continue(piece_point as f32 * 0.001f32)
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

                    let piece_point = match state.c.whose_turn {
                        cetkaik_core::absolute::Side::ASide => {
                            (res.f.a_side_hop1zuo1.len() as i32) - (state.c.f.a_side_hop1zuo1.len() as i32)
                        },
                        cetkaik_core::absolute::Side::IASide => {
                            (res.f.ia_side_hop1zuo1.len() as i32) - (state.c.f.ia_side_hop1zuo1.len() as i32)
                        },
                    };

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

                    ActionResult::Continue(piece_point as f32 * 0.001f32)
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
                                ActionResult::Continue(0f32)
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
