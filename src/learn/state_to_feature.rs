use std::collections::HashMap;

use cetkaik_core::{
    absolute::{Coord, NonTam2Piece, Piece, Side},
    Color,
};
use cetkaik_full_state_transition::{message::{
    AfterHalfAcceptance, InfAfterStep, NormalMove, PureMove,
}, state::Phase};

pub const STATE_SIZE: usize = 42 * 81 + 2 * 2 * (2 + 9 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 2);
pub const ACTION_SIZE: usize = 20 * 81 + 81 * 81 + 81 + 3; // hand + normal move + half_acceptance + pass + tymok + taxot

fn coord_to_num(coord: &Coord) -> usize {
    (match coord.0 {
        cetkaik_core::absolute::Row::A => 0,
        cetkaik_core::absolute::Row::E => 1,
        cetkaik_core::absolute::Row::I => 2,
        cetkaik_core::absolute::Row::U => 3,
        cetkaik_core::absolute::Row::O => 4,
        cetkaik_core::absolute::Row::Y => 5,
        cetkaik_core::absolute::Row::AI => 6,
        cetkaik_core::absolute::Row::AU => 7,
        cetkaik_core::absolute::Row::IA => 8,
        _ => unreachable!(),
    }) * 9usize
        + match coord.1 {
            cetkaik_core::absolute::Column::K => 0,
            cetkaik_core::absolute::Column::L => 1,
            cetkaik_core::absolute::Column::N => 2,
            cetkaik_core::absolute::Column::T => 3,
            cetkaik_core::absolute::Column::Z => 4,
            cetkaik_core::absolute::Column::X => 5,
            cetkaik_core::absolute::Column::C => 6,
            cetkaik_core::absolute::Column::M => 7,
            cetkaik_core::absolute::Column::P => 8,
            _ => unreachable!(),
        }
}

fn num_to_coord(index: &usize) -> Coord {
    let p = index / 9;
    let q = index % 9;
    Coord(
        match p {
            0 => cetkaik_core::absolute::Row::A,
            1 => cetkaik_core::absolute::Row::E,
            2 => cetkaik_core::absolute::Row::I,
            3 => cetkaik_core::absolute::Row::U,
            4 => cetkaik_core::absolute::Row::O,
            5 => cetkaik_core::absolute::Row::Y,
            6 => cetkaik_core::absolute::Row::AI,
            7 => cetkaik_core::absolute::Row::AU,
            8 => cetkaik_core::absolute::Row::IA,
            _ => unreachable!(),
        },
        match q {
            0 => cetkaik_core::absolute::Column::K,
            1 => cetkaik_core::absolute::Column::L,
            2 => cetkaik_core::absolute::Column::N,
            3 => cetkaik_core::absolute::Column::T,
            4 => cetkaik_core::absolute::Column::Z,
            5 => cetkaik_core::absolute::Column::X,
            6 => cetkaik_core::absolute::Column::C,
            7 => cetkaik_core::absolute::Column::M,
            8 => cetkaik_core::absolute::Column::P,
            _ => unreachable!(),
        },
    )
}

fn piece_to_num(piece: &Piece, player_side: &Side) -> usize {
    match piece {
        Piece::Tam2 => 41,
        Piece::NonTam2Piece { color, prof, side } => {
            (if *color == Color::Huok2 { 10 } else { 0 })
                + (if side == player_side { 20 } else { 0 })
                + (match prof {
                    cetkaik_core::Profession::Nuak1 => 0,
                    cetkaik_core::Profession::Kauk2 => 1,
                    cetkaik_core::Profession::Gua2 => 2,
                    cetkaik_core::Profession::Kaun1 => 3,
                    cetkaik_core::Profession::Dau2 => 4,
                    cetkaik_core::Profession::Maun1 => 5,
                    cetkaik_core::Profession::Kua2 => 6,
                    cetkaik_core::Profession::Tuk2 => 7,
                    cetkaik_core::Profession::Uai1 => 8,
                    cetkaik_core::Profession::Io => 9,
                    _ => unreachable!(),
                })
        }
    }
}

fn nontam_piece_to_num(piece: &NonTam2Piece, is_player_side: &bool) -> usize {
    let NonTam2Piece { color, prof } = piece;
    (if *color == Color::Huok2 { 10 } else { 0 })
        + (if *is_player_side { 20 } else { 0 })
        + (match prof {
            cetkaik_core::Profession::Nuak1 => 0,
            cetkaik_core::Profession::Kauk2 => 1,
            cetkaik_core::Profession::Gua2 => 2,
            cetkaik_core::Profession::Kaun1 => 3,
            cetkaik_core::Profession::Dau2 => 4,
            cetkaik_core::Profession::Maun1 => 5,
            cetkaik_core::Profession::Kua2 => 6,
            cetkaik_core::Profession::Tuk2 => 7,
            cetkaik_core::Profession::Uai1 => 8,
            cetkaik_core::Profession::Io => 9,
            _ => unreachable!(),
        })
}

fn num_to_nontam_piece(index: &usize) -> NonTam2Piece {
    let color = if *index > 10usize {
        Color::Huok2
    } else {
        Color::Kok1
    };
    let prof = match *index % 10usize {
        0 => cetkaik_core::Profession::Nuak1,
        1 => cetkaik_core::Profession::Kauk2,
        2 => cetkaik_core::Profession::Gua2,
        3 => cetkaik_core::Profession::Kaun1,
        4 => cetkaik_core::Profession::Dau2,
        5 => cetkaik_core::Profession::Maun1,
        6 => cetkaik_core::Profession::Kua2,
        7 => cetkaik_core::Profession::Tuk2,
        8 => cetkaik_core::Profession::Uai1,
        9 => cetkaik_core::Profession::Io,
        _ => unreachable!(),
    };
    NonTam2Piece { color, prof }
}

const KEY_TO_OFFSET: [usize; 20] = [
    0, 2, 11, 14, 17, 20, 23, 26, 29, 31, 33, 35, 44, 47, 50, 53, 56, 59, 62, 64,
];

fn get_piece_key(piece: &NonTam2Piece) -> usize {
    let NonTam2Piece { color, prof } = piece;
    (if *color == Color::Huok2 { 10 } else { 0 })
        + (match prof {
            cetkaik_core::Profession::Nuak1 => 0,
            cetkaik_core::Profession::Kauk2 => 1,
            cetkaik_core::Profession::Gua2 => 2,
            cetkaik_core::Profession::Kaun1 => 3,
            cetkaik_core::Profession::Dau2 => 4,
            cetkaik_core::Profession::Maun1 => 5,
            cetkaik_core::Profession::Kua2 => 6,
            cetkaik_core::Profession::Tuk2 => 7,
            cetkaik_core::Profession::Uai1 => 8,
            cetkaik_core::Profession::Io => 9,
        })
}

fn state_a_to_feature(
    board: &HashMap<Coord, Piece>,
    a_side_hop1zuo1: &Vec<NonTam2Piece>,
    ia_side_hop1zuo1: &Vec<NonTam2Piece>,
    whose_turn: &Side,
) -> [f32; STATE_SIZE] {
    /*  42 * 81 + 2 * 2 * (
        2 + 9 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 2
    ) */

    let mut res = [0f32; STATE_SIZE];
    let (turn_player, nonturn_player) = match whose_turn {
        Side::ASide => (a_side_hop1zuo1, ia_side_hop1zuo1),
        Side::IASide => (ia_side_hop1zuo1, a_side_hop1zuo1),
    };
    let mut hm = [0usize; 81];
    for (c, p) in board.iter() {
        hm[coord_to_num(c)] = piece_to_num(p, &whose_turn);
    }
    for i in 0..81 {
        res[i * 42 + hm[i]] = 1f32;
    }

    let offset = 42 * 81;
    let mut hm = [0usize; 20];
    for p in turn_player.iter() {
        hm[get_piece_key(p)] += 1;
    }
    for i in 0..20 {
        res[offset + KEY_TO_OFFSET[i] + hm[i]] = 1f32;
    }

    let mut hm = [0usize; 20];
    for p in nonturn_player.iter() {
        hm[get_piece_key(p)] += 1;
    }

    let offset = 42 * 81 + 2 * (2 + 9 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 2);
    for i in 0..20 {
        res[offset + KEY_TO_OFFSET[i] + hm[i]] = 1f32;
    }

    res
}

pub fn state_to_feature(state: &Phase) -> [f32; STATE_SIZE] {
    match state {
        Phase::Start(state) => state_a_to_feature(
            &state.f.board,
            &state.f.a_side_hop1zuo1,
            &state.f.ia_side_hop1zuo1,
            &state.whose_turn,
        ),
        Phase::AfterCiurl(state) => state_a_to_feature(
            &state.c.f.board,
            &state.c.f.a_side_hop1zuo1,
            &state.c.f.ia_side_hop1zuo1,
            &state.c.whose_turn,
        ),
        Phase::Moved(state) => state_a_to_feature(
            &state.f.board,
            &state.f.a_side_hop1zuo1,
            &state.f.ia_side_hop1zuo1,
            &state.whose_turn,
        ),
    }
}

#[test]
fn test_init_state() {
    let (mut e, _) = cetkaik_full_state_transition::initial_state().choose();
    e.whose_turn = Side::IASide;
    let v = state_to_feature(&Phase::Start(e));
    let v: Vec<usize> = v
        .iter()
        .enumerate()
        .filter(|(_i, x)| **x != 0f32)
        .map(|(i, _x)| i)
        .collect();
    let expect = vec![
        26, 67, 107, 154, 177, 218, 255, 299, 342, 385, 422, 462, 508, 546, 612, 630, 694, 741,
        777, 799, 861, 883, 924, 967, 1029, 1051, 1113, 1134, 1176, 1218, 1260, 1302, 1344, 1386,
        1428, 1470, 1512, 1554, 1596, 1638, 1721, 1722, 1764, 1806, 1848, 1890, 1932, 1974, 2016,
        2058, 2100, 2142, 2184, 2226, 2299, 2321, 2383, 2405, 2466, 2489, 2551, 2573, 2635, 2683,
        2720, 2730, 2806, 2814, 2870, 2898, 2952, 2999, 3040, 3081, 3121, 3168, 3231, 3272, 3309,
        3353, 3396, 3402, 3404, 3413, 3416, 3419, 3422, 3425, 3428, 3431, 3433, 3435, 3437, 3446,
        3449, 3452, 3455, 3458, 3461, 3464, 3466, 3470, 3472, 3481, 3484, 3487, 3490, 3493, 3496,
        3499, 3501, 3503, 3505, 3514, 3517, 3520, 3523, 3526, 3529, 3532, 3534,
    ];
    assert_eq!(v.len(), STATE_SIZE);
    assert_eq!(v, expect);
}

pub fn candidates_to_mask(
    hop1zuo1_candidates: &Vec<PureMove>,
    candidates: &Vec<PureMove>,
) -> [i8; ACTION_SIZE] {
    use cetkaik_full_state_transition::message::{NormalMove, PureMove};
    let mut mask = [0; ACTION_SIZE];

    for c in candidates.iter() {
        match c {
            PureMove::InfAfterStep(c) => {
                let InfAfterStep {
                    planned_direction: _,
                    src: _,
                    step: _,
                } = c;
            }
            PureMove::NormalMove(c) => match c {
                NormalMove::NonTamMoveSrcDst { src, dest } => {
                    mask[coord_to_num(src) * 81 + coord_to_num(dest)] = 1;
                }
                NormalMove::NonTamMoveSrcStepDstFinite { src, step: _, dest } => {
                    mask[coord_to_num(src) * 81 + coord_to_num(dest)] = 1;
                }
                NormalMove::NonTamMoveFromHopZuo {
                    color: _,
                    prof: _,
                    dest: _,
                } => {
                    unreachable!();
                }
                NormalMove::TamMoveNoStep {
                    src,
                    first_dest: _,
                    second_dest,
                } => {
                    mask[coord_to_num(src) * 81 + coord_to_num(second_dest)] = 1;
                }
                NormalMove::TamMoveStepsDuringFormer {
                    src,
                    step: _,
                    first_dest: _,
                    second_dest,
                } => {
                    mask[coord_to_num(src) * 81 + coord_to_num(second_dest)] = 1;
                }
                NormalMove::TamMoveStepsDuringLatter {
                    src,
                    step: _,
                    first_dest: _,
                    second_dest,
                } => {
                    mask[coord_to_num(src) * 81 + coord_to_num(second_dest)] = 1;
                }
            },
        }
    }

    let offset = 81 * 81;

    for c in hop1zuo1_candidates.iter() {
        match c {
            PureMove::NormalMove(NormalMove::NonTamMoveFromHopZuo { color, prof, dest }) => {
                mask[offset
                    + nontam_piece_to_num(
                        &NonTam2Piece {
                            color: *color,
                            prof: *prof,
                        },
                        &false,
                    ) * 81
                    + coord_to_num(dest)];
            }
            _ => unreachable!(),
        }
    }

    mask
}

pub fn get_candidate_by_index(
    index: usize,
    _hop1zuo1_candidates: &Vec<PureMove>,
    candidates: &Vec<PureMove>,
) -> PureMove {
    if index < 81 * 81 {
        let c_src = num_to_coord(&(index / 81));
        let c_dest = num_to_coord(&(index % 81));
        for c in candidates.iter() {
            match c {
                PureMove::InfAfterStep(mov) => {
                    let InfAfterStep {
                        planned_direction,
                        src,
                        step: _,
                    } = mov;
                    if src == &c_src && planned_direction == &c_dest {
                        return c.clone();
                    }
                }
                PureMove::NormalMove(mov) => match mov {
                    NormalMove::NonTamMoveSrcDst { src, dest } => {
                        if src == &c_src && dest == &c_dest {
                            return c.clone();
                        }
                    }
                    NormalMove::NonTamMoveSrcStepDstFinite { src, step: _, dest } => {
                        if src == &c_src && dest == &c_dest {
                            return c.clone();
                        }
                    }
                    NormalMove::NonTamMoveFromHopZuo {
                        color: _,
                        prof: _,
                        dest: _,
                    } => {
                        unreachable!();
                    }
                    NormalMove::TamMoveNoStep {
                        src,
                        first_dest: _,
                        second_dest,
                    } => {
                        if src == &c_src && second_dest == &c_dest {
                            return c.clone();
                        }
                    }
                    NormalMove::TamMoveStepsDuringFormer {
                        src,
                        step: _,
                        first_dest: _,
                        second_dest,
                    } => {
                        if src == &c_src && second_dest == &c_dest {
                            return c.clone();
                        }
                    }
                    NormalMove::TamMoveStepsDuringLatter {
                        src,
                        step: _,
                        first_dest: _,
                        second_dest,
                    } => {
                        if src == &c_src && second_dest == &c_dest {
                            return c.clone();
                        }
                    }
                },
            }
        }
        panic!()
    } else {
        let index = index - 81 * 81;
        let piece = num_to_nontam_piece(&(index / 81));
        let dest = num_to_coord(&(index % 81));
        PureMove::NormalMove(NormalMove::NonTamMoveFromHopZuo {
            color: piece.color,
            prof: piece.prof,
            dest,
        })
    }
}

pub fn afterhalf_candidates_to_mask(candidates: &Vec<AfterHalfAcceptance>) -> [i8; ACTION_SIZE] {
    let mut mask = [0; ACTION_SIZE];
    for c in candidates.iter() {
        match c.dest {
            Some(x) => mask[20 * 81 + 81 * 81 + coord_to_num(&x)] = 1,
            None => mask[20 * 81 + 81 * 81 + 81] = 1,
        }
    }
    mask
}

pub fn get_after_half_candidate_by_index(
    index: usize,
    candidates: &Vec<AfterHalfAcceptance>,
) -> AfterHalfAcceptance {
    if index < 20 * 81 + 81 * 81 {
        unreachable!()
    } else if index < 20 * 81 + 81 * 81 + 81 {
        let index = index - 20 * 81 + 81 * 81;
        let c_dest = num_to_coord(&(index % 81));
        for c in candidates.iter() {
            if c.dest == Some(c_dest) {
                return c.clone();
            }
        }
        panic!()
    } else if index == 20 * 81 + 81 * 81 + 81 {
        return AfterHalfAcceptance { dest: None };
    } else {
        unreachable!()
    }
}

pub fn tymok_mask() -> [i8; ACTION_SIZE] {
    let mut mask = [0; ACTION_SIZE];
    mask[20 * 81 + 81 * 81 + 81 + 1] = 1;
    mask[20 * 81 + 81 * 81 + 81 + 2] = 1;
    mask
}

pub fn get_tymok_candidate_by_index(index: usize) -> bool {
    if index == (20 * 81 + 81 * 81 + 81 + 1) {
        true
    } else if index == (20 * 81 + 81 * 81 + 81 + 2) {
        false
    } else {
        unreachable!()
    }
}
