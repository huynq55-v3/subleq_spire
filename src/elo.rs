/// ELO Rating System for SUBLEQ Gladiators
///
/// Simple ELO implementation for ranking bots based on battle outcomes.
/// Used by the Tournament Pipeline to identify truly elite programs.

/// Default starting ELO for new bots
pub const DEFAULT_ELO: f64 = 1500.0;

/// K-factor: how much a single match can shift ratings
pub const K_FACTOR: f64 = 32.0;

/// Calculate expected win probability for player A against player B.
///
/// Returns a value in [0.0, 1.0] — probability that A wins.
pub fn elo_expected(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf((rating_b - rating_a) / 400.0))
}

/// Update ELO ratings after a match.
///
/// `score_a`: 1.0 if A won, 0.0 if A lost, 0.5 for draw.
///
/// Returns `(new_rating_a, new_rating_b)`.
pub fn elo_update(rating_a: f64, rating_b: f64, score_a: f64, k: f64) -> (f64, f64) {
    let expected_a = elo_expected(rating_a, rating_b);
    let expected_b = 1.0 - expected_a;
    let score_b = 1.0 - score_a;

    let new_a = rating_a + k * (score_a - expected_a);
    let new_b = rating_b + k * (score_b - expected_b);

    (new_a, new_b)
}

/// Compute ELO ratings for all fighters based on elimination order.
///
/// `num_fighters`: total number participating
/// `elimination_order`: indices of eliminated fighters, earliest first
/// `winner_index`: the last survivor (if any)
/// `initial_elos`: starting ELO for each fighter (indexed by fighter index)
///
/// Returns updated ELO ratings (same indexing).
///
/// Logic: Each elimination is treated as a pairwise loss against all
/// survivors at that point. The winner gets pairwise wins against everyone.
pub fn compute_battle_elos(
    num_fighters: usize,
    elimination_order: &[usize],
    winner_index: Option<usize>,
    initial_elos: &[f64],
) -> Vec<f64> {
    let mut elos = initial_elos.to_vec();

    // Each eliminated fighter "lost" to everyone still alive at their elimination
    let mut alive: Vec<bool> = vec![true; num_fighters];

    for &eliminated in elimination_order {
        // The eliminated fighter loses pairwise to each currently alive fighter
        let survivors: Vec<usize> = (0..num_fighters)
            .filter(|&i| alive[i] && i != eliminated)
            .collect();

        for &survivor in &survivors {
            let (new_elim, new_surv) =
                elo_update(elos[eliminated], elos[survivor], 0.0, K_FACTOR);
            elos[eliminated] = new_elim;
            elos[survivor] = new_surv;
        }

        alive[eliminated] = false;
    }

    // If there's a winner, give them a small bonus for being last standing
    if let Some(w) = winner_index {
        elos[w] += K_FACTOR * 0.5; // Bonus for outright victory
    }

    elos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elo_expected_equal() {
        let e = elo_expected(1500.0, 1500.0);
        assert!((e - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_elo_expected_higher_wins() {
        let e = elo_expected(1800.0, 1500.0);
        assert!(e > 0.8); // Much stronger player should expect to win
    }

    #[test]
    fn test_elo_update_winner() {
        let (new_a, new_b) = elo_update(1500.0, 1500.0, 1.0, 32.0);
        assert!(new_a > 1500.0);
        assert!(new_b < 1500.0);
        assert!((new_a - new_b - 32.0).abs() < 1e-6); // Symmetric shift
    }

    #[test]
    fn test_compute_battle_elos() {
        // 4 fighters, fighter 2 eliminated first, then 0, then 1. Fighter 3 wins.
        let initial = vec![1500.0, 1500.0, 1500.0, 1500.0];
        let elimination = vec![2, 0, 1];
        let elos = compute_battle_elos(4, &elimination, Some(3), &initial);

        // Winner (3) should have highest ELO
        assert!(elos[3] > elos[0]);
        assert!(elos[3] > elos[1]);
        assert!(elos[3] > elos[2]);
        // First eliminated (2) should have lowest
        assert!(elos[2] < elos[0]);
    }
}
