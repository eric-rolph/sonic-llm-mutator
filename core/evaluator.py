ACT1_COMPLETION_X = 9700

# Fitness weights. These are the single source of truth: the dashboard renders
# FITNESS_FORMULA (below) from them, so the displayed formula can never drift
# from what calculate_fitness actually computes.
DISTANCE_WEIGHT = 2.0   # reward pure rightward distance heavily
SPEED_WEIGHT = 500.0    # distance-per-frame; makes a faster route win at equal distance
RING_WEIGHT = 1.0       # small tie-breaker
SCORE_WEIGHT = 0.01     # very small tie-breaker
COMPLETION_BONUS = 5000  # one-off bonus for reaching the level's end-zone X

FITNESS_FORMULA = (
    f"fitness = (distance * {DISTANCE_WEIGHT:g})"
    f" + ((distance / frames) * {SPEED_WEIGHT:g})"
    f" + (rings * {RING_WEIGHT:g})"
    f" + (score * {SCORE_WEIGHT:g})"
    f" + ({COMPLETION_BONUS:g} if completed else 0)"
)


def calculate_fitness(x_max, frames_alive, rings, score, completion_x=ACT1_COMPLETION_X):
    """
    Calculates fitness for speedrun-first evolution.

    Distance remains the base objective, speed is the primary tie-breaker
    for comparable routes, and rings/score are intentionally small bonuses.
    The completion threshold defaults to Green Hill Act 1, but callers can
    provide state-specific end coordinates for broader benchmark states.
    """
    # Prevent division by zero
    frames = max(1, frames_alive)

    # Speedrun metric: distance per frame, weighted strongly enough that a
    # faster route beats a slower collector at the same distance.
    speed_bonus = (x_max / frames) * SPEED_WEIGHT

    # Reward pure distance heavily
    distance_score = x_max * DISTANCE_WEIGHT

    # Secondary rewards should not dominate route speed.
    ring_score = rings * RING_WEIGHT
    game_score = score * SCORE_WEIGHT
    completion_target = int(completion_x or 0)
    completion_bonus = COMPLETION_BONUS if completion_target > 0 and x_max >= completion_target else 0

    fitness = distance_score + speed_bonus + ring_score + game_score + completion_bonus

    components = {
        "distance": distance_score,
        "speed": speed_bonus,
        "rings": ring_score,
        "score": game_score,
        "completion": completion_bonus,
        "completion_target": completion_target
    }
    return fitness, components
