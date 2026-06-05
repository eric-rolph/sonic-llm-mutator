ACT1_COMPLETION_X = 9700

# Fitness weights. These are the single source of truth: the dashboard renders
# FITNESS_FORMULA (below) from them, so the displayed formula can never drift
# from what calculate_fitness actually computes.
DISTANCE_WEIGHT = 2.0   # reward total rightward distance heavily
SPEED_WEIGHT = 500.0    # distance-per-frame; makes a faster route win at equal distance
RING_WEIGHT = 1.0       # small tie-breaker
SCORE_WEIGHT = 0.01     # very small tie-breaker
COMPLETION_BONUS = 5000  # one-off bonus for reaching the current level's end-zone X
LEVEL_CLEARED_BONUS = 25000  # per level fully cleared during a continuous play-through

FITNESS_FORMULA = (
    f"fitness = (levels_cleared * {LEVEL_CLEARED_BONUS:g})"
    f" + (total_distance * {DISTANCE_WEIGHT:g})"
    f" + ((total_distance / frames) * {SPEED_WEIGHT:g})"
    f" + (rings * {RING_WEIGHT:g})"
    f" + (score * {SCORE_WEIGHT:g})"
    f" + ({COMPLETION_BONUS:g} if current level completed else 0)"
)


def calculate_fitness(
    x_max,
    frames_alive,
    rings,
    score,
    completion_x=ACT1_COMPLETION_X,
    levels_cleared=0,
    cumulative_distance=0,
):
    """
    Calculates fitness for speedrun-first, multi-level evolution.

    Clearing whole levels dominates (a continuous run that beats more acts is
    always preferred), distance is the base objective within a level, speed is
    the primary tie-breaker for comparable routes, and rings/score are
    intentionally small bonuses.

    ``levels_cleared`` and ``cumulative_distance`` describe progress made in
    earlier acts of a continuous play-through; ``x_max`` is the furthest point
    reached in the *current* act. They default to 0 so single-level callers
    (the benchmark, unit tests) get the original behaviour unchanged.
    """
    # Prevent division by zero
    frames = max(1, frames_alive)

    # Total ground covered across every act played this episode.
    total_distance = cumulative_distance + x_max

    # Speedrun metric: distance per frame, weighted strongly enough that a
    # faster route beats a slower collector at the same distance.
    speed_bonus = (total_distance / frames) * SPEED_WEIGHT

    # Reward pure distance heavily.
    distance_score = total_distance * DISTANCE_WEIGHT

    # Each fully cleared act is worth more than maxing distance on a single one.
    levels_bonus = levels_cleared * LEVEL_CLEARED_BONUS

    # Secondary rewards should not dominate route speed.
    ring_score = rings * RING_WEIGHT
    game_score = score * SCORE_WEIGHT
    completion_target = int(completion_x or 0)
    completion_bonus = COMPLETION_BONUS if completion_target > 0 and x_max >= completion_target else 0

    fitness = levels_bonus + distance_score + speed_bonus + ring_score + game_score + completion_bonus

    components = {
        "levels_cleared": levels_cleared,
        "levels": levels_bonus,
        "distance": distance_score,
        "speed": speed_bonus,
        "rings": ring_score,
        "score": game_score,
        "completion": completion_bonus,
        "completion_target": completion_target,
    }
    return fitness, components
