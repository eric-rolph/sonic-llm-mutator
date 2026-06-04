ACT1_COMPLETION_X = 9700


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
    speed_bonus = (x_max / frames) * 500
    
    # Reward pure distance heavily
    distance_score = x_max * 2.0
    
    # Secondary rewards should not dominate route speed.
    ring_score = rings * 1.0
    game_score = score * 0.01
    completion_target = int(completion_x or 0)
    completion_bonus = 5000 if completion_target > 0 and x_max >= completion_target else 0
    
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
