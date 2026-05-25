def calculate_fitness(x_max, frames_alive, rings, score):
    """
    Calculates fitness based on distance traveled, speed (Time-to-Distance decay), 
    and secondary objectives like rings and score.
    """
    # Prevent division by zero
    frames = max(1, frames_alive)
    
    # Speedrun metric: distance per frame (multiplied by a constant to make it readable)
    speed_bonus = (x_max / frames) * 100
    
    # Reward pure distance heavily
    distance_score = x_max * 2.0
    
    # Secondary rewards
    ring_score = rings * 10
    
    fitness = distance_score + speed_bonus + ring_score + score
    return fitness
