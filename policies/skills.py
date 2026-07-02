def move_right_and_jump(state, _STATE):
    _STATE["jump_timer"] = 15
    return 'RIGHT,B'

def roll_forward(state, _STATE):
    _STATE["roll_timer"] = 5
    return 'RIGHT,DOWN'

def handle_boss_arena(state, _STATE):
    current_x = state.get("x_pos", 0)
    vision = state.get("vision_context", "CLEAR")
    
    if current_x > 9850:
        _STATE["boss_dir"] = "LEFT"
    elif current_x < 9600:
        _STATE["boss_dir"] = "RIGHT"
    else:
        _STATE["boss_dir"] = "RIGHT"
        
    if vision in ['PIT', 'DANGER', 'WATER', 'SPIKES']:
        _STATE["jump_timer"] = 15
        return f"{_STATE['boss_dir']},B"

    if _STATE.get("jump_timer", 0) > 0:
        return f"{_STATE['boss_dir']},B"
        
    if _STATE.get("is_grounded", True) and _STATE.get("boss_jump_timer", 0) == 0:
        _STATE["boss_jump_timer"] = 40
        _STATE["jump_timer"] = 10
        return f"{_STATE['boss_dir']},B"
        
    return _STATE.get("boss_dir", "RIGHT")

def recover_from_stuck(state, _STATE):
    if _STATE.get("stuck_timer", 0) > 12:
        _STATE["stuck_timer"] = 0
        _STATE["jump_timer"] = 12
        return "RIGHT,B"
    return None

def avoid_hazards(state, _STATE):
    vision = state.get("vision_context", "CLEAR")
    if vision in ['SPIKES', 'ENEMY', 'PIT', 'DANGER', 'WATER']:
        if _STATE.get("is_grounded", True):
            _STATE["jump_timer"] = 25
            return "RIGHT,B"
        elif _STATE.get("vertical_bounce_count", 0) == 0:
            return "RIGHT"
    return None

def maintain_momentum(state, _STATE):
    if _STATE.get("x_velocity", 0) > 3.5 and _STATE.get("is_grounded", True):
        _STATE["roll_timer"] = 5
        return "RIGHT,DOWN"
    return None

def descend_slope(state, _STATE):
    dy = _STATE.get("y_velocity", 0)
    if dy > 1.5 and _STATE.get("is_grounded", True) and _STATE.get("x_velocity", 0) > 1:
        _STATE["roll_timer"] = 8
        return "RIGHT,DOWN"
    return None

def semantic_memory_fix_3050_3200(state, _STATE):
    current_x = state.get("x_pos", 0)
    if 3050 <= current_x < 3200 and _STATE.get("is_grounded", True):
        return "RIGHT,B"
    return None

def recover_vertical_bounce(state, _STATE):
    if _STATE.get("vertical_bounce_count", 0) > 3:
        return "RIGHT"
    return None

def maintain_airborne_momentum(state, _STATE):
    if _STATE.get("jump_timer", 0) > 0 and not _STATE.get("in_boss_arena", False):
        return "RIGHT"
    return None

def maintain_roll(state, _STATE):
    if _STATE.get("roll_timer", 0) > 0:
        # Check if we are stuck while rolling to abort
        current_x = state.get("x_pos", 0)
        dx = current_x - _STATE.get("last_x_pos", 0)
        if abs(dx) < 0.5 and _STATE.get("is_grounded", True):
            return None # Signal to abort roll in get_action logic
        return "RIGHT,DOWN"
    return None

def early_level_safety(state, _STATE):
    current_x = state.get("x_pos", 0)
    if 70 < current_x < 110 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 15
        return "RIGHT,B"
    if 270 < current_x < 300 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 20
        return "RIGHT,B"
    return None

def specific_zone_hazards(state, _STATE):
    current_x = state.get("x_pos", 0)
    if 480 < current_x < 560 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 20
        return "RIGHT,B"
    if 690 < current_x < 770 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 30
        return "RIGHT,B"
    if 1120 < current_x < 1190 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 35
        return "RIGHT,B"
    if 1340 < current_x < 1460 and _STATE.get("is_grounded", True):
        # Only roll if we have some momentum
        current_x = state.get("x_pos", 0)
        dx = current_x - _STATE.get("last_x_pos", 0)
        if abs(dx) > 0.1:
            _STATE["roll_timer"] = 25
            return "RIGHT,DOWN"
    if 2180 < current_x < 2560 and _STATE.get("is_grounded", True):
        if _STATE.get("jump_timer", 0) == 0:
            _STATE["jump_timer"] = 20
            return "RIGHT,B"
    if 2620 < current_x < 2680 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 35
        return "RIGHT,B"
    return None

def handle_collapsing_platforms(state, _STATE):
    current_x = state.get("x_pos", 0)
    dx = current_x - _STATE.get("last_x_pos", 0)
    if 2980 < current_x < 3360:
        if _STATE.get("is_grounded", True) and _STATE.get("jump_timer", 0) == 0 and _STATE.get("vertical_bounce_count", 0) == 0:
            _STATE["jump_timer"] = 22
            return "RIGHT,B"
        if abs(dx) < 0.5:
            return "RIGHT"
    return None

def loop_prep(state, _STATE):
    current_x = state.get("x_pos", 0)
    if 3500 < current_x < 3820 and _STATE.get("is_grounded", True):
        _STATE["roll_timer"] = 40
        return "RIGHT,DOWN"
    return None

def post_loop_platforms(state, _STATE):
    current_x = state.get("x_pos", 0)
    if 4270 < current_x < 4720 and _STATE.get("is_grounded", True):
        if _STATE.get("jump_timer", 0) == 0:
            _STATE["jump_timer"] = 22
            return "RIGHT,B"
    return None

def recover_from_bonk(state, _STATE):
    current_x = state.get("x_pos", 0)
    dx = current_x - _STATE.get("last_x_pos", 0)
    if _STATE.get("x_velocity", 0) > 2 and dx < 0.5 and _STATE.get("is_grounded", True):
        _STATE["jump_timer"] = 12
        return "RIGHT,B"
    return None

def semantic_memory_fix_9767(state, _STATE):
    current_x = state.get("x_pos", 0)
    if 9755 <= current_x <= 9780:
        _STATE["jump_timer"] = 15
        _STATE["stuck_timer"] = 0
        return "RIGHT,B"
    return None

def semantic_memory_fix_stagnation(state, _STATE):
    current_x = state.get("x_pos", 0)
    dx = current_x - _STATE.get("last_x_pos", 0)
    if (3055 <= current_x <= 3065 or 2940 <= current_x <= 2960) and abs(dx) < 0.5:
        return "RIGHT,B"
    return None

def semantic_memory_fix_1077(state, _STATE):
    current_x = state.get("x_pos", 0)
    dx = current_x - _STATE.get("last_x_pos", 0)
    if 1070 <= current_x <= 1085 and abs(dx) < 0.5:
        _STATE["roll_timer"] = 0
        return "RIGHT,B"
    return None

def default_move(state, _STATE):
    return "RIGHT"


def frontier_guard_fix(state, _STATE):
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and 1052 <= state.get("x_pos", 0) <= 1102
        and abs(state.get("x_velocity", 0)) < 0.5
    ):
        return "RIGHT,B"
    return None

def semantic_memory_fix_2330_2385(state, _STATE):
    current_x = state.get("x_pos", 0)
    dx = current_x - _STATE.get("last_x_pos", 0)
    if state.get("zone") == 0 and state.get("act") == 1:
        if 2330 <= current_x <= 2385:
            if dx < -0.5:
                return "RIGHT,UP,B"
            if dx < 1.0:
                _STATE["jump_timer"] = 15
                return "RIGHT,B"
    return None


def semantic_memory_fix_zone0_act1_midsection(state, _STATE):
    current_x = state.get("x_pos", 0)
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and 2379 <= current_x < 3326
    ):
        if current_x >= 2498:
            return "RIGHT"
        if current_x >= 2453:
            return "RIGHT,B"
        return "RIGHT"
    return None


def replay_sequence_zone0_act1_3928(state, _STATE):
    current_x = state.get("x_pos", 0)
    if state.get("zone") == 0 and state.get("act") == 1:
        if "replay_3928" not in _STATE:
            _STATE["replay_3928"] = -1
        if 3903 <= current_x <= 3953:
            if _STATE["replay_3928"] < 0 or (3903 <= current_x <= 3953):
                _STATE["replay_3928"] += 1
                if _STATE["replay_3928"] < 200: return "RIGHT"
                if _STATE["replay_3928"] < 240: return "RIGHT,B"
                if _STATE["replay_3928"] < 340: return "RIGHT"
    return None

def replay_sequence_zone0_act1_2393(state, _STATE):
    current_x = state.get("x_pos", 0)
    if state.get("zone") == 0 and state.get("act") == 1:
        if "replay_2393" not in _STATE:
            _STATE["replay_2393"] = -1
        if 2368 <= current_x <= 2418:
            if _STATE["replay_2393"] < 0 or (2368 <= current_x <= 2418):
                _STATE["replay_2393"] += 1
                if _STATE["replay_2393"] < 120: return "RIGHT"
                if _STATE["replay_2393"] < 160: return "RIGHT,B"
                if _STATE["replay_2393"] < 280: return "RIGHT"
    return None


def get_action(state):
    return 'RIGHT'
