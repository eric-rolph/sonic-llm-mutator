def get_action(state):
    # FRONTIER_GUARD zone=0 act=1 x=1077
    # DIAGNOSIS_GUARD zone=0 act=1 x=2393
    # DIAGNOSIS_GUARD zone=0 act=1 x=3928
    # DIAGNOSIS_GUARD zone=0 act=1 x=3593
    # DIAGNOSIS_GUARD zone=0 act=1 x=4341
    # LLM_GUARD zone=0 act=1 x=4341
    global _LLM_REPLAY_0_1_4341
    if '_LLM_REPLAY_0_1_4341' not in globals():
        _LLM_REPLAY_0_1_4341 = -1
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and _LLM_REPLAY_0_1_4341 < 15
        and (_LLM_REPLAY_0_1_4341 >= 0 or 4316 <= state.get("x_pos", 0) <= 4366)
    ):
        _LLM_REPLAY_0_1_4341 = _LLM_REPLAY_0_1_4341 + 1
        return "RIGHT"

    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and 4316 <= state.get("x_pos", 0) < 4737
    ):
        return "RIGHT,B"

    global _DIAG_PHASE_0_1_3593, _DIAG_REPLAY_0_1_3593
    if '_DIAG_PHASE_0_1_3593' not in globals():
        _DIAG_PHASE_0_1_3593 = -1
        _DIAG_REPLAY_0_1_3593 = -1
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and _DIAG_REPLAY_0_1_3593 < 165
    ):
        _diag_x = state.get("x_pos", 0)
        if _DIAG_PHASE_0_1_3593 < 0 and 3568 <= _diag_x <= 3618:
            _DIAG_PHASE_0_1_3593 = 0
        if _DIAG_PHASE_0_1_3593 >= 0:
            if _DIAG_PHASE_0_1_3593 == 0:
                if _diag_x > 3559:
                    return "LEFT"
                _DIAG_PHASE_0_1_3593 = 1
            if _DIAG_PHASE_0_1_3593 == 1:
                if _diag_x < 3920:
                    return "RIGHT"
                _DIAG_PHASE_0_1_3593 = 2
            if _DIAG_PHASE_0_1_3593 == 2 and _DIAG_REPLAY_0_1_3593 < 0:
                _DIAG_REPLAY_0_1_3593 = 0
            if _DIAG_REPLAY_0_1_3593 >= 0:
                _DIAG_REPLAY_0_1_3593 = _DIAG_REPLAY_0_1_3593 + 1
                if _DIAG_REPLAY_0_1_3593 <= 45:
                    return "RIGHT,B"
                return "RIGHT"

    global _DIAG_REPLAY_0_1_3928
    if '_DIAG_REPLAY_0_1_3928' not in globals():
        _DIAG_REPLAY_0_1_3928 = -1
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and _DIAG_REPLAY_0_1_3928 < 165
    ):
        _diag_x = state.get("x_pos", 0)
        if _DIAG_REPLAY_0_1_3928 < 0 and 3903 <= _diag_x < 4053:
            return "RIGHT"
        if _DIAG_REPLAY_0_1_3928 < 0 and 4053 <= _diag_x <= 4078:
            _DIAG_REPLAY_0_1_3928 = 0
        if _DIAG_REPLAY_0_1_3928 >= 0:
            _DIAG_REPLAY_0_1_3928 = _DIAG_REPLAY_0_1_3928 + 1
            if _DIAG_REPLAY_0_1_3928 <= 45:
                return "RIGHT,B"
            return "RIGHT"

    global _DIAG_REPLAY_0_1_2393
    if '_DIAG_REPLAY_0_1_2393' not in globals():
        _DIAG_REPLAY_0_1_2393 = -1
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and _DIAG_REPLAY_0_1_2393 < 280
        and (_DIAG_REPLAY_0_1_2393 >= 0 or 2368 <= state.get("x_pos", 0) <= 2418)
    ):
        _DIAG_REPLAY_0_1_2393 = _DIAG_REPLAY_0_1_2393 + 1
        if _DIAG_REPLAY_0_1_2393 < 120:
            return "RIGHT"
        if _DIAG_REPLAY_0_1_2393 < 160:
            return "RIGHT,B"
        return "RIGHT"

    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and 1052 <= state.get("x_pos", 0) <= 1102
        and abs(state.get("x_velocity", 0)) < 0.5
    ):
        return "RIGHT,B"

    global _STATE
    if '_STATE' not in globals():
        _STATE = {
            "last_x_pos": 0, "last_y_pos": 0,
            "x_velocity": 0, "y_velocity": 0,
            "is_grounded": True,
            "jump_timer": 0, "roll_timer": 0, "stuck_timer": 0,
            "boss_jump_timer": 0, "in_boss_arena": False,
            "boss_dir": "RIGHT",
            "vertical_bounce_count": 0,
            "last_dy_sign": 0
        }

    current_x = state.get("x_pos", 0)
    current_y = state.get("y_pos", 0)
    vision = state.get("vision_context", "CLEAR")

    # --- Physics & State Update ---
    dx = current_x - _STATE["last_x_pos"]
    dy = current_y - _STATE["last_y_pos"]
    _STATE["x_velocity"] = dx
    _STATE["y_velocity"] = dy
    
    # Grounded detection: Sonic is grounded if vertical movement is minimal and he's not at the very top of map
    _STATE["is_grounded"] = abs(dy) < 1.0 and current_y > 10
    
    # Detect vertical bouncing (stuck in air loop/oscillation)
    if abs(dy) > 4.0:
        if _STATE["last_dy_sign"] != 0 and _STATE["last_dy_sign"] != (1 if dy > 0 else -1):
            _STATE["vertical_bounce_count"] += 1
        else:
            _STATE["vertical_bounce_count"] = max(0, _STATE["vertical_bounce_count"] - 1)
        _STATE["last_dy_sign"] = 1 if dy > 0 else -1
    else:
        _STATE["vertical_bounce_count"] = max(0, _STATE["vertical_bounce_count"] - 2)

    # Boss Arena Flag Update (Green Hill Zone end area)
    _STATE["in_boss_arena"] = 9500 < current_x < 10100

    # --- Timer Management ---
    if _STATE["jump_timer"] > 0: _STATE["jump_timer"] -= 1
    if _STATE["roll_timer"] > 0: _STATE["roll_timer"] -= 1
    if _STATE["boss_jump_timer"] > 0: _STATE["boss_jump_timer"] -= 1

    # Reset state at level start or transition
    if current_x < 50:
        _STATE.update({"jump_timer": 0, "roll_timer": 0, "stuck_timer": 0, 
                       "in_boss_arena": False, "boss_dir": "RIGHT", 
                       "vertical_bounce_count": 0, "last_dy_sign": 0})

    # --- Priority 1: CRITICAL SEMANTIC FIXES (Hard-coded Coordinates) ---
    # Fix for X=2330-2385 stagnation in Zone 0 Act 1 (The current frontier failure)
    if state.get("zone") == 0 and state.get("act") == 1:
        if 2330 <= current_x <= 2385:
            if dx < -0.5: # Bouncing back significantly, use high jump to clear obstacle
                return "RIGHT,UP,B"
            if dx < 1.0: # Stalled forward progress
                _STATE["jump_timer"] = 15
                return "RIGHT,B"

    # Fix for X=9767 Pit/Spikes
    if 9755 <= current_x <= 9780:
        _STATE["jump_timer"] = 15
        _STATE["stuck_timer"] = 0
        return "RIGHT,B"

    # Fix for X=3061 stagnation
    if 3055 <= current_x <= 3065 and abs(dx) < 0.5:
        _STATE["stuck_timer"] = 0 
        return "RIGHT,B"

    # Fix for X=2949 stagnation
    if 2940 <= current_x <= 2960 and abs(dx) < 0.5:
        return "RIGHT,B"

    # Specific fix for failure at x=1077 (Zone 0 Act 1)
    if 1070 <= current_x <= 1085 and abs(dx) < 0.5:
        _STATE["roll_timer"] = 0 # Kill any roll that's keeping us stuck
        return "RIGHT,B"

    # --- Priority 2: Recovery Mechanisms (General Stuck Detection) ---
    if abs(dx) < 0.5 and _STATE["is_grounded"] and not _STATE["in_boss_arena"]:
        _STATE["stuck_timer"] += 1
    else:
        _STATE["stuck_timer"] = 0
        
    if _STATE["stuck_timer"] > 12: 
        _STATE["stuck_timer"] = 0
        _STATE["roll_timer"] = 0 # Stop rolling into the wall
        _STATE["jump_timer"] = 12
        return "RIGHT,B"

    if _STATE["vertical_bounce_count"] > 3:
        return "RIGHT" 
        
    # --- Priority 3: Active Timers (Only if not stuck) ---
    if _STATE["jump_timer"] > 0 and not _STATE["in_boss_arena"]:
        return "RIGHT"
    
    # Ensure we are actually moving before committing to a roll timer action
    if _STATE["roll_timer"] > 0:
        if abs(dx) < 0.5 and _STATE["is_grounded"]:
            _STATE["roll_timer"] = 0 # Abort roll if stationary
            return "RIGHT,B"
        return "RIGHT,DOWN"

    # --- Priority 4: Boss Arena Logic ---
    if _STATE["in_boss_arena"]:
        if current_x > 9850:
            _STATE["boss_dir"] = "LEFT"
        elif current_x < 9600:
            _STATE["boss_dir"] = "RIGHT"
        else:
            _STATE["boss_dir"] = "RIGHT"
            
        if vision in ['PIT', 'DANGER', 'WATER', 'SPIKES']:
            _STATE["jump_timer"] = 15
            return f"{_STATE['boss_dir']},B"

        if _STATE["jump_timer"] > 0:
            return f"{_STATE['boss_dir']},B"
            
        if _STATE["is_grounded"] and _STATE["boss_jump_timer"] == 0:
            _STATE["boss_jump_timer"] = 40
            _STATE["jump_timer"] = 10
            return f"{_STATE['boss_dir']},B"
        return _STATE["boss_dir"]

    # --- Priority 5: Vision Context Hazard Detection ---
    if vision in ['SPIKES', 'ENEMY', 'PIT', 'DANGER', 'WATER']:
        if _STATE["is_grounded"]:
            _STATE["jump_timer"] = 25
            return "RIGHT,B"
        elif _STATE["vertical_bounce_count"] == 0:
            return "RIGHT"
            
    # --- Priority 6: Hardcoded Zone Geometry (Green Hill Act 1) ---
    if 3050 <= current_x < 3200 and _STATE["is_grounded"]:
        return "RIGHT,B"

    if 70 < current_x < 110 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 15
        return "RIGHT,B"
    if 270 < current_x < 300 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 20
        return "RIGHT,B"

    if 480 < current_x < 560 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 20
        return "RIGHT,B"
    if 690 < current_x < 770 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 30
        return "RIGHT,B"
    if 1120 < current_x < 1190 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 35
        return "RIGHT,B"
    if 1340 < current_x < 1460 and _STATE["is_grounded"]:
        if abs(dx) > 0.1:
            _STATE["roll_timer"] = 25
            return "RIGHT,DOWN"
        else:
            return "RIGHT"

    if 2180 < current_x < 2560 and _STATE["is_grounded"]:
        if _STATE["jump_timer"] == 0:
            _STATE["jump_timer"] = 20
            return "RIGHT,B"
    if 2620 < current_x < 2680 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 35
        return "RIGHT,B"
        
    if 2980 < current_x < 3360:
        if _STATE["is_grounded"] and _STATE["jump_timer"] == 0 and _STATE["vertical_bounce_count"] == 0:
            _STATE["jump_timer"] = 22
            return "RIGHT,B"
        if abs(dx) < 0.5:
            return "RIGHT"

    if 3500 < current_x < 3820 and _STATE["is_grounded"]:
        _STATE["roll_timer"] = 40
        return "RIGHT,DOWN"
    if 4270 < current_x < 4720 and _STATE["is_grounded"]:
        if _STATE["jump_timer"] == 0:
            _STATE["jump_timer"] = 22
            return "RIGHT,B"

    # --- Priority 7: Momentum & Physics Optimization ---
    if _STATE["x_velocity"] > 2 and dx < 0.5 and _STATE["is_grounded"]:
        _STATE["jump_timer"] = 12
        return "RIGHT,B"

    if _STATE["x_velocity"] > 3.5 and _STATE["is_grounded"]:
        _STATE["roll_timer"] = 5
        return "RIGHT,DOWN"
        
    if dy > 1.5 and _STATE["is_grounded"] and _STATE["x_velocity"] > 1:
        _STATE["roll_timer"] = 8
        return "RIGHT,DOWN"

    # --- Update Position Cache ---
    _STATE["last_x_pos"] = current_x
    _STATE["last_y_pos"] = current_y

    # --- Default Fallback ---
    return "RIGHT"