import sys
import subprocess

from emulator.sonic_env import make_retro_env, normalize_step_result, resolve_backend_module

def bk2_to_mp4(bk2_path, mp4_path):
    retro, _ = resolve_backend_module()
    movie = retro.Movie(bk2_path)
    movie.step()
    
    # We must explicitly load the ROM. retro.make uses the game name from the bk2
    game_name = movie.get_game()
    env = make_retro_env(
        retro,
        game=game_name,
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    env.initial_state = movie.get_state()
    env.reset()

    # Start ffmpeg
    width = env.observation_space.shape[1]
    height = env.observation_space.shape[0]
    
    command = [
        'ffmpeg',
        '-y', # Overwrite
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', '60',
        '-i', '-', # Read from stdin
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        mp4_path
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        obs, rew, done, info = normalize_step_result(env.step(keys))
        process.stdin.write(obs.tobytes())
        
    process.stdin.close()
    process.wait()
    env.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python render_video.py <input.bk2> <output.mp4>")
        sys.exit(1)
    bk2_to_mp4(sys.argv[1], sys.argv[2])
