"""FunSearch-style policy pool: a small, diversity-preserving set of parents.

Pool files are named ``pool_<fitness>_<hash>.py`` so fitness survives process
restarts without a separate index. Pruning keeps the best policy per action
signature (species) before falling back to raw fitness, so crossover always
has parents with different controller habits to draw from.
"""

import glob
import hashlib
import os

from core.fsio import atomic_write_text
from core.population import behavior_descriptor


def parse_pool_fitness(path):
    name = os.path.basename(path)
    if name.startswith("pool_"):
        name = name[len("pool_"):]
    if name.endswith(".py"):
        name = name[:-len(".py")]
    value = name.split("_", 1)[0]
    return float(value)


def update_pool(code, fitness, pool_dir="policies/pool", max_size=6):
    os.makedirs(pool_dir, exist_ok=True)
    code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()[:8]
    new_path = os.path.join(pool_dir, f"pool_{fitness:.2f}_{code_hash}.py")
    atomic_write_text(new_path, code, prefix=".pool-")

    pool = []
    for path in glob.glob(os.path.join(pool_dir, "pool_*.py")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                pool_code = f.read()
            pool.append(
                {
                    "fitness": parse_pool_fitness(path),
                    "path": path,
                    "signature": behavior_descriptor(pool_code),
                }
            )
        except (OSError, UnicodeError, ValueError):
            pass

    species_leaders = {}
    for entry in sorted(pool, key=lambda item: item["fitness"], reverse=True):
        species_leaders.setdefault(entry["signature"], entry)

    selected_paths = []
    for entry in sorted(species_leaders.values(), key=lambda item: item["fitness"], reverse=True):
        if len(selected_paths) < max_size:
            selected_paths.append(entry["path"])

    for entry in sorted(pool, key=lambda item: item["fitness"], reverse=True):
        if len(selected_paths) >= max_size:
            break
        if entry["path"] not in selected_paths:
            selected_paths.append(entry["path"])

    for entry in pool:
        if entry["path"] not in selected_paths:
            try:
                os.remove(entry["path"])
            except OSError:
                pass


def load_pool_codes(pool_dir="policies/pool"):
    """Read every policy currently in the FunSearch population pool."""
    codes = []
    for pf in glob.glob(os.path.join(pool_dir, "pool_*.py")):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                codes.append(f.read())
        except (OSError, UnicodeError):
            pass
    return codes
