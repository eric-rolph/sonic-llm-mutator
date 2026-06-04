import tempfile
import unittest
from pathlib import Path

from main import update_pool


def policy_returning(action):
    return f"""
def get_action(state):
    return "{action}"
"""


class PolicyPoolTests(unittest.TestCase):
    def test_update_pool_preserves_distinct_action_species(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool_dir = Path(tmp_dir)
            update_pool(policy_returning("RIGHT"), 300.0, pool_dir=str(pool_dir), max_size=3)
            update_pool(policy_returning("RIGHT"), 250.0, pool_dir=str(pool_dir), max_size=3)
            update_pool(policy_returning("RIGHT"), 200.0, pool_dir=str(pool_dir), max_size=3)
            update_pool(policy_returning("RIGHT,B"), 100.0, pool_dir=str(pool_dir), max_size=3)

            pool_code = "\n".join(path.read_text() for path in pool_dir.glob("pool_*.py"))
            pool_count = len(list(pool_dir.glob("pool_*.py")))

        self.assertIn('return "RIGHT,B"', pool_code)
        self.assertLessEqual(pool_count, 3)


if __name__ == "__main__":
    unittest.main()
