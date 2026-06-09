import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from core.policy_validator import validate_policy_source
from core.trace_context import trace_entry_x


def behavior_descriptor(code):
    """Describe a policy by the literal actions it may return."""
    actions = set()
    for match in re.finditer(r"return\s+['\"]([^'\"]*)['\"]", code or ""):
        actions.add(match.group(1).strip() or "NOOP")
    return "|".join(sorted(actions)) if actions else "dynamic"


def _failure_category(failure_reason):
    reason = str(failure_reason or "").lower()
    if "failed to load" in reason or "syntax" in reason:
        return "load-error"
    if "timeout" in reason:
        return "timeout"
    if "stuck" in reason or "forward progress" in reason:
        return "stuck"
    if "fatal" in reason or "lost a life" in reason:
        return "fatal"
    return "other"


def build_obstacle_key(failure_reason, trace, bucket_size=250):
    """Cluster failures by level, approximate position, and failure category."""
    if not trace:
        return "global-" + _failure_category(failure_reason)

    entry = trace[-1]
    x = trace_entry_x(entry)
    bucket_size = max(1, int(bucket_size))
    x_bucket = (x // bucket_size) * bucket_size
    if isinstance(entry, dict):
        try:
            zone = int(float(entry.get("zone", 0)))
        except (TypeError, ValueError):
            zone = 0
        try:
            act = int(float(entry.get("act", 0)))
        except (TypeError, ValueError):
            act = 0
    else:
        zone = 0
        act = 0
    return f"zone-{zone}-act-{act}-x-{x_bucket}-{_failure_category(failure_reason)}"


def p_ucb_score(normalized_fitness, visits, total_visits, exploration_constant=0.2):
    """Combine measured quality with an exploration bonus."""
    visits = max(0, int(visits or 0))
    total_visits = max(0, int(total_visits or 0))
    return float(normalized_fitness) + exploration_constant * math.sqrt(total_visits + 1) / (visits + 1)


# The index is loaded and fully rewritten on every evaluation, so records in
# it must stay small: long text is truncated and traces/components live in a
# per-policy details sidecar (details/<policy_id>.json) instead.
INDEX_TEXT_LIMIT = 300
_HEAVY_FIELDS = ("trace", "components")


def _truncate(text, limit=INDEX_TEXT_LIMIT):
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class PopulationArchive:
    """Persistent archive of every unique evaluated policy."""

    def __init__(self, root="artifacts/population"):
        self.root = Path(root)
        self.policy_dir = self.root / "policies"
        self.details_dir = self.root / "details"
        self.index_path = self.root / "index.json"

    def load_records(self):
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        return candidates if isinstance(candidates, list) else []

    def load_details(self, policy_id):
        """Full evaluation context (trace, components, untruncated text)."""
        try:
            payload = json.loads(
                (self.details_dir / f"{policy_id}.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_details(self, policy_id, details):
        self.details_dir.mkdir(parents=True, exist_ok=True)
        details_path = self.details_dir / f"{policy_id}.json"
        temp_path = details_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
        os.replace(str(temp_path), str(details_path))

    def _slim_record(self, record):
        """Strip heavy fields out of an index record, spilling them to details.

        Also migrates legacy records that stored the full trace inline in
        index.json the first time they are rewritten.
        """
        if not any(field in record for field in _HEAVY_FIELDS):
            return record
        policy_id = str(record.get("policy_id", ""))
        if policy_id:
            details = self.load_details(policy_id)
            for field in _HEAVY_FIELDS:
                if field in record:
                    details[field] = record[field]
            for field in ("failure_reason", "reasoning"):
                if field in record:
                    details.setdefault(field, record[field])
            details["policy_id"] = policy_id
            self._write_details(policy_id, details)
        slim = {key: value for key, value in record.items() if key not in _HEAVY_FIELDS}
        for field in ("failure_reason", "reasoning"):
            if field in slim:
                slim[field] = _truncate(slim[field])
        return slim

    def _save_records(self, records):
        self.root.mkdir(parents=True, exist_ok=True)
        slimmed = [self._slim_record(record) for record in records]
        records[:] = slimmed
        temp_path = self.index_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps({"candidates": slimmed}, indent=2),
            encoding="utf-8",
        )
        os.replace(str(temp_path), str(self.index_path))

    def record_evaluation(
        self,
        code,
        fitness,
        components=None,
        failure_reason="",
        trace=None,
        reasoning="",
    ):
        code = str(code or "")
        now = int(time.time())
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        policy_id = code_hash[:16]
        records = self.load_records()
        record = next((item for item in records if item.get("code_hash") == code_hash), None)

        improved = False
        if record is None:
            improved = True
            self.policy_dir.mkdir(parents=True, exist_ok=True)
            code_path = self.policy_dir / f"{policy_id}.py"
            code_path.write_text(code, encoding="utf-8")
            record = {
                "policy_id": policy_id,
                "code_hash": code_hash,
                "code_path": f"policies/{policy_id}.py",
                "fitness": float(fitness),
                "failure_reason": _truncate(failure_reason),
                "reasoning": _truncate(reasoning),
                "behavior_descriptor": behavior_descriptor(code),
                "obstacle_key": build_obstacle_key(failure_reason, trace or []),
                "evaluations": 1,
                "selection_visits": 0,
                "created_at": now,
                "updated_at": now,
            }
            records.append(record)
        else:
            record["evaluations"] = int(record.get("evaluations", 0)) + 1
            record["updated_at"] = now
            if float(fitness) > float(record.get("fitness", float("-inf"))):
                improved = True
                record.update(
                    {
                        "fitness": float(fitness),
                        "failure_reason": _truncate(failure_reason),
                        "reasoning": _truncate(reasoning),
                        "obstacle_key": build_obstacle_key(failure_reason, trace or []),
                    }
                )
                record.pop("trace", None)
                record.pop("components", None)

        if improved:
            # Details mirror the best observed evaluation, like the index.
            self._write_details(
                policy_id,
                {
                    "policy_id": policy_id,
                    "fitness": float(fitness),
                    "components": components or {},
                    "failure_reason": str(failure_reason or ""),
                    "trace": trace or [],
                    "reasoning": str(reasoning or ""),
                    "updated_at": now,
                },
            )

        self._save_records(records)
        return record

    def elite_candidates(self, limit=64):
        records = sorted(
            self.load_records(),
            key=lambda item: float(item.get("fitness", 0.0)),
            reverse=True,
        )
        if not records:
            return []

        leaders = {}
        for record in records:
            leaders.setdefault(("behavior", record.get("behavior_descriptor", "dynamic")), record)
            leaders.setdefault(("obstacle", record.get("obstacle_key", "global-other")), record)

        selected = []
        for record in sorted(
            leaders.values(),
            key=lambda item: float(item.get("fitness", 0.0)),
            reverse=True,
        ):
            if record not in selected:
                selected.append(record)
            if len(selected) >= limit:
                return selected

        for record in records:
            if record not in selected:
                selected.append(record)
            if len(selected) >= limit:
                break
        return selected

    def _read_code(self, record):
        code_path = self.root / str(record.get("code_path", ""))
        return code_path.read_text(encoding="utf-8")

    def select_parent_codes(self, rng=None, elite_limit=64, exploration_constant=0.2):
        rng = rng or __import__("random")
        records = self.load_records()
        elites = []
        for record in self.elite_candidates(limit=len(records)):
            try:
                code = self._read_code(record)
                validate_policy_source(code)
            except (OSError, UnicodeError, ValueError):
                continue
            elites.append((record, code))
            if len(elites) >= elite_limit:
                break
        if len(elites) < 2:
            return None

        fitnesses = [float(record.get("fitness", 0.0)) for record, _code in elites]
        low = min(fitnesses)
        spread = max(fitnesses) - low
        total_visits = sum(
            int(record.get("selection_visits", 0)) for record, _code in elites
        )

        available = list(elites)
        chosen = []
        while available and len(chosen) < 2:
            weights = []
            for record, _code in available:
                fitness = float(record.get("fitness", 0.0))
                normalized = (fitness - low) / spread if spread else 1.0
                weights.append(
                    p_ucb_score(
                        normalized,
                        record.get("selection_visits", 0),
                        total_visits,
                        exploration_constant,
                    )
                )
            threshold = rng.random() * sum(weights)
            cumulative = 0.0
            selected = available[-1]
            for candidate, weight in zip(available, weights):
                cumulative += weight
                if threshold <= cumulative:
                    selected = candidate
                    break
            chosen.append(selected)
            available.remove(selected)

        if len(chosen) < 2:
            return None

        chosen_ids = {record.get("policy_id") for record, _code in chosen}
        for record in records:
            if record.get("policy_id") in chosen_ids:
                record["selection_visits"] = int(record.get("selection_visits", 0)) + 1
        self._save_records(records)
        return chosen[0][1], chosen[1][1]
