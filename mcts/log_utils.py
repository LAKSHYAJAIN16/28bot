import os
import json
import sys
from datetime import datetime

# Paths
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
JSONL_LOG = os.path.join(PROJECT_ROOT, "training.log")
TEXT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "game28", "mcts_games")


def log_event(event_type: str, payload: dict) -> None:
    try:
        record = {"event": event_type, **payload}
        with open(JSONL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


class Tee:
    def __init__(self, file_obj, original):
        self._file = file_obj
        self._orig = original
    def write(self, data):
        try:
            self._orig.write(data)
        except Exception:
            pass
        try:
            self._file.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


def open_game_log(game_id: int):
    os.makedirs(TEXT_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(TEXT_LOG_DIR, f"game_{game_id}_{ts}.log")
    return path, ts


