"""
TIME features (spec Part 9) and Forex session classification (spec Part 10).
Session boundaries below are standard approximate UTC hours — the spec is
explicit that "session = better" must be *proven*, not assumed, so these are
just labels for the model to condition on, not a hardcoded trading rule.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum


class Session(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    OFF_PEAK = "OFF_PEAK"


def time_features(epoch: int) -> dict:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return {
        "hour": dt.hour,
        "day_of_week": dt.weekday(),  # Monday=0 .. Sunday=6
        "session": classify_session(epoch).value,
    }


def classify_session(epoch: int) -> Session:
    hour = datetime.fromtimestamp(epoch, tz=timezone.utc).hour
    if 13 <= hour < 17:
        return Session.LONDON_NY_OVERLAP
    if 8 <= hour < 17:
        return Session.LONDON
    if 13 <= hour < 22:
        return Session.NEW_YORK
    if 0 <= hour < 9:
        return Session.ASIA
    return Session.OFF_PEAK
