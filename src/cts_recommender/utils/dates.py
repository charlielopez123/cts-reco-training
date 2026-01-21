from datetime import date, datetime
import pandas as pd
from typing import Dict, Set, Tuple, List, Iterable

def _to_date(x: object) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        # expect ISO (YYYY-MM-DD)
        return date.fromisoformat(x)
    raise TypeError(f"Unsupported date-like value: {type(x)}")


def build_holiday_index(holidays: dict):
    """
    Build a fast index:
        index[canton]["holidays"] -> set[date]
        [canton]["vacations"] -> list[(start_date, end_date)]
    Accepts public_holidays as strings OR dicts (with 'date').
    """
    index: Dict[str, Dict[str, object]] = {}
    for c, cdata in holidays["cantons"].items():
        # normalize public holidays
        hset: Set[date] = {h["date"] for h in cdata.get("public_holidays", [])}
        # normalize vacations
        intervals: List[Tuple[date, date]] = [
            (v["start"], v["end"])
            for v in cdata.get("school_vacations", [])
        ]
        index[c] = {"holidays": hset, "vacations": intervals}
    return index

def is_holiday_indexed(x: object, index: dict, cantons: Iterable[str] | None = None) -> bool:

    d = _to_date(x)
    for c in (cantons or index.keys()):
        cidx = index[c]
        if any(_to_date(h) == d for h in cidx["holidays"]):
            return True
        for s, e in cidx["vacations"]:
            sd = _to_date(s)
            ed = _to_date(e)
            if sd <= d <= ed:
                return True
    return False

def get_season(x: date) -> str:
    d = _to_date(x)
    month = d.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"



