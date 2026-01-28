from dataclasses import dataclass
from enum import Enum

from cts_recommender.environments.curtains import CurtainType


class ContextMode(Enum):
    """
    Context mode determines how time is represented in the feature vector.

    RTS_CURTAIN: Uses 7 curtain types specific to RTS curator workflows.
                 21-dim context: curtain_type(7) + day(7) + weekend(1) + season(4) + channel(2)

    GENERAL: Uses 4 generic time slots (morning, afternoon, prime_time, late_night).
             18-dim context: time_slot(4) + day(7) + weekend(1) + season(4) + channel(2)
             Useful for experimentation outside RTS curator workflows.
    """

    RTS_CURTAIN = "rts_curtain"
    GENERAL = "general"


class TimeSlot(Enum):
    PRIME_TIME = "prime_time"  # 20:00-22:00
    LATE_NIGHT = "late_night"  # 22:00-00:00
    AFTERNOON = "afternoon"    # 14:00-18:00
    MORNING = "morning"        # 06:00-12:00


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"  # Using 'autumn' to match production codebase
    WINTER = "winter"


class Channel(Enum):
    RTS1 = "RTS 1"
    RTS2 = "RTS 2"


@dataclass
class Context:
    """Programming context for general mode (hour-based)."""

    hour: int  # 0-26 airing hour
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    season: Season
    channel: Channel

    def day_of_week_name(self) -> str:
        """Return the name of the day of the week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[self.day_of_week]


@dataclass
class CurtainContext:
    """Programming context for RTS curtain mode."""

    curtain_type: CurtainType
    curtain_id: str  # Full curtain ID (e.g., "rts1_rideau_2")
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    season: Season
    channel: Channel

    def day_of_week_name(self) -> str:
        """Return the name of the day of the week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[self.day_of_week]


