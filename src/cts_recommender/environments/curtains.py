"""
Curtain definitions for RTS TV programming.

Curtains (rideaux) are channel-specific programming timeslots used by RTS curators.
Each curtain has defined time boundaries for when movies can be scheduled.

The curtain_type is shared across channels (7 types) while the channel provides context.
This allows the model to share learning across channels (e.g., "Rideau 2 favors audience signal"
applies to both RTS 1 and RTS 2).
"""

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import List, Optional


class CurtainType(Enum):
    """
    Curtain types shared across channels.

    Each type represents a conceptual programming slot that may exist
    on one or both channels with different time boundaries.
    """

    FICTION_MATIN = "fiction_matin"  # Morning fiction slot (RTS 1 only)
    APRES_MIDI_1 = "apres_midi_1"  # Afternoon slot 1 (RTS 1 only)
    APRES_MIDI_2 = "apres_midi_2"  # Afternoon slot 2 (RTS 1 only)
    RIDEAU_1 = "rideau_1"  # Pre-prime time (both channels)
    RIDEAU_2 = "rideau_2"  # Prime time (both channels)
    RIDEAU_3 = "rideau_3"  # Late evening (both channels)
    RIDEAU_4 = "rideau_4"  # Late night (both channels)


@dataclass(frozen=True)
class CurtainDefinition:
    """
    Definition of a specific curtain on a specific channel.

    Attributes:
        curtain_id: Unique identifier (e.g., "rts1_rideau_2")
        curtain_type: The type of curtain (shared across channels)
        channel: The channel ("RTS 1" or "RTS 2")
        display_name: Human-readable name for UI
        end_time: End boundary for the curtain (movies must start before this)
        start_range_min: Earliest typical start time
        start_range_max: Latest typical start time
        allowed_days: Days when this curtain is available (None = all days)
        content_type: Description of typical content
    """

    curtain_id: str
    curtain_type: CurtainType
    channel: str
    display_name: str
    end_time: time
    start_range_min: Optional[time] = None
    start_range_max: Optional[time] = None
    allowed_days: Optional[tuple[int, ...]] = None  # 0=Monday, 5=Saturday, 6=Sunday
    content_type: Optional[str] = None


# ============================================================================
# RTS 1 Curtain Definitions (7 curtains)
# ============================================================================

RTS1_FICTION_MATIN = CurtainDefinition(
    curtain_id="rts1_fiction_matin",
    curtain_type=CurtainType.FICTION_MATIN,
    channel="RTS 1",
    display_name="Fiction Matin",
    end_time=time(11, 0),
    start_range_min=time(9, 20),
    start_range_max=time(10, 0),
    content_type="Films (no logo rouge)",
)

RTS1_APRES_MIDI_1 = CurtainDefinition(
    curtain_id="rts1_apres_midi_1",
    curtain_type=CurtainType.APRES_MIDI_1,
    channel="RTS 1",
    display_name="Apres-midi 1",
    end_time=time(14, 40),
    start_range_min=time(13, 15),
    start_range_max=time(13, 45),
    content_type="Telefilm",
)

RTS1_APRES_MIDI_2 = CurtainDefinition(
    curtain_id="rts1_apres_midi_2",
    curtain_type=CurtainType.APRES_MIDI_2,
    channel="RTS 1",
    display_name="Apres-midi 2",
    end_time=time(17, 40),
    start_range_min=time(14, 45),
    start_range_max=time(15, 15),
    content_type="Series, films rarement",
)

RTS1_RIDEAU_1 = CurtainDefinition(
    curtain_id="rts1_rideau_1",
    curtain_type=CurtainType.RIDEAU_1,
    channel="RTS 1",
    display_name="Rideau 1",
    end_time=time(20, 30),
    start_range_min=time(20, 0),
    start_range_max=time(20, 20),
    content_type="Magazine, emission",
)

RTS1_RIDEAU_2 = CurtainDefinition(
    curtain_id="rts1_rideau_2",
    curtain_type=CurtainType.RIDEAU_2,
    channel="RTS 1",
    display_name="Rideau 2",
    end_time=time(22, 20),
    start_range_min=time(20, 30),
    start_range_max=time(21, 8),
    content_type="Prime-time film",
)

RTS1_RIDEAU_3 = CurtainDefinition(
    curtain_id="rts1_rideau_3",
    curtain_type=CurtainType.RIDEAU_3,
    channel="RTS 1",
    display_name="Rideau 3",
    end_time=time(0, 0),  # Midnight
    start_range_min=time(22, 0),
    start_range_max=time(23, 0),
    content_type="Late evening",
)

RTS1_RIDEAU_4 = CurtainDefinition(
    curtain_id="rts1_rideau_4",
    curtain_type=CurtainType.RIDEAU_4,
    channel="RTS 1",
    display_name="Rideau 4",
    end_time=time(1, 50),
    start_range_min=time(0, 0),
    start_range_max=time(1, 10),
    content_type="Late night",
)


# ============================================================================
# RTS 2 Curtain Definitions (4 curtains)
# ============================================================================

RTS2_RIDEAU_1 = CurtainDefinition(
    curtain_id="rts2_rideau_1",
    curtain_type=CurtainType.RIDEAU_1,
    channel="RTS 2",
    display_name="Rideau 1",
    end_time=time(20, 50),
    start_range_min=time(20, 10),
    start_range_max=time(20, 30),
    content_type="Movies before 20:50",
)

RTS2_RIDEAU_2 = CurtainDefinition(
    curtain_id="rts2_rideau_2",
    curtain_type=CurtainType.RIDEAU_2,
    channel="RTS 2",
    display_name="Rideau 2",
    end_time=time(22, 50),
    start_range_min=time(20, 30),
    start_range_max=time(22, 0),
)

RTS2_RIDEAU_3 = CurtainDefinition(
    curtain_id="rts2_rideau_3",
    curtain_type=CurtainType.RIDEAU_3,
    channel="RTS 2",
    display_name="Rideau 3",
    end_time=time(0, 50),
    start_range_min=time(22, 30),
    start_range_max=time(0, 0),
)

RTS2_RIDEAU_4 = CurtainDefinition(
    curtain_id="rts2_rideau_4",
    curtain_type=CurtainType.RIDEAU_4,
    channel="RTS 2",
    display_name="Rideau 4",
    end_time=time(1, 30),
    start_range_min=time(0, 0),
    start_range_max=time(1, 0),
    allowed_days=(5,),  # Saturday only (5 = Saturday in Python weekday)
    content_type="Saturday only",
)


# ============================================================================
# Curtain Lookup Dictionaries
# ============================================================================

CURTAIN_DEFINITIONS: dict[str, CurtainDefinition] = {
    # RTS 1
    RTS1_FICTION_MATIN.curtain_id: RTS1_FICTION_MATIN,
    RTS1_APRES_MIDI_1.curtain_id: RTS1_APRES_MIDI_1,
    RTS1_APRES_MIDI_2.curtain_id: RTS1_APRES_MIDI_2,
    RTS1_RIDEAU_1.curtain_id: RTS1_RIDEAU_1,
    RTS1_RIDEAU_2.curtain_id: RTS1_RIDEAU_2,
    RTS1_RIDEAU_3.curtain_id: RTS1_RIDEAU_3,
    RTS1_RIDEAU_4.curtain_id: RTS1_RIDEAU_4,
    # RTS 2
    RTS2_RIDEAU_1.curtain_id: RTS2_RIDEAU_1,
    RTS2_RIDEAU_2.curtain_id: RTS2_RIDEAU_2,
    RTS2_RIDEAU_3.curtain_id: RTS2_RIDEAU_3,
    RTS2_RIDEAU_4.curtain_id: RTS2_RIDEAU_4,
}

CURTAINS_BY_CHANNEL: dict[str, List[CurtainDefinition]] = {
    "RTS 1": [
        RTS1_FICTION_MATIN,
        RTS1_APRES_MIDI_1,
        RTS1_APRES_MIDI_2,
        RTS1_RIDEAU_1,
        RTS1_RIDEAU_2,
        RTS1_RIDEAU_3,
        RTS1_RIDEAU_4,
    ],
    "RTS 2": [
        RTS2_RIDEAU_1,
        RTS2_RIDEAU_2,
        RTS2_RIDEAU_3,
        RTS2_RIDEAU_4,
    ],
}

# Ordered list of curtain types for one-hot encoding (7 types)
CURTAIN_TYPE_ORDER: List[CurtainType] = [
    CurtainType.FICTION_MATIN,
    CurtainType.APRES_MIDI_1,
    CurtainType.APRES_MIDI_2,
    CurtainType.RIDEAU_1,
    CurtainType.RIDEAU_2,
    CurtainType.RIDEAU_3,
    CurtainType.RIDEAU_4,
]


# ============================================================================
# Validation Functions
# ============================================================================


class CurtainValidationError(Exception):
    """Raised when curtain validation fails."""

    pass


def get_curtain_definition(curtain_id: str) -> CurtainDefinition:
    """
    Get a curtain definition by its ID.

    Args:
        curtain_id: The curtain identifier (e.g., "rts1_rideau_2")

    Returns:
        CurtainDefinition for the given curtain_id

    Raises:
        CurtainValidationError: If curtain_id is not found
    """
    if curtain_id not in CURTAIN_DEFINITIONS:
        available = list(CURTAIN_DEFINITIONS.keys())
        raise CurtainValidationError(
            f"Unknown curtain '{curtain_id}'. Available curtains: {available}"
        )
    return CURTAIN_DEFINITIONS[curtain_id]


def validate_curtain_for_day(curtain_id: str, day_of_week: int) -> None:
    """
    Validate that a curtain is available on the given day.

    Args:
        curtain_id: The curtain identifier
        day_of_week: Day of week as int (0=Monday, 5=Saturday, 6=Sunday)

    Raises:
        CurtainValidationError: If curtain is not available on the given day
    """
    curtain = get_curtain_definition(curtain_id)

    if curtain.allowed_days is not None and day_of_week not in curtain.allowed_days:
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        allowed_names = [day_names[day] for day in curtain.allowed_days]
        raise CurtainValidationError(
            f"Curtain '{curtain_id}' is only available on {allowed_names}. "
            f"Requested day: {day_names[day_of_week]}"
        )


def validate_curtain_for_channel(curtain_id: str, channel: str) -> None:
    """
    Validate that a curtain belongs to the specified channel.

    Args:
        curtain_id: The curtain identifier
        channel: Channel name ("RTS 1" or "RTS 2")

    Raises:
        CurtainValidationError: If curtain does not belong to the channel
    """
    curtain = get_curtain_definition(curtain_id)

    if curtain.channel != channel:
        available_for_channel = [c.curtain_id for c in CURTAINS_BY_CHANNEL.get(channel, [])]
        raise CurtainValidationError(
            f"Curtain '{curtain_id}' belongs to {curtain.channel}, not {channel}. "
            f"Available curtains for {channel}: {available_for_channel}"
        )


def get_curtain_type_one_hot(curtain_type: CurtainType) -> List[int]:
    """
    Get one-hot encoding for a curtain type (7 dims).

    Args:
        curtain_type: The CurtainType enum value

    Returns:
        List of 7 integers (0 or 1) representing one-hot encoding
    """
    return [1 if ct == curtain_type else 0 for ct in CURTAIN_TYPE_ORDER]


def get_curtains_for_channel(channel: str) -> List[CurtainDefinition]:
    """
    Get all curtains available for a given channel.

    Args:
        channel: Channel name ("RTS 1" or "RTS 2")

    Returns:
        List of CurtainDefinition objects for the channel

    Raises:
        CurtainValidationError: If channel is unknown
    """
    if channel not in CURTAINS_BY_CHANNEL:
        raise CurtainValidationError(f"Unknown channel '{channel}'. Expected 'RTS 1' or 'RTS 2'")
    return CURTAINS_BY_CHANNEL[channel]


def get_curtain_for_broadcast(
    start_time: str,
    channel: str,
    day_of_week: int
) -> Optional[str]:
    """
    Map broadcast timing to curtain_id using start_range boundaries.

    Uses start_range_min and start_range_max to determine valid curtain windows.
    A broadcast belongs to a curtain if its start time falls within [start_range_min, start_range_max].

    Args:
        start_time: Broadcast start time in "HH:MM:SS" format
        channel: Channel name ("RTS 1" or "RTS 2")
        day_of_week: Day of week as int (0=Monday, 5=Saturday, 6=Sunday)

    Returns:
        curtain_id if the broadcast fits a curtain slot, None otherwise
    """
    parts = start_time.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    broadcast_minutes = hours * 60 + minutes

    curtains = CURTAINS_BY_CHANNEL.get(channel, [])

    for curtain in curtains:
        if curtain.allowed_days is not None and day_of_week not in curtain.allowed_days:
            continue

        if curtain.start_range_min is None or curtain.start_range_max is None:
            continue

        lower_minutes = curtain.start_range_min.hour * 60 + curtain.start_range_min.minute
        upper_minutes = curtain.start_range_max.hour * 60 + curtain.start_range_max.minute

        # Handle midnight crossover (e.g., 22:30-00:00)
        if upper_minutes <= lower_minutes:
            if broadcast_minutes >= lower_minutes or broadcast_minutes < upper_minutes:
                return curtain.curtain_id
        else:
            if lower_minutes <= broadcast_minutes < upper_minutes:
                return curtain.curtain_id

    return None
