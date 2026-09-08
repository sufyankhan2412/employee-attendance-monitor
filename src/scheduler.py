"""
scheduler.py

Two things live here:

1. generate_daily_presence_schedule() - builds the day's list of random
   workplace-screenshot times from the time windows in config.yaml
   (e.g. 09:00-10:30, 10:30-12:00, 12:00-13:00, [lunch, no window],
   14:00-15:30, 15:30-17:00). One random instant per window, each
   guaranteed to be at least `min_gap_minutes` after the previous one.
   Five windows -> five screenshots/day/camera, matching "presence of
   5 screenshots" per employee/workspace.

2. gate_mode_for_time() - tells the main-gate daemon whether right now
   should be treated as CHECK_IN, CHECK_OUT, or idle (before opening
   time), based on Pakistan time (or whatever timezone you configure).

Both are pure functions with no I/O, so they're easy to unit test
before you ever point them at a real camera.
"""

import random
from datetime import datetime, timedelta, time as dtime


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def generate_daily_presence_schedule(windows_cfg, min_gap_minutes, tz, day=None):
    """
    windows_cfg: list of {"start": "HH:MM", "end": "HH:MM"} dicts, in order,
        e.g. config["presence"]["windows"]. Just omit the lunch hour from
        this list entirely - that's how "no screenshots 1-2pm" is expressed.
    min_gap_minutes: minimum spacing enforced between consecutive picks
        (e.g. 40), even across a window boundary.
    tz: a tzinfo object, e.g. ZoneInfo("Asia/Karachi") - pass this in so
        the module doesn't need to hardcode a timezone.
    day: a date to schedule for; defaults to "today" in tz.

    Returns: sorted list of tz-aware datetimes, one per window.
    """
    if day is None:
        day = datetime.now(tz).date()

    min_gap = timedelta(minutes=min_gap_minutes)
    chosen = []
    previous = None

    for w in windows_cfg:
        start_t, end_t = _parse_hhmm(w["start"]), _parse_hhmm(w["end"])
        start_dt = datetime.combine(day, start_t, tzinfo=tz)
        end_dt = datetime.combine(day, end_t, tzinfo=tz)

        lower_bound = start_dt
        if previous is not None and previous + min_gap > lower_bound:
            lower_bound = previous + min_gap

        if lower_bound >= end_dt:
            # Enforcing the gap pushed us past this window's end - take the
            # earliest legal instant instead of skipping the shot entirely.
            picked = min(lower_bound, end_dt)
        else:
            span_seconds = int((end_dt - lower_bound).total_seconds())
            picked = lower_bound + timedelta(seconds=random.randint(0, span_seconds))

        chosen.append(picked)
        previous = picked

    return chosen


def gate_mode_for_time(now: datetime, checkin_start: dtime, checkout_start: dtime):
    """
    Decide what the main gate camera should be doing right now.

    - Before checkin_start (e.g. 08:00): None -> gate is idle, nobody is
      expected yet, don't bother running recognition.
    - From checkin_start up to checkout_start (e.g. 08:00-17:00): "CHECK_IN"
      -> any newly-recognized employee today gets a check-in.
    - From checkout_start onward (e.g. 17:00 until midnight): "CHECK_OUT"
      -> any recognized employee gets a check-out (once per day). The
      daemon just keeps running through the evening, so it naturally
      stays "active" until whenever the last employee actually walks
      past the camera - there's no separate timer needed for that part.
    """
    t = now.time()
    if t < checkin_start:
        return None
    if t < checkout_start:
        return "CHECK_IN"
    return "CHECK_OUT"