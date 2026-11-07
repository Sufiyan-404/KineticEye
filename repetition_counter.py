"""
repetition_counter.py

Counts completed push-ups for the Sports Video Analytics & Action Counter
project, using the movement phases that form_analyzer.py already produces.

Where this module fits in the pipeline:

    landmark coordinates -> joint angles -> analyze_push_up_form()
                                                   |
                                    result["movement_phase"]
                                                   |
                                                   v
                                    repetition_counter.py   <-- THIS FILE
                                                   |
                                                   v
                                          total repetitions

This module does NOT measure angles, does NOT apply any form rules and does
NOT decide what TOP, MIDDLE or BOTTOM mean. It only watches the sequence of
phase labels, one per video frame, and counts completed cycles. The only
project file it imports is form_analyzer.py (for the phase name constants),
which itself uses only Python's standard library. It has no connection to
OpenCV, MediaPipe, the webcam or video files, and it prints nothing.

The phase labels (defined in form_analyzer.py):

    "TOP"      arms nearly straight (elbow angle >= 160 degrees)
    "MIDDLE"   anywhere between top and bottom (elbow angle 91 - 159)
    "BOTTOM"   arms strongly bent (elbow angle <= 90 degrees)
    "UNKNOWN"  the elbow angle was missing, so the phase could not be found

What counts as ONE repetition:

    TOP  ->  (any number of MIDDLE frames)  ->  BOTTOM  ->  (any MIDDLE)  ->  TOP

The repetition is counted at the moment the movement returns to TOP.

Simple usage:

    counter = RepetitionCounter()
    for phase in phases_one_per_frame:
        counter.update(phase)
    print(counter.count)

How each kind of input is treated (the counter never crashes on bad input):

    TOP / MIDDLE / BOTTOM   used by the state machine described below.
    "UNKNOWN"               the frame is IGNORED: the state and the count do
                            not change. (We know nothing about that frame, so
                            we neither move forward nor start over.) It is
                            tallied in `unknown_frames`.
    anything else           INVALID input (None, a number, a misspelled or
                            lower-case label such as "top", ...). It is also
                            IGNORED, exactly like "UNKNOWN", but tallied in
                            `invalid_frames` so a bug in the caller can be
                            noticed. Nothing is raised, because one bad frame
                            should not stop the processing of a whole video.

The three states (the "memory" of the counter):

    WAITING_FOR_TOP  Nothing is in progress. We have not yet seen a TOP frame
                     (start of the video, or the video began mid-movement).
                     Only a TOP frame moves us on. MIDDLE and BOTTOM are
                     ignored, because a cycle must START at the top.

    GOING_DOWN       We have seen TOP and are now at the top or on the way
                     down. A BOTTOM frame moves us on. TOP and MIDDLE frames
                     change nothing, so repeated TOP frames, or going
                     TOP -> MIDDLE -> TOP (a partial push-up), never count.

    COMING_UP        We have reached BOTTOM and are on the way back up.
                     A TOP frame completes the repetition: the count goes up
                     by one and we go back to GOING_DOWN (we are at the top,
                     ready for the next push-up). BOTTOM and MIDDLE frames
                     change nothing.
"""

from form_analyzer import (
    PHASE_TOP,
    PHASE_MIDDLE,
    PHASE_BOTTOM,
    PHASE_UNKNOWN,
)


# ===========================================================================
# 1. NAMES (kept as constants to avoid typos elsewhere)
# ===========================================================================

STATE_WAITING_FOR_TOP = "WAITING_FOR_TOP"
STATE_GOING_DOWN = "GOING_DOWN"
STATE_COMING_UP = "COMING_UP"

# The phase labels the state machine understands.
VALID_PHASES = (PHASE_TOP, PHASE_MIDDLE, PHASE_BOTTOM)


# ===========================================================================
# 2. THE COUNTER
# ===========================================================================

class RepetitionCounter:
    """
    Counts completed push-up cycles from a stream of movement phases.

    Call update(phase) once per frame. Read `count` at any time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Start again from zero (count, state and tallies)."""
        self._count = 0
        self._state = STATE_WAITING_FOR_TOP
        self._unknown_frames = 0
        self._invalid_frames = 0

    # ---- Read-only information -------------------------------------------

    @property
    def count(self):
        """Number of completed repetitions so far (a whole number)."""
        return self._count

    @property
    def state(self):
        """
        Where the counter is in a cycle: STATE_WAITING_FOR_TOP,
        STATE_GOING_DOWN or STATE_COMING_UP. Useful for debugging.
        """
        return self._state

    @property
    def unknown_frames(self):
        """How many "UNKNOWN" frames were ignored."""
        return self._unknown_frames

    @property
    def invalid_frames(self):
        """How many invalid (unrecognised) inputs were ignored."""
        return self._invalid_frames

    # ---- The one method to call for every frame --------------------------

    def update(self, phase):
        """
        Give the counter the movement phase of the next frame.

        Parameters
        ----------
        phase : str
            One of "TOP", "MIDDLE", "BOTTOM" or "UNKNOWN" (the values of
            analyze_push_up_form()["movement_phase"]). Anything else is
            treated as invalid and ignored (see the module description).

        Returns
        -------
        bool
            True if THIS frame completed a repetition, otherwise False.
            You can ignore the return value and just read `count`.
        """
        # An unhashable or odd object (list, None, number) is simply "not one
        # of our labels", so a type check keeps the comparisons below safe.
        if not isinstance(phase, str):
            self._invalid_frames += 1
            return False

        if phase == PHASE_UNKNOWN:
            self._unknown_frames += 1
            return False

        if phase not in VALID_PHASES:
            self._invalid_frames += 1
            return False

        # From here on, phase is exactly "TOP", "MIDDLE" or "BOTTOM".
        if self._state == STATE_WAITING_FOR_TOP:
            if phase == PHASE_TOP:
                self._state = STATE_GOING_DOWN

        elif self._state == STATE_GOING_DOWN:
            if phase == PHASE_BOTTOM:
                self._state = STATE_COMING_UP

        elif self._state == STATE_COMING_UP:
            if phase == PHASE_TOP:
                self._count += 1
                self._state = STATE_GOING_DOWN
                return True

        return False


# ===========================================================================
# 3. CONVENIENCE FUNCTION
# ===========================================================================

def count_repetitions(phases):
    """
    Count the completed repetitions in a whole sequence of phases.

    `phases` is any iterable of phase labels, one per frame. A fresh counter
    is used, so the result depends only on this sequence.
    """
    counter = RepetitionCounter()
    for phase in phases:
        counter.update(phase)
    return counter.count
