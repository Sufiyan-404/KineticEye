"""
test_repetition_counter.py

Tests for repetition_counter.py, using SIMULATED phase sequences.
No webcam, video file, OpenCV or MediaPipe is involved.

Run either way (put this file next to repetition_counter.py):
    python -m pytest -v test_repetition_counter.py
    python test_repetition_counter.py        (simple built-in runner)

Like the other test files in this project, these are plain functions that
use `assert`, so pytest collects them without needing anything extra.
"""

import ast
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

import repetition_counter
from repetition_counter import (
    RepetitionCounter,
    count_repetitions,
    STATE_WAITING_FOR_TOP,
    STATE_GOING_DOWN,
    STATE_COMING_UP,
)
from form_analyzer import (
    analyze_push_up_form,
    PHASE_TOP, PHASE_MIDDLE, PHASE_BOTTOM, PHASE_UNKNOWN,
)

T, M, B, U = PHASE_TOP, PHASE_MIDDLE, PHASE_BOTTOM, PHASE_UNKNOWN


# ---- Helpers (their names do not start with "test_") --------------------

def feed(phases):
    """Feed a whole sequence to a new counter and return the counter."""
    counter = RepetitionCounter()
    for phase in phases:
        counter.update(phase)
    return counter


def counts_after_each_frame(phases):
    """The running total after every frame (shows WHEN a rep is counted)."""
    counter = RepetitionCounter()
    totals = []
    for phase in phases:
        counter.update(phase)
        totals.append(counter.count)
    return totals


# ---- 0. The labels really come from the existing analyzer ----------------

def test_phase_labels_match_the_form_analyzer():
    assert (T, M, B, U) == ("TOP", "MIDDLE", "BOTTOM", "UNKNOWN")


def test_real_analyzer_phases_drive_the_counter():
    # Phases produced by the REAL analyzer from elbow angles (no webcam).
    elbow_angles = [175, 150, 120, 85, 120, 150, 175]     # one full push-up
    counter = RepetitionCounter()
    for angle in elbow_angles:
        phase = analyze_push_up_form(elbow_angle=angle)["movement_phase"]
        counter.update(phase)
    assert counter.count == 1


# ---- 1. One complete push-up ---------------------------------------------

def test_one_complete_push_up():
    assert feed([T, M, B, M, T]).count == 1


def test_one_complete_push_up_without_middle_frames():
    # Fast movement: the video skipped straight from TOP to BOTTOM.
    assert feed([T, B, T]).count == 1


def test_rep_is_counted_exactly_when_the_movement_returns_to_top():
    assert counts_after_each_frame([T, M, B, M, T]) == [0, 0, 0, 0, 1]


def test_update_returns_true_only_on_the_completing_frame():
    counter = RepetitionCounter()
    results = [counter.update(p) for p in [T, M, B, M, T, T]]
    assert results == [False, False, False, False, True, False]


def test_slow_push_up_with_many_repeated_frames():
    sequence = [T] * 5 + [M] * 4 + [B] * 6 + [M] * 4 + [T] * 5
    assert feed(sequence).count == 1


# ---- 2. Multiple complete push-ups ---------------------------------------

def test_three_complete_push_ups():
    one_rep = [T, M, B, M]
    assert feed(one_rep * 3 + [T]).count == 3


def test_ten_complete_push_ups():
    sequence = ([T, M, B, M] * 10) + [T]
    assert count_repetitions(sequence) == 10


def test_back_to_back_reps_share_the_top_frame():
    # The TOP that ends rep 1 is also the start of rep 2.
    assert counts_after_each_frame([T, B, T, B, T]) == [0, 0, 1, 1, 2]


def test_reps_separated_by_a_rest_at_the_top():
    sequence = [T, M, B, M, T] + [T] * 20 + [M, B, M, T]
    assert feed(sequence).count == 2


# ---- 3. Repeated TOP frames without a completed cycle --------------------

def test_repeated_top_frames_never_count():
    counter = feed([T] * 100)
    assert counter.count == 0
    assert counter.state == STATE_GOING_DOWN


def test_repeated_top_frames_after_a_rep_do_not_add_counts():
    counter = feed([T, M, B, M] + [T] * 50)
    assert counter.count == 1


def test_top_middle_top_is_not_a_rep():
    assert feed([T, M, T, M, T, M, T]).count == 0


# ---- 4. A partial movement that never returns to TOP ---------------------

def test_going_down_and_staying_down_is_not_counted():
    counter = feed([T, M, B, B, B])
    assert counter.count == 0
    assert counter.state == STATE_COMING_UP


def test_coming_halfway_back_up_is_not_counted():
    assert feed([T, M, B, M, M, M]).count == 0


def test_partial_movement_that_never_reaches_bottom():
    # Goes down only to MIDDLE and returns: too shallow, no BOTTOM seen.
    assert feed([T, M, M, M, T]).count == 0


def test_video_that_ends_mid_rep():
    # Two full reps, then the video stops while the person is at the bottom.
    sequence = [T, M, B, M, T, M, B, M, T, M, B]
    assert feed(sequence).count == 2


# ---- 5. Sequences beginning in the middle of a movement ------------------

def test_starting_in_the_middle_waits_for_a_top():
    counter = feed([M, M, M])
    assert counter.count == 0
    assert counter.state == STATE_WAITING_FOR_TOP


def test_starting_at_the_bottom_does_not_count_the_first_return_to_top():
    # M, B, M, T finishes a movement that did not START at the top.
    # The first TOP only "arms" the counter.
    assert counts_after_each_frame([M, B, M, T]) == [0, 0, 0, 0]


def test_a_full_rep_after_a_mid_movement_start_is_counted():
    sequence = [M, B, M, T, M, B, M, T]
    assert counts_after_each_frame(sequence) == [0, 0, 0, 0, 0, 0, 0, 1]


def test_bottom_frames_before_the_first_top_are_ignored():
    counter = feed([B, B, B])
    assert counter.count == 0
    assert counter.state == STATE_WAITING_FOR_TOP


# ---- 6. UNKNOWN phases ----------------------------------------------------

def test_unknown_frames_do_not_change_the_state_or_count():
    counter = RepetitionCounter()
    counter.update(T)
    counter.update(B)
    state_before = counter.state
    for _ in range(5):
        assert counter.update(U) is False
    assert counter.state == state_before
    assert counter.count == 0


def test_unknown_frames_inside_a_rep_do_not_break_it():
    # Wrist briefly hidden at the bottom and on the way up.
    assert feed([T, M, U, U, B, U, M, U, T]).count == 1


def test_unknown_frames_alone_never_count():
    counter = feed([U] * 20)
    assert counter.count == 0
    assert counter.state == STATE_WAITING_FOR_TOP


def test_unknown_frames_do_not_create_a_rep_by_themselves():
    assert feed([T, U, U, U, T]).count == 0


def test_unknown_frames_are_tallied():
    counter = feed([T, U, M, U, U, B, T])
    assert counter.unknown_frames == 3
    assert counter.invalid_frames == 0


# ---- 7. Invalid inputs ----------------------------------------------------

INVALID_INPUTS = [
    None, 5, 2.5, True, False, [], ["TOP"], ("TOP",), {"phase": "TOP"},
    b"TOP", object(), np.float64(1.0), float("nan"),
    "", " ", "top", "Top", "bottom", "MID", " TOP", "TOP ", "GOOD",
]


def test_invalid_inputs_do_not_crash_and_return_false():
    for bad in INVALID_INPUTS:
        counter = RepetitionCounter()
        assert counter.update(bad) is False, repr(bad)
        assert counter.count == 0, repr(bad)


def test_invalid_inputs_are_tallied_and_not_counted_as_unknown():
    counter = feed(INVALID_INPUTS)
    assert counter.invalid_frames == len(INVALID_INPUTS)
    assert counter.unknown_frames == 0
    assert counter.count == 0


def test_invalid_inputs_do_not_change_the_state():
    counter = RepetitionCounter()
    counter.update(T)
    counter.update(B)
    for bad in INVALID_INPUTS:
        counter.update(bad)
        assert counter.state == STATE_COMING_UP, repr(bad)
    counter.update(T)
    assert counter.count == 1


def test_invalid_inputs_inside_a_rep_do_not_break_it():
    assert feed([T, None, M, "top", B, 42, M, [], T]).count == 1


def test_invalid_inputs_cannot_complete_a_rep():
    # "top" in lower case is not the label TOP.
    assert feed([T, B, "top"]).count == 0


def test_whole_sequence_given_as_one_wrong_object_is_just_invalid():
    # A common mistake: passing the list instead of one phase.
    counter = RepetitionCounter()
    assert counter.update([T, M, B, M, T]) is False
    assert counter.count == 0
    assert counter.invalid_frames == 1


# ---- 8. Incomplete movements never increase the count --------------------

INCOMPLETE_SEQUENCES = {
    "empty": [],
    "only MIDDLE": [M, M, M, M],
    "only BOTTOM": [B, B, B],
    "only TOP": [T, T, T],
    "only UNKNOWN": [U, U, U],
    "TOP then MIDDLE": [T, M, M],
    "TOP, MIDDLE, TOP (too shallow)": [T, M, T],
    "TOP, BOTTOM (never came back)": [T, B],
    "BOTTOM then TOP (never started at the top)": [B, T],
    "MIDDLE, BOTTOM, TOP (started mid-movement)": [M, B, T],
    "BOTTOM, MIDDLE, TOP": [B, M, T],
    "TOP, UNKNOWN, TOP": [T, U, T],
    "TOP, invalid, BOTTOM, invalid": [T, None, B, "x"],
}


def test_incomplete_movements_do_not_increase_the_count():
    for name, sequence in INCOMPLETE_SEQUENCES.items():
        counter = feed(sequence)
        assert counter.count == 0, name
        assert counts_after_each_frame(sequence) == [0] * len(sequence), name


def test_the_count_never_goes_down():
    sequence = [T, M, B, M, T, M, M, U, None, T, B, B, M, T, T, M, T]
    totals = counts_after_each_frame(sequence)
    assert totals == sorted(totals)
    # Cycle 1: T,M,B,M,T.  Cycle 2: T,B,B,M,T.  The rest is shallow.
    assert totals[-1] == 2


def test_an_incomplete_movement_after_a_rep_keeps_the_old_count():
    counter = feed([T, M, B, M, T])               # 1 rep
    for phase in [M, M, T, M, T]:                 # shallow movements only
        counter.update(phase)
    assert counter.count == 1


# ---- 9. State transitions -------------------------------------------------

def test_initial_state():
    counter = RepetitionCounter()
    assert counter.count == 0
    assert counter.state == STATE_WAITING_FOR_TOP
    assert counter.unknown_frames == 0
    assert counter.invalid_frames == 0


def test_state_transitions_through_one_rep():
    counter = RepetitionCounter()
    steps = [
        (M, STATE_WAITING_FOR_TOP),
        (B, STATE_WAITING_FOR_TOP),
        (T, STATE_GOING_DOWN),
        (T, STATE_GOING_DOWN),
        (M, STATE_GOING_DOWN),
        (B, STATE_COMING_UP),
        (B, STATE_COMING_UP),
        (M, STATE_COMING_UP),
        (T, STATE_GOING_DOWN),      # rep counted here
    ]
    for phase, expected_state in steps:
        counter.update(phase)
        assert counter.state == expected_state, (phase, expected_state)
    assert counter.count == 1


def test_bouncing_between_bottom_and_middle_still_counts_once():
    # Flicker near the bottom threshold must not create extra reps.
    assert feed([T, B, M, B, M, B, T]).count == 1


# ---- 10. reset(), count_repetitions() and independence --------------------

def test_reset_clears_everything():
    counter = feed([T, M, B, M, T, B, U, None])
    assert counter.count == 1
    counter.reset()
    assert counter.count == 0
    assert counter.state == STATE_WAITING_FOR_TOP
    assert counter.unknown_frames == 0
    assert counter.invalid_frames == 0
    # And it works normally afterwards.
    for phase in [T, M, B, M, T]:
        counter.update(phase)
    assert counter.count == 1


def test_two_counters_do_not_share_state():
    first, second = RepetitionCounter(), RepetitionCounter()
    for phase in [T, M, B, M, T]:
        first.update(phase)
    assert first.count == 1
    assert second.count == 0
    assert second.state == STATE_WAITING_FOR_TOP


def test_count_repetitions_helper():
    assert count_repetitions([]) == 0
    assert count_repetitions([T, M, B, M, T]) == 1
    assert count_repetitions(iter([T, B, T, B, T])) == 2     # any iterable
    assert count_repetitions((p for p in [T, B, T])) == 1     # generator


def test_count_is_a_plain_int_and_read_only():
    counter = feed([T, B, T])
    assert type(counter.count) is int
    try:
        counter.count = 99
    except AttributeError:
        pass
    else:
        raise AssertionError("count should be read-only")
    assert counter.count == 1


def test_update_prints_nothing():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        feed([T, M, B, None, U, M, T])
    assert output.getvalue() == ""


def test_module_does_not_import_opencv_mediapipe_or_video_code():
    # The counter may use the standard library and form_analyzer (for the
    # phase labels) and nothing else.
    source = Path(repetition_counter.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported, "expected the module to import something"
    assert imported <= set(sys.stdlib_module_names) | {"form_analyzer"}, imported
    for forbidden in ("cv2", "mediapipe", "numpy", "pose_detector",
                      "pushup_angle_demo", "video_analysis_demo",
                      "single_frame_demo"):
        assert forbidden not in imported


# ---- 11. Focused checks of the counting rule -----------------------------

def test_normal_complete_cycle_counts_once():
    counter = feed([T, T, M, M, B, B, M, M, T, T])
    assert counter.count == 1


def test_multiple_complete_cycles_are_all_counted():
    counter = feed([T, M, B, M, T, M, B, M, T, B, T])
    assert counter.count == 3


def test_reaching_bottom_without_returning_to_top_does_not_count():
    # The count stays 0 on every frame, including the long stay at BOTTOM.
    sequence = [T, M, B, B, B, B, B]
    assert counts_after_each_frame(sequence) == [0] * len(sequence)


def test_moving_partly_up_and_dropping_back_to_bottom_does_not_count():
    # Up to MIDDLE (not TOP), then down to BOTTOM again, repeatedly.
    sequence = [T, B, M, B, M, B, M, M, B]
    assert counts_after_each_frame(sequence) == [0] * len(sequence)


def test_middle_and_bottom_bouncing_alone_never_counts():
    assert feed([M, B] * 10).count == 0
    assert feed([T] + [M, B] * 10).count == 0


def test_a_bounce_is_only_counted_when_top_is_finally_reached():
    # One cycle, however many times the person bounced before reaching TOP.
    sequence = [T, B, M, B, M, B, M, T]
    assert counts_after_each_frame(sequence) == [0, 0, 0, 0, 0, 0, 0, 1]


def test_many_repeated_top_frames_do_not_inflate_the_count():
    sequence = [T] * 30 + [B, M, T] + [T] * 30
    assert feed(sequence).count == 1


def test_unknown_phase_behaviour_is_unchanged():
    # UNKNOWN is ignored: no count, no state change, tallied separately.
    counter = RepetitionCounter()
    for phase in [T, U, B, U, T]:
        counter.update(phase)
    assert counter.count == 1
    assert counter.unknown_frames == 2
    assert counter.invalid_frames == 0
    # A long run of UNKNOWN neither resets nor times out the cycle.
    counter = feed([T, B] + [U] * 1000)
    assert counter.state == STATE_COMING_UP
    counter.update(T)
    assert counter.count == 1


def reference_count(phases):
    """
    An independent way to count cycles, written differently on purpose:
    it ignores UNKNOWN, then searches by position for TOP, then a later
    BOTTOM, then a later TOP. Each returning TOP starts the next search.
    """
    frames = [p for p in phases if p in (T, M, B)]
    start, reps = 0, 0
    while True:
        try:
            top = frames.index(T, start)
            bottom = frames.index(B, top + 1)
            back_at_top = frames.index(T, bottom + 1)
        except ValueError:
            return reps
        reps += 1
        start = back_at_top


def test_counter_agrees_with_an_independent_count_on_every_short_sequence():
    import itertools
    for length in range(0, 8):
        for sequence in itertools.product([T, M, B, U], repeat=length):
            assert count_repetitions(sequence) == reference_count(sequence), \
                sequence


# ---- Simple runner so `python test_repetition_counter.py` works ----------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as error:      # AssertionError or an unexpected crash
            failed += 1
            print(f"FAIL  {name}  ({type(error).__name__}: {error})")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
