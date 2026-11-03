"""In-video quiz checkpoints based on duration and completed sentences."""

CHECKPOINT_INTERVAL_SECONDS = 600
CHECKPOINT_MIN_DURATION_SECONDS = 1200


def final_question_count(duration: float) -> int:
    """Return the required number of questions for the whole-video quiz."""
    import math

    return max(10, math.ceil(duration / CHECKPOINT_INTERVAL_SECONDS) * 10)


def make_checkpoints(
    duration: float,
    sentences: list[dict],
    interval: int = CHECKPOINT_INTERVAL_SECONDS,
) -> list[dict]:
    """Schedule in-video quizzes, excluding the video-end Final Quiz."""
    if duration < CHECKPOINT_MIN_DURATION_SECONDS:
        return []

    points = []
    targets = list(range(interval, int(duration), interval))
    ends = []
    for sentence in sentences:
        cue_ends = sentence.get('cue_end_times', [])
        if cue_ends:
            ends.append((max(cue_ends), sentence['id']))

    previous = 0.0
    for target in targets:
        options = [
            (time, sentence_id)
            for time, sentence_id in ends
            if target - 30 <= time <= target + 60 and time <= duration and time > previous
        ]
        after = [item for item in options if item[0] >= target]
        chosen = min(after, key=lambda item: item[0]) if after else (
            max(options, key=lambda item: item[0]) if options else None
        )
        trigger = chosen[0] if chosen else float(target)
        if points and trigger - points[-1]['trigger_time'] < 60:
            continue
        points.append({
            'id': f'cp{len(points) + 1:04d}',
            'target_time': target,
            'trigger_time': trigger,
            'reason': 'sentence_boundary' if chosen else 'forced_interval',
            'interrupts_sentence': not bool(chosen),
            'is_final': False,
            'question_ids': [],
        })
        previous = trigger
    return points
