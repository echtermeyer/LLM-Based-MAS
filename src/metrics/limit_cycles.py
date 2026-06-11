from typing import Dict, List, Optional, Tuple


def _vote_tuple(phase_b: List[Dict]) -> Tuple:
    return tuple(ag['vote'] for ag in phase_b)


def _all_unanimous(vt: Tuple) -> bool:
    return len(set(vt)) == 1


def _trim_trajectory(trajectory: List[Dict]) -> int:
    T_r = len(trajectory) - 1
    terminal_vt = _vote_tuple(trajectory[T_r]['phase_b'])
    if not _all_unanimous(terminal_vt):
        return T_r
    k = 0
    for t in range(T_r, -1, -1):
        if _vote_tuple(trajectory[t]['phase_b']) == terminal_vt:
            k += 1
        else:
            break
    return T_r - max(0, k - 1)


def detect_limit_cycle(trajectory: List[Dict]) -> Optional[Dict]:
    """
    Returns a dict with:
      lc      — True iff a genuine oscillation was detected
      period  — rounds between first and second visit (None if no cycle)
      seq     — the trimmed vote-profile sequence
      L       — length of trimmed sequence

    A genuine oscillation requires:
      - period >= 2  (the group left the state before returning)
      - the recurring profile is non-unanimous  (unanimous recurrence is just
        early-stopping noise, not an oscillation)
    """
    T_tilde = _trim_trajectory(trajectory)
    seq = [_vote_tuple(trajectory[t]['phase_b']) for t in range(T_tilde + 1)]
    if len(seq) < 3:
        return None
    seen = {}
    for t, vt in enumerate(seq):
        if vt in seen:
            period = t - seen[vt]
            genuine = period >= 2 and not _all_unanimous(vt)
            return {
                'lc': genuine,
                'period': period if genuine else None,
                'seq': seq,
                'L': len(seq),
            }
        seen[vt] = t
    return {'lc': False, 'period': None, 'seq': seq, 'L': len(seq)}
