"""Stage 11 pilot (E30) -- EpisodeWindowStats episode-count double-counting fix.

Found 2026-08-07 auditing Stage 11 v4's results: ``window.episodes`` was
incremented TWICE per real episode -- once explicitly in the training loop
(right next to the completions/collisions/truncations counters) and again
inside ``EpisodeWindowStats.record_episode``, which was called
unconditionally every time the explicit increment also ran. Since
``completion_rate``/``collision_free_rate``/``truncation_rate``/
``mean_U_mean``/``min_U_mean`` all divide by ``self.episodes``, this silently
halved every reported rate across the entire Stage 11 pilot (v1-v4) -- the
completions/collisions/truncations counters (incremented once per episode)
never summed to the doubled ``episodes`` count.
"""

from __future__ import annotations

from thesis.pilots.stage11_dyad_merge_runner import EpisodeWindowStats


def test_record_episode_does_not_increment_episodes():
    window = EpisodeWindowStats()
    window.record_episode(welfare_by_stakeholder={"ramp": 0.5, "mainline": 0.5}, first_crosser=None)
    assert window.episodes == 0


def test_episodes_counter_matches_terminal_reason_counts_after_one_real_episode():
    """Mirrors the training loop's actual call pattern: the explicit
    increment plus completions/collisions/truncations bookkeeping, THEN
    record_episode. After one episode, episodes must equal exactly the sum
    of the three terminal-reason counters -- this is the invariant the
    double-counting bug violated (episodes was always double that sum)."""
    window = EpisodeWindowStats()
    window.episodes += 1
    window.completions += 1  # this episode ended in success
    window.record_episode(welfare_by_stakeholder={"ramp": 1.0, "mainline": 1.0}, first_crosser=("ramp", "V0"))

    assert window.episodes == 1
    assert window.episodes == window.completions + window.collisions + window.truncations


def test_window_as_dict_rates_are_not_halved_over_several_episodes():
    window = EpisodeWindowStats()
    # 4 episodes: 2 success, 1 collision, 1 truncation -- same call pattern
    # the training loop uses for every terminal episode.
    for is_success, is_collision, is_truncation in [
        (True, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ]:
        window.episodes += 1
        if is_success:
            window.completions += 1
        elif is_collision:
            window.collisions += 1
        elif is_truncation:
            window.truncations += 1
        window.record_episode(welfare_by_stakeholder={"ramp": 0.5, "mainline": 0.5}, first_crosser=None)

    d = window.as_dict()
    assert d["episodes"] == 4
    assert d["completion_rate"] == 0.5  # 2/4, not 2/8
    assert d["collision_free_rate"] == 0.75  # (4-1)/4, not (8-1)/8
    assert d["truncation_rate"] == 0.25  # 1/4, not 1/8
