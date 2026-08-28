"""Measure longitudinal mobility habits directly from raw YJMob observations.

The analysis never interpolates missing slots and never reconstructs the masked
civil calendar.  It uses the same deterministic user cohort as a prepared
dataset manifest, then reports observation gaps, P(location | user, timeslot),
chronological holdout prediction, lagged cross-day similarity, and a within-day
timeslot-shuffled null.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils.prepare_yjmob import (
    OFFICIAL_DATASET1_MD5,
    grid_cell_id,
    iter_observations,
    md5sum,
)
from data_utils.profile_yjmob import distribution_summary


MASKED_CALENDAR_SOURCE = "https://doi.org/10.1038/s41597-024-03237-9"


def _safe_distribution(values: np.ndarray | list[float]) -> dict[str, float | int] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return distribution_summary(values)


def _mode(values: np.ndarray) -> tuple[int, int]:
    values = np.asarray(values, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    max_count = int(counts.max())
    # The smallest cell id is a deterministic but otherwise neutral tie-break.
    return int(unique[counts == max_count].min()), max_count


def read_manifest_users(manifest_path: Path) -> tuple[np.ndarray, int]:
    with Path(manifest_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")
    required = {"uid", "day"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    users = np.asarray(sorted({int(row["uid"]) for row in rows}), dtype=np.int64)
    return users, len(rows)


def load_observation_cube(
    raw_path: Path,
    users: np.ndarray,
    *,
    day_count: int = 75,
    timeslot_count: int = 48,
) -> tuple[np.ndarray, dict[str, int]]:
    users = np.asarray(users, dtype=np.int64)
    if users.size == 0:
        raise ValueError("The cohort is empty")
    user_to_index = {int(uid): index for index, uid in enumerate(users)}
    cube = np.full(
        (len(users), int(day_count), int(timeslot_count)),
        -1,
        dtype=np.int32,
    )
    max_uid = int(users.max())
    rows_scanned = 0
    rows_retained = 0
    for observation in iter_observations(Path(raw_path)):
        rows_scanned += 1
        if observation.uid > max_uid:
            break
        user_index = user_to_index.get(observation.uid)
        if user_index is None:
            continue
        if not 0 <= observation.day < day_count:
            raise ValueError(
                f"day {observation.day} is outside [0, {day_count - 1}]"
            )
        if not 0 <= observation.timeslot < timeslot_count:
            raise ValueError(
                f"timeslot {observation.timeslot} is outside "
                f"[0, {timeslot_count - 1}]"
            )
        cube[user_index, observation.day, observation.timeslot] = grid_cell_id(
            observation.x,
            observation.y,
        )
        rows_retained += 1
    observed_users = np.unique(np.flatnonzero(np.any(cube >= 0, axis=(1, 2))))
    if len(observed_users) != len(users):
        missing = users[np.setdiff1d(np.arange(len(users)), observed_users)]
        raise ValueError(f"Raw source is missing {len(missing)} manifest users")
    return cube, {
        "rows_scanned_until_cohort_complete": rows_scanned,
        "rows_retained": rows_retained,
        "cohort_user_count": len(users),
    }


def missingness_profile(
    cube: np.ndarray,
    day_mask: np.ndarray,
) -> dict[str, object]:
    selected = np.asarray(cube)[:, np.asarray(day_mask, dtype=bool), :]
    observed = selected >= 0
    counts = observed.sum(axis=2)
    eligible = counts >= 2
    positive_internal_gaps: list[int] = []
    max_internal_gaps: list[int] = []
    interpolated_fractions: list[float] = []
    leading_missing: list[int] = []
    trailing_missing: list[int] = []
    for user_index, day_index in np.ndindex(counts.shape):
        slots = np.flatnonzero(observed[user_index, day_index])
        if len(slots):
            leading_missing.append(int(slots[0]))
            trailing_missing.append(int(selected.shape[2] - 1 - slots[-1]))
        if len(slots) < 2:
            continue
        gaps = np.diff(slots) - 1
        positive_internal_gaps.extend(int(value) for value in gaps if value > 0)
        max_gap = int(gaps.max(initial=0))
        max_internal_gaps.append(max_gap)
        span = int(slots[-1] - slots[0] + 1)
        interpolated_fractions.append(float((span - len(slots)) / span))

    max_internal = np.asarray(max_internal_gaps, dtype=np.int64)
    eligible_count = int(eligible.sum())
    return {
        "potential_user_days": int(counts.size),
        "raw_observation_count": int(observed.sum()),
        "raw_observation_rate_over_48_slots": float(observed.mean()),
        "observations_per_user_day": distribution_summary(counts.ravel()),
        "zero_observation_user_day_count": int((counts == 0).sum()),
        "one_observation_user_day_count": int((counts == 1).sum()),
        "converter_eligible_user_day_count": eligible_count,
        "converter_eligible_user_day_rate": float(eligible.mean()),
        "leading_missing_slots": _safe_distribution(leading_missing),
        "trailing_missing_slots": _safe_distribution(trailing_missing),
        "positive_internal_gap_slots": _safe_distribution(positive_internal_gaps),
        "max_internal_gap_slots_per_eligible_user_day": _safe_distribution(
            max_internal
        ),
        "unobserved_fraction_between_first_and_last_ping": _safe_distribution(
            interpolated_fractions
        ),
        "eligible_user_day_rate_with_internal_missing_slots": (
            float((max_internal >= 1).mean()) if eligible_count else None
        ),
        "eligible_user_day_rate_by_max_internal_gap_threshold": {
            str(threshold): (
                float((max_internal >= threshold).mean()) if eligible_count else None
            )
            for threshold in (1, 2, 4, 8, 12)
        },
        "gap_semantics": (
            "an internal gap of g means g unobserved 30-minute slots between two "
            "raw pings; no missing slot is filled for this profile"
        ),
    }


def user_slot_stability(
    cube: np.ndarray,
    users: np.ndarray,
    day_mask: np.ndarray,
    *,
    min_observed_days: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = np.asarray(cube)[:, np.asarray(day_mask, dtype=bool), :]
    users = np.asarray(users, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for user_index, uid in enumerate(users):
        for timeslot in range(selected.shape[2]):
            values = selected[user_index, :, timeslot]
            values = values[values >= 0]
            observed_days = int(len(values))
            if observed_days:
                top_cell, top_count = _mode(values)
                top_rate = float(top_count / observed_days)
                unique, counts = np.unique(values, return_counts=True)
                pair_total = observed_days * (observed_days - 1)
                same_pair_count = int(np.sum(counts * (counts - 1)))
                pair_match = (
                    float(same_pair_count / pair_total) if pair_total else None
                )
                unique_cells = int(len(unique))
            else:
                top_cell = None
                top_count = 0
                top_rate = None
                pair_match = None
                unique_cells = 0
            rows.append({
                "uid": int(uid),
                "timeslot": timeslot,
                "observed_days": observed_days,
                "eligible": observed_days >= min_observed_days,
                "top_cell": top_cell,
                "top_cell_count": top_count,
                "top1_location_rate": top_rate,
                "same_slot_cross_day_pair_match_probability": pair_match,
                "unique_observed_cells": unique_cells,
            })

    eligible = [row for row in rows if row["eligible"]]
    top_rates = np.asarray([row["top1_location_rate"] for row in eligible])
    pair_matches = np.asarray([
        row["same_slot_cross_day_pair_match_probability"]
        for row in eligible
        if row["same_slot_cross_day_pair_match_probability"] is not None
    ])
    total_observations = sum(int(row["observed_days"]) for row in eligible)
    total_top_hits = sum(int(row["top_cell_count"]) for row in eligible)
    by_timeslot: list[dict[str, object]] = []
    for timeslot in range(selected.shape[2]):
        slot_rows = [
            row for row in eligible if int(row["timeslot"]) == timeslot
        ]
        slot_observations = sum(int(row["observed_days"]) for row in slot_rows)
        by_timeslot.append({
            "timeslot": timeslot,
            "start_time": f"{timeslot // 2:02d}:{(timeslot % 2) * 30:02d}",
            "eligible_user_count": len(slot_rows),
            "observed_user_days": slot_observations,
            "weighted_top1_location_rate": (
                sum(int(row["top_cell_count"]) for row in slot_rows)
                / slot_observations
                if slot_observations else None
            ),
            "user_top1_location_rate": _safe_distribution([
                float(row["top1_location_rate"]) for row in slot_rows
            ]),
        })

    summary = {
        "minimum_observed_days_per_user_slot": int(min_observed_days),
        "total_user_slots": len(rows),
        "eligible_user_slots": len(eligible),
        "eligible_user_slot_rate": float(len(eligible) / len(rows)),
        "weighted_top1_location_rate": (
            float(total_top_hits / total_observations)
            if total_observations else None
        ),
        "user_slot_top1_location_rate": _safe_distribution(top_rates),
        "user_slot_same_slot_cross_day_pair_match_probability": (
            _safe_distribution(pair_matches)
        ),
        "eligible_user_slot_rate_at_top1_threshold": {
            str(threshold): float((top_rates >= threshold).mean())
            for threshold in (0.5, 0.75, 0.9)
        } if len(top_rates) else {},
        "by_timeslot": by_timeslot,
        "probability_semantics": (
            "P(location | user, 30-minute timeslot) is estimated from raw observed "
            "days only; missing slots are excluded, not imputed"
        ),
    }
    return summary, rows


def _cell_distance_km(left: np.ndarray, right: int, grid_height: int, cell_size_km: float) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    left_x, left_y = np.divmod(left, grid_height)
    right_x, right_y = divmod(int(right), grid_height)
    return np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2) * cell_size_km


def temporal_holdout_metrics(
    cube: np.ndarray,
    train_day_mask: np.ndarray,
    test_day_mask: np.ndarray,
    *,
    min_train_observations: int = 5,
    min_test_observations: int = 1,
    grid_height: int = 200,
    cell_size_km: float = 0.5,
) -> dict[str, object]:
    cube = np.asarray(cube)
    train = cube[:, np.asarray(train_day_mask, dtype=bool), :]
    test = cube[:, np.asarray(test_day_mask, dtype=bool), :]
    population_modes: list[int | None] = []
    for timeslot in range(cube.shape[2]):
        values = train[:, :, timeslot].ravel()
        values = values[values >= 0]
        population_modes.append(_mode(values)[0] if len(values) else None)

    personal_hits = 0
    population_hits = 0
    test_count = 0
    eligible_user_slots = 0
    personal_distances: list[float] = []
    population_distances: list[float] = []
    user_personal_hits = np.zeros(cube.shape[0], dtype=np.int64)
    user_population_hits = np.zeros(cube.shape[0], dtype=np.int64)
    user_test_counts = np.zeros(cube.shape[0], dtype=np.int64)
    user_slot_personal_rates: list[float] = []
    for user_index in range(cube.shape[0]):
        for timeslot in range(cube.shape[2]):
            train_values = train[user_index, :, timeslot]
            train_values = train_values[train_values >= 0]
            test_values = test[user_index, :, timeslot]
            test_values = test_values[test_values >= 0]
            if (
                len(train_values) < min_train_observations
                or len(test_values) < min_test_observations
                or population_modes[timeslot] is None
            ):
                continue
            personal_mode, _ = _mode(train_values)
            population_mode = int(population_modes[timeslot])
            personal_slot_hits = int((test_values == personal_mode).sum())
            population_slot_hits = int((test_values == population_mode).sum())
            eligible_user_slots += 1
            personal_hits += personal_slot_hits
            population_hits += population_slot_hits
            test_count += len(test_values)
            user_personal_hits[user_index] += personal_slot_hits
            user_population_hits[user_index] += population_slot_hits
            user_test_counts[user_index] += len(test_values)
            user_slot_personal_rates.append(personal_slot_hits / len(test_values))
            personal_distances.extend(
                _cell_distance_km(
                    test_values,
                    personal_mode,
                    grid_height,
                    cell_size_km,
                ).tolist()
            )
            population_distances.extend(
                _cell_distance_km(
                    test_values,
                    population_mode,
                    grid_height,
                    cell_size_km,
                ).tolist()
            )
    valid_users = user_test_counts > 0
    if test_count:
        personal_rate = float(personal_hits / test_count)
        population_rate = float(population_hits / test_count)
        user_advantage = (
            user_personal_hits[valid_users] / user_test_counts[valid_users]
            - user_population_hits[valid_users] / user_test_counts[valid_users]
        )
    else:
        personal_rate = None
        population_rate = None
        user_advantage = np.asarray([], dtype=np.float64)
    return {
        "train_day_count": int(np.asarray(train_day_mask, dtype=bool).sum()),
        "test_day_count": int(np.asarray(test_day_mask, dtype=bool).sum()),
        "minimum_train_observations_per_user_slot": int(min_train_observations),
        "minimum_test_observations_per_user_slot": int(min_test_observations),
        "eligible_user_slots": eligible_user_slots,
        "eligible_users": int(valid_users.sum()),
        "held_out_observations": int(test_count),
        "personalized_top1_exact_hit_rate": personal_rate,
        "population_timeslot_top1_exact_hit_rate": population_rate,
        "personalized_minus_population_exact_hit_rate": (
            personal_rate - population_rate if test_count else None
        ),
        "user_slot_personalized_exact_hit_rate": _safe_distribution(
            user_slot_personal_rates
        ),
        "per_user_personalized_minus_population_exact_hit_rate": (
            _safe_distribution(user_advantage)
        ),
        "personalized_top1_spatial_error_km": _safe_distribution(
            personal_distances
        ),
        "population_top1_spatial_error_km": _safe_distribution(
            population_distances
        ),
        "holdout_semantics": (
            "a modal cell learned from chronological training days predicts raw "
            "observations on later days; population control uses one modal cell per "
            "timeslot across all cohort users"
        ),
    }


def shuffle_timeslots_within_user_days(cube: np.ndarray, seed: int) -> np.ndarray:
    shuffled = np.asarray(cube).copy()
    rng = np.random.default_rng(int(seed))
    for user_index, day_index in np.ndindex(shuffled.shape[:2]):
        slots = np.flatnonzero(shuffled[user_index, day_index] >= 0)
        if len(slots) > 1:
            values = shuffled[user_index, day_index, slots].copy()
            rng.shuffle(values)
            shuffled[user_index, day_index, slots] = values
    return shuffled


def shuffled_null_profile(
    cube: np.ndarray,
    train_day_mask: np.ndarray,
    test_day_mask: np.ndarray,
    *,
    repeats: int,
    seed: int,
    min_train_observations: int,
) -> dict[str, object]:
    if repeats <= 0:
        raise ValueError("null repeats must be positive")
    observed = temporal_holdout_metrics(
        cube,
        train_day_mask,
        test_day_mask,
        min_train_observations=min_train_observations,
    )
    observed_rate = observed["personalized_top1_exact_hit_rate"]
    null_rates: list[float] = []
    null_advantages: list[float] = []
    for repeat in range(repeats):
        shuffled = shuffle_timeslots_within_user_days(cube, seed + repeat)
        metrics = temporal_holdout_metrics(
            shuffled,
            train_day_mask,
            test_day_mask,
            min_train_observations=min_train_observations,
        )
        if metrics["personalized_top1_exact_hit_rate"] is not None:
            null_rates.append(float(metrics["personalized_top1_exact_hit_rate"]))
            null_advantages.append(float(
                metrics["personalized_minus_population_exact_hit_rate"]
            ))
    if observed_rate is None or not null_rates:
        comparison = None
    else:
        comparison = {
            "observed_minus_null_mean_exact_hit_rate": float(
                observed_rate - np.mean(null_rates)
            ),
            "empirical_one_sided_p_value": float(
                (1 + np.sum(np.asarray(null_rates) >= observed_rate))
                / (len(null_rates) + 1)
            ),
            "observed_exceeds_null_p95": bool(
                observed_rate > np.quantile(null_rates, 0.95)
            ),
        }
    return {
        "observed_chronological_holdout": observed,
        "null_repeat_count": len(null_rates),
        "null_seed_start": int(seed),
        "null_personalized_top1_exact_hit_rate": _safe_distribution(null_rates),
        "null_personalized_minus_population_exact_hit_rate": _safe_distribution(
            null_advantages
        ),
        "comparison": comparison,
        "null_semantics": (
            "locations are randomly permuted only among each user-day's observed "
            "timeslots, preserving the observation mask and daily location multiset "
            "while destroying absolute-time assignment"
        ),
    }


def lagged_self_similarity(
    cube: np.ndarray,
    day_mask: np.ndarray,
    *,
    lags: tuple[int, ...] = (1, 7, 14, 28),
    grid_height: int = 200,
    cell_size_km: float = 0.5,
) -> list[dict[str, object]]:
    cube = np.asarray(cube)
    day_mask = np.asarray(day_mask, dtype=bool)
    results: list[dict[str, object]] = []
    for lag in lags:
        if lag <= 0 or lag >= cube.shape[1]:
            continue
        pair_day_mask = day_mask[:-lag] & day_mask[lag:]
        left = cube[:, :-lag, :][:, pair_day_mask, :]
        right = cube[:, lag:, :][:, pair_day_mask, :]
        overlap = (left >= 0) & (right >= 0)
        overlap_count = int(overlap.sum())
        matches = (left == right) & overlap
        per_user_rates = []
        for user_index in range(cube.shape[0]):
            user_count = int(overlap[user_index].sum())
            if user_count:
                per_user_rates.append(float(matches[user_index].sum() / user_count))
        if overlap_count:
            left_values = left[overlap].astype(np.int64)
            right_values = right[overlap].astype(np.int64)
            left_x, left_y = np.divmod(left_values, grid_height)
            right_x, right_y = np.divmod(right_values, grid_height)
            distances = np.sqrt(
                (left_x - right_x) ** 2 + (left_y - right_y) ** 2
            ) * cell_size_km
            exact_rate = float(matches.sum() / overlap_count)
        else:
            distances = np.asarray([], dtype=np.float64)
            exact_rate = None
        results.append({
            "day_lag": int(lag),
            "eligible_day_pairs": int(pair_day_mask.sum()),
            "overlapping_observed_slots": overlap_count,
            "exact_same_cell_rate": exact_rate,
            "per_user_exact_same_cell_rate": _safe_distribution(per_user_rates),
            "same_slot_spatial_distance_km": _safe_distribution(distances),
        })
    return results


def weekly_phase_profile(
    cube: np.ndarray,
    day_mask: np.ndarray,
    train_day_mask: np.ndarray,
    test_day_mask: np.ndarray,
) -> dict[str, object]:
    cube = np.asarray(cube)
    day_mask = np.asarray(day_mask, dtype=bool)
    phases = np.arange(cube.shape[1]) % 7
    phase_rows = []
    phase_activity = []
    for phase in range(7):
        selected_days = day_mask & (phases == phase)
        observed = cube[:, selected_days, :] >= 0
        potential = observed.size
        observations_per_user_day = (
            float(observed.sum() / (cube.shape[0] * selected_days.sum()))
            if selected_days.sum() else None
        )
        phase_activity.append(observations_per_user_day)
        phase_rows.append({
            "masked_day_mod_7": phase,
            "day_count": int(selected_days.sum()),
            "raw_observation_count": int(observed.sum()),
            "raw_observation_rate": (
                float(observed.sum() / potential) if potential else None
            ),
            "mean_observations_per_user_day": observations_per_user_day,
        })

    pair_scores = []
    for start in range(7):
        weekend_like = {start, (start + 1) % 7}
        weekend_mask = np.isin(phases, list(weekend_like))
        weekday_mask = ~weekend_mask
        weekend_metrics = temporal_holdout_metrics(
            cube,
            train_day_mask & weekend_mask,
            test_day_mask & weekend_mask,
            min_train_observations=2,
        )
        weekday_metrics = temporal_holdout_metrics(
            cube,
            train_day_mask & weekday_mask,
            test_day_mask & weekday_mask,
            min_train_observations=5,
        )
        activity_score = float(
            np.mean([phase_activity[value] for value in weekend_like])
        )
        pair_scores.append({
            "candidate_two_day_phase_pair": sorted(weekend_like),
            "mean_observations_per_user_day": activity_score,
            "five_phase_holdout": weekday_metrics,
            "two_phase_holdout": weekend_metrics,
        })
    inferred = min(
        pair_scores,
        key=lambda row: float(row["mean_observations_per_user_day"]),
    )
    return {
        "calendar_mapping_available": False,
        "official_calendar_semantics": (
            "the official data descriptor calls d a masked date in [0, 74] and "
            "does not publish the civil-date or weekday mapping"
        ),
        "official_source": MASKED_CALENDAR_SOURCE,
        "masked_day_mod_7": phase_rows,
        "all_contiguous_two_phase_sensitivity": pair_scores,
        "lowest_activity_contiguous_two_phase_candidate": inferred,
        "label_policy": (
            "results use masked day modulo 7 and sensitivity over all seven offsets; "
            "the lowest-activity pair is only 'weekend-like', never a civil weekday claim"
        ),
    }


def build_habit_profile(
    cube: np.ndarray,
    users: np.ndarray,
    *,
    manifest_sample_count: int,
    excluded_days: set[int] | None = None,
    min_observed_days: int = 10,
    min_train_observations: int = 5,
    null_repeats: int = 20,
    seed: int = 42,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cube = np.asarray(cube)
    users = np.asarray(users, dtype=np.int64)
    excluded_days = {27} if excluded_days is None else set(excluded_days)
    day_mask = np.ones(cube.shape[1], dtype=bool)
    for day in excluded_days:
        if not 0 <= day < cube.shape[1]:
            raise ValueError(f"Excluded day {day} is outside the observation cube")
        day_mask[day] = False
    eligible_days = np.flatnonzero(day_mask)
    split_at = len(eligible_days) // 2
    train_day_mask = np.zeros_like(day_mask)
    test_day_mask = np.zeros_like(day_mask)
    train_day_mask[eligible_days[:split_at]] = True
    test_day_mask[eligible_days[split_at:]] = True

    missingness = missingness_profile(cube, day_mask)
    stability, user_slot_rows = user_slot_stability(
        cube,
        users,
        day_mask,
        min_observed_days=min_observed_days,
    )
    null = shuffled_null_profile(
        cube,
        train_day_mask,
        test_day_mask,
        repeats=null_repeats,
        seed=seed,
        min_train_observations=min_train_observations,
    )
    observed_holdout = null["observed_chronological_holdout"]
    null_comparison = null["comparison"]
    personalized_advantage = observed_holdout[
        "personalized_minus_population_exact_hit_rate"
    ]
    habit_signal_detected = bool(
        personalized_advantage is not None
        and personalized_advantage > 0
        and null_comparison is not None
        and null_comparison["observed_exceeds_null_p95"]
    )
    profile = {
        "schema_version": "yjmob-longitudinal-habit-profile-v1",
        "analysis_scope": {
            "cohort_user_count": len(users),
            "manifest_user_day_sample_count": int(manifest_sample_count),
            "raw_day_count": cube.shape[1],
            "timeslots_per_day": cube.shape[2],
            "excluded_days": sorted(excluded_days),
            "raw_only_no_interpolation": True,
        },
        "missingness_and_interpolation_exposure": missingness,
        "per_user_timeslot_stability": stability,
        "chronological_holdout_and_shuffled_null": null,
        "lagged_cross_day_self_similarity": lagged_self_similarity(
            cube,
            day_mask,
        ),
        "weekly_phase_analysis": weekly_phase_profile(
            cube,
            day_mask,
            train_day_mask,
            test_day_mask,
        ),
        "habit_evidence_rule": {
            "criterion": (
                "chronological personalized top-1 exact hit rate exceeds the "
                "population-timeslot control and the 95th percentile of the "
                "within-user-day timeslot-shuffled null"
            ),
            "stable_personal_time_slot_habit_signal_detected": habit_signal_detected,
            "not_a_causal_or_identity_reidentification_claim": True,
        },
        "conversion_cross_check": {
            "raw_user_days_with_at_least_two_observations": missingness[
                "converter_eligible_user_day_count"
            ],
            "prepared_manifest_sample_count": int(manifest_sample_count),
            "counts_match": bool(
                missingness["converter_eligible_user_day_count"]
                == manifest_sample_count
            ),
        },
        "privacy_and_interpretation_notes": [
            "Only aggregate statistics should be published; anonymous user-slot rows are a reproducibility intermediate.",
            "High modal accuracy alone can reflect a dominant home cell; population and shuffled controls are therefore reported.",
            "The masked calendar is not reverse-engineered and no real city, date, or user identity is inferred.",
        ],
    }
    return profile, user_slot_rows


def write_timeslot_summary(path: Path, profile: dict[str, object]) -> None:
    rows = profile["per_user_timeslot_stability"]["by_timeslot"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "timeslot",
                "start_time",
                "eligible_user_count",
                "observed_user_days",
                "weighted_top1_location_rate",
                "user_top1_rate_p10",
                "user_top1_rate_p50",
                "user_top1_rate_p90",
            ),
        )
        writer.writeheader()
        for row in rows:
            distribution = row["user_top1_location_rate"] or {}
            writer.writerow({
                "timeslot": row["timeslot"],
                "start_time": row["start_time"],
                "eligible_user_count": row["eligible_user_count"],
                "observed_user_days": row["observed_user_days"],
                "weighted_top1_location_rate": row[
                    "weighted_top1_location_rate"
                ],
                "user_top1_rate_p10": distribution.get("p10"),
                "user_top1_rate_p50": distribution.get("p50"),
                "user_top1_rate_p90": distribution.get("p90"),
            })


def write_user_slot_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--day-count", type=int, default=75)
    parser.add_argument("--timeslot-count", type=int, default=48)
    parser.add_argument("--exclude-day", type=int, action="append", default=None)
    parser.add_argument("--min-observed-days", type=int, default=10)
    parser.add_argument("--min-train-observations", type=int, default=5)
    parser.add_argument("--null-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-md5", default=OFFICIAL_DATASET1_MD5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_md5:
        source_md5 = md5sum(args.input)
        if source_md5.lower() != args.expected_md5.lower():
            raise ValueError(
                f"MD5 mismatch: expected {args.expected_md5}, got {source_md5}"
            )
    else:
        source_md5 = md5sum(args.input)
    users, manifest_sample_count = read_manifest_users(args.manifest)
    cube, scan = load_observation_cube(
        args.input,
        users,
        day_count=args.day_count,
        timeslot_count=args.timeslot_count,
    )
    profile, rows = build_habit_profile(
        cube,
        users,
        manifest_sample_count=manifest_sample_count,
        excluded_days=set(args.exclude_day or [27]),
        min_observed_days=args.min_observed_days,
        min_train_observations=args.min_train_observations,
        null_repeats=args.null_repeats,
        seed=args.seed,
    )
    profile["source_provenance"] = {
        "raw_filename": args.input.name,
        "raw_md5": source_md5,
        "manifest_path": str(args.manifest.resolve()),
        "scan": scan,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "habit_profile.json"
    summary_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_timeslot_summary(args.output_dir / "timeslot_stability.csv", profile)
    write_user_slot_rows(args.output_dir / "user_slot_stability.csv", rows)
    print(json.dumps({
        "habit_profile": str(summary_path),
        "habit_signal_detected": profile["habit_evidence_rule"][
            "stable_personal_time_slot_habit_signal_detected"
        ],
        "conversion_counts_match": profile["conversion_cross_check"][
            "counts_match"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
