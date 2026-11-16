"""Cross-material structure for the learning-support profile.

Three material sources share no unit: interest strength, barrier strength and
participation share are recorded on different scales, and some learners submit
only one of them.  This module looks for dimensions supported in more than one
source (``shared``), keeps dimensions that exist in a single source
(``view_specific``), and refuses to label anything else.

Rules that keep the result honest:

* no cell is ever filled in - correlations are pairwise complete and a learner
  is positioned only from cells actually observed;
* features are standardized before estimation, so the raw unit of one source
  cannot decide how many dimensions exist or dominate a loading;
* the number of dimensions is chosen by parallel analysis against a
  permutation null that preserves each feature's missingness, plus a unit-root
  check - never by "try k and look";
* a source supports a dimension only when its loading energy clears the null
  and at least two of its features clear a business floor, so one saturated
  item cannot pose as a need;
* a learner with too little observed support stays unassigned instead of being
  forced into a type.

A fitted structure is versioned and stored inside the saved analysis result.
:func:`apply_structure` reuses it for new learners without re-estimating.
"""

from collections import Counter, defaultdict
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = "shared-barriers/1"
_MISSING_POLICY = {"estimation": "pairwise_complete", "scoring": "observed_only", "imputed_values": False}
_CONSISTENCY_FLOOR = 0.85

_DEFAULTS = {
    "permutations": 199,
    "alpha": 0.05,
    "max_dimensions": 5,
    "min_pairs": 10,
    "min_feature_loading": 0.30,
    "support_threshold": 0.50,
    "partial_support": 0.25,
    "seed": 20261115,
    "loo_limit": 200,
}
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# parameters and saved-structure validation
# --------------------------------------------------------------------------- #
def _parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("分析参数不是对象")
    unknown = set(parameters) - set(_DEFAULTS) - {"people"}
    if unknown:
        raise ValueError(f"未知分析参数：{sorted(unknown)}")
    settings = dict(_DEFAULTS)
    settings.update({key: value for key, value in parameters.items() if key in _DEFAULTS})
    for key in ("permutations", "max_dimensions", "min_pairs", "seed", "loo_limit"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} 必须是整数")
    if not 19 <= settings["permutations"] <= 1999:
        raise ValueError("permutations 必须在 19 到 1999 之间")
    if not 1 <= settings["max_dimensions"] <= 20:
        raise ValueError("max_dimensions 必须在 1 到 20 之间")
    if settings["min_pairs"] < 4:
        raise ValueError("min_pairs 至少为 4")
    if settings["seed"] < 0:
        raise ValueError("seed 不能为负")
    if settings["loo_limit"] < 10:
        raise ValueError("loo_limit 至少为 10")
    for key in ("alpha", "min_feature_loading", "support_threshold", "partial_support"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值")
        if not 0 < float(value) <= 1:
            raise ValueError(f"{key} 必须在 (0, 1] 之间")
    if settings["partial_support"] > settings["support_threshold"]:
        raise ValueError("partial_support 不能大于 support_threshold")
    people = parameters.get("people")
    if people is not None:
        if not isinstance(people, list) or any(not isinstance(item, str) or not item for item in people):
            raise ValueError("people 必须是学员编号列表")
        if len(set(people)) != len(people):
            raise ValueError("people 含重复学员编号")
        settings["people"] = list(people)
    return settings


def _check_structure(structure):
    if not isinstance(structure, dict):
        raise ValueError("已保存结构不是对象")
    if structure.get("version") != VERSION:
        raise ValueError("已保存结构版本不兼容")
    features, dimensions = structure.get("features"), structure.get("dimensions")
    if not isinstance(features, list) or not isinstance(dimensions, list):
        raise ValueError("已保存结构缺少特征或维度")
    keys = set()
    for feature in features:
        if not isinstance(feature, dict) or not {"view", "feature", "center", "scale"} <= feature.keys():
            raise ValueError("已保存结构特征不合法")
        key = (feature["view"], feature["feature"])
        if key in keys:
            raise ValueError("已保存结构特征重复")
        keys.add(key)
        if isinstance(feature["center"], bool) or not isinstance(feature["center"], (int, float)) \
                or not np.isfinite(feature["center"]):
            raise ValueError("已保存结构特征中心不是有限数值")
        if isinstance(feature["scale"], bool) or not isinstance(feature["scale"], (int, float)) \
                or not np.isfinite(feature["scale"]) or feature["scale"] <= 0:
            raise ValueError("已保存结构特征尺度不合法")
    for dimension in dimensions:
        if not isinstance(dimension, dict) or not {"id", "kind", "views", "loadings"} <= dimension.keys():
            raise ValueError("已保存结构维度不合法")
        if dimension["kind"] not in {"shared", "view_specific"}:
            raise ValueError("已保存结构含不可命名的维度")
        loadings = dimension["loadings"]
        if not isinstance(loadings, list) or len(loadings) != len(features):
            raise ValueError("已保存结构维度载荷与特征不一致")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
               for value in loadings):
            raise ValueError("已保存结构载荷不是有限数值")
    return structure


# --------------------------------------------------------------------------- #
# matrix assembly, summaries, standardization
# --------------------------------------------------------------------------- #
def _matrix(tables):
    people = sorted(row["person"] for row in tables["people"])
    sites = {row["person"]: row.get("site") or "未登记" for row in tables["people"]}
    ages = {row["person"]: row.get("age_band") or "未登记" for row in tables["people"]}
    features = sorted({(row["view"], row["feature"]) for row in tables["features"]})
    person_index = {person: index for index, person in enumerate(people)}
    feature_index = {key: index for index, key in enumerate(features)}
    values = np.full((len(people), len(features)), np.nan)
    for row in tables["observations"]:
        values[person_index[row["person"]], feature_index[(row["view"], row["feature"])]] = float(row["value"])
    views = sorted({view for view, unused in features})
    blocks = {view: [index for index, key in enumerate(features) if key[0] == view] for view in views}
    return {"people": people, "sites": sites, "ages": ages, "features": features,
            "values": values, "views": views, "blocks": blocks}


def _summaries(matrix):
    values = matrix["values"]
    features = []
    for index, (view, feature) in enumerate(matrix["features"]):
        observed = values[~np.isnan(values[:, index]), index]
        features.append({"view": view, "feature": feature, "count": int(observed.size),
                         "mean": float(observed.mean()) if observed.size else None,
                         "std": float(observed.std()) if observed.size else None})
    observed_by_view = {view: int(np.sum(~np.isnan(values[:, indexes]).all(axis=1)))
                        for view, indexes in matrix["blocks"].items()}
    patterns = Counter()
    for row in values:
        present = tuple(view for view in matrix["views"]
                        if not np.isnan(row[matrix["blocks"][view]]).all())
        patterns[present] += 1
    return {"features": features,
            "coverage": {"people": len(matrix["people"]),
                         "people_with_any_observation": int(np.sum(~np.isnan(values).all(axis=1))),
                         "people_without_observation": int(np.sum(np.isnan(values).all(axis=1))),
                         "complete_people_by_view": observed_by_view},
            "missing_view_patterns": [{"observed_views": list(key), "people": count}
                                      for key, count in sorted(patterns.items())]}


def _standardize(values):
    means = np.full(values.shape[1], np.nan)
    scales = np.full(values.shape[1], 1.0)
    kept, excluded = [], []
    for index in range(values.shape[1]):
        observed = values[~np.isnan(values[:, index]), index]
        if observed.size < 3:
            excluded.append({"index": index, "reason": "insufficient_observations"})
            continue
        spread = float(observed.std())
        if spread <= 1e-9:
            excluded.append({"index": index, "reason": "no_variation"})
            continue
        means[index] = float(observed.mean())
        scales[index] = spread
        kept.append(index)
    return (values - means) / scales, means, scales, kept, excluded


def _blocks(matrix, kept):
    blocks = defaultdict(list)
    for position, index in enumerate(kept):
        blocks[matrix["features"][index][0]].append(position)
    return dict(blocks)


# --------------------------------------------------------------------------- #
# correlation work on the pairwise-complete matrix
# --------------------------------------------------------------------------- #
def _correlations(values, minimum):
    width = values.shape[1]
    correlations = np.eye(width)
    counts = np.zeros((width, width), dtype=int)
    for left in range(width):
        first = values[:, left]
        for right in range(left + 1, width):
            observed = ~np.isnan(first) & ~np.isnan(values[:, right])
            count = int(observed.sum())
            counts[left, right] = counts[right, left] = count
            if count < minimum:
                continue
            one, two = first[observed], values[observed, right]
            spread_one, spread_two = one.std(), two.std()
            if spread_one <= _EPS or spread_two <= _EPS:
                continue
            value = float(np.mean((one - one.mean()) * (two - two.mean())) / (spread_one * spread_two))
            correlations[left, right] = correlations[right, left] = float(np.clip(value, -1.0, 1.0))
    return correlations, counts


def _psd(correlations):
    eigenvalues, vectors = np.linalg.eigh((correlations + correlations.T) / 2)
    repaired = (vectors * np.clip(eigenvalues, 0.0, None)) @ vectors.T
    adjustment = float(np.linalg.norm(repaired - correlations) / max(np.linalg.norm(correlations), _EPS))
    return (repaired + repaired.T) / 2, eigenvalues, adjustment


def _varimax(loadings):
    rows, columns = loadings.shape
    if columns == 0:
        return loadings.copy()
    if columns == 1:
        sign = 1.0 if loadings[int(np.argmax(np.abs(loadings[:, 0]))), 0] >= 0 else -1.0
        return loadings * sign
    rotation = np.eye(columns)
    previous = 0.0
    for _ in range(500):
        rotated = loadings @ rotation
        first, second, third = np.linalg.svd(
            loadings.T @ (rotated ** 3 - (1.0 / rows) * rotated @ np.diag(np.diag(rotated.T @ rotated))))
        rotation = first @ third
        total = float(second.sum())
        if previous and total / previous < 1.0 + 1e-8:
            break
        previous = total
    return loadings @ rotation


def _canonical(loadings):
    """Deterministic order and sign: variance descending, largest loading positive."""
    if loadings.shape[1] == 0:
        return loadings
    order = np.argsort((loadings ** 2).sum(axis=0))[::-1]
    result = loadings[:, order].copy()
    for column in range(result.shape[1]):
        if result[int(np.argmax(np.abs(result[:, column]))), column] < 0:
            result[:, column] *= -1
    return result


def _extract(correlations, dimensions):
    if dimensions <= 0:
        return np.zeros((correlations.shape[0], 0))
    eigenvalues, vectors = np.linalg.eigh((correlations + correlations.T) / 2)
    order = np.argsort(eigenvalues)[::-1][:dimensions]
    loadings = vectors[:, order] * np.sqrt(np.clip(eigenvalues[order], 0.0, None))[None, :]
    return _canonical(_varimax(loadings))


def _permute(values, random, groups=None):
    """Shuffle each feature within its own observed rows, keeping missingness.

    With ``groups`` the shuffle stays inside each group, which is the correct
    null when the statistic is computed on group-mean-adjusted data.
    """
    result = values.copy()
    rows = np.arange(values.shape[0])
    strata = [rows] if groups is None else [rows[np.asarray(groups) == group]
                                           for group in sorted(set(np.asarray(groups).tolist()))]
    for index in range(values.shape[1]):
        for stratum in strata:
            observed = stratum[~np.isnan(values[stratum, index])]
            if observed.size:
                result[observed, index] = values[random.permutation(observed), index]
    return result


def _view_statistics(loadings, blocks, floor):
    statistics = {}
    for view, indexes in blocks.items():
        block = loadings[indexes]
        statistics[view] = {"energy": [float(value) for value in (block ** 2).sum(axis=0)],
                            "count": [int(value) for value in (np.abs(block) >= floor).sum(axis=0)]}
    return statistics


def _match(reference, candidate):
    """Match dimensions by absolute cosine; sign and column order carry no identity."""
    if reference.shape[1] == 0 or candidate.shape[1] == 0:
        return []
    reference_norms = np.linalg.norm(reference, axis=0)
    candidate_norms = np.linalg.norm(candidate, axis=0)
    similarity = np.zeros((reference.shape[1], candidate.shape[1]))
    for left in range(reference.shape[1]):
        for right in range(candidate.shape[1]):
            denominator = reference_norms[left] * candidate_norms[right]
            similarity[left, right] = abs(float(reference[:, left] @ candidate[:, right])) / max(denominator, _EPS)
    rows, columns = linear_sum_assignment(-similarity)
    return [{"dimension": f"d{row + 1}", "matched": f"d{column + 1}",
             "congruence": round(float(similarity[row, column]), 6)}
            for row, column in zip(rows.tolist(), columns.tolist())]


# --------------------------------------------------------------------------- #
# dimension selection and interpretation
# --------------------------------------------------------------------------- #
def _select_dimensions(reduced, blocks, settings, seed_offset=0, residual_groups=None):
    """Parallel analysis plus a unit-root check.

    The null permutes each feature within its observed rows, so missingness
    patterns - including whole missing materials - are preserved.  A dimension
    is kept only while its eigenvalue exceeds both the 1-alpha null quantile at
    that position and 1.0.  Loading thresholds are taken per dimension, so a
    later, weaker factor is judged against its own null rather than the first.

    When ``residual_groups`` is given the same group-mean adjustment is applied
    to the observed matrix *and to every permutation*, so the null still
    describes the statistic that is actually tested instead of the raw one.
    """
    if residual_groups is not None:
        reduced = _residualize_matrix(reduced, residual_groups)
    correlations, _ = _correlations(reduced, settings["min_pairs"])
    repaired, raw_eigenvalues, adjustment = _psd(correlations)
    ordered = np.sort(raw_eigenvalues)[::-1]
    cap = max(0, min(settings["max_dimensions"], reduced.shape[1] - len(blocks), reduced.shape[1] - 1))
    if cap == 0:
        return {"dimensions": 0, "loadings": np.zeros((reduced.shape[1], 0)), "eigenvalues": [],
                "adjustment": adjustment, "cap": 0, "thresholds": {}}

    random = np.random.default_rng(settings["seed"] + seed_offset)
    null_eigenvalues = []
    null_energy = {view: [[] for _ in range(cap)] for view in blocks}
    null_loading = {view: [[] for _ in range(cap)] for view in blocks}
    for _ in range(settings["permutations"]):
        permuted = _permute(reduced, random, residual_groups)
        if residual_groups is not None:
            permuted = _residualize_matrix(permuted, residual_groups)
        permuted_correlations, _ = _correlations(permuted, settings["min_pairs"])
        permuted_repaired, _, _ = _psd(permuted_correlations)
        null_eigenvalues.append(np.sort(np.linalg.eigvalsh(permuted_repaired))[::-1])
        noise = _extract(permuted_repaired, cap)
        statistics = _view_statistics(noise, blocks, settings["min_feature_loading"])
        for view in blocks:
            for column in range(cap):
                null_energy[view][column].append(statistics[view]["energy"][column])
                null_loading[view][column].append(float(np.abs(noise[blocks[view], column]).max()))
    null_quantile = np.quantile(np.array(null_eigenvalues), 1.0 - settings["alpha"], axis=0)
    thresholds = {view: [{"energy": float(np.quantile(null_energy[view][column], 1.0 - settings["alpha"])),
                          "loading": float(np.quantile(null_loading[view][column], 1.0 - settings["alpha"]))}
                         for column in range(cap)] for view in blocks}

    dimensions, table = 0, []
    for index in range(min(cap, len(ordered))):
        above_null = bool(ordered[index] > null_quantile[index])
        above_one = bool(ordered[index] > 1.0)
        table.append({"index": index + 1, "eigenvalue": round(float(ordered[index]), 6),
                      "null_quantile": round(float(null_quantile[index]), 6),
                      "above_null": above_null, "above_one": above_one})
        if not (above_null and above_one):
            break
        dimensions = index + 1
    return {"dimensions": dimensions, "loadings": _extract(repaired, dimensions),
            "eigenvalues": table, "adjustment": adjustment, "cap": cap, "thresholds": thresholds}


def _classify(loadings, blocks, thresholds, floor, matrix):
    """Give every factor a kind, or leave it unidentified.

    A source supports a factor when its squared loading energy exceeds the null
    quantile for that factor and at least two of its features clear the
    business floor.  Two or more supporting sources make it ``shared``; one
    makes it ``view_specific``; none means the factor cannot be named.
    """
    statistics = _view_statistics(loadings, blocks, floor)
    total_energy = np.clip((loadings ** 2).sum(axis=0), _EPS, None)
    rows = []
    for column in range(loadings.shape[1]):
        supported, evidence = [], {}
        for view, indexes in blocks.items():
            energy = statistics[view]["energy"][column]
            count = statistics[view]["count"][column]
            significant = int(np.sum(np.abs(loadings[indexes, column]) >= thresholds[view][column]["loading"]))
            supported_view = energy >= thresholds[view][column]["energy"] and count >= 2
            evidence[view] = {
                "share_of_loading_energy": round(float(energy / total_energy[column]), 6),
                "features_above_floor": count,
                "features_above_null_loading": significant,
                "supported": bool(supported_view),
                "null_energy_quantile": round(thresholds[view][column]["energy"], 6),
                "null_loading_quantile": round(thresholds[view][column]["loading"], 6),
            }
            if supported_view:
                supported.append(view)
        kind = ("shared" if len(supported) >= 2
                else "view_specific" if len(supported) == 1 else "unidentified")
        rows.append({"id": f"d{column + 1}", "kind": kind, "views": supported,
                     "loadings": [float(value) for value in loadings[:, column]],
                     "evidence": evidence,
                     "top_features": {view: [
                         {"feature": matrix["features"][index][1],
                          "loading": round(float(loadings[index, column]), 6)}
                         for index in sorted(blocks[view],
                                             key=lambda item: -abs(loadings[item, column]))[:3]]
                         for view in supported}})
    return rows


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def _split_half(reduced, loadings, blocks, thresholds, people, settings):
    """Hold out by learner hash, refit each half, and check correspondence."""
    if loadings.shape[1] == 0:
        return {"status": "no_dimensions"}
    halves = [[], []]
    for index, person in enumerate(people):
        halves[int(hashlib.sha256(person.encode()).hexdigest()[:8], 16) % 2].append(index)
    if min(len(half) for half in halves) < max(15, reduced.shape[1] + 5):
        return {"status": "insufficient_split"}
    results = []
    for number, members in enumerate(halves):
        correlations, _ = _correlations(reduced[members], settings["min_pairs"])
        repaired, _, adjustment = _psd(correlations)
        candidate = _extract(repaired, loadings.shape[1])
        matches = _match(loadings, candidate)
        statistics = _view_statistics(candidate, blocks, settings["min_feature_loading"])
        kinds = {}
        for column in range(candidate.shape[1]):
            supports = [view for view in blocks
                        if statistics[view]["energy"][column] >= thresholds[view][column]["energy"]
                        and statistics[view]["count"][column] >= 2]
            kinds[f"d{column + 1}"] = ("shared" if len(supports) >= 2
                                       else "view_specific" if len(supports) == 1 else "unidentified")
        values = [item["congruence"] for item in matches]
        results.append({"half": number + 1, "people": len(members),
                        "psd_adjustment": round(adjustment, 6), "matches": matches,
                        "minimum_congruence": round(float(min(values)), 6) if values else None,
                        "kinds": kinds})
    values = [item["minimum_congruence"] for item in results if item["minimum_congruence"] is not None]
    minimum = min(values) if values else None
    return {"status": "estimated", "halves": results, "minimum_congruence": minimum,
            "stable": bool(minimum is not None and minimum >= _CONSISTENCY_FLOOR)}


def _sufficient(reduced):
    """Per-pair sums over the rows where both features are observed.

    The sums must be taken on the pairwise-complete row set; using each
    feature's full column would mix different denominators into one variance.
    """
    width = reduced.shape[1]
    sums_left = np.zeros((width, width))
    sums_right = np.zeros((width, width))
    squares_left = np.zeros((width, width))
    squares_right = np.zeros((width, width))
    products = np.zeros((width, width))
    counts = np.zeros((width, width), dtype=int)
    for left in range(width):
        for right in range(left + 1, width):
            observed = ~np.isnan(reduced[:, left]) & ~np.isnan(reduced[:, right])
            one, two = reduced[observed, left], reduced[observed, right]
            sums_left[left, right] = float(one.sum())
            sums_right[left, right] = float(two.sum())
            squares_left[left, right] = float((one ** 2).sum())
            squares_right[left, right] = float((two ** 2).sum())
            products[left, right] = float((one * two).sum())
            counts[left, right] = int(observed.sum())
    return sums_left, sums_right, squares_left, squares_right, products, counts


def _leave_one_out(reduced, loadings, people, settings):
    """Refit after holding out one whole learner; check the dimensions correspond."""
    if loadings.shape[1] == 0:
        return {"status": "no_dimensions"}
    sums_left, sums_right, squares_left, squares_right, products, counts = _sufficient(reduced)
    width = reduced.shape[1]
    eligible = list(range(len(people)))
    if len(eligible) > settings["loo_limit"]:
        eligible = sorted(eligible, key=lambda index: hashlib.sha256(people[index].encode()).hexdigest()
                          )[:settings["loo_limit"]]
    matches = []
    for index in eligible:
        correlations = np.eye(width)
        for left in range(width):
            for right in range(left + 1, width):
                count = int(counts[left, right])
                sum_left = float(sums_left[left, right])
                sum_right = float(sums_right[left, right])
                square_left = float(squares_left[left, right])
                square_right = float(squares_right[left, right])
                product = float(products[left, right])
                if not np.isnan(reduced[index, left]) and not np.isnan(reduced[index, right]):
                    count -= 1
                    sum_left -= reduced[index, left]
                    sum_right -= reduced[index, right]
                    square_left -= reduced[index, left] ** 2
                    square_right -= reduced[index, right] ** 2
                    product -= reduced[index, left] * reduced[index, right]
                if count < settings["min_pairs"]:
                    continue
                covariance = product - sum_left * sum_right / count
                variance_left = square_left - sum_left ** 2 / count
                variance_right = square_right - sum_right ** 2 / count
                if variance_left > _EPS and variance_right > _EPS:
                    correlations[left, right] = correlations[right, left] = float(
                        np.clip(covariance / np.sqrt(variance_left * variance_right), -1.0, 1.0))
        repaired, _, _ = _psd(correlations)
        matched = _match(loadings, _extract(repaired, loadings.shape[1]))
        values = [item["congruence"] for item in matched]
        matches.append({"person": people[index], "minimum_congruence":
                        round(float(min(values)), 6) if values else None})
    values = [item["minimum_congruence"] for item in matches if item["minimum_congruence"] is not None]
    return {"status": "estimated", "people_evaluated": len(matches),
            "minimum_congruence": round(float(min(values)), 6) if values else None,
            "median_congruence": round(float(np.median(values)), 6) if values else None,
            "share_at_least_0_85": round(float(np.mean([value >= _CONSISTENCY_FLOOR for value in values])), 6)
            if values else None,
            "reproduced": bool(values and min(values) >= _CONSISTENCY_FLOOR)}


def _eta_squared(values, labels, random, permutations):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    present = ~np.isnan(values)
    values, labels = values[present], labels[present]
    if values.size < 6:
        return None
    groups = sorted(set(labels.tolist()))
    if len(groups) < 2:
        return None
    total = float(((values - values.mean()) ** 2).sum())
    if total <= _EPS:
        return {"eta_squared": 0.0, "permutation_p": None, "groups": len(groups), "people": int(values.size)}
    between = sum(float((labels == group).sum() * (values[labels == group].mean() - values.mean()) ** 2)
                  for group in groups)
    statistic = between / total
    exceed = 0
    for _ in range(permutations):
        shuffled = random.permutation(labels)
        null = sum(float((shuffled == group).sum() * (values[shuffled == group].mean() - values.mean()) ** 2)
                   for group in groups)
        if null / total >= statistic:
            exceed += 1
    return {"eta_squared": round(float(statistic), 6),
            "permutation_p": round((1 + exceed) / (permutations + 1), 6),
            "groups": len(groups), "people": int(values.size)}


def _group_diagnostic(scores, labels, settings):
    random = np.random.default_rng(settings["seed"] + 7919)
    results = {}
    for name, values in scores.items():
        result = _eta_squared(values, labels, random, settings["permutations"])
        if result is not None:
            result["flag"] = bool(result["eta_squared"] >= 0.10 and result["permutation_p"] is not None
                                  and result["permutation_p"] <= settings["alpha"])
            results[name] = result
    return results


def _residualize_matrix(reduced, groups):
    """Group-mean adjustment on an already standardized matrix."""
    groups = np.asarray(groups)
    residual = reduced.copy()
    for position in range(residual.shape[1]):
        column = residual[:, position]
        for group in sorted(set(groups.tolist())):
            members = np.where((groups == group) & ~np.isnan(column))[0]
            if members.size:
                column[members] -= float(np.mean(column[members]))
        spread = float(np.nanstd(column))
        if spread > _EPS:
            column /= spread
        residual[:, position] = column
    return residual


def _scale_diagnostic(matrix, kept, notes):
    values = matrix["values"][:, kept]
    complete = ~np.isnan(values).any(axis=1)
    rows = values[complete]
    if rows.shape[0] < max(10, values.shape[1] + 5) or values.shape[1] < 2:
        return {"status": "insufficient_complete_cases", "complete_people": int(rows.shape[0]),
                "dominated_by": [], "view_share_raw": {}, "view_share_standardized": {}}
    spread = np.where(rows.std(axis=0) <= _EPS, 1.0, rows.std(axis=0))
    raw_vector = np.linalg.svd(rows - rows.mean(axis=0), full_matrices=False)[2][0]
    standardized_vector = np.linalg.svd((rows - rows.mean(axis=0)) / spread, full_matrices=False)[2][0]
    blocks = _blocks(matrix, kept)
    raw_share = {view: float((raw_vector[indexes] ** 2).sum() / max((raw_vector ** 2).sum(), _EPS))
                 for view, indexes in blocks.items()}
    std_share = {view: float((standardized_vector[indexes] ** 2).sum() /
                             max((standardized_vector ** 2).sum(), _EPS)) for view, indexes in blocks.items()}
    dominated = [view for view in raw_share
                 if raw_share[view] >= 0.50 and raw_share[view] >= 2 * std_share[view]]
    if dominated:
        notes.append("原始量纲下第一主成分主要由 " + "、".join(dominated) +
                     " 支配；解释使用标准化载荷，不按原始量纲")
    return {"status": "estimated", "complete_people": int(rows.shape[0]), "dominated_by": dominated,
            "view_share_raw": {view: round(value, 6) for view, value in raw_share.items()},
            "view_share_standardized": {view: round(value, 6) for view, value in std_share.items()}}


def _diagnostics(matrix, settings, selection, structure, feature_rows, notes):
    kept = [row["index"] for row in feature_rows]
    blocks = _blocks(matrix, kept)
    centers = np.array([row["center"] for row in feature_rows])
    scales = np.array([row["scale"] for row in feature_rows])
    standardized = (matrix["values"][:, kept] - centers) / scales
    diagnostics = {"mode": "fit", "recomputed": True, "consistency_floor": _CONSISTENCY_FLOOR}

    if selection is not None and selection["dimensions"] > 0:
        loadings = selection["loadings"]
        thresholds = {view: [dict(item) for item in items] for view, items in selection["thresholds"].items()}
        diagnostics["split_half"] = _split_half(standardized, loadings, blocks, thresholds,
                                                matrix["people"], settings)
        diagnostics["leave_one_out"] = _leave_one_out(standardized, loadings, matrix["people"], settings)
        scores = {}
        for dimension in structure["dimensions"]:
            column = int(dimension["id"][1:]) - 1
            vector = []
            for row_index in range(len(matrix["people"])):
                row = matrix["values"][row_index, kept]
                mask = ~np.isnan(row)
                if np.sum(mask) < 2:
                    vector.append(np.nan)
                    continue
                weights = loadings[:, column][mask]
                if np.sum(np.abs(weights) >= settings["min_feature_loading"]) < 2:
                    vector.append(np.nan)
                    continue
                computed = ((row - centers) / scales)[mask]
                vector.append(float(np.linalg.lstsq(weights[:, None], computed, rcond=None)[0][0]))
            scores[dimension["id"]] = vector
        diagnostics["site_influence"] = _group_diagnostic(
            scores, [matrix["sites"][person] for person in matrix["people"]], settings)
        diagnostics["age_influence"] = _group_diagnostic(
            scores, [matrix["ages"][person] for person in matrix["people"]], settings)
        for label in ("site_influence", "age_influence"):
            for name, item in diagnostics[label].items():
                if item["flag"]:
                    notes.append(f"{name} 的定位分数与分组变量关联较强"
                                 f"（eta²={item['eta_squared']}，置换 p={item['permutation_p']}），解释需谨慎")
        groups = [matrix["sites"][person] for person in matrix["people"]]
        adjudicated = _select_dimensions(standardized, blocks, settings, seed_offset=101,
                                          residual_groups=groups)
        diagnostics["site_adjustment"] = {
            "dimension_count_before": selection["dimensions"],
            "dimension_count_within_site": adjudicated["dimensions"],
            "eigenvalues_within_site": adjudicated["eigenvalues"],
            "interpretation": ("控制校区后维度消失，说明该关联可由校区差异解释" if adjudicated["dimensions"] == 0
                               else "控制校区后仍有维度成立，关联不是只由校区构成造成"),
        }
        if adjudicated["dimensions"] == 0:
            notes.append("控制校区后没有维度成立；现有共同结构可能只是校区差异，需另找证据")
    else:
        diagnostics["split_half"] = {"status": "no_dimensions"}
        diagnostics["leave_one_out"] = {"status": "no_dimensions"}
        diagnostics["site_influence"] = {}
        diagnostics["age_influence"] = {}
        diagnostics["site_adjustment"] = {"status": "no_dimensions"}

    composites = {}
    for view, indexes in blocks.items():
        columns = standardized[:, indexes]
        present = ~np.isnan(columns).all(axis=1)
        scores = np.full(columns.shape[0], np.nan)
        scores[present] = np.nanmean(columns[present], axis=1)
        composites[view] = scores
    diagnostics["view_composite_site"] = _group_diagnostic(
        composites, [matrix["sites"][person] for person in matrix["people"]], settings)
    diagnostics["view_composite_age"] = _group_diagnostic(
        composites, [matrix["ages"][person] for person in matrix["people"]], settings)
    diagnostics["scale"] = _scale_diagnostic(matrix, kept, notes)
    return diagnostics


# --------------------------------------------------------------------------- #
# positioning
# --------------------------------------------------------------------------- #
def _observed_views(matrix, row):
    return [view for view in matrix["views"] if not np.isnan(row[matrix["blocks"][view]]).all()]


def _position(matrix, structure, settings):
    """Place learners on a saved structure using observed cells only.

    ``support`` is the share of a dimension's squared loadings the learner
    actually has measurements for.  A shared dimension is only ``supported``
    when at least two of its supporting sources are observed; otherwise the
    learner gets a ``partial`` position explicitly limited in scope.  A learner
    with too little observed support is left unassigned.
    """
    if "people" in settings:
        missing = [person for person in settings["people"] if person not in set(matrix["people"])]
        if missing:
            raise ValueError(f"资料中不存在学员：{missing[:5]}")
        selected = sorted(settings["people"])
    else:
        selected = matrix["people"]

    structure_position = {(feature["view"], feature["feature"]): index
                          for index, feature in enumerate(structure["features"])}
    current_position = {(view, feature): index for index, (view, feature) in enumerate(matrix["features"])}
    shared_keys = sorted(set(structure_position) & set(current_position))
    unused_structure = [{"view": view, "feature": feature}
                        for view, feature in sorted(set(structure_position) - set(current_position))]
    unused_current = [{"view": view, "feature": feature}
                      for view, feature in sorted(set(current_position) - set(structure_position))]

    if not structure["dimensions"]:
        positions, person_status = [], Counter()
        for person in selected:
            row = matrix["values"][matrix["people"].index(person)]
            observed_features = int(np.sum(~np.isnan(row)))
            status = "no_observation" if observed_features == 0 else "no_supported_dimensions"
            person_status[status] += 1
            positions.append({"person": person, "observed_views": _observed_views(matrix, row),
                              "observed_features": observed_features,
                              "status": status, "positions": [],
                              "reason": ("该学员没有任何已观测资料" if observed_features == 0
                                         else "没有通过验证与可识别性检查的维度")})
        return positions, {"policy": "no_forced_type",
                           "person_status": dict(sorted(person_status.items())),
                           "people": len(positions), "dimension_status": {},
                           "unused_structure_features": unused_structure,
                           "unused_current_features": unused_current,
                           "scope_note": "没有可命名维度，不生成类型"}

    positions = []
    person_status, dimension_status = Counter(), Counter()
    for person in selected:
        row = matrix["values"][matrix["people"].index(person)]
        observed_views = _observed_views(matrix, row)
        observed_features = int(np.sum(~np.isnan(row)))
        entries = []
        for dimension in structure["dimensions"]:
            full = np.array(dimension["loadings"], dtype=float)
            total_energy = float((full ** 2).sum())
            weights, computed, viewed = [], [], set()
            for view, feature in shared_keys:
                source = structure_position[(view, feature)]
                value = row[current_position[(view, feature)]]
                if np.isnan(value):
                    continue
                weight = float(full[source])
                if abs(weight) < 1e-8:
                    continue
                weights.append(weight)
                computed.append((value - structure["features"][source]["center"]) /
                                structure["features"][source]["scale"])
                if abs(weight) >= settings["min_feature_loading"]:
                    viewed.add(view)
            if not weights:
                entries.append({"dimension": dimension["id"], "kind": dimension["kind"],
                                "status": "unassigned", "score": None, "support": 0.0,
                                "observed_views": [], "observed_features": 0,
                                "reason": "没有该维度载荷上的观测值"})
                dimension_status["unassigned"] += 1
                continue
            weights = np.array(weights)
            support = float((weights ** 2).sum() / max(total_energy, _EPS))
            count = int(np.sum(np.abs(weights) >= settings["min_feature_loading"]))
            in_scope = sorted(viewed & set(dimension["views"]))
            score = (float(np.linalg.lstsq(weights[:, None], np.array(computed), rcond=None)[0][0])
                     if count >= 2 else None)
            reason = None
            if support < settings["partial_support"] or count == 0:
                status, score, reason = "unassigned", None, "观测到的载荷覆盖不足"
            elif count < 2:
                status, reason = "partial", "只有一个有效特征，无法稳定定位"
            elif dimension["kind"] == "shared" and len(in_scope) < 2:
                status, reason = "partial", "只观测到一类相关资料，解释仅限该资料范围"
            elif support < settings["support_threshold"]:
                status, reason = "partial", "观测载荷覆盖低于完整定位阈值"
            else:
                status = "supported"
            dimension_status[status] += 1
            entries.append({"dimension": dimension["id"], "kind": dimension["kind"], "status": status,
                            "score": None if score is None else round(score, 6),
                            "support": round(support, 6), "observed_views": in_scope,
                            "observed_features": count,
                            **({"reason": reason} if reason else {})})
        if observed_features == 0:
            overall, reason = "no_observation", "该学员没有任何已观测资料"
        elif any(entry["status"] == "supported" for entry in entries):
            overall, reason = "supported", None
        elif any(entry["status"] == "partial" for entry in entries):
            overall, reason = "partial", None
        else:
            overall, reason = "unassigned", "观测不足以定位任何维度"
        person_status[overall] += 1
        positions.append({"person": person, "observed_views": observed_views,
                          "observed_features": observed_features, "status": overall,
                          "positions": entries, **({"reason": reason} if reason else {})})
    summary = {"policy": "no_forced_type", "person_status": dict(sorted(person_status.items())),
               "people": len(positions), "dimension_status": dict(sorted(dimension_status.items())),
               "unused_structure_features": unused_structure, "unused_current_features": unused_current,
               "scope_note": "shared 维度需至少在两类资料上同时有观测才标为 supported；"
                             "否则只给适用资料范围内的 partial，不足以定位时不分配类型"}
    return positions, summary


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _fit(matrix, settings):
    values = matrix["values"]
    standardized, means, scales, kept, excluded = _standardize(values)
    feature_rows = [{"view": matrix["features"][index][0], "feature": matrix["features"][index][1],
                     "index": index, "center": float(means[index]), "scale": float(scales[index])}
                    for index in kept]
    removed = [{"view": matrix["features"][item["index"]][0],
                "feature": matrix["features"][item["index"]][1], "reason": item["reason"]}
               for item in excluded]
    notes = []
    structure = {"version": VERSION,
                 "features": [{"view": row["view"], "feature": row["feature"],
                               "center": row["center"], "scale": row["scale"]} for row in feature_rows],
                 "dimensions": [], "missing_policy": _MISSING_POLICY,
                 "estimation": {"removed_features": removed, "people": len(matrix["people"]),
                                "usable_features": len(kept), "missing_policy": _MISSING_POLICY}}
    if len(kept) < 4 or len(matrix["people"]) < 10 or len(matrix["views"]) < 2:
        notes.append("可用特征、学员或资料类不足，不能估计跨资料结构；不生成维度，也不分配类型")
        return structure, None, feature_rows, [], notes

    reduced = standardized[:, kept]
    blocks = _blocks(matrix, kept)
    selection = _select_dimensions(reduced, blocks, settings)
    if selection["adjustment"] > 0.05:
        notes.append(f"成对相关矩阵非正定，已做最近正定投影（相对调整 {selection['adjustment']:.3f}）")
    structure["estimation"].update({
        "dimension_cap": selection["cap"], "permutations": settings["permutations"],
        "alpha": settings["alpha"], "min_pairs": settings["min_pairs"], "seed": settings["seed"],
        "eigenvalues": selection["eigenvalues"], "psd_adjustment": round(selection["adjustment"], 6),
        "dimension_count": selection["dimensions"],
        "thresholds": {view: [{"energy": round(item["energy"], 6), "loading": round(item["loading"], 6)}
                              for item in items]
                       for view, items in selection["thresholds"].items()},
    })
    if selection["dimensions"] == 0:
        notes.append("没有维度同时超过置换零分布与单位根阈值；不生成共同因子或资料独有因子")
        return structure, None, feature_rows, [], notes

    rows = _classify(selection["loadings"], blocks, selection["thresholds"],
                     settings["min_feature_loading"], matrix)
    named = [row for row in rows if row["kind"] != "unidentified"]
    if len(named) < len(rows):
        notes.append("部分维度缺少至少两类资料或两个特征的支持，保留为不可识别，不当作需求类型")
    structure["dimensions"] = [{key: value for key, value in row.items() if key != "evidence"}
                               for row in named]
    structure["estimation"].update({
        "shared_dimensions": [row["id"] for row in named if row["kind"] == "shared"],
        "view_specific_dimensions": [row["id"] for row in named if row["kind"] == "view_specific"],
        "unidentified_dimensions": [row["id"] for row in rows if row["kind"] == "unidentified"],
        "dimension_evidence": [{"id": row["id"], "kind": row["kind"], "views": row["views"],
                                "evidence": row["evidence"]} for row in rows],
    })
    if not structure["estimation"]["shared_dimensions"]:
        notes.append("没有跨两类以上资料同时成立的维度；共同需求证据不足")
    return structure, selection, feature_rows, named, notes


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def _public(structure, positions, summary, diagnostics, notes, extra):
    result = {
        "structure_version": structure["version"],
        "structure": structure,
        "shared_needs": [{"dimension": dimension["id"], "views": dimension["views"],
                          "top_features": dimension["top_features"],
                          "direction_note": "同一维度内相对符号有含义，维度整体符号可翻转"}
                         for dimension in structure["dimensions"] if dimension["kind"] == "shared"],
        "view_specific": [{"dimension": dimension["id"], "views": dimension["views"],
                           "top_features": dimension["top_features"]}
                          for dimension in structure["dimensions"] if dimension["kind"] == "view_specific"],
        "positions": positions, "position_summary": summary,
        "missing_policy": _MISSING_POLICY, "diagnostics": diagnostics, "notes": notes,
    }
    result.update(extra)
    return result


def shared_barriers(tables, parameters):
    """Fit the structure on the supplied release and position every learner."""
    settings = _parameters(parameters)
    matrix = _matrix(tables)
    summaries = _summaries(matrix)
    structure, selection, feature_rows, named, notes = _fit(matrix, settings)
    diagnostics = _diagnostics(matrix, settings, selection, structure, feature_rows, notes)
    positions, summary = _position(matrix, structure, settings)
    return _public(structure, positions, summary, diagnostics, notes, {
        "mode": "fit",
        "view_summaries": summaries["features"],
        "coverage": summaries["coverage"],
        "missing_view_patterns": summaries["missing_view_patterns"],
    })


def apply_structure(tables, parameters, structure, source=None):
    """Position learners on a previously saved structure without re-estimating."""
    settings = _parameters(parameters)
    structure = _check_structure(structure)
    matrix = _matrix(tables)
    summaries = _summaries(matrix)
    positions, summary = _position(matrix, structure, settings)
    source = source or {}
    notes = ["沿用已保存结构定位，缺失资料不补值，只按已观测资料给出适用范围内的解释"]
    structure_note = structure.get("estimation", {}).get("dimension_count", len(structure["dimensions"]))
    if not structure["dimensions"]:
        notes.append("来源结构没有可命名维度，因此只报告观测覆盖，不分配类型")
    diagnostics = {"mode": "apply", "recomputed": False,
                   "reason": "沿用已保存结构定位，不重新估计；稳定性判断保留在结构来源分析中"}
    return _public(structure, positions, summary, diagnostics, notes, {
        "mode": "apply",
        "structure_source": {"analysis": source.get("id"), "revision": source.get("revision"),
                             "structure_version": structure["version"],
                             "source_dimension_count": structure_note},
        "view_summaries": summaries["features"],
        "coverage": summaries["coverage"],
        "missing_view_patterns": summaries["missing_view_patterns"],
    })
