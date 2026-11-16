"""Synthetic releases with known cross-material structure.

The shipped fixtures happen to carry no cross-material signal, so these tests
build a release where two dimensions are genuinely shared and one is specific
to a single material.  That lets the shared/partial/identifiability paths be
exercised instead of skipped.
"""

import numpy as np

FEATURES = [("access", "travel"), ("access", "hearing"), ("access", "schedule"),
            ("interests", "art"), ("interests", "digital"), ("interests", "fitness"),
            ("interests", "history"),
            ("participation", "attendance"), ("participation", "practice"),
            ("participation", "volunteer")]


def release(seed=7, people=260, site_shift=None):
    """Build a release.

    True structure: a shared interests+access factor, a shared
    access+participation factor, and an interests-only factor.  ``site_shift``
    adds a per-site level shift to the first three and next four features.
    """
    random = np.random.default_rng(seed)
    loadings = np.zeros((10, 3))
    loadings[3:7, 0] = [.72, .70, .68, .66]
    loadings[0:3, 0] = [.55, .53, .50]
    loadings[7:10, 1] = [.78, .75, .72]
    loadings[0:3, 1] = [.60, .58, .55]
    loadings[4:6, 2] = [.60, .55]
    factors = random.normal(size=(people, 3))
    noise = random.normal(size=(people, 10)) * np.sqrt(np.clip(1 - (loadings ** 2).sum(1), 1e-3, None))
    values = factors @ loadings.T + noise
    sites = random.integers(0, 4, people)
    if site_shift is not None:
        shift = np.array([0.0, 0.8, 1.6, 2.4])[sites] * site_shift
        values[:, :3] += shift[:, None]
        values[:, 3:7] += shift[:, None]
    for index, (view, feature) in enumerate(FEATURES):
        if view == "participation":
            values[:, index] = np.clip(values[:, index] * 28 + 50, 0, 100)
        else:
            values[:, index] = np.clip(values[:, index] * 1.45 + 2.5, 0, 5)
    for person in range(people):
        draw = random.random()
        if draw < .06:
            values[person, 0:3] = np.nan
        elif draw < .12:
            values[person, 3:7] = np.nan
        elif draw < .18:
            values[person, 7:10] = np.nan
        elif draw < .21:
            values[person, :] = np.nan
    rows = [{"person": f"q{index:03d}", "site": f"s{sites[index]}",
             "age_band": "70+" if index % 3 == 0 else "60-69"} for index in range(people)]
    observations = [{"person": rows[person]["person"], "view": view, "feature": feature,
                     "value": round(float(values[person, index]), 3)}
                    for person in range(people)
                    for index, (view, feature) in enumerate(FEATURES)
                    if not np.isnan(values[person, index])]
    features = [{"view": view, "feature": feature, "minimum": 0,
                 "maximum": 100 if view == "participation" else 5} for view, feature in FEATURES]
    return {"people": rows, "observations": observations, "features": features}


def subset(tables, people):
    """Keep only the named learners, preserving the feature registry."""
    keep = set(people)
    return {"people": [row for row in tables["people"] if row["person"] in keep],
            "observations": [row for row in tables["observations"] if row["person"] in keep],
            "features": tables["features"]}


def one_material(tables, view):
    """Keep only observations from a single material."""
    return {"people": tables["people"],
            "observations": [row for row in tables["observations"] if row["view"] == view],
            "features": tables["features"]}


FAST = {"permutations": 49, "seed": 4}
