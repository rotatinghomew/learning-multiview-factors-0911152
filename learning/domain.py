from collections import Counter, defaultdict
import math

import numpy as np

from learning.codec import numeric, require_fields, unique

PRODUCT = '学习支持画像'
KEYS = {'people': ['person'], 'observations': ['person', 'view', 'feature'], 'features': ['view', 'feature']}


def validate(tables):
    if set(tables) != set(KEYS):
        raise ValueError("资料表集合与产品契约不一致")
    for name, fields in KEYS.items():
        require_fields(tables[name], fields)
        unique(tables[name], fields)
    people = {row["person"] for row in tables["people"]}
    features = {(row["view"], row["feature"]): row for row in tables["features"]}
    for row in tables["observations"]:
        key = (row["view"], row["feature"])
        if row["person"] not in people or key not in features:
            raise ValueError("画像观测引用不存在")
        feature = features[key]
        if not feature["minimum"] <= numeric(row["value"], "画像观测") <= feature["maximum"]:
            raise ValueError("画像观测超界")


def overview(tables, parameters):
    observations = defaultdict(list)
    coverage = defaultdict(set)
    for row in tables["observations"]:
        observations[(row["view"], row["feature"])].append(row["value"])
        coverage[row["view"]].add(row["person"])
    return {"features": [{"view": key[0], "feature": key[1], "count": len(values),
                          "mean": float(np.mean(sorted(values))), "std": float(np.std(sorted(values)))}
                         for key, values in sorted(observations.items())],
            "coverage": {view: len(members) for view, members in coverage.items()}}


ANALYSES = {"overview": overview}
