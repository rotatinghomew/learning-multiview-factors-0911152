from collections import defaultdict

import numpy as np

from learning.codec import numeric, require_fields, unique
from learning.shared_barriers import apply_structure, shared_barriers

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


ANALYSES = {"overview": overview, "shared_barriers": shared_barriers}


def run_analysis(name, tables, parameters, lookup):
    """Run a registered analysis, resolving ``structure_from`` for reuse.

    ``parameters.structure_from`` names an earlier shared_barriers analysis id.
    Its saved structure is applied to the current tables; the original fit is
    never recomputed or overwritten.
    """
    if name not in ANALYSES:
        raise ValueError("分析类型未登记")
    if name != "shared_barriers" or "structure_from" not in parameters:
        return ANALYSES[name](tables, parameters)
    record = lookup(parameters["structure_from"])
    if record is None:
        raise ValueError("引用的结构分析不存在")
    if record["request"].get("analysis") != "shared_barriers" or record["result"].get("mode") != "fit":
        raise ValueError("引用的分析不是可复用的结构拟合")
    structure = record["result"].get("structure")
    if not isinstance(structure, dict):
        raise ValueError("引用的分析没有保存结构")
    clean = {key: value for key, value in parameters.items() if key != "structure_from"}
    return apply_structure(tables, clean, structure,
                           {"id": record["id"], "revision": record["revision"]})
