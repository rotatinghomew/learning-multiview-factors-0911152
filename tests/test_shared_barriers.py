import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic  # noqa: E402
from learning.codec import digest, encode  # noqa: E402
from learning.domain import ANALYSES, run_analysis, validate  # noqa: E402
from learning.shared_barriers import apply_structure, shared_barriers  # noqa: E402

FAST = {"permutations": 29, "seed": 3}


def load(release="term-a"):
    return json.loads(Path("data/releases.json").read_text(encoding="utf-8"))[release]


def observations(tables):
    rows = []
    for row in tables["observations"]:
        rows.append((row["person"], row["view"], row["feature"], row["value"]))
    return rows


def structured_release():
    """A release whose true structure has two shared and one specific dimension."""
    return synthetic.release(seed=11, people=320)


def fit_structured():
    return shared_barriers(structured_release(), {"permutations": 99, "seed": 6})


class SharedStructureTests(unittest.TestCase):
    """Paths that need real cross-material signal to be exercised."""

    @classmethod
    def setUpClass(cls):
        cls.result = fit_structured()
        cls.structure = cls.result["structure"]

    def test_shared_and_specific_are_separated(self):
        kinds = {dimension["kind"] for dimension in self.structure["dimensions"]}
        self.assertIn("shared", kinds)
        self.assertTrue(self.result["shared_needs"])
        for need in self.result["shared_needs"]:
            self.assertGreaterEqual(len(need["views"]), 2)
        for specific in self.result["view_specific"]:
            self.assertEqual(len(specific["views"]), 1)

    def test_shared_dimension_loads_on_both_sources(self):
        for dimension in self.structure["dimensions"]:
            if dimension["kind"] != "shared":
                continue
            for view in dimension["views"]:
                self.assertIn(view, dimension["top_features"])
                self.assertGreater(len(dimension["top_features"][view]), 0)

    def test_all_three_views_can_appear(self):
        views = set()
        for dimension in self.structure["dimensions"]:
            views.update(dimension["views"])
        self.assertEqual(views, {"access", "interests", "participation"})

    def test_refit_corresponds_and_loo_reproduces(self):
        self.assertTrue(self.result["diagnostics"]["split_half"]["stable"])
        self.assertTrue(self.result["diagnostics"]["leave_one_out"]["reproduced"])

    def test_raw_scale_does_not_dominate_structure(self):
        scale = self.result["diagnostics"]["scale"]
        self.assertEqual(scale["view_share_raw"].keys(), scale["view_share_standardized"].keys())
        self.assertGreater(scale["view_share_raw"]["participation"], 0.9)

    def test_rerun_is_deterministic(self):
        again = fit_structured()
        self.assertEqual(digest(again["structure"]), digest(self.structure))
        self.assertEqual(digest(again["positions"]), digest(self.result["positions"]))

    def test_apply_matches_fit_and_does_not_refit(self):
        applied = apply_structure(structured_release(), {}, self.structure, {"id": "src", "revision": "r"})
        self.assertEqual(applied["mode"], "apply")
        self.assertEqual(digest(applied["positions"]), digest(self.result["positions"]))
        self.assertFalse(applied["diagnostics"]["recomputed"])

    def test_single_material_learner_is_partial_not_full(self):
        tables = structured_release()
        shared = [dimension for dimension in self.structure["dimensions"] if dimension["kind"] == "shared"]
        self.assertTrue(shared, "夹具应包含 shared 维度")
        people = [row["person"] for row in tables["people"]]
        for person, view in zip(people, ("access", "participation", "interests")):
            trimmed = synthetic.subset(
                {"people": tables["people"],
                 "observations": [row for row in tables["observations"]
                                  if row["person"] == person and row["view"] == view],
                 "features": tables["features"]}, [person])
            position = apply_structure(trimmed, {}, self.structure)["positions"][0]
            self.assertEqual(position["observed_views"], [view])
            for entry in position["positions"]:
                if entry["kind"] == "shared" and entry["status"] != "unassigned":
                    self.assertEqual(entry["status"], "partial", person)
                    self.assertEqual(entry["observed_views"], [view])
                    self.assertLess(len(entry["observed_views"]), 2)
            # A view_specific dimension may legitimately be supported by that one
            # material; a shared one must never be, whatever the overall label.
            self.assertFalse(any(entry["kind"] == "shared" and entry["status"] == "supported"
                                 for entry in position["positions"]), person)

    def test_missing_material_is_not_zero(self):
        tables = structured_release()
        person = next(row["person"] for row in tables["people"]
                      if sum(item["person"] == row["person"] for item in tables["observations"]) > 0)
        only_interests = synthetic.subset(
            {"people": tables["people"],
             "observations": [row for row in tables["observations"]
                              if row["person"] == person and row["view"] == "interests"],
             "features": tables["features"]}, [person])
        position = apply_structure(only_interests, {}, self.structure)["positions"][0]
        self.assertEqual(position["observed_views"], ["interests"])
        expected = sum(1 for row in only_interests["observations"] if row["person"] == person)
        self.assertEqual(position["observed_features"], expected)
        self.assertEqual(position["observed_features"],
                         sum(1 for row in only_interests["observations"] if row["person"] == person))
        self.assertLess(position["observed_features"], len(tables["features"]))

    def test_minimal_support_is_unassigned(self):
        tables = structured_release()
        person = tables["people"][0]["person"]
        rows = [row for row in tables["observations"] if row["person"] == person][:1]
        sparse = synthetic.subset({"people": tables["people"], "observations": rows,
                                   "features": tables["features"]}, [person])
        position = apply_structure(sparse, {}, self.structure)["positions"][0]
        self.assertEqual(position["status"], "unassigned")
        self.assertFalse(any(entry["status"] == "supported" for entry in position["positions"]))

    def test_single_material_release_claims_no_shared_need(self):
        tables = synthetic.one_material(structured_release(), "interests")
        result = shared_barriers(tables, {"permutations": 49, "seed": 6})
        self.assertEqual(result["shared_needs"], [])
        self.assertEqual(result["structure"]["estimation"]["shared_dimensions"], [])
        self.assertTrue(any("共同需求证据不足" in note for note in result["notes"]))

    def test_single_material_apply_never_supports_shared_dimension(self):
        tables = synthetic.one_material(structured_release(), "interests")
        applied = apply_structure(tables, {}, self.structure)
        for position in applied["positions"]:
            self.assertFalse(any(entry["kind"] == "shared" and entry["status"] == "supported"
                                 for entry in position["positions"]))

    def test_site_adjustment_detects_pure_site_confound(self):
        tables = synthetic.release(seed=5, people=300, site_shift=1.0)
        result = shared_barriers(tables, {"permutations": 49, "seed": 8})
        adjustment = result["diagnostics"]["site_adjustment"]
        self.assertIn("dimension_count_before", adjustment)
        self.assertIn("dimension_count_within_site", adjustment)
        flagged = [name for name, item in result["diagnostics"]["site_influence"].items() if item["flag"]]
        self.assertTrue(flagged, "站点平移应当被识别为分组关联")


class ReusableStructureTests(unittest.TestCase):
    """A saved fit is reused by id; the replay must not re-estimate."""

    def test_apply_reproduces_fit_positions_on_reordered_release(self):
        fit = shared_barriers(load("term-a"), FAST)
        applied = apply_structure(load("term-b"), FAST, fit["structure"], {"id": "x", "revision": "y"})
        self.assertEqual(applied["mode"], "apply")
        self.assertFalse(applied["diagnostics"]["recomputed"])
        self.assertEqual(digest(applied["positions"]), digest(fit["positions"]))
        self.assertEqual(applied["structure_source"]["structure_version"], fit["structure_version"])

    def test_apply_does_not_recompute_structure(self):
        fit = shared_barriers(load("term-a"), FAST)
        applied = apply_structure(load("term-b"), FAST, fit["structure"], {"id": "x", "revision": "y"})
        self.assertEqual(digest(applied["structure"]), digest(fit["structure"]))
        self.assertEqual(applied["shared_needs"], fit["shared_needs"])
        self.assertEqual(applied["view_specific"], fit["view_specific"])

    def test_rejects_structure_from_other_version(self):
        fit = shared_barriers(load("term-a"), FAST)
        broken = json.loads(json.dumps(fit["structure"]))
        broken["version"] = "shared-barriers/0"
        with self.assertRaises(ValueError):
            apply_structure(load("term-b"), FAST, broken)

    def test_rejects_malformed_structure(self):
        fit = shared_barriers(load("term-a"), FAST)
        for patch in (
            lambda s: s.pop("features"),
            lambda s: s["features"][0].__setitem__("scale", 0),
            lambda s: s["dimensions"][0].__setitem__("loadings", [0.0]) if s["dimensions"] else s.update(dimensions=[{"id": "d1", "kind": "shared", "views": ["a"], "loadings": [1.0]}]),
        ):
            broken = json.loads(json.dumps(fit["structure"]))
            patch(broken)
            with self.subTest(patch=broken):
                with self.assertRaises(ValueError):
                    apply_structure(load("term-b"), FAST, broken)

    def test_missing_person_rejected(self):
        fit = shared_barriers(load("term-a"), FAST)
        with self.assertRaises(ValueError):
            apply_structure(load("term-b"), {**FAST, "people": ["nobody"]}, fit["structure"])

    def test_apply_for_unknown_feature_is_not_a_trained_column(self):
        fit = shared_barriers(load("term-a"), FAST)
        tables = load("term-b")
        tables["features"].append({"view": "interests", "feature": "cooking", "minimum": 0, "maximum": 5})
        tables["observations"].append({"person": "p001", "view": "interests", "feature": "cooking", "value": 4.0})
        validate(tables)
        applied = apply_structure(tables, FAST, fit["structure"])
        self.assertIn({"view": "interests", "feature": "cooking"},
                      applied["position_summary"]["unused_current_features"])
        self.assertEqual(digest(applied["structure"]), digest(fit["structure"]))


class StructureTests(unittest.TestCase):
    def test_analysis_is_registered(self):
        self.assertIn("shared_barriers", ANALYSES)

    def test_input_order_does_not_change_structure(self):
        before = shared_barriers(load("term-a"), FAST)
        changed = {name: list(reversed(rows)) for name, rows in load("term-a").items()}
        after = shared_barriers(changed, FAST)
        self.assertEqual(digest(before["structure"]), digest(after["structure"]))
        self.assertEqual(digest(before["positions"]), digest(after["positions"]))

    def test_summary_does_not_mutate_evidence(self):
        tables = load("term-a")
        before = digest(tables)
        validate(tables)
        result = shared_barriers(tables, FAST)
        encode(result)
        self.assertEqual(before, digest(tables))
        self.assertEqual(observations(tables), observations(load("term-a")))

    def test_missing_view_is_not_zero(self):
        tables = load("term-a")
        result = shared_barriers(tables, FAST)
        counts = {row["feature"]: row["count"] for row in result["view_summaries"]}
        expected = {}
        for row in tables["observations"]:
            expected[row["feature"]] = expected.get(row["feature"], 0) + 1
        self.assertEqual(counts, expected)
        self.assertFalse(result["missing_policy"]["imputed_values"])
        self.assertEqual(result["missing_policy"]["estimation"], "pairwise_complete")

    def test_no_imputation_value_is_reported(self):
        result = shared_barriers(load("term-a"), FAST)
        self.assertNotIn("imputed", json.dumps(result["positions"], ensure_ascii=False))

    def test_limited_release_does_not_force_types(self):
        result = shared_barriers(load("limited"), FAST)
        self.assertEqual(result["structure"]["dimensions"], [])
        self.assertEqual(result["shared_needs"], [])
        self.assertEqual(result["position_summary"]["policy"], "no_forced_type")
        self.assertFalse(any(position["status"] in {"supported", "partial"} for position in result["positions"]))

    def test_outcome_is_json_encodable(self):
        encode(shared_barriers(load("term-a"), FAST))

    def test_unknown_parameter_rejected(self):
        with self.assertRaises(ValueError):
            shared_barriers(load("term-a"), {"impute": "mean"})

    def test_parameter_bounds(self):
        for parameters in ({"permutations": 1}, {"alpha": 0}, {"max_dimensions": 0},
                           {"partial_support": .9, "support_threshold": .5}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    shared_barriers(load("term-a"), parameters)

    def test_no_dimensions_means_no_scores(self):
        result = shared_barriers(load("term-a"), FAST)
        if not result["structure"]["dimensions"]:
            for position in result["positions"]:
                self.assertEqual(position["positions"], [])
                self.assertIn(position["status"], {"no_observation", "no_supported_dimensions"})

    def test_reused_structure_keeps_source_reference(self):
        record = {"id": "abc", "revision": "rev-1",
                  "request": {"analysis": "shared_barriers"},
                  "result": {"mode": "fit", "structure": shared_barriers(load("term-a"), FAST)["structure"]}}
        result = run_analysis("shared_barriers", load("term-b"), {**FAST, "structure_from": "abc"},
                              lambda analysis_id: record if analysis_id == "abc" else None)
        self.assertEqual(result["mode"], "apply")
        self.assertEqual(result["structure_source"]["analysis"], "abc")
        self.assertEqual(result["structure_source"]["revision"], "rev-1")

    def test_structure_from_must_reference_a_fit(self):
        record = {"id": "abc", "revision": "rev-1",
                  "request": {"analysis": "shared_barriers"}, "result": {"mode": "apply", "structure": {}}}
        with self.assertRaises(ValueError):
            run_analysis("shared_barriers", load("term-b"), {**FAST, "structure_from": "abc"},
                         lambda analysis_id: record)
        with self.assertRaises(ValueError):
            run_analysis("shared_barriers", load("term-b"), {**FAST, "structure_from": "gone"},
                         lambda analysis_id: None)


class SupportScopeTests(unittest.TestCase):
    """Positioning must respect which material the learner actually submitted."""

    @classmethod
    def setUpClass(cls):
        tables = structured_release()
        cls.fit = shared_barriers(tables, {"permutations": 99, "seed": 6})
        cls.tables = tables
        cls.people = sorted({row["person"] for row in tables["observations"]})

    def position(self, person, structure=None):
        structure = structure or self.fit["structure"]
        result = apply_structure(self.tables, {**FAST, "people": [person]}, structure)
        return result["positions"][0]

    def test_people_without_observations_are_not_typed(self):
        everyone = {row["person"] for row in self.tables["people"]}
        empty = sorted(everyone - set(self.people))
        self.assertTrue(empty)
        for person in empty:
            position = self.position(person)
            self.assertEqual(position["status"], "no_observation")
            self.assertTrue(all(entry["status"] == "unassigned" for entry in position["positions"]))

    def test_single_material_learner_gets_partial_scope_not_full_type(self):
        self.assertTrue(any(dimension["kind"] == "shared"
                            for dimension in self.fit["structure"]["dimensions"]))
        counts = {}
        for row in self.tables["observations"]:
            counts[row["person"]] = counts.get(row["person"], 0) + 1
        for person in (person for person, count in counts.items() if count <= 3):
            position = self.position(person)
            for entry in position["positions"]:
                if entry["kind"] == "shared" and entry["status"] != "unassigned":
                    self.assertEqual(entry["status"], "partial")
                    self.assertLess(len(entry["observed_views"]), 2)


if __name__ == "__main__":
    unittest.main()
