from collections import Counter
import json
from pathlib import Path
import unittest

from learning.domain import overview, validate


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def test_feature_coverage_counts_people_not_cells(self):
        result = overview(self.tables, {})
        for view,coverage in result["coverage"].items():
            expected = {row["person"] for row in self.tables["observations"] if row["view"] == view}
            self.assertEqual(coverage, len(expected))
        self.assertEqual(sum(row["count"] for row in result["features"]), len(self.tables["observations"]))

    def test_no_view_is_not_all_zero(self):
        self.tables["observations"] = []
        validate(self.tables)
        self.assertEqual(overview(self.tables, {})["features"], [])

    def test_out_of_range_feature_rejected(self):
        self.tables["observations"][0]["value"] = 1000
        with self.assertRaises(ValueError):
            validate(self.tables)
