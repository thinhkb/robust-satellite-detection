from __future__ import annotations

import ast
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.split_domain import (
    SPLITS,
    grouped_split,
    source_image_id,
    write_split_manifest,
)


class GroupedSplitTests(unittest.TestCase):
    def setUp(self):
        self.files = [
            Path(f"scene_{scene}_{x}_{y}.txt")
            for scene in range(12)
            for x, y in [(0, 0), (0, 410), (410, 0), (410, 410)]
        ]

    def test_source_image_id(self):
        self.assertEqual(source_image_id("123_0_410.txt"), "123")
        self.assertEqual(source_image_id("scene_part_410_820.jpg"), "scene_part")

    def test_grouped_split_has_no_source_overlap(self):
        split_files = dict(zip(SPLITS, grouped_split(self.files, seed=42)))
        source_sets = {
            split: {source_image_id(path) for path in files}
            for split, files in split_files.items()
        }
        self.assertFalse(source_sets["train"] & source_sets["val"])
        self.assertFalse(source_sets["train"] & source_sets["test"])
        self.assertFalse(source_sets["val"] & source_sets["test"])
        self.assertEqual(sum(map(len, split_files.values())), len(self.files))

    def test_manifest_records_every_tile(self):
        split_files = dict(zip(SPLITS, grouped_split(self.files, seed=7)))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_split_manifest(Path(tmp), split_files)
            with open(manifest, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(self.files))
        self.assertEqual({row["split"] for row in rows}, set(SPLITS))


class WiringTests(unittest.TestCase):
    def test_dg_aug_uses_custom_trainer(self):
        source = Path("scripts/train_dg_aug.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("DGAugTrainer", classes)
        self.assertIn("trainer=DGAugTrainer", source)
        self.assertIn("self.args.augmentations", source)

    def test_suite_contains_acs_and_multi_seed_reporting(self):
        source = Path("scripts/run_experiments.py").read_text(encoding="utf-8")
        self.assertIn('"ACS-YOLO"', source)
        self.assertIn('"--seeds"', source)
        self.assertIn("summary_mean_std.csv", source)


if __name__ == "__main__":
    unittest.main()
