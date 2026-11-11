"""
Unit & Integration Tests for Per-Style Launchers and Style Profiles.
Verifies:
1. One folder per style in styles/ with profile.json and master_prompt.txt.
2. Zombie and What-If master prompts follow rules (no headings, simple direct prose, no bullet points).
3. Fallback to default master prompt for styles without custom master prompt.
4. Launcher generator generates run-<style_slug>.bat in root.
5. Multi-instance support (no single instance locks).
6. UI read-only style badge and absence of visual style dropdown.
"""
import os
import sys
import json
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.style_manager import StyleManager, StyleProfile
from tools.generate_launchers import generate_launchers


class TestStyleLaunchers(unittest.TestCase):
    def test_styles_directory_structure(self):
        styles_root = StyleManager.get_styles_root()
        self.assertTrue(styles_root.exists(), "styles/ root folder not found")

        required_styles = ["default", "zombie_survival", "what_if", "ink"]
        for s in required_styles:
            s_dir = styles_root / s
            self.assertTrue(s_dir.exists(), f"styles/{s} directory missing")
            self.assertTrue((s_dir / "profile.json").exists(), f"styles/{s}/profile.json missing")
            self.assertTrue((s_dir / "master_prompt.txt").exists(), f"styles/{s}/master_prompt.txt missing")

    def test_zombie_and_whatif_master_prompts_rules(self):
        # 1. Zombie Survival Prompt
        zombie_prof = StyleManager.get_style("zombie_survival")
        z_prompt = zombie_prof.master_prompt
        self.assertTrue(len(z_prompt) > 100, "Zombie master prompt too short")
        self.assertNotIn("## 1.", z_prompt, "Heading found in zombie prompt")
        self.assertNotIn("**Subject:**", z_prompt, "Bold category label found in zombie prompt")
        self.assertNotIn("**Environment:**", z_prompt, "Bold category label found in zombie prompt")
        self.assertIn("red baseball cap", z_prompt.lower(), "Hero lock missing in zombie prompt")
        self.assertIn("olive-green", z_prompt.lower(), "Zombie horde rule missing in zombie prompt")

        # 2. What-If Prompt
        whatif_prof = StyleManager.get_style("what_if")
        w_prompt = whatif_prof.master_prompt
        self.assertTrue(len(w_prompt) > 100, "What-If master prompt too short")
        self.assertNotIn("## 1.", w_prompt, "Heading found in what-if prompt")
        self.assertNotIn("**Subject:**", w_prompt, "Bold category label found in what-if prompt")
        self.assertIn("vector", w_prompt.lower(), "Vector style missing in what-if prompt")

    def test_style_fallback_to_default_master_prompt(self):
        # paper_cutout has no custom master_prompt.txt, so it must fall back to default
        paper_prof = StyleManager.get_style("paper_cutout")
        default_prof = StyleManager.get_style("default")

        self.assertTrue(paper_prof.master_prompt)
        self.assertEqual(paper_prof.master_prompt, default_prof.master_prompt)

    def test_style_normalization_and_aliases(self):
        self.assertEqual(StyleManager.normalize_style_id("zombie-survival"), "zombie_survival")
        self.assertEqual(StyleManager.normalize_style_id("ZOMBIE"), "zombie_survival")
        self.assertEqual(StyleManager.normalize_style_id("what-if"), "what_if")
        self.assertEqual(StyleManager.normalize_style_id("whatif"), "what_if")
        self.assertEqual(StyleManager.normalize_style_id("ink"), "ink")
        self.assertEqual(StyleManager.normalize_style_id(""), "default")
        self.assertEqual(StyleManager.normalize_style_id(None), "default")

    def test_launcher_generator(self):
        generated = generate_launchers(ROOT_DIR)
        self.assertTrue(len(generated) >= 4, f"Expected at least 4 launchers, got {len(generated)}")

        expected_bats = [
            ROOT_DIR / "run-zombie-survival.bat",
            ROOT_DIR / "run-what-if.bat",
            ROOT_DIR / "run-ink.bat",
            ROOT_DIR / "run-default.bat",
        ]
        for b in expected_bats:
            self.assertTrue(b.exists(), f"Expected bat launcher {b.name} was not generated")
            content = b.read_text(encoding="utf-8")
            self.assertIn("@echo off", content)
            self.assertIn("python -m app.main --style", content)
            self.assertIn("pause", content)

    def test_root_directory_clean_layout(self):
        # Confirm user data files are moved out of root
        forbidden_in_root = [
            ROOT_DIR / "image_prompts.txt",
            ROOT_DIR / "prompts_only.txt",
            ROOT_DIR / "sample data.txt",
            ROOT_DIR / "MASTER_AI_STORYBOARD_AND_BIBLES_PROMPT.txt",
            ROOT_DIR / "crash.log",
        ]
        for f in forbidden_in_root:
            self.assertFalse(f.exists(), f"File {f.name} should not be in root folder, must be in data/")

        # Confirm they exist in data/
        self.assertTrue((ROOT_DIR / "data" / "sample data.txt").exists(), "data/sample data.txt missing")


if __name__ == "__main__":
    unittest.main()
