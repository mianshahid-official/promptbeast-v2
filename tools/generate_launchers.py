"""
PromptBeast Visual Style Launcher Generator.
Reads style profiles from styles/*/profile.json and generates one dedicated .bat file
per style in the application root directory.
"""
import os
import sys
import json
from pathlib import Path


def generate_launchers(root_dir: Path = None):
    """Generates run-<style_slug>.bat in root_dir for each style in styles/."""
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent

    styles_dir = root_dir / "styles"
    if not styles_dir.exists():
        print(f"[Error] Styles directory not found at: {styles_dir}")
        return []

    generated = []
    print(f"Scanning styles directory: {styles_dir}")

    for style_folder in sorted(styles_dir.iterdir()):
        if not style_folder.is_dir():
            continue

        profile_path = style_folder / "profile.json"
        style_id = style_folder.name

        # Read style name if profile.json exists
        style_name = style_id.replace("_", " ").title()
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                style_name = data.get("name", style_name)
            except Exception:
                pass

        # Convert slug: zombie_survival -> zombie-survival
        slug = style_id.replace("_", "-")
        bat_name = f"run-{slug}.bat"
        bat_path = root_dir / bat_name

        # .bat script matching specification:
        # @echo off
        # cd /d %~dp0
        # python -m app.main --style <style_id>
        # pause
        bat_content = (
            "@echo off\n"
            f"title PromptBeast Studio — {style_name}\n"
            "cd /d %~dp0\n"
            "set PYTHONPATH=%~dp0\n"
            f"python -m app.main --style {style_id}\n"
            "pause\n"
        )

        bat_path.write_text(bat_content, encoding="utf-8")
        print(f"  + Generated: {bat_name} (Style: {style_name} [{style_id}])")
        generated.append(bat_path)

    print(f"Successfully generated {len(generated)} style launchers in {root_dir}.")
    return generated


if __name__ == "__main__":
    generate_launchers()

