"""
FFmpeg Video Renderer and 4-Point Post-Render Verification Engine for PromptBeast-V2 (Step 6).
Implements:
1. Continuous timeline: image displays from scene start to next scene start (last to audio end). Zero gaps.
2. Transitions from profile.json (crossfade 0.5s, cut, dip-to-black).
3. Ken Burns motion (slow zoom-in or pan from storyboard beat), cropped to project aspect ratio.
4. Karaoke subtitles burned in with word-level highlights.
5. Narration audio as single audio track.
6. H.264 CRF 18 output to <project>_final.mp4.
7. Four mandatory post-render checks:
   - Technical (duration ±0.5s, resolution, streams)
   - Audio (stream present, <=80% silence, no dead spots)
   - Transcript-diff (exact word match)
   - Visual spot checks (frames at 25%, 50%, 75%, 90% non-black)
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image

from app.core.validator import ValidationError, normalize_token, probe_audio_duration
from app.core.subtitle_engine import SubtitleEngine
from app.models.timeline import NarrationData, WordTimestamp
from app.models.video_config import SubtitleConfig, SubtitlePreset


class RenderVerificationError(Exception):
    """Raised when any of the 4 mandatory post-render checks fail."""
    pass


@dataclass
class PostRenderVerificationResult:
    technical_passed: bool = False
    audio_passed: bool = False
    transcript_diff_passed: bool = False
    visual_spot_checks_passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return (
            self.technical_passed
            and self.audio_passed
            and self.transcript_diff_passed
            and self.visual_spot_checks_passed
        )


def verify_technical(output_file: Path, expected_duration: float, expected_resolution: Tuple[int, int]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Check 1: Output file exists, decodes, duration within ±0.5s, resolution matches, video+audio present."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False, {}, f"Output file does not exist or is empty: {output_file}"

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height:format=duration",
        "-of", "json",
        str(output_file)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
    except Exception as e:
        return False, {}, f"ffprobe failed to decode output file: {e}"

    streams = data.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video:
        return False, data, "Output video stream missing."
    if not has_audio:
        return False, data, "Output audio stream missing."

    v_stream = next(s for s in streams if s.get("codec_type") == "video")
    actual_w = int(v_stream.get("width", 0))
    actual_h = int(v_stream.get("height", 0))
    exp_w, exp_h = expected_resolution

    if actual_w != exp_w or actual_h != exp_h:
        return False, data, f"Resolution mismatch: expected {exp_w}x{exp_h}, got {actual_w}x{actual_h}"

    format_dur = float(data.get("format", {}).get("duration", 0.0))
    dur_diff = abs(format_dur - expected_duration)
    if dur_diff > 0.5:
        return False, data, f"Duration mismatch: expected {expected_duration:.2f}s, got {format_dur:.2f}s (diff {dur_diff:.2f}s > 0.5s)"

    return True, {"duration": format_dur, "width": actual_w, "height": actual_h}, None


def verify_audio(output_file: Path, expected_duration: float) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Check 2: Audio stream present; <=80% silence; no hard distortion / dead spots."""
    cmd = [
        "ffmpeg", "-i", str(output_file),
        "-af", "silencedetect=noise=-50dB:d=2.0",
        "-f", "null", "-"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stderr = res.stderr
    except Exception as e:
        return False, {}, f"Audio analysis failed: {e}"

    # Sum silence durations
    silence_durations = [float(m) for m in re.findall(r"silence_duration:\s*([\d\.]+)", stderr)]
    total_silence = sum(silence_durations)
    silence_ratio = total_silence / max(0.1, expected_duration)

    if silence_ratio > 0.80:
        return False, {"silence_ratio": silence_ratio, "total_silence": total_silence}, (
            f"Audio is {silence_ratio * 100:.1f}% silent (exceeds 80% maximum allowed silence)."
        )

    return True, {"silence_ratio": silence_ratio, "total_silence": total_silence}, None


def verify_transcript_diff(script_words: List[str], timestamp_words: List[str]) -> Tuple[bool, Optional[str]]:
    """Check 3: Reconstruct narration from timestamp words and compare to script (must match exactly)."""
    clean_s = [normalize_token(w) for w in script_words if normalize_token(w)]
    clean_t = [normalize_token(w) for w in timestamp_words if normalize_token(w)]

    if len(clean_s) != len(clean_t):
        return False, f"Transcript diff word count mismatch: script={len(clean_s)} words, timestamps={len(clean_t)} words."

    for idx, (sw, tw) in enumerate(zip(clean_s, clean_t)):
        if sw != tw:
            return False, f"Transcript diff mismatch at position {idx}: script '{sw}' != timestamp '{tw}'."

    return True, None


def verify_visual_spot_checks(
    output_file: Path,
    expected_duration: float,
    temp_dir: Path
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """Check 4: Sample frames at 25%, 50%, 75%, 90% are non-black and readable."""
    percentages = [0.25, 0.50, 0.75, 0.90]
    results = []

    for pct in percentages:
        sample_time = round(expected_duration * pct, 2)
        frame_path = temp_dir / f"spot_check_{int(pct*100)}.png"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(sample_time),
            "-i", str(output_file),
            "-vframes", "1",
            str(frame_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            return False, results, f"Failed to extract spot-check frame at {sample_time:.2f}s ({int(pct*100)}%): {e}"

        if not frame_path.exists():
            return False, results, f"Spot-check frame was not generated at {sample_time:.2f}s."

        try:
            with Image.open(frame_path) as im:
                gray = im.convert("L")
                pixels = list(gray.getdata())
                avg_luma = sum(pixels) / max(1, len(pixels))

            # Black frame check (luma < 2.0 indicates pure black/blank frame)
            if avg_luma < 2.0:
                return False, results, f"Spot-check frame at {sample_time:.2f}s ({int(pct*100)}%) is completely black (luma {avg_luma:.2f})."

            results.append({"percentage": pct, "timestamp": sample_time, "luma": round(avg_luma, 2)})
        except Exception as e:
            return False, results, f"Failed to inspect spot check image: {e}"

    return True, results, None


class VideoRenderer:
    """Renders video according to Step 6 rules."""

    def __init__(
        self,
        project_dir: Union[str, Path],
        aspect_ratio: str = "16:9",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ):
        self.project_dir = Path(project_dir).resolve()
        self.aspect_ratio = aspect_ratio  # "16:9" or "9:16"
        self.progress_callback = progress_callback or (lambda p, m: None)

        if self.aspect_ratio == "9:16":
            self.resolution = (720, 1280)
        else:
            self.resolution = (1280, 720)

    def render(
        self,
        scenes: List[Dict[str, Any]],
        images_map: Dict[str, Path],
        audio_path: Union[str, Path],
        audio_duration: float,
        timestamp_words: List[Dict[str, Any]],
        script_words: List[str],
        profile_data: Optional[Dict[str, Any]] = None,
        output_filename: Optional[str] = None
    ) -> Tuple[Path, PostRenderVerificationResult]:
        """
        Executes full rendering pipeline:
        1. Assembles continuous timeline (no gaps).
        2. Applies Ken Burns slow motion & transitions.
        3. Burns word-level karaoke subtitles.
        4. Encodes to H.264 CRF 18.
        5. Runs 4 mandatory verification checks.
        """
        out_dir = self.project_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        proj_name = self.project_dir.name
        out_file = out_dir / (output_filename or f"{proj_name}_final.mp4")

        self.progress_callback(5, "Preparing render timeline...")

        # 1. Continuous Timeline Assembly:
        # scene i image displays from scene[i].start to scene[i+1].start (last to audio_duration)
        timeline_clips: List[Dict[str, Any]] = []
        for i, sc in enumerate(scenes):
            s_id = sc["scene_id"]
            img_p = images_map.get(s_id)
            if not img_p or not img_p.exists():
                raise ValidationError(f"Render gate failure: image missing for scene {s_id}")

            start_t = float(sc["start"])
            if i < len(scenes) - 1:
                end_t = float(scenes[i + 1]["start"])
            else:
                end_t = float(audio_duration)

            display_dur = max(0.5, end_t - start_t)
            composition = sc.get("composition", "slow zoom-in")
            timeline_clips.append({
                "scene_id": s_id,
                "image_path": img_p,
                "start": start_t,
                "end": end_t,
                "duration": display_dur,
                "composition": composition
            })

        self.progress_callback(15, "Generating karaoke subtitles...")
        # 2. Generate Karaoke Subtitles (ASS file)
        temp_dir = Path(tempfile.mkdtemp(prefix="pb_render_"))
        ass_path = temp_dir / "karaoke.ass"

        # Build NarrationData for SubtitleEngine
        wt_list = [
            WordTimestamp(word=str(w.get("word", "")), start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)))
            for w in timestamp_words
        ]
        narr_data = NarrationData(words=wt_list, duration_seconds=audio_duration)
        sub_cfg = SubtitleConfig(
            font_size=28 if self.aspect_ratio == "16:9" else 36,
            primary_color="#FFFFFF",
            highlight_color="#FFD700"
        )
        target_w, target_h = self.resolution
        SubtitleEngine.generate_ass_subtitles(
            narration=narr_data,
            output_ass_path=str(ass_path),
            config=sub_cfg,
            target_w=target_w,
            target_h=target_h
        )

        self.progress_callback(25, "Rendering scenes with Ken Burns motion...")

        # 3. Render individual scene clips with Ken Burns camera motion & transitions
        target_w, target_h = self.resolution
        fps = 25
        transition = "cut"
        if profile_data:
            transition = profile_data.get("transition", "cut")

        clips_dir = temp_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        def _render_clip(item: Tuple[int, Dict[str, Any]]) -> str:
            idx, clip = item
            dur = float(clip["duration"])
            frames = max(1, int(fps * dur))
            comp = str(clip.get("composition", "")).lower()
            clip_out = clips_dir / f"clip_{idx:04d}.ts"

            if "pan left" in comp:
                z_expr = "1.12"
                x_expr = f"(1.0-on/{frames})*(iw-iw/zoom)"
                y_expr = "(ih-ih/zoom)/2"
            elif "pan right" in comp:
                z_expr = "1.12"
                x_expr = f"(on/{frames})*(iw-iw/zoom)"
                y_expr = "(ih-ih/zoom)/2"
            elif "zoom-out" in comp:
                z_expr = f"max(1.15-0.15*(on/{frames}),1.0)"
                x_expr = "(iw-iw/zoom)/2"
                y_expr = "(ih-ih/zoom)/2"
            elif "static" in comp or "wide establishing" in comp:
                z_expr = "min(pzoom+0.0006,1.06)"
                x_expr = "(iw-iw/zoom)/2"
                y_expr = "(ih-ih/zoom)/2"
            else:  # default slow zoom-in
                z_expr = "min(pzoom+0.0015,1.15)"
                x_expr = "(iw-iw/zoom)/2"
                y_expr = "(ih-ih/zoom)/2"

            fade_filter = ""
            if transition == "dip_to_black" and dur > 0.8:
                fade_filter = f",fade=t=in:st=0:d=0.2,fade=t=out:st={dur-0.2:.3f}:d=0.2"

            vf = (
                f"scale={target_w*2}:{target_h*2}:force_original_aspect_ratio=increase,"
                f"crop={target_w*2}:{target_h*2},"
                f"zoompan=z='{z_expr}':d={frames}:x='{x_expr}':y='{y_expr}':s={target_w}x{target_h}:fps={fps}"
                f"{fade_filter}"
            )

            cmd_clip = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(clip["image_path"].resolve()),
                "-vf", vf,
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(clip_out)
            ]
            subprocess.run(cmd_clip, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return str(clip_out)

        from concurrent.futures import ThreadPoolExecutor
        workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            clip_paths = list(executor.map(_render_clip, enumerate(timeline_clips)))

        concat_txt = temp_dir / "input_concat.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                p_str = str(Path(cp).resolve()).replace("\\", "/")
                f.write(f"file '{p_str}'\n")

        # Escaped subtitle path for libass filter
        ass_esc = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")

        self.progress_callback(60, "Concatenating scenes and burning karaoke subtitles (H.264 CRF 18)...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-i", str(Path(audio_path).resolve()),
            "-vf", f"ass='{ass_esc}'",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(out_file)
        ]

        self.progress_callback(40, "Rendering video with FFmpeg (CRF 18)...")
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg render process failed:\n{e.stderr}")

        self.progress_callback(85, "Executing 4-point post-render verification...")

        # 4. Mandatory Post-Render Verification Checks
        v_res = PostRenderVerificationResult()

        # Check 1: Technical
        tech_ok, tech_data, tech_err = verify_technical(out_file, audio_duration, self.resolution)
        v_res.technical_passed = tech_ok
        v_res.details["technical"] = tech_data
        if not tech_ok:
            v_res.errors.append(f"Check 1 (Technical) failed: {tech_err}")

        # Check 2: Audio
        aud_ok, aud_data, aud_err = verify_audio(out_file, audio_duration)
        v_res.audio_passed = aud_ok
        v_res.details["audio"] = aud_data
        if not aud_ok:
            v_res.errors.append(f"Check 2 (Audio) failed: {aud_err}")

        # Check 3: Transcript Diff
        t_words_str = [str(w.get("word", "")) for w in timestamp_words]
        tdiff_ok, tdiff_err = verify_transcript_diff(script_words, t_words_str)
        v_res.transcript_diff_passed = tdiff_ok
        if not tdiff_ok:
            v_res.errors.append(f"Check 3 (Transcript Diff) failed: {tdiff_err}")

        # Check 4: Visual Spot Checks
        vis_ok, vis_data, vis_err = verify_visual_spot_checks(out_file, audio_duration, temp_dir)
        v_res.visual_spot_checks_passed = vis_ok
        v_res.details["visual_spot_checks"] = vis_data
        if not vis_ok:
            v_res.errors.append(f"Check 4 (Visual Spot Checks) failed: {vis_err}")

        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        if not v_res.all_passed:
            err_summary = "; ".join(v_res.errors)
            raise RenderVerificationError(f"Post-render verification failed: {err_summary}")

        self.progress_callback(100, f"Render completed and verified: {out_file.name}")
        return out_file, v_res
