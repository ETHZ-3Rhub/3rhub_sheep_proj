"""Sheep treadmill (top view) — bespoke third-party analysis script.

Self-contained on purpose: this is the "user-supplied pipeline" path, so it
only imports py3r.behaviour/matplotlib, never app.scripts._shared. Does the
standard preprocessing (strip instance qualifiers, filter by confidence,
interpolate, smooth) and renders one skeleton-overlay QC animation per group.

Keypoints come from user/models/sheep_top/output_mapping.csv. fl/fr/hl/hr are
the front-left/front-right/hind-left/hind-right leg-base markers visible from
above.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import py3r.behaviour as p3b

INTERPOLATION_LIMIT = 5
SMOOTH_WINDOW = 3

_REF_POINT = "withers"  # used for the speed-colour overlay

_SKELETON_LINES = [
    ("atlas", "ear_base_l"),
    ("ear_base_l", "ear_tip_l"),
    ("atlas", "ear_base_r"),
    ("ear_base_r", "ear_tip_r"),
    ("atlas", "neck_base"),
    ("neck_base", "withers"),
    ("withers", "tail_orig"),
    ("tail_orig", "tail_mid"),
    ("tail_mid", "tail_tip"),
    ("withers", "fl"),
    ("withers", "fr"),
    ("tail_orig", "hl"),
    ("tail_orig", "hr"),
]

_SKELETON_POINTS = sorted({p for pair in _SKELETON_LINES for p in pair})
_STATIC_RIG_POINTS = ["0", "1", "2", "X", "Y", "3", "Z", "1A", "1B", "1C", "2A", "2B", "2C"]


def run(
    *,
    tc: p3b.TrackingCollection,
    output_dir: Path,
    comparisons: list[tuple[str, str]] | None = None,
    group_tag: str = "group",
    likelihood_min: float = 0.5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    anim_dir = Path(output_dir) / "qc" / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    group_names = list(dict.fromkeys(tc[h].tags[group_tag] for h in tc))

    print("Preprocessing...")
    tc.each.strip_column_names()
    tc.each.filter_likelihood(threshold=likelihood_min)
    tc.each.interpolate(limit=INTERPOLATION_LIMIT)
    tc.each.smooth_all(window=SMOOTH_WINDOW, method="mean")

    print("Computing reference speed...")
    fc = tc.to_features()
    fc.each.speed(_REF_POINT).store()

    print("Rendering QC animations (one video per group)...")
    fc_grouped = fc.groupby(tags=[group_tag])
    style = {
        "points": {
            "default": {"color": (0, 255, 255), "radius": 3},
            _REF_POINT: {
                "radius": 5,
                "color": {
                    "from": f"speed_of_{_REF_POINT}_in_xy",
                    "cmap": "plasma",
                    "vmin": 0.0,
                    "vmax": 0.5,
                    "nan_color": (80, 80, 80),
                },
            },
            **{p: {"color": (150, 150, 150), "radius": 2} for p in _STATIC_RIG_POINTS},
        }
    }

    for group_name in group_names:
        group_fc = fc_grouped.get((group_name,))
        if not group_fc:
            continue

        feat = next(iter(group_fc.values()))
        video_path = feat.tracking.meta.get("video_path")
        out_path = anim_dir / f"{group_name}.mp4"
        print(f"  {group_name} ({'with video' if video_path else 'no video'})...")

        try:
            has_video = video_path is not None
            stream = feat.animation_stream(
                points=_SKELETON_POINTS + _STATIC_RIG_POINTS,
                lines=_SKELETON_LINES,
                features={"Speed (m/s)": f"speed_of_{_REF_POINT}_in_xy"},
                pixel_coords=has_video,
                undo_meta_scaling=has_video,
                style=style,
            )
            save_kwargs = {"out_path": str(out_path)}
            if has_video:
                save_kwargs["video_path"] = str(video_path)
            stream.save(**save_kwargs)
        except Exception as exc:
            print(f"  Warning: animation failed for {group_name}: {exc}")

    print("Pipeline complete.")
