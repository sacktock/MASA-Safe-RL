"""Configuration dataclasses + YAML loader for the probshield plot pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VariantStyle:
    name: str
    label: str
    colour: Optional[str] = None
    order: int = 0


@dataclass(frozen=True)
class RunSchema:
    """Regex extracting (env, variant, seed) named groups from a W&B run name."""
    pattern: str = r"^(?P<env>.+?)_(?P<variant>[a-z]+_v\d+)_seed(?P<seed>\d+)$"

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern)


@dataclass(frozen=True)
class Config:
    wandb_entity: str
    wandb_project: str
    run_schema: RunSchema
    variants: list[VariantStyle]
    cache_dir: Path
    output_dir: Path
    metrics: Optional[list[str]] = None
    envs: Optional[list[str]] = None
    palette: str = "colorblind"
    force_download: bool = False
    seed_for_logged_quantiles: int = 1

    def variant_by_name(self) -> dict[str, VariantStyle]:
        return {v.name: v for v in self.variants}

    def ordered_variant_names(self) -> list[str]:
        return [v.name for v in sorted(self.variants, key=lambda v: v.order)]


def load_config(path: str | Path) -> Config:
    """Parse a YAML file into a Config object, resolving colour palette gaps."""
    from ruamel.yaml import YAML

    path = Path(path)
    yaml = YAML(typ="safe")
    raw = yaml.load(path.read_text(encoding="utf-8"))

    base_dir = path.parent

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base_dir / pp).resolve()

    variants_raw = raw.get("variants", [])
    if not variants_raw:
        raise ValueError(f"{path}: at least one variant must be declared.")

    variants = [
        VariantStyle(
            name=v["name"],
            label=v.get("label", v["name"]),
            colour=v.get("colour"),
            order=v.get("order", i),
        )
        for i, v in enumerate(variants_raw)
    ]
    variants = _fill_palette(variants, raw.get("palette", "colorblind"))

    schema_raw = raw.get("run_schema") or {}
    schema = RunSchema(pattern=schema_raw.get("pattern", RunSchema.pattern))

    return Config(
        wandb_entity=raw["wandb_entity"],
        wandb_project=raw["wandb_project"],
        run_schema=schema,
        variants=variants,
        cache_dir=_resolve(raw["cache_dir"]),
        output_dir=_resolve(raw["output_dir"]),
        metrics=raw.get("metrics"),
        envs=raw.get("envs"),
        palette=raw.get("palette", "colorblind"),
        force_download=raw.get("force_download", False),
        seed_for_logged_quantiles=raw.get("seed_for_logged_quantiles", 1),
    )


def _fill_palette(variants: list[VariantStyle], palette_name: str) -> list[VariantStyle]:
    """Assign colours from a seaborn palette to any variant where colour is None."""
    missing_idx = [i for i, v in enumerate(variants) if v.colour is None]
    if not missing_idx:
        return variants

    import seaborn as sns
    palette = sns.color_palette(palette_name, n_colors=len(variants))
    palette_hex = [_to_hex(c) for c in palette]
    out = list(variants)
    for i in missing_idx:
        out[i] = VariantStyle(
            name=variants[i].name,
            label=variants[i].label,
            colour=palette_hex[i],
            order=variants[i].order,
        )
    return out


def _to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(round(c * 255)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"
