"""Configuration dataclasses + YAML loader for the probshield plot pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Default run-directory schema for TensorBoard sources: the leaf run dir is
# named ``{variant}_{seed}`` (env comes from the parent directory).
_TB_RUN_SCHEMA = r"^(?P<variant>.+)_(?P<seed>\d+)$"


@dataclass(frozen=True)
class VariantStyle:
    name: str
    label: str
    colour: Optional[str] = None
    order: int = 0
    linestyle: str = "-"


@dataclass(frozen=True)
class RunSchema:
    """Regex extracting (env, variant, seed) named groups from a W&B run name."""
    pattern: str = r"^(?P<env>.+?)_(?P<variant>[a-z]+_v\d+)_seed(?P<seed>\d+)$"

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern)


@dataclass(frozen=True)
class Config:
    run_schema: RunSchema
    variants: list[VariantStyle]
    cache_dir: Path
    output_dir: Path
    source_kind: str = "wandb"  # "wandb" or "tensorboard"
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    tensorboard_logdir: Optional[Path] = None
    metrics: Optional[list[str]] = None
    envs: Optional[list[str]] = None
    palette: str = "colorblind"
    force_download: bool = False
    seed_for_logged_quantiles: int = 1
    max_step: Optional[int] = None

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
            linestyle=v.get("linestyle", "-"),
        )
        for i, v in enumerate(variants_raw)
    ]
    variants = _fill_palette(variants, raw.get("palette", "colorblind"))

    source_kind, wandb_entity, wandb_project, tb_logdir = _resolve_source(path, raw, _resolve)

    schema_raw = raw.get("run_schema") or {}
    if "pattern" in schema_raw:
        schema = RunSchema(pattern=schema_raw["pattern"])
    elif source_kind == "tensorboard":
        schema = RunSchema(pattern=_TB_RUN_SCHEMA)
    else:
        schema = RunSchema()

    output_dir = _resolve(raw["output_dir"])
    cache_raw = raw.get("cache_dir")
    cache_dir = _resolve(cache_raw) if cache_raw else output_dir / "_cache"

    return Config(
        run_schema=schema,
        variants=variants,
        cache_dir=cache_dir,
        output_dir=output_dir,
        source_kind=source_kind,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        tensorboard_logdir=tb_logdir,
        metrics=raw.get("metrics"),
        envs=raw.get("envs"),
        palette=raw.get("palette", "colorblind"),
        force_download=raw.get("force_download", False),
        seed_for_logged_quantiles=raw.get("seed_for_logged_quantiles", 1),
        max_step=raw.get("max_step"),
    )


def _resolve_source(
    path: Path, raw: dict, resolve
) -> tuple[str, Optional[str], Optional[str], Optional[Path]]:
    """Return ``(source_kind, wandb_entity, wandb_project, tensorboard_logdir)``.

    Accepts an explicit ``source:`` block, or falls back to legacy top-level
    ``wandb_entity`` / ``wandb_project`` keys (treated as a W&B source).
    """
    src = raw.get("source")
    if src:
        kind = src.get("kind", "wandb")
        if kind == "wandb":
            project = src.get("project")
            if not project:
                raise ValueError(f"{path}: source.kind=wandb requires source.project.")
            return "wandb", src.get("entity"), project, None
        if kind == "tensorboard":
            logdir = src.get("logdir")
            if not logdir:
                raise ValueError(f"{path}: source.kind=tensorboard requires source.logdir.")
            return "tensorboard", None, None, resolve(logdir)
        raise ValueError(
            f"{path}: unknown source.kind={kind!r}; expected 'wandb' or 'tensorboard'."
        )
    project = raw.get("wandb_project")
    if not project:
        raise ValueError(f"{path}: provide a 'source:' block, or top-level 'wandb_project'.")
    return "wandb", raw.get("wandb_entity"), project, None


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
            linestyle=variants[i].linestyle,
        )
    return out


def _to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(round(c * 255)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"
