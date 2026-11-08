"""Tests for dynamic colour generation."""

from __future__ import annotations

from geolens.utils.colors import build_color_map, color_for, generate_color, text_color_for


def test_colour_is_deterministic() -> None:
    assert generate_color("Shale") == generate_color("Shale")


def test_unknown_labels_get_colours() -> None:
    for label in ("Shale", "Unobtainium", "Lithology 42", "cluster-xyz"):
        colour = generate_color(label)
        assert colour.startswith("#") and len(colour) == 7


def test_colour_map_has_no_collisions() -> None:
    labels = [f"Lithology {i}" for i in range(60)]
    mapping = build_color_map(labels)
    assert len(set(mapping.values())) == len(labels)


def test_colour_map_is_stable_across_calls() -> None:
    labels = ["Shale", "Sandstone", "Siltstone"]
    assert build_color_map(labels) == build_color_map(labels)


def test_color_for_falls_back_to_generation() -> None:
    mapping = build_color_map(["Shale"])
    assert color_for("Shale", mapping) == mapping["Shale"]
    assert color_for("Brand New Type", mapping).startswith("#")


def test_text_colour_contrast() -> None:
    assert text_color_for("#FFFFFF") == "#0F172A"
    assert text_color_for("#0F172A") == "#FFFFFF"
