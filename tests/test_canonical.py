from __future__ import annotations

import math

import pytest

from ascend_maze.core.canonical import (
    FrozenMap,
    canonical_bytes,
    canonical_digest,
    freeze_canonical,
    freeze_literal,
)
from ascend_maze.core.errors import CanonicalizationError, LiteralSizeError


def test_mapping_and_set_order_do_not_change_canonical_bytes() -> None:
    left = {"items": {3, 1, 2}, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "items": {2, 3, 1}}
    assert canonical_bytes(left) == canonical_bytes(right)
    assert canonical_digest(left) == canonical_digest(right)


def test_list_and_tuple_normalize_to_the_same_immutable_value() -> None:
    assert freeze_canonical([1, [2, 3]]) == (1, (2, 3))
    assert freeze_canonical((1, (2, 3))) == (1, (2, 3))


def test_type_tags_keep_logically_different_scalars_distinct() -> None:
    assert canonical_bytes(True) != canonical_bytes(1)
    assert canonical_bytes("1") != canonical_bytes(1)
    assert canonical_bytes(b"1") != canonical_bytes("1")
    assert canonical_bytes(-0.0) != canonical_bytes(0.0)


def test_unicode_is_normalized() -> None:
    assert canonical_bytes("e\u0301") == canonical_bytes("\u00e9")


def test_frozen_mapping_has_no_mutable_backing_container() -> None:
    source = {"outer": {"numbers": [1, 2]}}
    frozen = freeze_canonical(source)
    assert isinstance(frozen, FrozenMap)
    source["outer"]["numbers"].append(3)
    assert frozen["outer"]["numbers"] == (1, 2)
    with pytest.raises(TypeError):
        frozen["new"] = 1  # type: ignore[index]
    assert not hasattr(frozen, "_lookup")


def test_freezing_does_not_trust_a_prebuilt_frozen_map() -> None:
    nested = [1, 2]
    unsafe = FrozenMap((("nested", nested),))
    frozen = freeze_canonical(unsafe)
    nested.append(3)
    assert frozen["nested"] == (1, 2)


def test_custom_non_finite_and_recursive_values_are_rejected() -> None:
    class Custom:
        pass

    with pytest.raises(CanonicalizationError, match="workflow.input"):
        freeze_canonical(Custom())
    with pytest.raises(CanonicalizationError, match="non-finite"):
        freeze_canonical(math.nan)
    recursive: list[object] = []
    recursive.append(recursive)
    with pytest.raises(CanonicalizationError, match="recursive"):
        freeze_canonical(recursive)


def test_literal_byte_limit_is_enforced_after_canonicalization() -> None:
    with pytest.raises(LiteralSizeError, match="max_literal_value_bytes"):
        freeze_literal("x" * 100, max_bytes=16)
