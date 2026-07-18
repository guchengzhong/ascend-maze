"""Resource anchoring independent of placement and runtime backends."""

from ascend_maze.resources.anchors import (
    DeclaredOnlyAnchorProvider,
    ResourceAnchor,
)

__all__ = ["DeclaredOnlyAnchorProvider", "ResourceAnchor"]
