from __future__ import annotations


def test_branch_points_color_index_and_cycle() -> None:
    from regiongrow._widget import (
        BRANCH_POINTS_COLOR_CYCLE,
        DEFAULT_BRANCH_POINTS_COLOR,
        _branch_points_color_for_layer_name,
        _branch_points_layer_color_index,
        _sanitize_branch_display_color,
    )

    assert _branch_points_layer_color_index("BranchPoints") == 0
    assert _branch_points_layer_color_index("BranchPoints_1") == 1
    assert _branch_points_layer_color_index("BranchPoints_12") == 12

    assert _branch_points_color_for_layer_name("BranchPoints") == DEFAULT_BRANCH_POINTS_COLOR
    assert (
        _branch_points_color_for_layer_name("BranchPoints_1")
        == BRANCH_POINTS_COLOR_CYCLE[1]
    )
    assert "magenta" not in BRANCH_POINTS_COLOR_CYCLE
    assert _sanitize_branch_display_color("magenta") == DEFAULT_BRANCH_POINTS_COLOR


def test_draft_branch_archive_color_rotates() -> None:
    from regiongrow._widget import (
        BRANCH_POINTS_COLOR_CYCLE,
        _draft_branch_archive_color,
    )

    assert _draft_branch_archive_color("Draft_Branch (1)") == BRANCH_POINTS_COLOR_CYCLE[1]
    assert _draft_branch_archive_color("Draft_Branch (2)") == BRANCH_POINTS_COLOR_CYCLE[2]
