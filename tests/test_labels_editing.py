from __future__ import annotations

import numpy as np


def test_binary_colormap_does_not_force_use_selection() -> None:
    from regiongrow._widget import _binary_segmentation_colormap

    cmap = _binary_segmentation_colormap("cyan")
    assert cmap.use_selection is False


def test_sync_colormap_keeps_background_label_for_erase() -> None:
    import napari
    from napari.layers.labels._labels_constants import Mode

    from regiongrow._widget import (
        _configure_binary_labels_layer,
        _sync_binary_labels_colormap,
    )

    data = np.zeros((12, 12, 12), dtype=np.int32)
    data[4:8, 4:8, 4:8] = 1
    lyr = napari.layers.Labels(data.copy(), name="Segmentation")
    _configure_binary_labels_layer(lyr)
    _sync_binary_labels_colormap(lyr, "cyan")
    lyr.swap_selected_and_background_labels()
    assert int(lyr.selected_label) == 0

    _sync_binary_labels_colormap(lyr, "lime")
    assert int(lyr.selected_label) == 0

    lyr.mode = Mode.PAINT
    lyr.paint(np.array([6.0, 6.0, 6.0]), int(lyr.selected_label))
    assert int(lyr.data[6, 6, 6]) == 0


def test_configure_binary_labels_layer_enables_volume_editing() -> None:
    import napari

    from regiongrow._widget import _configure_binary_labels_layer

    lyr = napari.layers.Labels(np.zeros((8, 8, 8), dtype=np.int32))
    _configure_binary_labels_layer(lyr)
    assert lyr.editable is True
    assert lyr.preserve_labels is False
    assert int(lyr.n_edit_dimensions) == 3
