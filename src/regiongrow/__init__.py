"""Region Grow: A napari plugin for 3D vessel segmentation."""

__version__ = "0.1.0"

import sys as _sys

_MACOS_WORKAROUND_INSTALLED = False


def apply_macos_multiprocessing_workaround() -> None:
    """Suppress macOS false-positive "leaked semaphore" warnings.

    Python's multiprocessing ``resource_tracker`` over-reports leaked
    semaphores from framework internals when napari/Qt spins up worker
    threads. This patches the process-global tracker, so it is applied
    **lazily** (only when the widget is constructed) rather than at import
    time, to avoid imposing a global side effect on the whole napari process
    for plugins that are merely discovered but never opened. Idempotent.
    """
    global _MACOS_WORKAROUND_INSTALLED
    if _MACOS_WORKAROUND_INSTALLED or _sys.platform != "darwin":
        return
    _MACOS_WORKAROUND_INSTALLED = True

    import multiprocessing as _mp
    import warnings as _warnings

    try:
        _mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass  # already set — ignore

    try:
        from multiprocessing import resource_tracker as _resource_tracker

        _rt_register = _resource_tracker.register
        _rt_unregister = _resource_tracker.unregister

        def _register(name, rtype):
            if rtype == "semaphore":
                return
            _rt_register(name, rtype)

        def _unregister(name, rtype):
            if rtype == "semaphore":
                return
            _rt_unregister(name, rtype)

        _resource_tracker.register = _register
        _resource_tracker.unregister = _unregister
    except Exception:
        pass

    _warnings.filterwarnings(
        "ignore",
        message=(
            r"resource_tracker: There appear to be \d+ leaked semaphore "
            r"objects to clean up at shutdown"
        ),
        category=UserWarning,
    )
