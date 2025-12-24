"""Provide lightweight dependency fallbacks for offline environments.

This module is automatically imported by Python when present on sys.path.
It registers minimal placeholder modules for optional dependencies when they
are not installed. The placeholders are only intended to keep imports working
in lightweight demo mode and should not be used for full training.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def _ensure_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


def _register_numpy_stub() -> None:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.__FAKE__ = True

    class _RandomStub:
        def rand(self, *shape):
            raise RuntimeError("numpy is required for rand")

        def default_rng(self, seed=None):
            raise RuntimeError("numpy is required for default_rng")

    numpy_stub.random = _RandomStub()

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("numpy is required for this operation")

    numpy_stub.array = _unavailable
    numpy_stub.concatenate = _unavailable
    numpy_stub.full = _unavailable
    numpy_stub.max = _unavailable

    _ensure_module("numpy", numpy_stub)


def _register_pandas_stub() -> None:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.__FAKE__ = True
    _ensure_module("pandas", pandas_stub)


def _register_sklearn_stub() -> None:
    sklearn_stub = types.ModuleType("sklearn")
    sklearn_stub.__FAKE__ = True
    model_selection_stub = types.ModuleType("sklearn.model_selection")
    model_selection_stub.__FAKE__ = True

    def train_test_split(*_args, **_kwargs):
        raise RuntimeError("scikit-learn is required for train_test_split")

    model_selection_stub.train_test_split = train_test_split
    sklearn_stub.model_selection = model_selection_stub

    _ensure_module("sklearn", sklearn_stub)
    _ensure_module("sklearn.model_selection", model_selection_stub)


def _register_tensorflow_stub() -> None:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__FAKE__ = True
    _ensure_module("tensorflow", tf_stub)


def _maybe_register(name: str, register_fn) -> None:
    if importlib.util.find_spec(name) is None:
        register_fn()


def _install_stubs() -> None:
    _maybe_register("numpy", _register_numpy_stub)
    _maybe_register("pandas", _register_pandas_stub)
    _maybe_register("sklearn", _register_sklearn_stub)
    _maybe_register("tensorflow", _register_tensorflow_stub)


_install_stubs()
