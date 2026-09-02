"""Built-in target-model adapters."""

from .base import ModelAdapter, adapter_names, create_adapter, register_adapter
from . import anomalyclip as _anomalyclip  # noqa: F401
from . import aaclip as _aaclip  # noqa: F401
from . import adaclip as _adaclip  # noqa: F401
from . import faprompt as _faprompt  # noqa: F401
from . import crane as _crane  # noqa: F401

__all__ = ["ModelAdapter", "adapter_names", "create_adapter", "register_adapter"]
