"""Built-in target-model adapters."""

from .base import ModelAdapter, adapter_names, create_adapter, register_adapter
from . import anomalyclip as _anomalyclip  # noqa: F401
from . import aaclip as _aaclip  # noqa: F401
from . import adaclip as _adaclip  # noqa: F401
from . import faprompt as _faprompt  # noqa: F401
from . import crane as _crane  # noqa: F401
from . import aprilgan as _aprilgan  # noqa: F401
from . import tipsomaly as _tipsomaly  # noqa: F401
from . import fbclip as _fbclip  # noqa: F401
from . import vcpclip as _vcpclip  # noqa: F401
from . import filo as _filo  # noqa: F401
from . import bayespfl as _bayespfl  # noqa: F401
from . import afclip as _afclip  # noqa: F401
from . import cops as _cops  # noqa: F401
from . import mrad as _mrad  # noqa: F401
from . import winclip as _winclip  # noqa: F401

__all__ = ["ModelAdapter", "adapter_names", "create_adapter", "register_adapter"]
