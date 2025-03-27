import os
import sys

from typing import TYPE_CHECKING

# Disable autoloading before running 'import torch' to avoid circular dependencies
ORG_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

import torch_cuone  # noqa: E402
import torch_cuone._C  # noqa: E402
import torch_cuone.cuone  # noqa: E402

acc = torch._C._get_accelerator()
if acc.type != "cpu":
    raise RuntimeError(
        f"Two accelerators cannot be used at the same time "
        f"in PyTorch: cuone and {acc.type}. You can install "
        f"the cpu version of PyTorch to use your cuone device, "
        f"or use the {acc.type} device with "
        f"'export TORCH_DEVICE_BACKEND_AUTOLOAD=0'."
    )

__all__ = []

torch.utils.rename_privateuse1_backend("cuone")
torch._register_device_module("cuone", torch_cuone.cuone)
unsupported_dtype = [
    torch.quint8,
    torch.quint4x2,
    torch.quint2x4,
    torch.qint32,
    torch.qint8,
]
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
    unsupported_dtype=unsupported_dtype,
)


# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
def _autoload():
    # We should restore this switch as sub processes need to inherit its value
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ORG_AUTOLOAD
