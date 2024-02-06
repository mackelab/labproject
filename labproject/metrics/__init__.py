"""
Best practices for developing metrics:

1. Please do everything in torch, and if that is not possible, cast the output to torch.Tensor.
2. The function should be well-documented, including type hints.
3. The function should be tested with a simple example.
4. Add an assert at the beginning for shape checking (N,D), see examples. 
5. Register the function by importing `labrpoject.metrics.utils.regiter_metric` and give your function a meaningful name.
"""

import importlib
import pkgutil

# Get the list of all submodules in the metrics package
package_name = "labproject.metrics"
package = importlib.import_module(package_name)
module_names = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]

# Import all the metrics modules that have a register_metric decorator
for module_name in module_names:
    module = importlib.import_module(f"{package_name}.{module_name}")
    if hasattr(module, "register_metric"):
        globals().update(module.__dict__)
