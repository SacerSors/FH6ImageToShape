## 2024-06-10 - PyTorch anti-pattern: `torch.tensor` wraps existing tensors
**Learning:** Wrapping an existing tensor with `torch.tensor()` (e.g., `torch.tensor(torch.abs(x))`) is a major anti-pattern in PyTorch. It detaches the tensor from the computation graph (preventing backpropagation) and introduces graph breaks in `@torch.compile` which severely hurts ROCm/GPU performance.
**Action:** Always use PyTorch operations directly without redundant `torch.tensor()` wrappers, e.g. `torch.mean(torch.abs(blended_tile - target_tile))` instead of `torch.mean(torch.tensor(torch.abs(blended_tile - target_tile)))`.

## 2024-06-10 - Removing Unused GPU Variables
**Learning:** Unused tensors like `max_qx` and `max_qy` in kernel computations (e.g., `sdf_rectangle`) waste GPU memory and compute cycles, especially during high-throughput optimization loops.
**Action:** Ensure that variables declared inside heavily called GPU kernels are actually used in the computation, and clean up any unused intermediate outputs.

## 2024-06-10 - Dynamo Recompile Limit (Changing Scalars)
**Learning:** Passing changing Python scalars (like loop progress counters or dynamically changing floats) into a `@torch.compile` function causes Dynamo to treat them as specialized constants, triggering a costly full graph recompilation for *every single distinct value*. This quickly hits the cache limit and crashes if `fullgraph=True`.
**Action:** Convert changing scalars to 0-dimensional PyTorch tensors (e.g. `torch.tensor(val, device=...)`) *before* passing them to the compiled function, so Dynamo traces them as dynamic inputs rather than specialized constants.