"""PackNet manager (conservative / paper-faithful) for DreamerV3 RSSM.

Faithfully follows the original PackNet (Mallya & Lazebnik, CVPR 2018):
  - Only WEIGHT tensors (ndim >= 2) of Linear/Conv layers are pruned and masked.
  - Bias and normalization parameters (ndim == 1) are frozen after the first
    task's prune-retrain cycle and shared across all subsequent tasks.
  - This matches the paper: "we did not find it necessary to learn
    task-specific biases" and "we do not update the [batch normalization]
    parameters after the first round of pruning and re-training."

Parameter categories:
  PRUNABLE  (ndim >= 2): Linear/Conv weight matrices
    -> Full PackNet: prune by magnitude, retrain, freeze surviving, per-task masks
  SHARED    (ndim == 1): biases, LayerNorm weight/bias
    -> Frozen after task 1 prune+retrain; shared across all tasks unchanged
"""

import torch
from typing import Dict, Optional, Tuple


class PackNetManager:
    """Paper-faithful PackNet manager for DreamerV3 RSSM parameters."""

    def __init__(
        self,
        prune_ratio: float = 0.75,
        rssm_prefixes: tuple = ("encoder.", "dynamics.", "decoder."),
    ):
        self.prune_ratio = prune_ratio
        self.rssm_prefixes = rssm_prefixes

        # --- Prunable (weight) masks ---
        # frozen_mask[name] = float tensor, 1.0 = frozen, 0.0 = free
        self.frozen_mask: Dict[str, torch.Tensor] = {}
        # task_masks[task_idx][name] = float tensor, 1.0 = active, 0.0 = inactive
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}

        # --- Shared (bias/norm) state ---
        # After task 0 prune+retrain, all 1D params are frozen.
        # We store their names so we can zero their gradients.
        self._shared_param_names: list = []
        self._shared_params_frozen: bool = False

        self.num_tasks_packed: int = 0

        # Retrain state
        self._retrain_mode: bool = False
        self._retrain_mask: Dict[str, torch.Tensor] = {}
        self._retrain_task_mask: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Parameter classification
    # ------------------------------------------------------------------
    def _is_rssm_param(self, name: str) -> bool:
        """Check if a parameter name belongs to the RSSM."""
        norm = name.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        return any(norm.startswith(p) for p in self.rssm_prefixes)

    def _is_prunable(self, name: str, param: torch.Tensor) -> bool:
        """Prunable = RSSM weight matrix (ndim >= 2).

        These are Linear/Conv weight tensors that get the full PackNet
        treatment: prune, mask, freeze surviving, per-task eval masks.
        """
        return self._is_rssm_param(name) and param.ndim >= 2

    def _is_shared(self, name: str, param: torch.Tensor) -> bool:
        """Shared = RSSM bias or norm parameter (ndim == 1).

        These are frozen after the first task and shared across all tasks.
        Not pruned, not masked per-task.
        """
        return self._is_rssm_param(name) and param.ndim == 1

    def _norm_name(self, name: str) -> str:
        return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")

    # ------------------------------------------------------------------
    # Initialization: discover shared params
    # ------------------------------------------------------------------
    def register_rssm_params(self, rssm):
        """Call once after RSSM is created to discover shared (1D) params."""
        self._shared_param_names = []
        n_prunable = 0
        n_prunable_params = 0
        n_shared = 0
        n_shared_params = 0

        for name, param in rssm.named_parameters():
            if self._is_prunable(name, param):
                n_prunable += 1
                n_prunable_params += param.numel()
            elif self._is_shared(name, param):
                self._shared_param_names.append(name)
                n_shared += 1
                n_shared_params += param.numel()

        print(f">>> PackNet: {n_prunable} prunable param groups "
              f"({n_prunable_params:,} params, ndim>=2)")
        print(f">>> PackNet: {n_shared} shared param groups "
              f"({n_shared_params:,} params, ndim==1, frozen after task 1)")

    # ------------------------------------------------------------------
    # Gradient masking (called every training step)
    # ------------------------------------------------------------------
    def apply_gradient_mask(self, rssm):
        """Zero out gradients on frozen weights and shared params.

        Called AFTER backward + unscale, BEFORE clip + optimizer.step().

        Uses indexing assignment (grad[mask] = 0) instead of multiplication
        (grad *= mask) to avoid nan from AMP inf * 0 = nan (IEEE 754).
        """
        for name, param in rssm.named_parameters():
            if param.grad is None:
                continue

            # Prunable weights: mask based on frozen_mask or retrain_mask
            if self._is_prunable(name, param):
                if self._retrain_mode and name in self._retrain_mask:
                    # Zero grads where retrain_mask == 0 (frozen + pruned)
                    param.grad.data[self._retrain_mask[name] == 0] = 0.0
                elif name in self.frozen_mask:
                    # Zero grads where frozen_mask == 1 (frozen weights)
                    param.grad.data[self.frozen_mask[name].bool()] = 0.0

            # Shared (1D) params: frozen after first task
            elif self._shared_params_frozen and self._is_shared(name, param):
                param.grad.data.zero_()

    def apply_weight_mask(self, rssm):
        """Re-zero pruned weights after optimizer step.

        During retrain: momentum or weight decay may revive pruned weights.
        Only applies to prunable (weight) params.
        """
        if not self._retrain_mode:
            return
        for name, param in rssm.named_parameters():
            if name in self._retrain_task_mask:
                param.data.mul_(self._retrain_task_mask[name])

    # ------------------------------------------------------------------
    # Pruning (called once after training a task)
    # ------------------------------------------------------------------
    def prune(self, rssm, task_idx: int) -> Dict[str, torch.Tensor]:
        """Prune current task's free weight params by magnitude (per-layer).

        Only operates on prunable params (ndim >= 2). Shared params (ndim == 1)
        are untouched.

        Returns:
            task_mask: {param_name: float tensor} for prunable params only.
                       1.0 = active (frozen + surviving), 0.0 = pruned.
        """
        task_mask = {}

        for name, param in rssm.named_parameters():
            if not self._is_prunable(name, param):
                continue

            frozen = self.frozen_mask.get(name, torch.zeros_like(param.data))
            free_mask = (1.0 - frozen).bool()
            num_free = free_mask.sum().item()

            if num_free > 0:
                free_magnitudes = param.data.abs()[free_mask]
                k = int(num_free * self.prune_ratio)
                if k > 0 and k < num_free:
                    threshold = torch.kthvalue(free_magnitudes, k).values.item()
                    survive = frozen.bool() | (
                        free_mask & (param.data.abs() > threshold)
                    )
                elif k >= num_free:
                    survive = frozen.bool()
                else:
                    survive = torch.ones_like(param.data, dtype=torch.bool)
            else:
                survive = frozen.bool()

            mask = survive.float()
            task_mask[name] = mask
            # Zero out pruned weights immediately
            param.data.mul_(mask)

        # Print statistics
        n_total = sum(
            p.numel() for n, p in rssm.named_parameters()
            if self._is_prunable(n, p)
        )
        n_frozen = sum(v.sum().item() for v in self.frozen_mask.values())
        n_surviving = sum(v.sum().item() for v in task_mask.values())
        n_pruned = n_total - n_surviving
        n_task_surviving = n_surviving - n_frozen
        print(
            f">>> PackNet: Pruned task {task_idx+1} weights "
            f"(ratio={self.prune_ratio:.2f}): "
            f"{int(n_total):,} total weight params, "
            f"{int(n_frozen):,} frozen, "
            f"{int(n_task_surviving):,} task surviving, "
            f"{int(n_pruned):,} pruned (free for future)"
        )

        return task_mask

    # ------------------------------------------------------------------
    # Retrain mode (called after pruning, before freezing)
    # ------------------------------------------------------------------
    def start_retrain(self, task_mask: Dict[str, torch.Tensor]):
        """Enter retrain mode after pruning.

        During retrain:
          - Frozen weight params: no gradients
          - Surviving current-task weight params: get gradients
          - Pruned weight params: no gradients, stay zero
          - Shared (1D) params: frozen (no gradients) if shared_params_frozen
        """
        self._retrain_mode = True
        self._retrain_task_mask = {k: v.clone() for k, v in task_mask.items()}
        self._retrain_mask = {}

        for name, mask in task_mask.items():
            frozen = self.frozen_mask.get(name, torch.zeros_like(mask))
            # Trainable = survived pruning AND not frozen from previous tasks
            self._retrain_mask[name] = mask * (1.0 - frozen)

    def end_retrain(self):
        """Exit retrain mode."""
        self._retrain_mode = False
        self._retrain_mask = {}
        self._retrain_task_mask = {}

    # ------------------------------------------------------------------
    # Freeze (called after retrain, finalises the task)
    # ------------------------------------------------------------------
    def freeze_task(self, task_mask: Dict[str, torch.Tensor], task_idx: int):
        """Freeze surviving weight params and store the task mask for eval.

        Also freezes shared (1D) params after the first task.
        """
        # Store evaluation mask for prunable params
        self.task_masks[task_idx] = {k: v.clone() for k, v in task_mask.items()}

        # Update frozen mask for prunable params
        for name, mask in task_mask.items():
            if name in self.frozen_mask:
                self.frozen_mask[name] = torch.max(
                    self.frozen_mask[name], mask
                )
            else:
                self.frozen_mask[name] = mask.clone()

        # Freeze shared (1D) params after the first task
        if not self._shared_params_frozen:
            self._shared_params_frozen = True
            print(f">>> PackNet: Freezing {len(self._shared_param_names)} shared "
                  f"(bias/norm) param groups after task {task_idx+1}")

        self.num_tasks_packed += 1

        n_frozen_w = sum(v.sum().item() for v in self.frozen_mask.values())
        n_total_w = sum(v.numel() for v in self.frozen_mask.values())
        print(
            f">>> PackNet: Frozen task {task_idx+1}: "
            f"{int(n_frozen_w):,}/{int(n_total_w):,} weight params frozen "
            f"({100*n_frozen_w/max(n_total_w,1):.1f}%), "
            f"shared params frozen={self._shared_params_frozen}"
        )

    # ------------------------------------------------------------------
    # Evaluation mask application
    # ------------------------------------------------------------------
    def save_rssm_weights(self, rssm) -> Dict[str, torch.Tensor]:
        """Save a copy of prunable RSSM weight data for later restoration.

        Only saves prunable (ndim >= 2) params since shared params don't
        change between tasks.
        """
        saved = {}
        for name, param in rssm.named_parameters():
            if self._is_prunable(name, param):
                saved[name] = param.data.clone()
        return saved

    def restore_rssm_weights(self, rssm, saved: Dict[str, torch.Tensor]):
        """Restore prunable RSSM weight data from a saved copy."""
        for name, param in rssm.named_parameters():
            if name in saved:
                param.data.copy_(saved[name])

    def apply_eval_mask(self, rssm, task_idx: int):
        """Apply task-specific mask for evaluation.

        Only masks prunable (weight) params. Shared params are identical
        across tasks so no masking needed.

        Must call save_rssm_weights() before and restore_rssm_weights() after.
        """
        if task_idx not in self.task_masks:
            return
        mask = self.task_masks[task_idx]
        for name, param in rssm.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])

    # ------------------------------------------------------------------
    # Save / Load (for resume support)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "frozen_mask": {k: v.cpu() for k, v in self.frozen_mask.items()},
            "task_masks": {
                tid: {k: v.cpu() for k, v in masks.items()}
                for tid, masks in self.task_masks.items()
            },
            "num_tasks_packed": self.num_tasks_packed,
            "prune_ratio": self.prune_ratio,
            "shared_params_frozen": self._shared_params_frozen,
            "shared_param_names": self._shared_param_names,
        }

    def load_state_dict(self, state: dict):
        self.frozen_mask = state["frozen_mask"]
        self.task_masks = state["task_masks"]
        self.num_tasks_packed = state["num_tasks_packed"]
        self.prune_ratio = state.get("prune_ratio", self.prune_ratio)
        self._shared_params_frozen = state.get("shared_params_frozen", False)
        self._shared_param_names = state.get("shared_param_names", [])

    def to_device(self, device: str):
        """Move all masks to the specified device."""
        self.frozen_mask = {
            k: v.to(device) for k, v in self.frozen_mask.items()
        }
        self.task_masks = {
            tid: {k: v.to(device) for k, v in masks.items()}
            for tid, masks in self.task_masks.items()
        }
        if self._retrain_mode:
            self._retrain_mask = {
                k: v.to(device) for k, v in self._retrain_mask.items()
            }
            self._retrain_task_mask = {
                k: v.to(device) for k, v in self._retrain_task_mask.items()
            }
