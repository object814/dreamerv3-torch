"""EWC (Elastic Weight Consolidation) manager for DreamerV3 RSSM.

Standard per-task EWC (Kirkpatrick et al. 2017):
  L_total = L_task(θ) + λ · Σ_t Σ_i F_{t,i} · (θ_i − θ*_{t,i})²

Design choices (see README.md for full rationale):
  - AMP kept enabled during Fisher computation; FP32 only for squaring step
  - Training batch_size reused (not per-sample) for speed
  - Pre-flattened vectorised tensors for fast per-step penalty computation
  - Standard per-task (not online) EWC — memory OK for ≤ 10 tasks
"""

import torch
from typing import Dict, List, Tuple, Optional


class EWCManager:
    """EWC regularization manager for DreamerV3 RSSM.

    Designed for efficiency with large models (300M+ params):
    - Fisher computation keeps AMP enabled, casts to FP32 only for squaring
    - Penalty uses pre-flattened tensors for fast vectorised computation
    - Standard per-task EWC (matching skezle/owl)
    """

    def __init__(self, lambda_ewc: float = 5000.0,
                 rssm_prefixes: tuple = ("encoder.", "dynamics.", "heads.decoder.")):
        self.lambda_ewc = lambda_ewc
        self.rssm_prefixes = rssm_prefixes

        # Per-task storage: {task_idx: {'importance': {...}, 'task_param': {...}}}
        self.regularization_terms: Dict[int, dict] = {}
        self.num_tasks_consolidated: int = 0

        # Pre-flattened penalty cache (rebuilt after each consolidation)
        self._penalty_cache_valid = False
        self._cached_fishers: List[torch.Tensor] = []
        self._cached_params: List[torch.Tensor] = []
        self._param_name_to_slice: Dict[str, Tuple[int, int]] = {}
        self._rssm_param_names: List[str] = []

    def _is_rssm_param(self, name: str) -> bool:
        """Check if a parameter name belongs to the RSSM (encoder/dynamics/decoder)."""
        norm = name.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        return any(norm.startswith(p) for p in self.rssm_prefixes)

    # ------------------------------------------------------------------
    # Fisher computation (called once per task boundary)
    # ------------------------------------------------------------------
    def compute_fisher(self, world_model, dataset, config,
                       num_batches: int = 50, device: str = "cuda"):
        """Compute diagonal Fisher for RSSM parameters.

        Keeps AMP enabled for forward+backward (critical for 300M params).
        Casts gradients to FP32 BEFORE squaring to avoid FP16 overflow.
        Uses eval() mode and no gradient clipping.
        """
        importance = {}
        task_param = {}
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name):
                importance[name] = torch.zeros_like(param.data, dtype=torch.float32)
                task_param[name] = param.data.clone().float()

        was_training = world_model.training
        world_model.eval()
        world_model.requires_grad_(True)

        use_amp = (config.precision == 16)

        for _ in range(num_batches):
            data = next(dataset)
            world_model.zero_grad()

            with torch.cuda.amp.autocast(use_amp):
                data = world_model.preprocess(data)
                embed = world_model.encoder(data)
                post, prior = world_model.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_loss, _, _, _ = world_model.dynamics.kl_loss(
                    post, prior,
                    config.kl_free, config.dyn_scale, config.rep_scale,
                )
                feat = world_model.dynamics.get_feat(post)
                losses = {}
                for head_name, head in world_model.heads.items():
                    grad_head = head_name in config.grad_heads
                    f = feat if grad_head else feat.detach()
                    pred = head(f)
                    if isinstance(pred, dict):
                        for k, v in pred.items():
                            losses[k] = -v.log_prob(data[k])
                    else:
                        losses[head_name] = -pred.log_prob(data[head_name])
                scaled = {
                    k: v * world_model._scales.get(k, 1.0)
                    for k, v in losses.items()
                }
                model_loss = torch.mean(sum(scaled.values()) + kl_loss)

            # Backward without AMP scaler — we only want raw gradients
            model_loss.backward()

            # Accumulate squared gradients in FP32
            for name, param in world_model.named_parameters():
                if name in importance and param.grad is not None:
                    grad_fp32 = param.grad.data.float()
                    importance[name] += (grad_fp32 ** 2) / num_batches

        world_model.requires_grad_(False)
        if was_training:
            world_model.train()

        return importance, task_param

    # ------------------------------------------------------------------
    # Consolidation (called once per task boundary)
    # ------------------------------------------------------------------
    def consolidate(self, world_model, importance, task_param, task_idx: int):
        """Store Fisher + param snapshot and rebuild penalty cache."""
        self.regularization_terms[task_idx] = {
            "importance": {k: v.clone() for k, v in importance.items()},
            "task_param": {k: v.clone() for k, v in task_param.items()},
        }
        self.num_tasks_consolidated += 1
        self._rebuild_penalty_cache(world_model)

    def _rebuild_penalty_cache(self, world_model):
        """Pre-flatten Fisher diagonals and param snapshots for fast penalty."""
        if self.num_tasks_consolidated == 0:
            self._penalty_cache_valid = False
            return

        device = next(iter(
            next(iter(self.regularization_terms.values()))["importance"].values()
        )).device

        # Collect RSSM param names in deterministic order
        self._rssm_param_names = []
        self._param_name_to_slice = {}
        offset = 0
        first_reg = next(iter(self.regularization_terms.values()))
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name) and name in first_reg["importance"]:
                n = param.numel()
                self._rssm_param_names.append(name)
                self._param_name_to_slice[name] = (offset, offset + n)
                offset += n

        total_params = offset

        self._cached_fishers = []
        self._cached_params = []

        for task_idx in sorted(self.regularization_terms.keys()):
            reg = self.regularization_terms[task_idx]
            fisher_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            params_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            for name in self._rssm_param_names:
                s, e = self._param_name_to_slice[name]
                fisher_flat[s:e] = reg["importance"][name].flatten()
                params_flat[s:e] = reg["task_param"][name].flatten()
            self._cached_fishers.append(fisher_flat)
            self._cached_params.append(params_flat)

        self._penalty_cache_valid = True

    # ------------------------------------------------------------------
    # Penalty computation (called every training step — must be fast)
    # ------------------------------------------------------------------
    def penalty(self, world_model) -> torch.Tensor:
        """Compute EWC penalty using pre-flattened vectorised tensors.

        Returns scalar tensor (already multiplied by lambda).
        """
        if not self._penalty_cache_valid or self.num_tasks_consolidated == 0:
            return torch.tensor(0.0, device=next(world_model.parameters()).device)

        # Build a normalised-name → param lookup to handle _orig_mod. mismatches
        param_dict = {}
        for n, p in world_model.named_parameters():
            norm = n.replace("._orig_mod.", ".").replace("_orig_mod.", "")
            param_dict[norm] = p

        # Per-parameter penalty avoids concatenating all RSSM params into one
        # giant flat tensor. Peak VRAM drops from ~O(N_tasks * total_params) to
        # ~O(max_single_param) with identical gradients.
        device = next(world_model.parameters()).device
        reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for name in self._rssm_param_names:
            norm = name.replace("._orig_mod.", ".").replace("_orig_mod.", "")
            s, e = self._param_name_to_slice[name]
            param_flat = param_dict[norm].flatten().float()
            for fisher_flat, params_flat in zip(self._cached_fishers, self._cached_params):
                diff = param_flat - params_flat[s:e]
                reg_loss = reg_loss + (fisher_flat[s:e] * diff ** 2).sum()

        return self.lambda_ewc * reg_loss

    # ------------------------------------------------------------------
    # Save / Load (for resume support)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "regularization_terms": self.regularization_terms,
            "num_tasks_consolidated": self.num_tasks_consolidated,
            "lambda_ewc": self.lambda_ewc,
        }

    def load_state_dict(self, state, world_model=None):
        self.regularization_terms = state["regularization_terms"]
        self.num_tasks_consolidated = state["num_tasks_consolidated"]
        self.lambda_ewc = state.get("lambda_ewc", self.lambda_ewc)
        self._penalty_cache_valid = False
        if world_model is not None and self.num_tasks_consolidated > 0:
            self._rebuild_penalty_cache(world_model)
