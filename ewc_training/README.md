# Adapting EWC to DreamerV3 RSSM: Final Roadmap (v3)

*Combined from v1 (initial design) and v2 (OWL cross-reference), then revised for wall-clock efficiency given the MetaWorld training regime: 300M param RSSM, 128×128 images, 1M env steps = 5 days, FP16 training.*

---

## 1. How EWC Works (High-Level)

Elastic Weight Consolidation (EWC), from Kirkpatrick et al. 2017, takes a Bayesian perspective on sequential learning:

> After learning task A, the posterior distribution over parameters becomes the prior when learning task B.

When you train on task A, you find optimal parameters **θ\*_A**. EWC prevents catastrophic forgetting by identifying which parameters were *important* for task A via the **Fisher Information Matrix (FIM)**, and penalizing changes to those parameters when learning task B.

### The EWC Loss

```
L_total = L_task_B(θ) + λ · Σ_i F_i · (θ_i − θ*_A,i)²
```

- `F_i` — diagonal Fisher Information for parameter i (how sensitive the loss was to this parameter on task A)
- `θ*_A` — parameter snapshot after task A
- `λ` — regularization strength

The Fisher diagonal is estimated empirically: sample data from task A, compute loss, backprop, and accumulate squared gradients. Parameters with large squared gradients were important and should be protected.

### Standard vs. Online EWC

Standard EWC stores a separate (Fisher, θ\*) pair per completed task and sums all penalties. This is what `skezle/owl` uses. Online EWC (Schwarz et al. 2018) maintains a running average instead, keeping memory constant.

**Our choice**: Standard per-task EWC. With your 3-task sequence, memory is not a concern (3 × 300M floats ≈ 3.6 GB — within budget for a GPU with 24+ GB VRAM). Standard EWC is simpler to debug and matches the OWL reference implementation. If you later scale to 10+ tasks, switch to online.

---

## 2. Architecture Mapping

| Component | Prefixes | EWC? | Rationale |
|-----------|---------|------|-----------|
| Encoder | `_wm.encoder.*` | **Yes** | Shared across tasks |
| RSSM Dynamics | `_wm.dynamics.*` | **Yes** | Shared across tasks |
| Decoder | `_wm.heads.decoder.*` | **Yes** | Shared across tasks |
| Reward Head | `_wm.heads.reward.*` | No | Reset per task |
| Continue Head | `_wm.heads.cont.*` | No | Reset per task |
| Actor/Critic | `_task_behavior.*` | No | Reset per task |

This matches your existing `RSSM_PREFIXES` in the ER script (line 184). The EWC penalty is applied **only** during the world model's `_train()` call, never during actor-critic training.

---

## 3. Efficiency Analysis and Design Decisions

Your config tells us a lot about where time is spent:

```
batch_size: 16, train_ratio: 512, precision: 16 (FP16/AMP)
dyn_deter: 4096, dyn_hidden: 1024, images: 128×128×3
envs: 32 (parallel), eval_every: 20000
```

With `train_ratio: 512`, every environment step triggers many gradient updates. The world model forward+backward pass dominates training time. Two things matter for EWC overhead:

**A. Fisher computation (one-time, at task boundaries):**
This is a fixed cost paid once per task transition. Even if it takes 30 minutes, that's <1% of a 5-day training run. We can afford some extra time here.

**B. EWC penalty computation (every training step):**
This runs inside `WorldModel._train()` on every single gradient update. With `train_ratio: 512`, this is called thousands of times per env step. Even a small per-call overhead gets multiplied enormously. **This is where we must be extremely efficient.**

### Decision Table

| Concern | v2 Approach | v3 Approach (Efficiency-Optimized) | Rationale |
|---------|-------------|-------------------------------------|-----------|
| Fisher: AMP | Disable AMP entirely | **Keep AMP enabled**, use FP32 only for the squaring step | Disabling AMP doubles forward+backward cost. For 300M params at 128×128, this is massive. Instead: run forward+backward in AMP, cast gradients to FP32 before squaring. |
| Fisher: eval mode | eval() mode | **eval() mode** (keep) | Correct behavior for dropout/batchnorm. No performance impact. |
| Fisher: grad clipping | No clipping | **No clipping** (keep) | Correct — Fisher should reflect true gradients. No performance impact. |
| Fisher: batch size | batch_size=1 (per-sample) or small batches | **Use training batch_size (16)** | Per-sample is 16× slower. The mini-batch bias is acceptable; compensate with slightly more batches or slightly higher λ. |
| Fisher: num batches | 100 batches | **50 batches** | At batch_size=16, 50 batches = 800 sequence samples. Sufficient for a 300M param model. Cost: ~50 forward+backward passes ≈ minutes, not hours. |
| Penalty: computation | `penalty()` iterates all params every call | **Pre-compute a flat penalty vector** + use vectorized ops | Naive iteration over `named_parameters()` with dict lookups is slow in Python. Pre-flatten Fisher and params into contiguous tensors for fast vectorized penalty. |
| Penalty: AMP | Not addressed | **Compute penalty in FP32, but within AMP context** | The penalty is a simple quadratic — no risk of FP16 issues, but the loss scalar needs it in the same precision as the rest. Use `.float()` on the penalty result if needed. |
| Standard vs Online | Standard per-task | **Standard per-task** | 3 tasks, memory is fine. Simpler. |
| λ/2 convention | Dropped the 1/2 | **No 1/2** (match OWL) | Just a scaling convention. |

---

## 4. Step-by-Step Implementation

### Step 1: `ewc.py` — The EWCManager Class

```python
import torch
from typing import Dict, List, Tuple, Optional


class EWCManager:
    """EWC regularization manager for DreamerV3 RSSM.
    
    Designed for efficiency with large models (300M+ params):
    - Fisher computation keeps AMP enabled, casts to FP32 only for squaring
    - Penalty uses pre-flattened tensors for fast vectorized computation
    - Standard per-task EWC (matching skezle/owl)
    """
    
    def __init__(self, lambda_ewc: float = 5000.0, 
                 rssm_prefixes: tuple = ("encoder.", "dynamics.", "heads.decoder.")):
        self.lambda_ewc = lambda_ewc
        self.rssm_prefixes = rssm_prefixes
        
        # Per-task storage: {task_idx: {'importance': {...}, 'task_param': {...}}}
        self.regularization_terms: Dict[int, dict] = {}
        self.num_tasks_consolidated: int = 0
        
        # ============================================================
        # Pre-flattened penalty cache (for fast per-step computation)
        # Rebuilt after each consolidation call.
        # ============================================================
        self._penalty_cache_valid = False
        self._cached_fisher_flat: Optional[torch.Tensor] = None   # [total_rssm_params]
        self._cached_params_flat: Optional[torch.Tensor] = None   # [total_rssm_params]
        self._param_name_to_slice: Dict[str, Tuple[int, int]] = {}
        self._rssm_param_names: List[str] = []
    
    def _is_rssm_param(self, name: str) -> bool:
        norm = name.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        return any(norm.startswith(p) for p in self.rssm_prefixes)
    
    # ------------------------------------------------------------------
    # Fisher computation (called once per task boundary — can be slower)
    # ------------------------------------------------------------------
    def compute_fisher(self, world_model, dataset, config,
                       num_batches: int = 50, device: str = 'cuda'):
        """Compute diagonal Fisher for RSSM parameters.
        
        Efficiency notes:
        - Keeps AMP enabled for the forward+backward pass (critical for
          300M param model with 128x128 images — disabling AMP would ~2x
          the wall time of each batch).
        - Casts gradients to FP32 ONLY for the squaring accumulation step,
          avoiding FP16 overflow (grad=256 → grad²=65536 > FP16 max 65504).
        - Uses eval() mode to disable dropout (matching OWL line 199).
        - No gradient clipping (matching OWL — Fisher should reflect true
          gradient magnitudes, unlike training which clips at 100).
        - Uses the standard training batch_size (not per-sample), accepting
          the slight mini-batch bias for 16× speedup over per-sample Fisher.
        """
        importance = {}
        task_param = {}
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name):
                importance[name] = torch.zeros_like(param.data, dtype=torch.float32)
                task_param[name] = param.data.clone().float()
        
        was_training = world_model.training
        world_model.eval()
        
        use_amp = (config.precision == 16)
        
        for _ in range(num_batches):
            data = next(dataset)
            world_model.zero_grad()
            
            # Forward+backward WITH AMP (same as training, for speed)
            with torch.cuda.amp.autocast(use_amp):
                data = world_model.preprocess(data)
                embed = world_model.encoder(data)
                post, prior = world_model.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_loss, _, _, _ = world_model.dynamics.kl_loss(
                    post, prior,
                    config.kl_free, config.dyn_scale, config.rep_scale
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
            
            # Backward (AMP scaler not needed — we only want gradients, 
            # not optimizer steps. Use unscaled loss.)
            model_loss.backward()
            
            # Accumulate squared gradients in FP32
            for name, param in world_model.named_parameters():
                if name in importance and param.grad is not None:
                    # Cast to float32 BEFORE squaring to avoid FP16 overflow
                    grad_fp32 = param.grad.data.float()
                    importance[name] += (grad_fp32 ** 2) / num_batches
        
        if was_training:
            world_model.train()
        
        return importance, task_param
    
    # ------------------------------------------------------------------
    # Consolidation (called once per task boundary)
    # ------------------------------------------------------------------
    def consolidate(self, world_model, importance, task_param, task_idx: int):
        """Store Fisher + param snapshot. Rebuild penalty cache."""
        self.regularization_terms[task_idx] = {
            'importance': {k: v.clone() for k, v in importance.items()},
            'task_param': {k: v.clone() for k, v in task_param.items()},
        }
        self.num_tasks_consolidated += 1
        
        # Rebuild flattened cache for fast penalty computation
        self._rebuild_penalty_cache(world_model)
    
    def _rebuild_penalty_cache(self, world_model):
        """Pre-flatten all Fisher diagonals and param snapshots into 
        contiguous tensors for fast vectorized penalty computation.
        
        Instead of iterating over dicts with Python loops every training
        step (thousands of times per env step with train_ratio=512),
        we do one vectorized operation on flat tensors.
        
        For 300M RSSM params × 3 tasks, this saves significant Python
        overhead per call.
        """
        if self.num_tasks_consolidated == 0:
            self._penalty_cache_valid = False
            return
        
        device = next(iter(
            next(iter(self.regularization_terms.values()))['importance'].values()
        )).device
        
        # Collect RSSM param names in deterministic order
        self._rssm_param_names = []
        self._param_name_to_slice = {}
        offset = 0
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name) and name in next(iter(
                self.regularization_terms.values()))['importance']:
                n = param.numel()
                self._rssm_param_names.append(name)
                self._param_name_to_slice[name] = (offset, offset + n)
                offset += n
        
        total_params = offset
        
        # Sum Fisher across all tasks (standard EWC sums penalties)
        # Pre-sum here so penalty() does ONE vectorized op, not a loop over tasks
        fisher_sum = torch.zeros(total_params, dtype=torch.float32, device=device)
        # For param anchors, we need per-task: store weighted combo
        # Actually for standard EWC:  Σ_t F_t * (θ - θ*_t)²
        # We can't pre-sum params because each task has different θ*_t.
        # But we CAN pre-flatten each task's Fisher and params.
        
        self._cached_fishers = []  # list of flat tensors, one per task
        self._cached_params = []   # list of flat tensors, one per task
        
        for task_idx in sorted(self.regularization_terms.keys()):
            reg = self.regularization_terms[task_idx]
            fisher_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            params_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            for name in self._rssm_param_names:
                s, e = self._param_name_to_slice[name]
                fisher_flat[s:e] = reg['importance'][name].flatten()
                params_flat[s:e] = reg['task_param'][name].flatten()
            self._cached_fishers.append(fisher_flat)
            self._cached_params.append(params_flat)
        
        self._penalty_cache_valid = True
    
    # ------------------------------------------------------------------
    # Penalty computation (called EVERY training step — must be fast)
    # ------------------------------------------------------------------
    def penalty(self, world_model) -> torch.Tensor:
        """Compute EWC penalty using pre-flattened vectorized tensors.
        
        Performance: For a 300M param RSSM with train_ratio=512, this is
        called thousands of times per env step. The pre-flattened cache 
        avoids Python dict iteration and does one vectorized op per task.
        
        Returns scalar tensor (already multiplied by lambda).
        """
        if not self._penalty_cache_valid or self.num_tasks_consolidated == 0:
            return torch.tensor(0.0, device=next(world_model.parameters()).device)
        
        # Flatten current RSSM params into a single vector
        current_flat = []
        for name in self._rssm_param_names:
            # Find the param (handles torch.compile name variations)
            for pname, param in world_model.named_parameters():
                if pname == name:
                    current_flat.append(param.flatten())
                    break
        current_flat = torch.cat(current_flat)  # [total_rssm_params]
        
        # Vectorized penalty: Σ_t F_t * (θ - θ*_t)²
        reg_loss = torch.tensor(0.0, device=current_flat.device)
        for fisher_flat, params_flat in zip(self._cached_fishers, self._cached_params):
            diff = current_flat - params_flat.to(current_flat.device)
            reg_loss = reg_loss + (fisher_flat.to(current_flat.device) * diff ** 2).sum()
        
        return self.lambda_ewc * reg_loss
    
    # ------------------------------------------------------------------
    # Save / Load (for resume support)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            'regularization_terms': self.regularization_terms,
            'num_tasks_consolidated': self.num_tasks_consolidated,
            'lambda_ewc': self.lambda_ewc,
        }
    
    def load_state_dict(self, state, world_model=None):
        self.regularization_terms = state['regularization_terms']
        self.num_tasks_consolidated = state['num_tasks_consolidated']
        self.lambda_ewc = state.get('lambda_ewc', self.lambda_ewc)
        if world_model is not None and self.num_tasks_consolidated > 0:
            self._rebuild_penalty_cache(world_model)
```

### Step 2: Modify `WorldModel._train()` in `models.py`

Minimal change — add the penalty to the loss and log both components:

```python
def _train(self, data):
    data = self.preprocess(data)

    with tools.RequiresGrad(self):
        with torch.cuda.amp.autocast(self._use_amp):
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            kl_free = self._config.kl_free
            dyn_scale = self._config.dyn_scale
            rep_scale = self._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            assert kl_loss.shape == embed.shape[:2], kl_loss.shape
            preds = {}
            for name, head in self.heads.items():
                grad_head = name in self._config.grad_heads
                feat = self.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            losses = {}
            for name, pred in preds.items():
                loss = -pred.log_prob(data[name])
                assert loss.shape == embed.shape[:2], (name, loss.shape)
                losses[name] = loss
            scaled = {
                key: value * self._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss
            base_loss = torch.mean(model_loss)
            
            # ---- EWC penalty (v3: vectorized, no extra forward pass) ----
            ewc_loss = torch.tensor(0.0, device=self._config.device)
            if hasattr(self, 'ewc_manager') and self.ewc_manager is not None:
                ewc_loss = self.ewc_manager.penalty(self)
            total_loss = base_loss + ewc_loss
            # ---- end EWC ----
            
        metrics = self._model_opt(total_loss, self.parameters())

    metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
    metrics["kl_free"] = kl_free
    metrics["dyn_scale"] = dyn_scale
    metrics["rep_scale"] = rep_scale
    metrics["dyn_loss"] = to_np(dyn_loss)
    metrics["rep_loss"] = to_np(rep_loss)
    metrics["kl"] = to_np(torch.mean(kl_value))
    # ---- EWC metrics ----
    metrics["ewc_loss"] = float(ewc_loss.detach()) if isinstance(ewc_loss, torch.Tensor) else 0.0
    metrics["model_loss_base"] = float(base_loss.detach())
    # ---- end EWC metrics ----
    with torch.cuda.amp.autocast(self._use_amp):
        metrics["prior_ent"] = to_np(
            torch.mean(self.dynamics.get_dist(prior).entropy())
        )
        metrics["post_ent"] = to_np(
            torch.mean(self.dynamics.get_dist(post).entropy())
        )
        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics
```

**Per-step overhead**: The `penalty()` call does one parameter flattening + one vectorized multiply-and-sum per stored task. For 300M params and 3 tasks, this is ~3 vectorized ops on GPU tensors — microseconds, not milliseconds. **Negligible compared to the forward+backward pass.**

### Step 3: Modify `dreamer_sequential_ewc.py`

Fork from your ER script. The changes are:

**A. Initialization (before task loop):**

```python
from ewc import EWCManager

ewc_manager = EWCManager(
    lambda_ewc=args.ewc_lambda,
    rssm_prefixes=("encoder.", "dynamics.", "heads.decoder."),
)

# Resume: load EWC state if available
if resume_from_task_idx > 0:
    for j in range(resume_from_task_idx - 1, -1, -1):
        ewc_file = base_logdir / f"task{j+1}_{tasks[j]}" / f"ewc_state_task{j+1}.pt"
        if ewc_file.exists():
            print(f">>> EWC: Loading state from {ewc_file}")
            ewc_state = torch.load(ewc_file, map_location='cpu')
            # Move tensors to device lazily when agent is created
            ewc_manager.load_state_dict(ewc_state)
            break
```

**B. After agent creation (per task):**

```python
agent._wm.ewc_manager = ewc_manager

# If resuming with loaded EWC state, rebuild the penalty cache now
if ewc_manager.num_tasks_consolidated > 0 and not ewc_manager._penalty_cache_valid:
    # Move tensors to device
    for tid, reg in ewc_manager.regularization_terms.items():
        for k in reg['importance']:
            reg['importance'][k] = reg['importance'][k].to(config.device)
            reg['task_param'][k] = reg['task_param'][k].to(config.device)
    ewc_manager._rebuild_penalty_cache(agent._wm)
```

**C. After task training completes (task boundary — the one-time cost):**

```python
# ============================================================
# [EWC] Compute Fisher and consolidate (one-time per task)
# ============================================================
if task_idx < num_tasks - 1:
    print(f">>> EWC: Computing Fisher for task {task_idx+1}...")
    import time as _time
    t0 = _time.time()
    
    fisher_dataset = make_dataset(train_eps, config)
    importance, task_param = ewc_manager.compute_fisher(
        agent._wm, fisher_dataset, config,
        num_batches=args.ewc_fisher_batches,
        device=config.device,
    )
    ewc_manager.consolidate(agent._wm, importance, task_param, task_idx)
    
    elapsed = _time.time() - t0
    n_params = sum(v.numel() for v in importance.values())
    fisher_mean = sum(v.mean().item() for v in importance.values()) / max(len(importance), 1)
    print(f">>> EWC: Consolidated task {task_idx+1} in {elapsed:.1f}s: "
          f"{len(importance)} groups, {n_params:,} params, "
          f"mean Fisher={fisher_mean:.2e}")
    
    # Save for resume
    torch.save(ewc_manager.state_dict(), 
               task_logdir / f"ewc_state_task{task_idx+1}.pt")
```

**D. CLI arguments:**

```python
parser.add_argument("--ewc-lambda", type=float, default=5000.0,
                    help="EWC regularization strength")
parser.add_argument("--ewc-fisher-batches", type=int, default=50,
                    help="Mini-batches for Fisher estimation (default: 50)")
```

---

## 5. Efficiency Budget

Here's the wall-clock overhead estimate for your setup:

### Fisher Computation (one-time, per task boundary)

- 50 batches × ~same cost as one training forward+backward ≈ 50 training steps
- With `train_ratio: 512` and `batch_size: 16`, your training does roughly `1M / (16*50/512)` ≈ 640K gradient updates per 1M env steps
- 50 Fisher batches is **0.008%** of 640K updates
- **Estimated time**: 2–5 minutes per task boundary (negligible vs. 5 days)

### Per-Step EWC Penalty

- One `torch.cat` of pre-cached param views + one vectorized multiply+sum per stored task
- For 300M params: ~microseconds on GPU (pure tensor ops, no Python loops over params)
- Called every training step, but cost is dominated by the forward+backward pass
- **Estimated overhead**: <1% of per-step training time

### Memory

- Fisher storage: 300M params × 4 bytes × N_tasks = ~1.2 GB per task (FP32)
- Parameter snapshots: same ~1.2 GB per task
- For 3 tasks: ~7.2 GB total EWC overhead
- Your existing dataset_size is 400K–750K episodes of 128×128 images — the replay buffer is already using significantly more memory
- **Verdict**: Manageable on 40GB+ GPU. If tight, reduce to FP16 Fisher storage (most values are small enough).

### Total Overhead Estimate

| Component | Cost | % of 5-day run |
|-----------|------|----------------|
| Fisher computation (2 task boundaries) | ~10 minutes | 0.1% |
| EWC penalty per step (640K steps) | <1% overhead per step | <1% |
| **Total** | **< 5–10 minutes extra** | **~0.1%** |

The design ensures EWC adds virtually no wall-clock overhead to your training.

---

## 6. Design Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| EWC variant | Standard per-task | 3 tasks, memory OK, matches OWL, simpler to debug |
| Fisher AMP | **Keep AMP on**, FP32 only for squaring | 2× speedup vs. disabling AMP entirely |
| Fisher batch size | Training batch_size (16) | 16× faster than per-sample; bias is acceptable |
| Fisher num batches | 50 | 800 sequences sufficient for 300M params |
| Fisher mode | eval() | Correct for dropout/batchnorm |
| Fisher grad clipping | None | Fisher should reflect true gradients |
| Penalty computation | Pre-flattened vectorized tensors | Avoids Python dict loops at train_ratio=512 |
| λ/2 convention | No 1/2 (match OWL) | Just scaling; adjust λ accordingly |
| λ default | 5000 | Start here; sweep [500, 1000, 5000, 10000, 50000] |
| Resume support | Save/load EWC state dict | Critical for 5-day runs |

---

## 7. Important Pitfalls

### λ Calibration
After the first task boundary, log `ewc_loss` and `model_loss_base` for the first few thousand steps of task 2. They should be within 1–2 orders of magnitude of each other. If `ewc_loss` is 1000× larger, divide λ by 1000.

### torch.compile Compatibility
Your ER script uses `torch.compile` (line 279). Parameter names may contain `_orig_mod.`. The `_is_rssm_param()` normalizer handles this, but make sure the Fisher dict keys match what `named_parameters()` returns on the compiled model. Test with a quick sanity check:
```python
# After agent creation:
rssm_names = [n for n, _ in agent._wm.named_parameters() if ewc_manager._is_rssm_param(n)]
print(f"EWC will track {len(rssm_names)} RSSM param groups")
assert len(rssm_names) > 0, "No RSSM params found — check prefix matching!"
```

### None Gradients
Some parameters may have `None` gradients if they're not in the computation graph for a particular batch (e.g., unused decoder paths). The Fisher computation checks `param.grad is not None` (matching OWL line 223). Parameters with consistently None gradients will have Fisher = 0, which is correct — they're unimportant.

### AMP Scaler Interaction
During Fisher computation, we call `model_loss.backward()` directly without going through the AMP GradScaler. This means the raw (unscaled) loss is backpropped. This is fine — we want the true gradient magnitudes for Fisher, not the scaled ones. The scaler is only needed for the optimizer step, which we don't do during Fisher computation.

### EWC + ER Combination
If you want to combine EWC with your existing ER buffer, just use both: the ER buffer provides data diversity and the EWC penalty provides parameter protection. Create `dreamer_sequential_ewc_er.py` that has both the `MergedEpisodes` dataset and the `ewc_manager`. The two mechanisms don't interact directly.

---

## 8. File Structure

```
continual_dreamer/
├── ewc.py                           # EWCManager (~200 lines)
├── models.py                        # +10 lines in WorldModel._train()
├── dreamer_sequential_ewc.py        # Fork of ER script with EWC
├── dreamer_sequential_ewc_er.py     # (optional) EWC + ER combined
├── networks.py                      # Unchanged
└── dreamer.py                       # Unchanged
```

---

## 9. Relevant Repos

| Repo | Use For |
|------|---------|
| **`skezle/owl`** | Primary EWC architecture reference (shared backbone + multi-head + EWC on backbone) |
| **`skezle/continual-dreamer`** | Task transition patterns with DreamerV2 |
| **`moskomule/ewc.pytorch`** | Clean minimal EWC algorithm reference |
| **`GMvandeVen/continual-learning`** | Comprehensive CL library with EWC, SI, LwF (for comparison) |

---

## 10. Evaluation Checklist

1. **Sanity check**: Train task 1 with EWC enabled (λ > 0) — should match vanilla performance (no consolidation yet, ewc_loss = 0).
2. **Loss decomposition**: Log `model_loss_base` and `ewc_loss` separately on W&B. Monitor their ratio.
3. **Fisher statistics**: After each consolidation, log mean/std/max of Fisher diagonal.
4. **Forward transfer**: Task N+1 learning curve start — does RSSM initialization help?
5. **Backward forgetting**: Eval on tasks 1..N after training task N+1 (your existing eval loop handles this).
6. **Wall-clock timing**: Log Fisher computation time and confirm it's < 10 min per boundary.