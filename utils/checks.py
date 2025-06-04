# -*- coding: utf-8 -*-
"""
通用張量／梯度檢查、穩定 loss 與權重初始化工具
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
#  1. 張量合法性檢查
# ---------------------------------------------------------------------
def assert_valid(tensor: torch.Tensor,
                 name: str = "tensor",
                 abort: bool = True) -> None:
    """
    檢查 tensor 是否含 NaN / Inf。若 abort=True 遇錯即丟例外。
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        msg = f"[check] {name} contains NaN / Inf"
        if abort:
            raise ValueError(msg)
        print(msg)

# ---------------------------------------------------------------------
#  2. 梯度檢查
# ---------------------------------------------------------------------
@torch.no_grad()
def check_gradients(model: nn.Module,
                    top_n: int = 3,
                    abort: bool = False) -> None:
    """
    列出梯度 norm，並可偵測 NaN / Inf。
    """
    bad = []
    stats = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if torch.isnan(g).any() or torch.isinf(g).any():
            bad.append(n)
        stats.append((n, g.norm().item()))
    if bad:
        msg = f"[check] Grad NaN/Inf in: {bad}"
        if abort:
            raise ValueError(msg)
        print(msg)
    # 印前後 top-n
    stats.sort(key=lambda x: x[1])
    print(f"=== grad norms (lowest {top_n}) ===")
    for n, v in stats[:top_n]:
        print(f"{n:60s}: {v:.3e}")
    print(f"=== grad norms (highest {top_n}) ===")
    for n, v in stats[-top_n:]:
        print(f"{n:60s}: {v:.3e}")

# ---------------------------------------------------------------------
#  3. 穩定 RankNet / margin loss
# ---------------------------------------------------------------------
def stable_rank_loss(pos: torch.Tensor,
                     neg: torch.Tensor,
                     margin: float = 0.2,
                     clamp: float = 10.0) -> torch.Tensor:
    """
    若差距過大導致 overflow，可使用 margin loss；
    同時將輸入分數 clamp 於 [-clamp, +clamp]。
    """
    pos_c = torch.clamp(pos, -clamp, clamp)
    neg_c = torch.clamp(neg, -clamp, clamp)
    return F.relu(margin - (pos_c.unsqueeze(1) - neg_c)).mean()

# ---------------------------------------------------------------------
#  4. Xavier 初始化
# ---------------------------------------------------------------------
def xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
