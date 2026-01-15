# profile_thop.py
import os
import torch
from thop import profile, clever_format

# 1) 匯入你的模型（自己改成正確路徑）
# 例如：from models.dcae_rwkv import DCAE
from models import DCAE


def try_cuda():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def patch_rwkv_to_identity(model: torch.nn.Module):
    """
    若 thop 跑你會因為 RwkvBlock_BiV4 / biwkv4 op 崩掉，
    可以把 RWKV block 暫時 patch 成 identity，先得到可比較的 FLOPs。
    """
    import torch.nn as nn

    for name, m in model.named_modules():
        # 你這裡用的是 RwkvBlock_BiV4
        if m.__class__.__name__ == "RwkvBlock_BiV4":
            # 直接替換該 module 的 forward（不改參數結構）
            def _id_forward(x, _m=m):
                return x
            m.forward = _id_forward


@torch.no_grad()
def main():
    device = try_cuda()
    model = DCAE(N=192, M=320, num_slices=5, max_support_slices=5).eval().to(device)

    # ====== 你要測的輸入大小（報告務必寫清楚）======
    B = 1
    H = 256
    W = 256
    x = torch.randn(B, 3, H, W, device=device)

    # ====== (A) 直接算：含 RWKV（可能漏算 / 或 thop 會跳過未知 op）======
    try:
        macs, params = profile(model, inputs=(x,), verbose=False)
        macs_s, params_s = clever_format([macs, params], "%.3f")
        print(f"[THOP] (raw) MACs: {macs_s}  Params: {params_s}")
        print(f"[THOP] (raw) FLOPs(≈2*MACs): {clever_format([2*macs], '%.3f')[0]}")
    except Exception as e:
        print("[THOP] (raw) failed due to:", repr(e))

    # ====== (B) Patch RWKV -> Identity 再算一次：得到「不含 RWKV」的穩定基線 ======
    model2 = DCAE(N=192, M=320, num_slices=5, max_support_slices=5).eval().to(device)
    patch_rwkv_to_identity(model2)

    macs2, params2 = profile(model2, inputs=(x,), verbose=False)
    macs2_s, params2_s = clever_format([macs2, params2], "%.3f")
    print(f"[THOP] (RWKV->Id) MACs: {macs2_s}  Params: {params2_s}")
    print(f"[THOP] (RWKV->Id) FLOPs(≈2*MACs): {clever_format([2*macs2], '%.3f')[0]}")

    # ====== 你報告可以這樣寫 ======
    # - 我們用 THOP 計算 forward 推論 MACs/FLOPs，輸入為 Bx3xHxW
    # - RANS 熵編碼與自訂 CUDA kernel（biwkv4）不屬於標準 Conv/Linear FLOPs 統計範圍
    # - 因此同時回報：完整模型（thop 可能略過未知 op）以及 RWKV->Identity 的基線，
    #   RWKV FLOPs 另以理論估算/實測時間補充。
    diff_macs = macs - macs2 if "macs" in locals() else None
    if diff_macs is not None:
        print(f"[THOP] (approx RWKV accounted by THOP) extra MACs: {clever_format([diff_macs], '%.3f')[0]}")
        print(f"[THOP] (approx RWKV accounted by THOP) extra FLOPs: {clever_format([2*diff_macs], '%.3f')[0]}")


if __name__ == "__main__":
    main()
