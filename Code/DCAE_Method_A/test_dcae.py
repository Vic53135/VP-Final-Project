import os
import sys
import torch

# ===== ensure project root in path =====
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
print("PROJECT_ROOT =", PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from DCAE.models.dcae import DCAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = DCAE().to(device)
    model.eval()
    model.update()
    x = torch.rand(1, 3, 256, 256).to(device)

    print("=== Forward test ===")
    with torch.no_grad():
        out = model(x)
    print("x_hat shape:", out["x_hat"].shape)

    print("=== Compress test ===")
    with torch.no_grad():
        strings = model.compress(x)
    print("Compress OK")

    print("=== Decompress test ===")
    with torch.no_grad():
        x_hat = model.decompress(
            strings["strings"], strings["shape"]
        )["x_hat"]
    print("x_hat shape:", x_hat.shape)

    print("✅ Sanity check PASSED")

if __name__ == "__main__":
    main()
