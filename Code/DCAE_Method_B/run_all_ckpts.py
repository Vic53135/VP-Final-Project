import os
import subprocess
import csv

CHECKPOINT_ROOT = "/home/user/Desktop/Victor/VP_Final_Project/Code/DCAE_Method_B/checkpoints"
DATASET_DIR = "/home/user/Desktop/Victor/VP_Final_Project/Dataset/Dataset/test/kodak"
OUTPUT_CSV = "methodB_results.csv"

# === 1. 找出所有 full_dataset/*checkpoint_best.pth.tar ===
CHECKPOINTS = []

for root, dirs, files in os.walk(CHECKPOINT_ROOT):
    if os.path.basename(root) == "full_dataset":
        for f in files:
            if f.endswith("checkpoint_latest.pth.tar"):
                CHECKPOINTS.append(os.path.join(root, f))

CHECKPOINTS = sorted(CHECKPOINTS)

print(f"Found {len(CHECKPOINTS)} checkpoints:")
for c in CHECKPOINTS:
    print("  ", c)

results = []

# === 2. 逐一做 inference ===
for ckpt_path in CHECKPOINTS:
    ckpt_name = os.path.basename(ckpt_path).replace("checkpoint_latest.pth.tar", "")
    print(f"\n=== Running checkpoint: {ckpt_name} ===")

    cmd = [
        "python", "DCAE_Method_B/eval.py",
        "--real",
        "--cuda",
        "--checkpoint", ckpt_path,
        "--data", DATASET_DIR
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    avg_psnr = avg_msssim = avg_bpp = None
    enc_t = dec_t = total_t = None

    for line in proc.stdout:
        print(line, end="")

        if "average_PSNR" in line:
            avg_psnr = float(line.split(":")[1].replace("dB", "").strip())
        elif "average_MS-SSIM" in line:
            avg_msssim = float(line.split(":")[1].strip())
        elif "average_Bit-rate" in line:
            avg_bpp = float(line.split(":")[1].replace("bpp", "").strip())
        elif "average_encode_time" in line:
            enc_t = float(line.split(":")[1].replace("ms", "").strip())
        elif "average_decode_time" in line:
            dec_t = float(line.split(":")[1].replace("ms", "").strip())
        elif "average_time" in line:
            total_t = float(line.split(":")[1].replace("ms", "").strip())

    # lambda 可以直接從資料夾名取
    lambda_value = os.path.basename(os.path.dirname(ckpt_path))

    results.append([
        lambda_value,
        avg_psnr, avg_msssim, avg_bpp,
        enc_t, dec_t, total_t
    ])

# === 3. 寫入 CSV ===
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "lambda",
        "PSNR(dB)",
        "MS-SSIM",
        "BPP",
        "EncodeTime(ms)",
        "DecodeTime(ms)",
        "TotalTime(ms)"
    ])
    writer.writerows(results)

print(f"\n✅ All results saved to {OUTPUT_CSV}")
