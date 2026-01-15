import os
import subprocess
import csv

CHECKPOINT_DIR = "/home/user/Desktop/Victor/VP_Final_Project/pre-train"
DATASET_DIR = "/home/user/Desktop/Victor/VP_Final_Project/Dataset/Dataset/test"
OUTPUT_CSV = "baseline_results.csv"

CHECKPOINTS = sorted([
    f for f in os.listdir(CHECKPOINT_DIR)
    if f.endswith("checkpoint_best.pth.tar")
])

results = []

for ckpt in CHECKPOINTS:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
    print(f"\n=== Running checkpoint: {ckpt} ===")

    cmd = [
        "python", "eval.py",     # ← 你的原本測試檔名
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
            avg_psnr = float(line.split(":")[1].replace("dB",""))
        elif "average_MS-SSIM" in line:
            avg_msssim = float(line.split(":")[1])
        elif "average_Bit-rate" in line:
            avg_bpp = float(line.split(":")[1].replace("bpp",""))
        elif "average_encode_time" in line:
            enc_t = float(line.split(":")[1].replace("ms",""))
        elif "average_decode_time" in line:
            dec_t = float(line.split(":")[1].replace("ms",""))
        elif "average_time" in line:
            total_t = float(line.split(":")[1].replace("ms",""))

    results.append([
        ckpt.replace("checkpoint_best.pth.tar",""),
        avg_psnr, avg_msssim, avg_bpp,
        enc_t, dec_t, total_t
    ])

# === write CSV ===
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "lambda",
        "PSNR(dB)",
        "MS-SSIM(dB)",
        "BPP",
        "EncodeTime(ms)",
        "DecodeTime(ms)",
        "TotalTime(ms)"
    ])
    writer.writerows(results)

print(f"\nAll results saved to {OUTPUT_CSV}")
