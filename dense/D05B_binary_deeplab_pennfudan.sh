%%bash
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# D05B — PennFudan DeepLab — part 2 of 2
# CORRECTED STANDALONE KAGGLE CELL
# ============================================================

SESSION_ID="D05B"
TARGET="binary_deeplab_pennfudan"

METHODS="bam,lora_conv,residual_adapter,conv_adapter"
GROUP0="bam,residual_adapter"
GROUP1="lora_conv,conv_adapter"
SEEDS="0,1,2"

REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"

WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"

DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"

export \
    SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS \
    REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

export PYTHONUNBUFFERED=1


# ============================================================
# Helper: package partial/full results
# ============================================================
pack_results() {
    [[ -d "$OUTPUT_ROOT" ]] || return 0

    python - <<'PYZIP'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
dst = Path(os.environ["RESULT_ZIP"])

dst.unlink(missing_ok=True)

with zipfile.ZipFile(
    dst,
    "w",
    zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as z:
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() in {".pth", ".pt", ".ckpt"}:
            continue

        z.write(p, p.relative_to(root.parent))

print("RESULT_ZIP =", dst)
PYZIP
}


trap '
rc=$?
trap - EXIT
[[ -f "$RESULT_ZIP" ]] || pack_results || true
exit $rc
' EXIT


# ============================================================
# 1. Clone repository
# ============================================================
echo
echo "============================================================"
echo "1. CLONE REPOSITORY"
echo "============================================================"

rm -rf "$REPO_DIR"

CLONED=0

for n in 1 2 3; do
    if git clone --depth 1 "$REPO_URL" "$REPO_DIR"; then
        CLONED=1
        break
    fi

    echo "Clone attempt $n failed."
    sleep "$((n * 5))"
done

if (( CLONED == 0 )); then
    echo "ERROR: unable to clone repository after 3 attempts."
    exit 1
fi

cd "$REPO_DIR"

if [[ -n "$REPO_COMMIT" ]]; then
    echo "Checking out frozen commit: $REPO_COMMIT"
    git fetch --depth 1 origin "$REPO_COMMIT"
    git checkout --detach "$REPO_COMMIT"
fi

SOURCE_COMMIT="$(git rev-parse HEAD)"
export SOURCE_COMMIT

echo "SOURCE_COMMIT=$SOURCE_COMMIT"


# ============================================================
# 2. Correct PennFudan image/mask matching
# ============================================================
echo
echo "============================================================"
echo "2. APPLY PENNFUDAN PAIRING FIX"
echo "============================================================"

python - <<'PYPATCH'
from pathlib import Path

path = Path("dense_prediction/datasets.py")
text = path.read_text(encoding="utf8")

old = """        images = {p.stem: p for p in _files(image_dir)}
        masks = {p.stem: p for p in _files(mask_dir)}
        keys = sorted(set(images).intersection(masks))
"""

new = """        images = {p.stem: p for p in _files(image_dir)}

        def mask_key(path: Path) -> str:
            stem = path.stem
            return stem[:-5] if stem.lower().endswith("_mask") else stem

        masks = {mask_key(p): p for p in _files(mask_dir)}
        keys = sorted(set(images).intersection(masks))
"""

already_fixed = (
    'stem.lower().endswith("_mask")' in text
    and "masks = {mask_key(p): p for p in _files(mask_dir)}" in text
)

if already_fixed:
    print("PennFudan pairing is already corrected.")
else:
    if old not in text:
        raise RuntimeError(
            "Expected PennFudan code was not found. "
            "Unsafe automatic modification refused."
        )

    path.write_text(
        text.replace(old, new, 1),
        encoding="utf8",
    )

    print("Applied PennFudan *_mask pairing correction.")

check = path.read_text(encoding="utf8")

assert 'stem.lower().endswith("_mask")' in check
assert "masks = {mask_key(p): p for p in _files(mask_dir)}" in check

print("PATCH CHECK: PASS")
PYPATCH

python -m py_compile dense_prediction/datasets.py


# ============================================================
# 3. Dependencies
# ============================================================
echo
echo "============================================================"
echo "3. INSTALL REQUIREMENTS"
echo "============================================================"

python -m pip install \
    -q \
    --upgrade-strategy only-if-needed \
    -r requirements-kaggle.txt


# ============================================================
# 4. Validate repository
# ============================================================
echo
echo "============================================================"
echo "4. VALIDATE REPOSITORY"
echo "============================================================"

python tools/validate_dt1d.py
python tools/validate_all_configs.py

python -m pytest -q \
    tests/test_proposal_contract.py \
    tests/test_dense_manifest.py \
    tests/test_dense_direct_configs.py \
    tests/test_run_dense_paper_selection.py


# ============================================================
# 5. GPU validation
# ============================================================
echo
echo "============================================================"
echo "5. CHECK GPU"
echo "============================================================"

python - <<'PYGPU'
import torch

assert torch.cuda.is_available(), "CUDA GPU is required."

count = torch.cuda.device_count()

print("GPU count =", count)

for i in range(count):
    print(i, torch.cuda.get_device_name(i))

assert count >= 1
PYGPU


# ============================================================
# 6. Restore previous D05B ZIP
# ============================================================
echo
echo "============================================================"
echo "6. RESTORE PREVIOUS D05B RESULTS IF AVAILABLE"
echo "============================================================"

rm -rf "$OUTPUT_ROOT"

python - <<'PYRESTORE'
import os
import shutil
import zipfile
from pathlib import Path

sid = os.environ["SESSION_ID"]
out = Path(os.environ["OUTPUT_ROOT"])

input_root = Path("/kaggle/input")

archives = (
    list(input_root.rglob(f"{sid}_results.zip"))
    if input_root.exists()
    else []
)

if not archives:
    print("No previous D05B ZIP found; starting fresh.")
    raise SystemExit(0)

archive = max(
    archives,
    key=lambda p: p.stat().st_mtime,
)

tmp = (
    Path("/kaggle/working")
    / f"_restore_{sid}"
)

shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as z:
    z.extractall(tmp)

matches = list(
    tmp.rglob(f"dense_{sid}")
)

if len(matches) == 1:
    shutil.move(
        str(matches[0]),
        str(out),
    )
    print("RESTORED:", archive)
else:
    print(
        f"Found {len(matches)} candidate result directories; "
        "archive ignored."
    )

shutil.rmtree(tmp, ignore_errors=True)
PYRESTORE

mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"


# ============================================================
# 7. Download PennFudan and pretrained backbone
# ============================================================
echo
echo "============================================================"
echo "7. PRELOAD PENNFUDAN + PRETRAINED BACKBONE"
echo "============================================================"

python - <<'PYPRELOAD'
import os
import shutil
import urllib.request
from pathlib import Path

import yaml
from torchvision import datasets, models

target = os.environ["TARGET"]

config_path = (
    Path("configs")
    / "dense"
    / "experiments"
    / f"{target}.yaml"
)

cfg = yaml.safe_load(
    config_path.read_text(encoding="utf8")
)

root = Path(os.environ["DATA_ROOT"])
root.mkdir(parents=True, exist_ok=True)

dataset_name = cfg["dataset"]
pipeline = cfg["pipeline"]

print("PRELOAD dataset =", dataset_name)
print("PRELOAD pipeline =", pipeline)

if dataset_name == "drive":

    dst = root / "DRIVE"

    required = [
        dst / "training/images",
        dst / "training/1st_manual",
        dst / "test/images",
        dst / "test/1st_manual",
    ]

    if not all(p.is_dir() for p in required):

        inp = Path("/kaggle/input")
        found = None

        if inp.exists():
            for training in inp.rglob("training"):

                base = training.parent

                candidate = [
                    base / "training/images",
                    base / "training/1st_manual",
                    base / "test/images",
                    base / "test/1st_manual",
                ]

                if all(p.is_dir() for p in candidate):
                    found = base
                    break

        if found is None:
            raise RuntimeError(
                "DRIVE requires an authorized Kaggle Input."
            )

        shutil.copytree(
            found,
            dst,
            dirs_exist_ok=True,
        )

elif dataset_name == "pennfudan":

    dst = root / "PennFudanPed"

    images = dst / "PNGImages"
    masks = dst / "PedMasks"

    if not (
        images.is_dir()
        and masks.is_dir()
    ):
        archive = root / "PennFudanPed.zip"

        url = (
            "https://www.cis.upenn.edu/"
            "~jshi/ped_html/PennFudanPed.zip"
        )

        print("Downloading PennFudan:", url)

        urllib.request.urlretrieve(
            url,
            archive,
        )

        print("Extracting:", archive)

        shutil.unpack_archive(
            str(archive),
            str(root),
        )

    assert images.is_dir(), f"Missing {images}"
    assert masks.is_dir(), f"Missing {masks}"

elif dataset_name in {
    "oxford_pet_segmentation",
    "oxford_pet_detection",
}:

    for split in ("trainval", "test"):
        datasets.OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="segmentation",
            download=True,
        )

else:
    raise RuntimeError(
        f"Unsupported bootstrap dataset: {dataset_name}"
    )


if pipeline == "vit_b16_dense":

    models.vit_b_16(
        weights=models.ViT_B_16_Weights.DEFAULT
    )

elif pipeline in {
    "deeplab_mobilenet_v3",
    "fasterrcnn_mobilenet_v3_fpn",
}:

    models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT
    )

else:
    raise RuntimeError(
        f"Unsupported pipeline: {pipeline}"
    )

print("PRELOAD PASS")
PYPRELOAD


# ============================================================
# 8. Validate real PennFudan data through repository class
# ============================================================
echo
echo "============================================================"
echo "8. VERIFY REAL PENNFUDAN DATA"
echo "============================================================"

python - <<'PYCHECK'
import os
from pathlib import Path

import torch

from dense_prediction.datasets import (
    PennFudanBinaryDataset,
)
from dense_prediction.transforms import (
    SegmentationTransform,
)

root = (
    Path(os.environ["DATA_ROOT"])
    / "PennFudanPed"
)

raw = PennFudanBinaryDataset(
    root=root,
    transform=None,
)

print(
    "Matched PennFudan samples =",
    len(raw),
)

if len(raw) < 100:
    raise RuntimeError(
        f"Suspicious PennFudan sample count: {len(raw)}"
    )

image_path, mask_path = raw.samples[0]

print("First image =", image_path.name)
print("First mask  =", mask_path.name)

assert mask_path.stem.lower().endswith("_mask")

dataset = PennFudanBinaryDataset(
    root=root,
    transform=SegmentationTransform(
        size=224,
        train=False,
    ),
)

image, mask = dataset[0]

print(
    "Image shape =",
    tuple(image.shape),
)

print(
    "Mask shape =",
    tuple(mask.shape),
)

print(
    "Mask values =",
    torch.unique(mask).tolist(),
)

assert tuple(image.shape) == (
    3,
    224,
    224,
)

assert tuple(mask.shape) == (
    224,
    224,
)

values = set(
    torch.unique(mask).tolist()
)

assert values.issubset({0, 1})

if int(mask.sum()) <= 0:
    raise RuntimeError(
        "PennFudan mask contains no foreground."
    )

print("PENNFUDAN DATA CHECK: PASS")
PYCHECK


# ============================================================
# 9. Validate exact execution plan
# ============================================================
echo
echo "============================================================"
echo "9. BUILD D05B EXECUTION PLAN"
echo "============================================================"

python tools/run_dense_paper.py \
    --target "$TARGET" \
    --methods "$METHODS" \
    --seeds "$SEEDS" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device cpu \
    --plan-only \
    --download


# ============================================================
# 10. Runner helper
# ============================================================
run_group() {

    local group="$1"
    local device="$2"
    local logfile="$3"

    [[ -z "$group" ]] && return 0

    echo "Starting group=$group on $device"
    echo "Log=$logfile"

    python tools/run_dense_paper.py \
        --target "$TARGET" \
        --methods "$group" \
        --seeds "$SEEDS" \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --device "$device" \
        --download \
        --skip-if-complete \
        --continue-on-error \
        >"$logfile" 2>&1
}

export -f run_group


# ============================================================
# 11. Execute groups
# ============================================================
echo
echo "============================================================"
echo "10. START D05B TRAINING"
echo "============================================================"

set +e

PIDS=()

NGPU="$(
    python -c \
    'import torch; print(torch.cuda.device_count())'
)"

echo "Detected GPUs: $NGPU"


if (( NGPU >= 2 )); then

    if [[ -n "$GROUP0" ]]; then
        setsid bash -c \
            'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' &
        PIDS+=("$!")
    fi

    if [[ -n "$GROUP1" ]]; then
        setsid bash -c \
            'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' &
        PIDS+=("$!")
    fi

else

    if [[ -n "$GROUP0" ]]; then
        setsid bash -c \
            'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' &
        PIDS+=("$!")
    fi
fi


RUN_RC=0


while ((${#PIDS[@]})); do

    ALIVE=0

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ALIVE=1
        fi
    done

    (( ALIVE == 0 )) && break

    if (( $(date +%s) >= DEADLINE_EPOCH )); then

        echo "Kaggle safety deadline reached."

        for pid in "${PIDS[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || true
        done

        sleep 20

        for pid in "${PIDS[@]}"; do
            kill -KILL -- "-$pid" 2>/dev/null || true
        done

        RUN_RC=124
        break
    fi

    sleep 10
done


for pid in "${PIDS[@]}"; do

    wait "$pid"
    rc=$?

    if (( rc != 0 && RUN_RC == 0 )); then
        RUN_RC=$rc
    fi
done


# One-GPU fallback.
if (( NGPU < 2 )) \
    && [[ -n "$GROUP1" ]] \
    && (( $(date +%s) < DEADLINE_EPOCH )); then

    echo "Only one GPU; running GROUP1 on cuda:0."

    setsid bash -c \
        'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' &

    pid=$!

    while kill -0 "$pid" 2>/dev/null; do

        if (( $(date +%s) >= DEADLINE_EPOCH )); then

            kill -TERM -- "-$pid" 2>/dev/null || true
            sleep 20
            kill -KILL -- "-$pid" 2>/dev/null || true

            RUN_RC=124
            break
        fi

        sleep 10
    done

    wait "$pid"
    rc=$?

    if (( rc != 0 && RUN_RC == 0 )); then
        RUN_RC=$rc
    fi
fi

set -e


# ============================================================
# 12. Aggregate
# ============================================================
echo
echo "============================================================"
echo "11. AGGREGATE RESULTS"
echo "============================================================"

python tools/aggregate_dense_results.py \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/aggregated" \
    --require-seeds "$SEEDS" \
    || true


# ============================================================
# 13. Authoritative status
# ============================================================
python - <<'PYSTATUS'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
target = os.environ["TARGET"]

methods = [
    value
    for value in os.environ["METHODS"].split(",")
    if value
]

seeds = [
    int(value)
    for value in os.environ["SEEDS"].split(",")
    if value
]

missing = []

for method in methods:
    for seed in seeds:

        run_dir = (
            root
            / target
            / method
            / f"seed_{seed}"
        )

        accepted = (
            "metrics.json",
            "summary.json",
            "test_summary.json",
            "run_summary.json",
        )

        if not any(
            (run_dir / filename).is_file()
            for filename in accepted
        ):
            missing.append(
                [method, seed]
            )

status = {
    "session": os.environ["SESSION_ID"],
    "family": "dense",
    "target": target,
    "methods": methods,
    "seeds": seeds,
    "complete": not missing,
    "missing": missing,
    "source_commit": os.environ.get(
        "SOURCE_COMMIT"
    ),
}

status_file = (
    root
    / "SESSION_STATUS.json"
)

status_file.write_text(
    json.dumps(
        status,
        indent=2,
    )
    + "\n",
    encoding="utf8",
)

print(
    json.dumps(
        status,
        indent=2,
    )
)
PYSTATUS


# ============================================================
# 14. Diagnose failures automatically
# ============================================================
COMPLETE="$(
python - <<'PY'
import json
import os
from pathlib import Path

path = (
    Path(os.environ["OUTPUT_ROOT"])
    / "SESSION_STATUS.json"
)

status = json.loads(
    path.read_text()
)

print(
    "true"
    if status["complete"]
    else "false"
)
PY
)"


if [[ "$COMPLETE" != "true" ]]; then

    echo
    echo "============================================================"
    echo "SESSION INCOMPLETE — DIAGNOSTIC LOGS"
    echo "============================================================"

    for log in \
        "$OUTPUT_ROOT/gpu0.log" \
        "$OUTPUT_ROOT/gpu1.log"
    do

        if [[ -f "$log" ]]; then

            echo
            echo "---------------- $log ----------------"

            tail -n 120 "$log" || true

        fi
    done
fi


# ============================================================
# 15. Save result ZIP
# ============================================================
pack_results

trap - EXIT


echo
echo "============================================================"

if [[ "$COMPLETE" == "true" ]]; then

    echo "D05B COMPLETE"
    echo "SOURCE_COMMIT=$SOURCE_COMMIT"
    echo "RESULT_ZIP=$RESULT_ZIP"
    echo "============================================================"

    exit 0
fi


if (( RUN_RC == 124 )); then

    echo "D05B INCOMPLETE ONLY BECAUSE OF KAGGLE TIME LIMIT."
    echo "Attach D05B_results.zip and rerun this same cell."
    echo "Completed runs will automatically be skipped."
    echo "RESULT_ZIP=$RESULT_ZIP"
    echo "============================================================"

    exit 0
fi


echo "D05B FAILED WITH A REAL TRAINING ERROR."
echo "The relevant GPU log tail has been printed above."
echo "Partial results are available at:"
echo "$RESULT_ZIP"
echo "============================================================"

FINAL_RC="$RUN_RC"

if (( FINAL_RC == 0 )); then
    FINAL_RC=1
fi

exit "$FINAL_RC"
