%%bash

#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# V02 — VTAB-EuroSAT / ViT-B16 / BS32
# FULL CORRECTED KAGGLE CELL
# ============================================================

SESSION_ID="V02"
DATASET="vtab-eurosat"
BATCH_SIZE="32"
VPT_TOKENS="10"
METHODS="dt1d,vpt,pfeiffer,full,linear"
RESULT_MODE="table"
FINAL_SEEDS="0,1,2"

REPO_URL="${DT1D_VIT_REPO_URL:-https://github.com/tydeptrai21042004/whc-vit.git}"
REPO_COMMIT="${DT1D_VIT_COMMIT:-}"

WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/whc-vit-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
MODEL_ROOT="$WORKDIR/models_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/vit_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"

# 710 minutes ≈ safely below Kaggle 12-hour session
DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"

export \
    SESSION_ID \
    DATASET \
    BATCH_SIZE \
    VPT_TOKENS \
    METHODS \
    RESULT_MODE \
    FINAL_SEEDS \
    REPO_DIR \
    DATA_ROOT \
    MODEL_ROOT \
    OUTPUT_ROOT \
    RESULT_ZIP \
    DEADLINE_EPOCH


# ============================================================
# PACK RESULTS
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

        # Do not put large checkpoints into result zip.
        if p.suffix.lower() in {".pth", ".pt", ".ckpt"}:
            continue

        z.write(
            p,
            p.relative_to(root.parent),
        )

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
# CLONE REPOSITORY
# ============================================================

rm -rf "$REPO_DIR"

for n in 1 2 3; do

    git clone \
        --depth 1 \
        "$REPO_URL" \
        "$REPO_DIR" \
    && break

    sleep $((n * 5))
done

cd "$REPO_DIR"


if [[ -n "$REPO_COMMIT" ]]; then

    git fetch \
        --depth 1 \
        origin \
        "$REPO_COMMIT"

    git checkout \
        --detach \
        "$REPO_COMMIT"
fi


SOURCE_COMMIT="$(git rev-parse HEAD)"
export SOURCE_COMMIT

echo "SOURCE COMMIT: $SOURCE_COMMIT"


# ============================================================
# DEPENDENCIES
# ============================================================

python -m pip install -q \
    --upgrade-strategy only-if-needed \
    scipy \
    scikit-learn \
    pandas \
    Pillow \
    fvcore \
    iopath \
    yacs \
    simplejson \
    termcolor \
    tabulate \
    tqdm \
    ml-collections \
    'timm>=1.0.0,<2' \
    PyYAML \
    tensorflow-datasets \
    six


if ! python -c \
    'import tensorflow; print("TensorFlow:", tensorflow.__version__)'
then

    python -m pip install -q \
        'tensorflow>=2.16,<2.20'

fi


# ============================================================
# REPOSITORY VALIDATION
# ============================================================

python validate_dt1d_vit.py

python verify_vpt_original.py

python verify_fair_protocol.py


python -m pytest -q \
    tests/test_dt1d_token_adapter.py \
    tests/test_fair_protocol.py \
    tests/test_repository_contracts.py


# ============================================================
# GPU CHECK
# ============================================================

python - <<'PYGPU'
import torch

assert torch.cuda.is_available(), "CUDA GPU is not available."

print("GPU count =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(
        i,
        torch.cuda.get_device_name(i),
    )
PYGPU


# ============================================================
# CLEAN OUTPUT
# ============================================================

rm -rf "$OUTPUT_ROOT"


# ============================================================
# RESTORE PREVIOUS V02 RESULTS IF ATTACHED TO KAGGLE INPUT
# ============================================================

python - <<'PYRESTORE'
import os
import shutil
import zipfile
from pathlib import Path

sid = os.environ["SESSION_ID"]
out = Path(os.environ["OUTPUT_ROOT"])

input_root = Path("/kaggle/input")

if input_root.exists():

    zips = list(
        input_root.rglob(
            f"{sid}_results.zip"
        )
    )

else:

    zips = []


if zips:

    z = max(
        zips,
        key=lambda p: p.stat().st_mtime,
    )

    tmp = (
        Path("/kaggle/working")
        / f"_restore_{sid}"
    )

    shutil.rmtree(
        tmp,
        ignore_errors=True,
    )

    tmp.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)

    found = list(
        tmp.rglob(
            f"vit_{sid}"
        )
    )

    if len(found) == 1:

        if out.exists():
            shutil.rmtree(
                out,
                ignore_errors=True,
            )

        shutil.move(
            str(found[0]),
            str(out),
        )

        print(
            "RESTORED",
            z,
        )

    shutil.rmtree(
        tmp,
        ignore_errors=True,
    )
PYRESTORE


mkdir -p \
    "$OUTPUT_ROOT" \
    "$DATA_ROOT" \
    "$MODEL_ROOT"


# ============================================================
# ViT-B/16 PRETRAINED WEIGHTS
# ============================================================

WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz"


if [[ ! -s "$WEIGHT_FILE" ]]; then

    tmp="$WEIGHT_FILE.part"

    rm -f "$tmp"

    curl \
        -L \
        --fail \
        --retry 5 \
        --retry-delay 5 \
        --connect-timeout 30 \
        'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz' \
        -o "$tmp"

    mv \
        "$tmp" \
        "$WEIGHT_FILE"

fi


python - "$WEIGHT_FILE" <<'PYW'
import sys
import numpy as np

weight_file = sys.argv[1]

z = np.load(weight_file)

assert len(z.files) == 200, (
    f"Unexpected ViT checkpoint: "
    f"{len(z.files)} tensors"
)

print(
    "ViT tensors =",
    len(z.files),
)
PYW


# ============================================================
# DATASET PREPARATION
#
# V02 FIX:
#
# Old TFDS EuroSAT source:
#   http://madm.dfki.de/files/sentinel/EuroSAT.zip
#   -> HTTP 403
#
# New source:
#   Zenodo EuroSAT_RGB.zip
#
# Important directory difference:
#
# OLD archive expected by TFDS:
#   2750/<class>/*.jpg
#
# ZENODO archive:
#   EuroSAT_RGB/<class>/*.jpg
#
# Therefore:
#   download_url -> Zenodo
#   subdir       -> EuroSAT_RGB
#
# Nothing about train/test protocol is changed.
# ============================================================

if [[ "$DATASET" == "flowers102" ]]; then

    FLOWERS_ROOT="$DATA_ROOT/flowers_download"

    mkdir -p "$FLOWERS_ROOT"


    DATA_PATH="$(
    python - "$FLOWERS_ROOT" <<'PYF'
import json
import sys
from pathlib import Path

from torchvision.datasets import Flowers102


root = Path(sys.argv[1])


datasets = {
    split: Flowers102(
        str(root),
        split=split,
        download=True,
    )
    for split in (
        "train",
        "val",
        "test",
    )
}


img_root = (
    root
    / "flowers-102"
    / "jpg"
)


assert img_root.is_dir(), img_root


for split, ds in datasets.items():

    mapping = {
        Path(str(path)).name: int(label)
        for path, label in zip(
            ds._image_files,
            ds._labels,
        )
    }

    (
        img_root
        / f"{split}.json"
    ).write_text(
        json.dumps(mapping)
    )


print(img_root)
PYF
    )"


else

    DATA_PATH="$DATA_ROOT/tfds"

    mkdir -p "$DATA_PATH"

    export \
        DATA_PATH \
        DATASET


    python - <<'PYD'
import os
import shutil
from pathlib import Path

import tensorflow_datasets as tfds


dataset = os.environ["DATASET"]
data_path = Path(
    os.environ["DATA_PATH"]
)


spec = {
    "vtab-caltech101":
        "caltech101:3.*.*",

    "vtab-dtd":
        "dtd:3.*.*",

    "vtab-eurosat":
        "eurosat/rgb:2.*.*",
}[dataset]


# ------------------------------------------------------------
# Remove ONLY an incomplete/prepared EuroSAT dataset generated
# by the previous failed attempt.
#
# Keep TFDS downloads/extractions cache when possible.
# ------------------------------------------------------------

if dataset == "vtab-eurosat":

    rgb_root = (
        data_path
        / "eurosat"
        / "rgb"
    )

    prepared = (
        rgb_root
        / "2.0.0"
    )

    if prepared.exists():

        print(
            "Removing previous incomplete "
            "EuroSAT prepared directory:",
            prepared,
        )

        shutil.rmtree(
            prepared,
            ignore_errors=True,
        )


    if rgb_root.exists():

        for p in rgb_root.glob(
            "2.0.0.incomplete*"
        ):

            print(
                "Removing incomplete TFDS directory:",
                p,
            )

            shutil.rmtree(
                p,
                ignore_errors=True,
            )


# ------------------------------------------------------------
# Create the SAME TFDS dataset builder.
# ------------------------------------------------------------

builder = tfds.builder(
    spec,
    data_dir=str(data_path),
)


# ------------------------------------------------------------
# V02 DOWNLOAD FIX ONLY
# ------------------------------------------------------------

if dataset == "vtab-eurosat":

    builder.builder_config.download_url = (
        "https://zenodo.org/"
        "records/7711810/files/"
        "EuroSAT_RGB.zip?download=1"
    )

    # Critical fix for the Zenodo archive layout.
    builder.builder_config.subdir = (
        "EuroSAT_RGB"
    )

    print()
    print(
        "EuroSAT download URL :",
        builder.builder_config.download_url,
    )

    print(
        "EuroSAT archive dir  :",
        builder.builder_config.subdir,
    )

    print()


# ------------------------------------------------------------
# Prepare dataset.
# ------------------------------------------------------------

builder.download_and_prepare()


print()
print(
    "TFDS READY       :",
    builder.info.full_name,
)

print(
    "TFDS data dir    :",
    builder.data_dir,
)


# ------------------------------------------------------------
# EuroSAT sanity checks
# ------------------------------------------------------------

if dataset == "vtab-eurosat":

    assert (
        "train"
        in builder.info.splits
    ), builder.info.splits


    n = (
        builder
        .info
        .splits["train"]
        .num_examples
    )


    print(
        "EuroSAT examples :",
        n,
    )


    assert n == 27000, (
        "EuroSAT should contain "
        f"27000 RGB images, got {n}"
    )


    # Make sure dataset can actually yield examples.
    ds = tfds.load(
        "eurosat/rgb:2.*.*",
        data_dir=str(data_path),
        split="train[:1]",
    )

    count = sum(
        1
        for _ in tfds.as_numpy(ds)
    )

    assert count == 1

    print(
        "EuroSAT read test : PASS"
    )

PYD

fi


export DATA_PATH


# ============================================================
# EXPERIMENT PARAMETERS — UNCHANGED
# ============================================================

COMMON=(

    --dataset "$DATASET"

    --data-path "$DATA_PATH"

    --model-root "$MODEL_ROOT"

    --batch-sizes "$BATCH_SIZE"

    --methods "$METHODS"

    --epochs 10

    --resolution 224

    --result-mode table

    --seeds 0,1,2

    --tune-seed 42

    --weight-decay 1e-4

    --warmup-epoch 1

    --patience 20

    --vpt-tokens "$VPT_TOKENS"

    --allow-boundary-best

)


# ============================================================
# CPU DRY-RUN / PREFLIGHT
# ============================================================

PREFLIGHT="$WORKDIR/_preflight_$SESSION_ID"

rm -rf "$PREFLIGHT"


echo
echo "========================================"
echo "V02 PREFLIGHT"
echo "========================================"
echo


python run_fair_vit_comparison.py \
    "${COMMON[@]}" \
    --output-root "$PREFLIGHT" \
    --gpus cpu \
    --dry-run


rm -rf "$PREFLIGHT"


# ============================================================
# ACTUAL V02 EXPERIMENT
# ============================================================

LOG="$OUTPUT_ROOT/session.log"

mkdir -p "$OUTPUT_ROOT"


echo
echo "========================================"
echo "STARTING V02 TRAINING"
echo "Dataset : $DATASET"
echo "BS      : $BATCH_SIZE"
echo "Methods : $METHODS"
echo "Seeds   : $FINAL_SEEDS"
echo "========================================"
echo


set +e


setsid python run_fair_vit_comparison.py \
    "${COMMON[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --gpus auto \
    >"$LOG" 2>&1 &


p=$!


# ============================================================
# WATCH PROCESS + KAGGLE TIME LIMIT
# ============================================================

while kill -0 "$p" 2>/dev/null; do

    if (( $(date +%s) >= DEADLINE_EPOCH )); then

        echo
        echo "========================================"
        echo "TIME CAP REACHED"
        echo "Saving partial V02 results."
        echo "========================================"
        echo

        kill \
            -TERM \
            -- "-$p" \
            2>/dev/null \
            || true

        sleep 20

        kill \
            -KILL \
            -- "-$p" \
            2>/dev/null \
            || true

        break

    fi

    sleep 10

done


wait "$p"

RUN_RC=$?


set -e


echo
echo "TRAIN PROCESS RETURN CODE = $RUN_RC"
echo


# ============================================================
# SHOW FINAL PART OF TRAINING LOG
# ============================================================

if [[ -f "$LOG" ]]; then

    echo
    echo "========================================"
    echo "LAST 80 LINES OF SESSION LOG"
    echo "========================================"

    tail -n 80 "$LOG"

fi


# ============================================================
# SESSION STATUS
# ============================================================

python - <<'PYSTATUS'
import json
import os
from pathlib import Path

import pandas as pd


root = Path(
    os.environ["OUTPUT_ROOT"]
)

dataset = os.environ["DATASET"]

batch_size = int(
    os.environ["BATCH_SIZE"]
)

expected = set(
    os.environ["METHODS"].split(",")
)


stem = dataset.replace(
    "-",
    "_",
)


aggregate_csv = (
    root
    / "aggregated"
    / f"{stem}_fair_three_seed.csv"
)


actual = set()
complete = False


if aggregate_csv.is_file():

    df = pd.read_csv(
        aggregate_csv
    )

    df = df[
        df["batch_size"].astype(int)
        == batch_size
    ]

    actual = set(
        df["method_key"]
        .astype(str)
    )

    complete = (
        actual == expected
        and
        len(df) == len(expected)
    )


status = {

    "session":
        os.environ["SESSION_ID"],

    "family":
        "vit",

    "dataset":
        dataset,

    "batch_size":
        batch_size,

    "methods":
        sorted(expected),

    "final_seeds":
        [0, 1, 2],

    "complete":
        complete,

    "actual_methods":
        sorted(actual),

    "aggregate_csv":
        str(aggregate_csv),

    "source_commit":
        os.environ.get(
            "SOURCE_COMMIT"
        ),

}


status_path = (
    root
    / "SESSION_STATUS.json"
)


status_path.write_text(
    json.dumps(
        status,
        indent=2,
    )
)


print()
print(
    json.dumps(
        status,
        indent=2,
    )
)

print()

PYSTATUS


# ============================================================
# PACK RESULTS
# ============================================================

pack_results


trap - EXIT


echo
echo "========================================"
echo "V02 FINISHED"
echo "RUN_RC=$RUN_RC"
echo "RESULT ZIP:"
echo "$RESULT_ZIP"
echo "========================================"
echo


# The experiment may stop because of Kaggle's time cap.
# We still intentionally finish the notebook cell normally,
# because partial results are packed and can be resumed.
exit 0
