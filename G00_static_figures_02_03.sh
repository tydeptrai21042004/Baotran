#!/usr/bin/env bash
set -Eeuo pipefail
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"; W=/kaggle/working; R="$W/whc-dt1d-static"; O="$W/static_figures"
rm -rf "$R" "$O"; git clone --depth 1 "$REPO_URL" "$R"; cd "$R"; if [[ -n "$REPO_COMMIT" ]]; then git fetch --depth 1 origin "$REPO_COMMIT"; git checkout --detach "$REPO_COMMIT"; fi
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt; mkdir -p "$O"; python tools/plot_figure_02_spectral.py --output "$O/figure_02.png"; python tools/plot_figure_03_architecture.py --output "$O/figure_03.png"; cd "$W"; zip -qr static_figures_results.zip static_figures; ls -lh static_figures_results.zip
