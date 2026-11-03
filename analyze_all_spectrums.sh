#!/usr/bin/env bash
# Analyze top-level MCA/CSV spectra; save element plots and four-line TXT blocks (last line blank).
set -euo pipefail
XRF_PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$XRF_PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$XRF_PROJECT_DIR/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
export OPENBLAS_NUM_THREADS="${XRF_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${XRF_BLAS_THREADS:-1}"
exec "$PYTHON_BIN" -m xrf_analyzer.batch \
  --input-dir "${XRF_INPUT_DIR:-$XRF_PROJECT_DIR/XRF_SDD}" \
  --output-dir "${XRF_OUTPUT_DIR:-$XRF_PROJECT_DIR/XRF_Analyze_results}" \
  --calibration "${CALIBRATION_FILE:-$XRF_PROJECT_DIR/data/calibration.csv}" \
  --sensitivity "${SENSITIVITY_FILE:-$XRF_PROJECT_DIR/data/xrf_relative_sensitivity.csv}" \
  --config "${XRF_CONFIG:-$XRF_PROJECT_DIR/configs/default.json}" \
  "$@"
