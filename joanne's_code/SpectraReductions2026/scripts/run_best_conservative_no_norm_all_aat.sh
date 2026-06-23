#!/usr/bin/env bash
# ------------------------------------------------------------
# Best current SP_Ace / SPACE run for NGC 6569 AAT spectra
#
# Purpose:
#   Run all aat_*.txt spectra with the wavelength setup that
#   best matches the Voigt CaT / Dias-Parisi [Fe/H]DP scale:
#
#       Conservative + no_norm + ABD
#
# Adopted conservative wavelength limits:
#   8450-8493
#   8503-8535
#   8550-8660
#   8670-8745
#
# These avoid the CaT cores and the redder problematic/sky-heavy
# region while keeping useful nearby metal-line information.
#
# Important:
#   - no_norm is ON
#   - alpha is OFF / commented
#   - ABD_loop is ON
#   - Salaris_MH is ON
#   - fwhm = 0.8
# ------------------------------------------------------------

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

SPACEDIR="${SPACE_DIR:-${PROJECT_ROOT}/space}"
SPECTRADIR="${NORMALIZED_AAT_DIR:-${PROJECT_ROOT}/data/interim/aat_normalized_lab_frame}"
GCOG_LIBRARY="${GCOG_DIR:-${SPACEDIR}/GCOGlibrarySPACE1}"

MASTER_OUT="${SPACEDIR}/space_TGM_ABD_best_conservative_no_norm_all_aat.txt"
LOGFILE="${SPACEDIR}/run_best_conservative_no_norm_all_aat.log"

cd "$SPACEDIR" || {
    echo "ERROR: Cannot cd to $SPACEDIR"
    exit 1
}

if [[ ! -x ./SPACE ]]; then
    echo "ERROR: ./SPACE not found or not executable in $SPACEDIR"
    echo "Try: chmod +x SPACE"
    exit 1
fi

echo "Running best conservative no_norm ABD SPACE batch" | tee "$LOGFILE"
echo "SPACE dir:   $SPACEDIR" | tee -a "$LOGFILE"
echo "Spectra dir: $SPECTRADIR" | tee -a "$LOGFILE"
echo "Master out:  $MASTER_OUT" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Master output header.
# This matches the standard SPACE/SP_Ace TGM_ABD first columns.
printf "fiberid\tconv\tRV\tFWHM\tSNR\tchisq\tTeff\tT_l\tT_h\tlogg\tL_l\tL_h\tMH\tMH_l\tMH_h\tFeH\tFeH_l\tFeH_h\tFeH_N\talphaH\talphaH_l\talphaH_h\talpha_N\talphaFe\n" > "$MASTER_OUT"

shopt -s nullglob
spectra=( "$SPECTRADIR"/aat_*.txt )

if (( ${#spectra[@]} == 0 )); then
    echo "ERROR: No spectra found matching $SPECTRADIR/aat_*.txt" | tee -a "$LOGFILE"
    exit 1
fi

for SPEC in "${spectra[@]}"; do
    BASE="$(basename "$SPEC")"
    FIBER="${BASE#aat_}"
    FIBER="${FIBER%.txt}"

    echo "------------------------------------------------------------" | tee -a "$LOGFILE"
    echo "Running fiber $FIBER: $SPEC" | tee -a "$LOGFILE"

    # Write a clean, reproducible space.par for this star.
    cat > space.par <<EOF
#obs_sp_file 'ESO_Spec/eso_592.txt'
obs_sp_file '${SPEC}'
GCOGlib '${GCOG_LIBRARY}'
#fwhm 0.4
fwhm 0.8
#
#wave_lims 8445 8845
#wave_lims 8450 8493 8503 8535 8550 8660 8670 8800
#wave_lims 8490 8662 8700 8760
#wave_lims 8450 8493 8503 8535 8550 8660 8670 8745 8775 8875
#Conservative:
wave_lims 8450 8493 8503 8535 8550 8660 8670 8745
#
#Preferred:
#wave_lims 8450 8493 8503 8535 8550 8660 8670 8745 8775 8850
#
#Wide:
#wave_lims 8450 8493 8503 8535 8550 8660 8670 8745 8775 8875
# this is a comment
#
no_norm
#norm_rad 30
error_est
null_value 'NaN'
#alpha
#G_force 0.6
ABD_loop
Salaris_MH
EOF

    # Save the exact parameter file used for this star.
    cp space.par "space_${FIBER}_best_conservative_no_norm.par"

    # Remove old generic outputs so stale files cannot be reused accidentally.
    rm -f space_TGM_ABD.dat space_TGM.dat space_model.dat space_residuals.dat

    ./SPACE >> "$LOGFILE" 2>&1
    STATUS=$?

    if [[ $STATUS -ne 0 ]]; then
        echo "WARNING: SPACE returned non-zero status $STATUS for fiber $FIBER" | tee -a "$LOGFILE"
    fi

    if [[ -s space_TGM_ABD.dat ]]; then
        cp space_TGM_ABD.dat "space_TGM_ABD_${FIBER}_best_conservative_no_norm.dat"

        # Append numeric result line(s), prepending fiberid.
        # This skips headers, comments, and blank lines.
        awk -v fid="$FIBER" '
            /^[[:space:]]*$/ {next}
            /^[[:space:]]*#/ {next}
            $1 ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ {
                print fid "\t" $0
            }
        ' space_TGM_ABD.dat >> "$MASTER_OUT"

        echo "Saved space_TGM_ABD_${FIBER}_best_conservative_no_norm.dat" | tee -a "$LOGFILE"
    else
        echo "WARNING: No usable space_TGM_ABD.dat for fiber $FIBER; writing NaN/failed row" | tee -a "$LOGFILE"
        printf "%s\t2\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\t0\tNaN\tNaN\tNaN\t0\tNaN\n" "$FIBER" >> "$MASTER_OUT"
    fi

    if [[ -s space_model.dat ]]; then
        cp space_model.dat "space_model_${FIBER}_best_conservative_no_norm.dat"
    fi

    if [[ -s space_residuals.dat ]]; then
        cp space_residuals.dat "space_residuals_${FIBER}_best_conservative_no_norm.dat"
    fi

done

echo "" | tee -a "$LOGFILE"
echo "Done." | tee -a "$LOGFILE"
echo "Master output: $MASTER_OUT" | tee -a "$LOGFILE"
echo "Log file:      $LOGFILE" | tee -a "$LOGFILE"
echo "Saved parameter files: space_*_best_conservative_no_norm.par" | tee -a "$LOGFILE"
