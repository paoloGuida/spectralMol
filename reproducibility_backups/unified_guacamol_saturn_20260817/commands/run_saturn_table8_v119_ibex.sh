#!/usr/bin/env bash
set -euo pipefail

cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/saturn_table8_v119.env

bash auto_saturn_theta_tune_ibex.sh
