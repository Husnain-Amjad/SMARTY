
set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
bash run_full_experiment.sh "vanillaOVO/WizardMath-7B-V1.0" "${1:-full}" "${2:-full}"
