set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
bash run_full_experiment.sh "AI-MO/NuminaMath-7B-CoT" "${1:-full}" "${2:-full}"
