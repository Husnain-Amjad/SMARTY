set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
bash run_full_experiment.sh "deepseek-ai/deepseek-math-7b-r1" "${1:-full}" "${2:-full}"
