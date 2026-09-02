

set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
bash run_full_experiment.sh "deepseek-ai/deepseek-math-7b-base" "${1:-full}" "${2:-full}"
