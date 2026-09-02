set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
bash run_full_experiment.sh "Qwen/Qwen2.5-Math-7B-Instruct" "${1:-full}" "${2:-full}"
