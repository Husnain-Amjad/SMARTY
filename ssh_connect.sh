#!/usr/bin/env bash
# ssh_connect.sh - connect to your remote GPU box and (optionally) sync the
# pipeline + launch training inside a tmux session that survives disconnects.
#
# This is parameterized via environment variables so no host/credentials are
# hardcoded here - set them once (e.g. in a local .env you source, NOT committed
# to version control) or pass them inline:
#
#   REMOTE_HOST=203.0.113.10 REMOTE_USER=ubuntu REMOTE_KEY=~/.ssh/id_ed25519 \
#     ./ssh_connect.sh connect
#
# Usage:
#   ./ssh_connect.sh connect          - open an interactive SSH session
#   ./ssh_connect.sh sync             - rsync this local directory to the remote box
#   ./ssh_connect.sh setup            - sync + install requirements.txt remotely
#   ./ssh_connect.sh train <cmd...>   - run <cmd...> remotely INSIDE a tmux session
#                                        that keeps running even if your connection drops
#   ./ssh_connect.sh attach           - reattach to the training tmux session later
#   ./ssh_connect.sh status           - check whether the tmux training session is still running
#   ./ssh_connect.sh download <path>  - pull a file/dir back from the remote box (e.g. a checkpoint)

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST (e.g. export REMOTE_HOST=203.0.113.10)}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_KEY="${REMOTE_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/DeepMATH}"
LOCAL_DIR="${LOCAL_DIR:-$(pwd)}"
TMUX_SESSION="${TMUX_SESSION:-training}"

SSH_OPTS=(-i "$REMOTE_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=accept-new)
SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

cmd="${1:-}"
shift || true

case "$cmd" in
  connect)
    echo "[ssh_connect] connecting to $SSH_TARGET ..."
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET"
    ;;

  sync)
    echo "[ssh_connect] syncing $LOCAL_DIR -> $SSH_TARGET:$REMOTE_DIR ..."
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"
    rsync -avz --progress \
      -e "ssh -i $REMOTE_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new" \
      --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
      --exclude 'outputs/' --exclude 'ckpts/' \
      "$LOCAL_DIR"/ "$SSH_TARGET:$REMOTE_DIR"/
    ;;

  setup)
    "$0" sync
    echo "[ssh_connect] installing requirements.txt on remote (see requirements.txt "
    echo "for the CUDA vs ROCm torch/vllm install split - run the right one manually"
    echo "first if this is a fresh box, then re-run setup for the rest)."
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "cd '$REMOTE_DIR' && pip install --break-system-packages -q -r requirements.txt"
    ;;

  train)
    if [ "$#" -eq 0 ]; then
      echo "usage: $0 train <command to run remotely>" >&2
      exit 1
    fi
    remote_cmd="$*"
    echo "[ssh_connect] launching inside tmux session '$TMUX_SESSION' on $SSH_TARGET:"
    echo "  $remote_cmd"
    echo "[ssh_connect] this keeps running even if this SSH connection drops - use "
    echo "'$0 attach' to reconnect and watch progress, or '$0 status' to check on it."
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "tmux new-session -d -s '$TMUX_SESSION' -c '$REMOTE_DIR' \"$remote_cmd; echo; echo '[training finished, exit code:' \\\$?; exec bash\""
    ;;

  attach)
    echo "[ssh_connect] attaching to tmux session '$TMUX_SESSION' on $SSH_TARGET ..."
    echo "(Ctrl-b then d to detach without stopping the training run)"
    ssh -t "${SSH_OPTS[@]}" "$SSH_TARGET" "tmux attach -t '$TMUX_SESSION'"
    ;;

  status)
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "tmux has-session -t '$TMUX_SESSION' 2>/dev/null && echo 'RUNNING: $TMUX_SESSION is active' || echo 'NOT RUNNING: no session named $TMUX_SESSION'"
    ;;

  download)
    if [ "$#" -eq 0 ]; then
      echo "usage: $0 download <remote path under $REMOTE_DIR>" >&2
      exit 1
    fi
    remote_path="$1"
    dest="./$(basename "$remote_path")"
    echo "[ssh_connect] downloading $SSH_TARGET:$REMOTE_DIR/$remote_path -> $dest"
    rsync -avz --progress \
      -e "ssh -i $REMOTE_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new" \
      "$SSH_TARGET:$REMOTE_DIR/$remote_path" "$dest"
    ;;

  *)
    cat <<'USAGE'
Usage: ./ssh_connect.sh <command> [args...]

Commands:
  connect            open an interactive SSH session
  sync               rsync the local pipeline directory to the remote box
  setup              sync + pip install requirements.txt remotely
  train <cmd...>     run <cmd...> remotely inside a tmux session that survives
                      SSH disconnects (e.g. ./ssh_connect.sh train python sft_train.py
                      --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl
                      --mode lora --output_dir ckpts/run1 --save_every_epochs 0.5
                      --push_every_checkpoint --push_to hf --hf_repo_id you/run1)
  attach             reattach to the training tmux session to watch progress
  status             check whether the tmux training session is still running
  download <path>    pull a file/dir back from the remote box (e.g. a checkpoint)

Required environment variables:
  REMOTE_HOST   (required) remote box IP or hostname
  REMOTE_USER   (default: ubuntu)
  REMOTE_PORT   (default: 22)
  REMOTE_KEY    (default: ~/.ssh/id_ed25519)
  REMOTE_DIR    (default: /workspace/DeepMATH)
  TMUX_SESSION  (default: training)
USAGE
    exit 1
    ;;
esac
