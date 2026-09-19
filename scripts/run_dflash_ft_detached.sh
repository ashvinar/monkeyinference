#!/bin/bash
# Detached LoRA distill: tmux + caffeinate so a dropped SSH/agent session
# or idle sleep cannot silently kill the job. Resume from adapters if a
# previous trainer died. Does not start a second copy while one is live.
#
#   scripts/run_dflash_ft_detached.sh
#   scripts/run_dflash_ft_detached.sh --watch 67496
#   scripts/run_dflash_ft_detached.sh --eval-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${DFLASH_FT_DIR:-$HOME/.monkey/dflash-ft2}"
SESSION="${TMUX_SESSION:-dflash-ft2}"
PYTHON="${PYTHON:-$HOME/.monkey/mlx-venv/bin/python}"
LOG="$DIR/run.log"
WATCH_PID=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH_PID="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$DIR"
cd "$ROOT"
export PYTHONPATH=src

trainer_pids() {
  pgrep -f 'finetune_dflash.py' || true
}

launch_train_teed() {
  if trainer_pids | grep -q .; then
    echo "finetune_dflash.py already running ($(trainer_pids | tr '\n' ' ')); not starting a second copy"
    return 0
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
  fi
  local argstr="--dir $DIR --max-new 48 --steps 2000 --resume --lr 3e-5 --rank 8 --weight-decay 0.1"
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    argstr="$argstr ${EXTRA[*]}"
  fi
  tmux new-session -d -s "$SESSION" -c "$ROOT" -- \
    /bin/zsh -lc "export PYTHONPATH=src
      mkdir -p '$DIR'
      echo \"==== detached \$(date) extra=[$argstr] ====\" >> '$LOG'
      caffeinate -dims '$PYTHON' -u scripts/finetune_dflash.py $argstr 2>&1 | tee -a '$LOG'
      echo EXIT:\$? \$(date) >> '$LOG'"
  echo "started tmux session $SESSION (caffeinate -dims). log: $LOG"
  tmux ls
}

if [[ -n "$WATCH_PID" ]]; then
  WATCH_SESSION="${SESSION}-watch"
  if tmux has-session -t "$WATCH_SESSION" 2>/dev/null; then
    echo "watchdog session $WATCH_SESSION already exists"
    exit 0
  fi
  tmux new-session -d -s "$WATCH_SESSION" -c "$ROOT" -- \
    /bin/zsh -lc "
      echo \"==== watchdog for PID $WATCH_PID \$(date) ====\" >> '$LOG'
      while kill -0 $WATCH_PID 2>/dev/null; do
        sleep 20
      done
      echo \"==== PID $WATCH_PID exited \$(date) ====\" >> '$LOG'
      if [[ -f '$DIR/eval.json' ]]; then
        echo 'eval.json present; watchdog idle' >> '$LOG'
        exit 0
      fi
      if pgrep -f 'finetune_dflash.py' >/dev/null; then
        echo 'another trainer is live; watchdog idle' >> '$LOG'
        exit 0
      fi
      echo 'no eval.json; launching resume under caffeinate' >> '$LOG'
      export PYTHONPATH=src DFLASH_FT_DIR='$DIR' TMUX_SESSION='$SESSION' PYTHON='$PYTHON'
      '$ROOT/scripts/run_dflash_ft_detached.sh'
    "
  echo "watchdog $WATCH_SESSION waiting on PID $WATCH_PID"
  exit 0
fi

launch_train_teed
