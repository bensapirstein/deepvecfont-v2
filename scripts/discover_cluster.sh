#!/usr/bin/env bash
# One-shot probe of the training host (labserver). Answers the questions the
# runbooks currently assume rather than verify: how many GPUs and whose they
# are, whether a scheduler owns them, whether a detached job survives logout,
# and whether the env and storage match what PROJECT_PLAN.md says.
#
# Read-only. Launches nothing, writes only to /tmp.
#
# Run it either way:
#   ssh labserver 'bash -s' < scripts/discover_cluster.sh      # from the Mac
#   ./scripts/discover_cluster.sh                              # already on the host
#
# Paste the output back into the session that asked for it. Each section says
# what its answer decides, so a reader who does not know the project can still
# route the result.

set -uo pipefail   # deliberately no -e: a failing probe is data, not a stop

REPO="${DVF_REPO:-$HOME/deepvecfont-v2}"
CONDA_ENV="${DVF_ENV:-dvf_v2}"
STORE="${DVF_STORE:-/data/bens}"

hr()  { printf '\n=== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }
have() { command -v "$1" >/dev/null 2>&1; }

hr "HOST"
printf 'hostname : %s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'user     : %s\n' "$(whoami)"
printf 'uname    : %s\n' "$(uname -srm)"
printf 'distro   : %s\n' "$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo unknown)"
printf 'uptime   : %s\n' "$(uptime -p 2>/dev/null || uptime)"
printf 'cpus     : %s   mem: %s\n' "$(nproc 2>/dev/null)" "$(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
printf 'shell    : %s   login shell: %s\n' "$SHELL" "$(getent passwd "$(whoami)" | cut -d: -f7)"

# --- Decides: whether COMMANDS.md's "at least three GPUs" is right, which ids
# --- are actually free, and whether run_experiments.sh can use parallel mode.
hr "GPUS"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi present but query failed"
  echo
  echo "-- processes currently holding a GPU (who else is on this box) --"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r uuid pid mem; do
        pid="$(echo "$pid" | tr -d ' ')"
        owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        cmd="$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-70)"
        printf '  pid %-8s user %-12s %-10s %s\n' "$pid" "${owner:-?}" "$(echo "$mem" | tr -d ' ')" "$cmd"
      done
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && echo "  (none — all GPUs idle)"
  echo
  printf 'driver/cuda : %s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
  echo "no nvidia-smi on PATH"
fi

# --- Decides: whether launching with nohup on the login shell is legitimate or
# --- whether jobs are supposed to go through a queue. If any of these exist,
# --- the whole launch pattern in run_experiments.sh needs revisiting.
hr "SCHEDULER"
found_sched=0
for s in sinfo squeue sbatch qsub bsub condor_q pbsnodes; do
  if have "$s"; then echo "FOUND: $s -> $(command -v $s)"; found_sched=1; fi
done
if [ "$found_sched" -eq 1 ]; then
  echo
  echo "A scheduler is installed. Direct nohup + CUDA_VISIBLE_DEVICES may violate"
  echo "local policy even if it technically works. Check with whoever runs the box."
  have sinfo && { echo; echo "-- sinfo --"; sinfo 2>&1 | head -20; }
  have squeue && { echo; echo "-- squeue --"; squeue 2>&1 | head -20; }
else
  echo "none found (no slurm/pbs/lsf/condor)."
  echo "=> bare multi-GPU box: CUDA_VISIBLE_DEVICES + nohup is the right pattern,"
  echo "   and GPU etiquette is social rather than enforced. Check the GPUS section"
  echo "   above for other users before claiming an id."
fi

# --- Decides: whether a job launched over a non-interactive ssh survives the
# --- connection closing. nohup detaches from the terminal but the job still
# --- dies if systemd-logind reaps the session scope on logout.
hr "DETACHED JOB SURVIVAL"
have tmux    && echo "tmux    : $(tmux -V 2>/dev/null)"    || echo "tmux    : NOT INSTALLED"
have screen  && echo "screen  : $(screen -v 2>&1 | head -1)" || echo "screen  : NOT INSTALLED"
have setsid  && echo "setsid  : yes" || echo "setsid  : NOT INSTALLED"
if have loginctl; then
  kill_mode="$(loginctl show-session "$(loginctl 2>/dev/null | awk -v u="$(whoami)" '$3==u{print $1; exit}')" -p KillUserProcesses 2>/dev/null)"
  echo "logind  : ${kill_mode:-KillUserProcesses=unknown}"
  echo "          KillUserProcesses=yes means bare nohup DIES on logout; use tmux or setsid."
  echo "          =no (the default) means nohup survives."
else
  echo "logind  : loginctl absent, likely no systemd session reaping"
fi
echo
echo "-- live check: detach a 60s sleep, confirm it is orphaned and alive --"
rm -f /tmp/dvf_probe.out
nohup bash -c 'sleep 60' > /tmp/dvf_probe.out 2>&1 &
probe_pid=$!
sleep 1
if kill -0 "$probe_pid" 2>/dev/null; then
  echo "  launched pid $probe_pid, ppid $(ps -o ppid= -p $probe_pid | tr -d ' '), alive"
  echo "  NOTE: this only proves nohup works within the session. To prove survival,"
  echo "  disconnect, reconnect, and run: ps -p $probe_pid -o pid,ppid,etime,args"
else
  echo "  launched pid $probe_pid but it is already gone — investigate"
fi

# --- Decides: whether the env in PROJECT_PLAN.md still matches reality, and
# --- whether the SSIM work in 2.2 needs a scikit-image install.
hr "CONDA ENV: $CONDA_ENV"
if have conda; then
  echo "conda   : $(conda --version 2>/dev/null), base $(conda info --base 2>/dev/null)"
  conda env list 2>/dev/null | sed 's/^/  /'
  echo
  # shellcheck disable=SC1091
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" 2>/dev/null
  if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    echo "activated $CONDA_ENV, python $(python -V 2>&1)"
    python - <<'PY' 2>&1 | sed 's/^/  /'
import importlib, sys
for mod in ("torch","numpy","cairosvg","wandb","tensorboardX","skimage","matplotlib","scipy"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"{mod:14s} MISSING ({type(e).__name__})")
try:
    import torch
    print(f"{'cuda avail':14s} {torch.cuda.is_available()}  devices={torch.cuda.device_count()}  torch-cuda={torch.version.cuda}")
except Exception as e:
    print("torch cuda check failed:", e)
PY
    echo
    echo "  skimage MISSING means PROJECT_PLAN 2.2's SSIM needs: pip install scikit-image"
  else
    echo "could not activate $CONDA_ENV"
  fi
else
  echo "no conda on PATH (non-interactive ssh may not source ~/.bashrc —"
  echo "if so, cluster commands must source conda.sh explicitly)"
fi

# --- Decides: whether the 7.3 storage arithmetic still holds and how many
# --- runs of headroom remain.
hr "STORAGE"
echo "-- quota / free on $STORE --"
df -h "$STORE" 2>&1 | sed 's/^/  /'
have quota && { echo; echo "-- quota -s --"; quota -s 2>&1 | sed 's/^/  /'; }
echo
echo "-- $STORE/deepvecfont-v2 breakdown --"
du -sh "$STORE"/deepvecfont-v2/* 2>/dev/null | sort -h | sed 's/^/  /' || echo "  (path not found)"
echo
echo "-- home --"
df -h "$HOME" 2>&1 | sed 's/^/  /'

# --- Decides: whether the Mac and the cluster are actually looking at the same
# --- commit, which is the usual cause of a command that works in one place only.
hr "REPO: $REPO"
if [ -d "$REPO/.git" ]; then
  cd "$REPO" || exit
  printf 'branch   : %s\n' "$(git rev-parse --abbrev-ref HEAD)"
  printf 'head     : %s\n' "$(git log -1 --format='%h %ad %s' --date=short)"
  printf 'upstream : %s\n' "$(git rev-parse --abbrev-ref '@{u}' 2>/dev/null || echo none)"
  echo
  echo "-- ahead/behind origin --"
  git fetch --quiet origin 2>/dev/null
  git rev-list --left-right --count '@{u}...HEAD' 2>/dev/null \
    | awk '{printf "  behind %s, ahead %s\n", $1, $2}' || echo "  (no upstream)"
  echo
  echo "-- uncommitted --"
  git status --short 2>/dev/null | head -20 | sed 's/^/  /'
  [ -z "$(git status --short)" ] && echo "  (clean)"
  echo
  echo "-- symlinks --"
  for d in data experiments; do
    if [ -L "$d" ]; then printf '  %-12s -> %s\n' "$d" "$(readlink "$d")"
    elif [ -d "$d" ]; then printf '  %-12s real directory, NOT a symlink\n' "$d"
    else printf '  %-12s missing\n' "$d"; fi
  done
  echo
  echo "-- gitignore lines for data/experiments --"
  grep -n '^data\|^experiments' .gitignore 2>/dev/null | sed 's/^/  /' || echo "  (none)"
  echo "  a trailing slash does NOT match a symlink — PROJECT_PLAN 1.5"
  echo
  echo "-- experiments present --"
  ls -1 experiments/ 2>/dev/null | sed 's/^/  /' | head -30 || echo "  (none)"
  echo
  echo "-- checkpoints per experiment --"
  for e in experiments/*/; do
    n=$(ls -1 "$e/checkpoints"/*.ckpt 2>/dev/null | wc -l)
    sz=$(du -sh "$e" 2>/dev/null | cut -f1)
    [ "$n" -gt 0 ] && printf '  %-40s %2s ckpt  %s\n' "$(basename "$e")" "$n" "$sz"
  done
  echo
  echo "-- running train/test processes --"
  pgrep -af "python (train|test_few_shot)\.py" 2>/dev/null | cut -c1-120 | sed 's/^/  /' || echo "  (none)"
  echo
  echo "-- recent nohup logs --"
  ls -lt nohup_*.out 2>/dev/null | head -10 | sed 's/^/  /' || echo "  (none)"
else
  echo "no git repo at $REPO"
fi

hr "WANDB"
if [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "credentials present in ~/.netrc"
else
  echo "no wandb credentials in ~/.netrc — 'wandb login' may be needed"
fi
echo "outbound https to api.wandb.ai:"
timeout 8 bash -c 'cat < /dev/null > /dev/tcp/api.wandb.ai/443' 2>/dev/null \
  && echo "  reachable (online logging works)" \
  || echo "  BLOCKED — wandb must run with WANDB_MODE=offline and be synced later"

hr "DONE"
echo "Sections that change the plan if they surprise: SCHEDULER, DETACHED JOB"
echo "SURVIVAL, GPUS (how many are really free), and STORAGE headroom."
