#!/usr/bin/env bash
# Collect a read-only Ubuntu preflight report for V1 acceptance.
#
# Usage:
#   bash scripts/collect_acceptance_preflight.sh [output-file]
#
# Optional path overrides:
#   REALTIME_ROOT=/path/to/duet-edge-realtime
#   DUET_EDGE_ROOT=/path/to/duet-edge
#   EDGE_CHECKPOINT=/path/to/train-1800.pt
#   AIST_RAW=/path/to/aist_plusplus_final/motions/example.pkl
#   EDGE_ENV_NAME=edge

set -u

timestamp="$(date +%Y%m%d-%H%M%S 2>/dev/null || printf 'unknown-time')"
output_file="${1:-acceptance-preflight-${timestamp}.txt}"
edge_env_name="${EDGE_ENV_NAME:-edge}"

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd -P)"
detected_realtime_root="$(CDPATH= cd -- "${script_dir}/.." 2>/dev/null && pwd -P)"
realtime_root="${REALTIME_ROOT:-${detected_realtime_root}}"
workspace_root="$(dirname -- "${realtime_root}")"
duet_edge_root="${DUET_EDGE_ROOT:-${workspace_root}/duet-edge}"
checkpoint_path="${EDGE_CHECKPOINT:-${workspace_root}/data+checkpoint/train-1800.pt}"

if [[ -n "${AIST_RAW:-}" ]]; then
  aist_raw="${AIST_RAW}"
else
  aist_raw=""
  motion_dir="${workspace_root}/data+checkpoint/aist_plusplus_final/motions"
  if [[ -d "${motion_dir}" ]]; then
    while IFS= read -r -d '' candidate; do
      aist_raw="${candidate}"
      break
    done < <(find "${motion_dir}" -maxdepth 1 -type f -name '*.pkl' -print0 2>/dev/null)
  fi
fi

# Open the report before redirecting the rest of the script. Failure here should
# be explicit because otherwise the user may assume a report was saved.
if ! : >"${output_file}" 2>/dev/null; then
  printf 'ERROR: cannot write report: %s\n' "${output_file}" >&2
  exit 1
fi

# Keep a copy of the original terminal output descriptor. Writing the report
# directly is more portable than process substitution/tee on restricted hosts.
exec 3>&1
printf 'Collecting acceptance preflight information...\n' >&3
exec >>"${output_file}" 2>&1

section() {
  printf '\n================================================================\n'
  printf '%s\n' "$1"
  printf '================================================================\n'
}

have() {
  command -v "$1" >/dev/null 2>&1
}

run() {
  local label="$1"
  shift
  printf '\n--- %s\n' "${label}"
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  "$@" 2>&1
  local status=$?
  if (( status != 0 )); then
    printf '[WARN] command exited with status %d\n' "${status}"
  fi
  return 0
}

missing() {
  printf '\n--- %s\n' "$1"
  printf '[NOT FOUND] %s\n' "$2"
}

path_status() {
  local label="$1"
  local path="$2"
  printf '%-24s %s\n' "${label}:" "${path:-<not detected>}"
  if [[ -z "${path}" ]]; then
    printf '%-24s %s\n' "${label} status:" 'NOT FOUND'
  elif [[ -e "${path}" ]]; then
    printf '%-24s %s\n' "${label} status:" 'EXISTS'
  else
    printf '%-24s %s\n' "${label} status:" 'NOT FOUND'
  fi
}

hash_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf '[NOT FOUND] %s\n' "${path}"
  elif have sha256sum; then
    sha256sum "${path}" 2>&1 || true
  elif have shasum; then
    shasum -a 256 "${path}" 2>&1 || true
  else
    printf '[NOT FOUND] neither sha256sum nor shasum is installed\n'
  fi
}

git_report() {
  local label="$1"
  local repo="$2"
  printf '\n--- %s: %s\n' "${label}" "${repo}"
  if ! have git; then
    printf '[NOT FOUND] git command\n'
  elif [[ ! -d "${repo}/.git" ]]; then
    printf '[NOT FOUND] Git repository\n'
  else
    git -C "${repo}" rev-parse HEAD 2>&1 || true
    printf 'branch: '
    git -C "${repo}" branch --show-current 2>&1 || true
    printf 'latest: '
    git -C "${repo}" log -1 --oneline 2>&1 || true
    printf 'working tree status:\n'
    local status
    status="$(git -C "${repo}" status --short 2>&1)"
    if [[ -n "${status}" ]]; then
      printf '%s\n' "${status}"
    else
      printf '(clean)\n'
    fi
  fi
}

python_probe_code='import platform
print("Python:", platform.python_version())
modules = ["torch", "numpy", "pytorch3d", "websockets", "pytest", "accelerate"]
for name in modules:
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, \"__version__\", \"installed; version unavailable\")}")
    except Exception as exc:
        print(f"{name}: NOT AVAILABLE ({type(exc).__name__}: {exc})")
try:
    import torch
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN:", torch.backends.cudnn.version())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {prop.name}; VRAM={prop.total_memory / 1024**3:.1f} GiB; capability={prop.major}.{prop.minor}")
except Exception as exc:
    print(f"PyTorch GPU probe failed: {type(exc).__name__}: {exc}")'

section 'V1 ACCEPTANCE PRE-FLIGHT REPORT'
printf 'Generated:              %s\n' "$(date --iso-8601=seconds 2>/dev/null || date 2>/dev/null || printf unknown)"
printf 'Hostname:               %s\n' "$(hostname 2>/dev/null || printf unknown)"
printf 'User:                   %s\n' "$(id -un 2>/dev/null || printf unknown)"
printf 'Current directory:      %s\n' "$(pwd -P 2>/dev/null || pwd)"
printf 'Report file:            %s\n' "${output_file}"
printf 'CUDA_VISIBLE_DEVICES:   %s\n' "${CUDA_VISIBLE_DEVICES:-<not set>}"

section '1. DETECTED PROJECT PATHS'
path_status 'REALTIME_ROOT' "${realtime_root}"
path_status 'DUET_EDGE_ROOT' "${duet_edge_root}"
path_status 'EDGE_CHECKPOINT' "${checkpoint_path}"
path_status 'AIST_RAW' "${aist_raw}"
printf '%-24s %s\n' 'Conda environment:' "${edge_env_name}"

printf '\nExpected key files:\n'
for path in \
  "${realtime_root}/docs/V1_ACCEPTANCE_EXECUTION_CN.md" \
  "${realtime_root}/configs/v1.cuda.json" \
  "${realtime_root}/configs/v1.fake.json" \
  "${duet_edge_root}/EDGE.py"; do
  if [[ -f "${path}" ]]; then
    printf '[OK]        %s\n' "${path}"
  else
    printf '[NOT FOUND] %s\n' "${path}"
  fi
done

section '2. OPERATING SYSTEM AND HARDWARE'
if [[ -r /etc/os-release ]]; then
  run '/etc/os-release' sed -n '1,80p' /etc/os-release
else
  missing '/etc/os-release' 'file is unavailable'
fi
have uname && run 'Kernel and architecture' uname -a || missing 'Kernel' 'uname command'
have lscpu && run 'CPU information' lscpu || missing 'CPU information' 'lscpu command'
have free && run 'Memory' free -h || missing 'Memory' 'free command'
have df && run 'Filesystem capacity' df -h || missing 'Filesystem capacity' 'df command'
have timedatectl && run 'Clock and timezone' timedatectl status || missing 'Clock and timezone' 'timedatectl command'
have uptime && run 'System load' uptime || missing 'System load' 'uptime command'

section '3. NVIDIA GPU AND DRIVER'
if have nvidia-smi; then
  run 'nvidia-smi overview' nvidia-smi
  run 'GPU inventory' nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv
  run 'GPU compute processes' nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
else
  missing 'NVIDIA information' 'nvidia-smi command; driver may be absent or not on PATH'
fi

section '4. CONDA AND PYTHON ENVIRONMENT'
conda_cmd=''
if have conda; then
  conda_cmd="$(command -v conda)"
else
  for candidate in \
    "${HOME:-}/miniconda3/bin/conda" \
    "${HOME:-}/anaconda3/bin/conda" \
    /opt/conda/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      conda_cmd="${candidate}"
      break
    fi
  done
fi

if [[ -n "${conda_cmd}" ]]; then
  run 'Conda version' "${conda_cmd}" --version
  run 'Conda environments' "${conda_cmd}" env list
  printf '\n--- Python probe via conda environment %s\n' "${edge_env_name}"
  "${conda_cmd}" run -n "${edge_env_name}" python -c "${python_probe_code}" 2>&1
  probe_status=$?
  if (( probe_status != 0 )); then
    printf '[WARN] Could not run Python in Conda environment %s (status %d)\n' "${edge_env_name}" "${probe_status}"
  fi
  printf '\n--- pip check via conda environment %s\n' "${edge_env_name}"
  "${conda_cmd}" run -n "${edge_env_name}" python -m pip check 2>&1 || printf '[WARN] pip check failed\n'
else
  missing 'Conda' 'conda command and common Conda installation paths'
fi

if have python3; then
  run 'System/default python3' python3 --version
elif have python; then
  run 'System/default python' python --version
else
  missing 'System Python' 'python3 and python commands'
fi

section '5. REPOSITORY VERSIONS'
git_report 'duet-edge-realtime' "${realtime_root}"
git_report 'duet-edge' "${duet_edge_root}"

section '6. CHECKPOINT AND INPUT MOTION'
printf '%s\n' 'Checkpoint:'
if [[ -f "${checkpoint_path}" ]]; then
  run 'Checkpoint file metadata' ls -lh "${checkpoint_path}"
  printf '\nCheckpoint SHA256:\n'
  hash_file "${checkpoint_path}"
else
  printf '[NOT FOUND] %s\n' "${checkpoint_path}"
fi

printf '\nAIST++ raw motion:\n'
if [[ -n "${aist_raw}" && -f "${aist_raw}" ]]; then
  run 'Motion file metadata' ls -lh "${aist_raw}"
  printf '\nMotion SHA256:\n'
  hash_file "${aist_raw}"
  case "${aist_raw}" in
    */motions_sliced/*) printf '[IMPORTANT] path is motions_sliced; verify whether --root-scaled true is required\n' ;;
    */motions/*) printf '[INFO] path is raw motions; expected acceptance setting is --root-scaled false\n' ;;
    *) printf '[WARN] path is not clearly under motions/ or motions_sliced/; verify --root-scaled manually\n' ;;
  esac
else
  printf '[NOT FOUND] No AIST_RAW file detected. Set AIST_RAW explicitly and run again.\n'
fi

section '7. PORTS, BROWSER, AND SLEEP SETTINGS'
if have ss; then
  printf '\n--- Listening status for 8080, 8765, and 18765\n'
  port_lines="$(ss -ltnp 2>&1 | awk '$4 ~ /:(8080|8765|18765)$/ {print}')"
  if [[ -n "${port_lines}" ]]; then
    printf '%s\n' "${port_lines}"
    printf '[WARN] One or more acceptance ports appear to be in use.\n'
  else
    printf '[OK] No listeners detected on acceptance ports.\n'
  fi
elif have lsof; then
  run 'Acceptance ports via lsof' lsof -nP -iTCP:8080 -iTCP:8765 -iTCP:18765 -sTCP:LISTEN
else
  missing 'Port check' 'ss and lsof commands'
fi

printf '\n--- Installed browsers\n'
browser_found=0
for browser in google-chrome google-chrome-stable chromium chromium-browser firefox; do
  if have "${browser}"; then
    printf '[OK] %s -> %s\n' "${browser}" "$(command -v "${browser}")"
    browser_found=1
  fi
done
if (( browser_found == 0 )); then
  printf '[NOT FOUND] No common browser command detected.\n'
fi

if have systemctl; then
  printf '\n--- Sleep-related targets\n'
  for target in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    printf '%-24s enabled=%-12s active=%s\n' \
      "${target}" \
      "$(systemctl is-enabled "${target}" 2>/dev/null || printf unknown)" \
      "$(systemctl is-active "${target}" 2>/dev/null || printf unknown)"
  done
else
  missing 'Sleep settings' 'systemctl command'
fi

section '8. CONFIGURATION SNAPSHOT'
for config in "${realtime_root}/configs/v1.cuda.json" "${realtime_root}/configs/v1.fake.json"; do
  printf '\n--- %s\n' "${config}"
  if [[ ! -f "${config}" ]]; then
    printf '[NOT FOUND]\n'
  elif have python3; then
    python3 -m json.tool "${config}" 2>&1 || printf '[WARN] invalid JSON or json.tool failed\n'
  elif have python; then
    python -m json.tool "${config}" 2>&1 || printf '[WARN] invalid JSON or json.tool failed\n'
  else
    sed -n '1,240p' "${config}" 2>&1 || true
  fi
done

section '9. SUMMARY / ITEMS TO REVIEW'
[[ -d "${realtime_root}" ]] || printf '[ACTION] Set REALTIME_ROOT to the realtime repository.\n'
[[ -f "${duet_edge_root}/EDGE.py" ]] || printf '[ACTION] Set DUET_EDGE_ROOT to the model repository containing EDGE.py.\n'
[[ -f "${checkpoint_path}" ]] || printf '[ACTION] Set EDGE_CHECKPOINT to train-1800.pt.\n'
[[ -n "${aist_raw}" && -f "${aist_raw}" ]] || printf '[ACTION] Set AIST_RAW to an actual AIST++ .pkl motion.\n'
[[ -n "${conda_cmd}" ]] || printf '[ACTION] Conda was not found; install/locate Conda before following the current acceptance guide.\n'
have nvidia-smi || printf '[ACTION] nvidia-smi was not found; check the NVIDIA driver and PATH.\n'
printf '\nReview warnings above, especially CUDA availability, GPU contention, VRAM, disk space, Git changes, root scaling, and occupied ports.\n'
printf 'This script is read-only except for writing this report file.\n'

section 'REPORT COMPLETE'
printf 'Saved to: %s\n' "${output_file}"
printf 'Preflight report saved to: %s\n' "${output_file}" >&3
