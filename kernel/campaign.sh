#!/usr/bin/env bash
# Kernel Baseline campaign: replays workload_seed42.csv under each Linux scheduling policy.
#
#   Full run  (~3.4 h):  sudo nohup ./campaign.sh > campaign.log 2>&1 &
#   Smoke run (~4 min):  sudo JOBS=10 ./campaign.sh
#
# 3 lanes = 3 CPUs on separate physical cores; each batch runs one policy's 3 replicates in
# parallel. SCHED_RR's quantum is one global sysctl, so each RR quantum is its own batch.
# Changes 4 kernel settings and restores them on any exit (trap):
#   sched_rr_timeslice_ms         RR quantum (1..1100 ms)
#   sched_rt_runtime_us = -1      no RT throttling, no SCHED_DEADLINE admission control
#                                 (lets DL tasks be admitted at rho ~ 30 and pinned to one CPU)
#   sched_deadline_period_max_us  raised so Relative Deadlines up to ~150 s are accepted
#   cpufreq governor = performance
set -euo pipefail
cd "$(dirname "$0")"

LANES=(6 8 10)                 # CPUs on physical cores 3, 4, 5 (SMT pairs are 2k, 2k+1); core 0 left to the system
HOUSEKEEPING=0,1               # where the launchers run
REPS=${#LANES[@]}
RR_QUANTA_MS=(1 2 5 10 20 50 100 1000 1100)   # 1 s sits just below the 1 s modal burst (tick-based slice), 1.1 s just above
UNIT_NS=100000                 # one Time Unit = 100 us
JOBS=${JOBS:-500}
WORKLOAD=workload_seed42.csv
OUT=${OUT:-results/$(date +%Y%m%d-%H%M%S)$([ "$JOBS" = 500 ] || echo "-smoke$JOBS")}

log() { echo "[$(date '+%F %T')] $*"; }

[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
[ -x replay ] || { echo "build first: gcc -O2 -Wall -o replay replay.c"; exit 1; }
load=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($load > 1.5)}"; then
    echo "machine is busy (load $load); stop other jobs first. Top CPU users:"
    ps -eo pid,pcpu,comm --sort=-pcpu | head -6
    exit 1
fi

K=/proc/sys/kernel
orig_rr=$(cat $K/sched_rr_timeslice_ms)
orig_rt=$(cat $K/sched_rt_runtime_us)
orig_dlmax=$(cat $K/sched_deadline_period_max_us)
declare -A orig_gov
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do orig_gov[$g]=$(cat "$g"); done

restore() {
    set +e
    pkill -KILL -x replay              # launchers; their jobs die with them (PR_SET_PDEATHSIG)
    sleep 1
    echo "$orig_rr" > $K/sched_rr_timeslice_ms
    for i in 1 2 3 4 5; do             # EBUSY while any DL task still exists
        echo "$orig_rt" > $K/sched_rt_runtime_us 2>/dev/null && break
        sleep 1
    done
    echo "$orig_dlmax" > $K/sched_deadline_period_max_us
    for g in "${!orig_gov[@]}"; do echo "${orig_gov[$g]}" > "$g"; done
    [ -n "${SUDO_USER:-}" ] && [ -d results ] && chown -R "$SUDO_USER": results   # so analyze.py can write
    log "restored: rr_timeslice=$(cat $K/sched_rr_timeslice_ms) rt_runtime=$(cat $K/sched_rt_runtime_us)" \
        "dl_period_max=$(cat $K/sched_deadline_period_max_us) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

echo -1        > $K/sched_rt_runtime_us
echo 200000000 > $K/sched_deadline_period_max_us   # 200 s
for g in "${!orig_gov[@]}"; do echo performance > "$g"; done
renice --priority 0 -p $$ > /dev/null                # absolute nice 0, so EEVDF jobs run at nice 0

mkdir -p "$OUT"
{
    date; uname -a; lscpu
    echo "jobs=$JOBS unit_ns=$UNIT_NS lanes=${LANES[*]} rr_quanta_ms=${RR_QUANTA_MS[*]}"
    sha256sum "$WORKLOAD" replay.c
    for f in sched_rt_runtime_us sched_rt_period_us sched_deadline_period_max_us sched_autogroup_enabled; do
        echo "$f=$(cat $K/$f)"
    done
    mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug || true
    echo "eevdf_base_slice_ns=$(cat /sys/kernel/debug/sched/base_slice_ns 2>&1)"
    echo "loadavg=$(cat /proc/loadavg)"
} > "$OUT/env.txt" 2>&1

# batch NAME POLICY [lanes...]: one replicate per lane, in parallel
batch() {
    local name=$1 policy=$2; shift 2
    local lanes=("$@") pids=() r=0
    log "start $name on CPUs ${lanes[*]}"
    for cpu in "${lanes[@]}"; do
        r=$((r + 1))
        taskset -c $HOUSEKEEPING ./replay "$WORKLOAD" "$policy" "$cpu" \
            "$OUT/${name}_rep${r}_cpu${cpu}.csv" "$JOBS" "$UNIT_NS" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { log "FAILED: $name"; exit 1; }; done
    log "done  $name"
}

echo 1 > $K/sched_rr_timeslice_ms   # irrelevant to non-RR policies; set for the record
batch fifo        fifo        "${LANES[@]}"
batch fifo-prio   fifo-prio   "${LANES[@]}"
batch deadline    deadline    "${LANES[@]}"
batch eevdf       other       "${LANES[@]}"
batch eevdf-slice other-slice "${LANES[@]}"
for q in "${RR_QUANTA_MS[@]}"; do
    echo "$q" > $K/sched_rr_timeslice_ms
    batch "rr-q${q}ms" rr "${LANES[@]}"
done
batch fifo-alone  fifo "${LANES[0]}"     # interference check: same policy, no parallel lanes

log "campaign complete: $OUT"
