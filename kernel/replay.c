/*
 * replay.c: Kernel Baseline for the simulator in cpu-scheduling-algorithm-comparison.ipynb.
 *
 * Replays a job stream (workload CSV written by the notebook) as real processes pinned to one
 * CPU under one Linux scheduling policy. Every job is forked before T0, pinned, given its policy,
 * and sleeps until its arrival (T0 + arrival * unit). It then spins until it has consumed
 * burst * unit of its own CPU time and exits. One Time Unit = 100 us by default.
 *
 *   replay WORKLOAD.csv POLICY CPU OUT.csv [JOBS] [UNIT_NS]
 *
 * POLICY: fifo        SCHED_FIFO, all jobs priority 1                  (FCFS)
 *         fifo-prio   SCHED_FIFO, priority 1 + rank of Borg Priority   (Priority, preemptive)
 *         rr          SCHED_RR, priority 1; quantum = sched_rr_timeslice_ms (RR)
 *         deadline    SCHED_DEADLINE, runtime 1.05*burst, deadline = period = Relative Deadline (EDF)
 *         other       SCHED_OTHER, default slice                       (Shipped EEVDF)
 *         other-slice SCHED_OTHER, 1 ms custom slice via sched_runtime (Matched-slice EEVDF)
 *
 * Output: one row per job. Times are ns relative to T0; the context-switch counts are deltas
 * from the job's arrival to its completion (getrusage).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef SCHED_DEADLINE
#define SCHED_DEADLINE 6
#endif

struct attr {                 /* struct sched_attr, declared locally to avoid libc version skew */
    uint32_t size, sched_policy;
    uint64_t sched_flags;
    int32_t sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime, sched_deadline, sched_period;
    uint32_t sched_util_min, sched_util_max;
};

struct job {
    double arrival, burst, deadline;   /* Time Units; deadline is absolute, as in the CSV */
    long borg_prio;
    int rt_prio, err, ready, done;
    long long release, start, finish, cpu; /* ns */
    long nvcsw, nivcsw;
};

static long long now(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}

static int cmp_long(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}

static int set_policy(const char *policy, struct job *j, long long unit) {
    struct attr a = {.size = sizeof a};
    if (!strcmp(policy, "fifo") || !strcmp(policy, "rr") || !strcmp(policy, "fifo-prio")) {
        a.sched_policy = strcmp(policy, "rr") ? SCHED_FIFO : SCHED_RR;
        a.sched_priority = strcmp(policy, "fifo-prio") ? 1 : j->rt_prio;
    } else if (!strcmp(policy, "deadline")) {
        a.sched_policy = SCHED_DEADLINE;
        a.sched_runtime = (uint64_t)(1.05 * j->burst * unit);
        a.sched_deadline = a.sched_period = (uint64_t)((j->deadline - j->arrival) * unit);
    } else if (!strcmp(policy, "other")) {
        a.sched_policy = SCHED_OTHER;
    } else if (!strcmp(policy, "other-slice")) {
        a.sched_policy = SCHED_OTHER;
        a.sched_runtime = 1000000;            /* 1 ms custom slice (EEVDF, Linux >= 6.12) */
    } else {
        errno = EINVAL;
        return -1;
    }
    return syscall(SYS_sched_setattr, 0, &a, 0);
}

static void run_job(struct job *j, const char *policy, int cpu, long long unit, long long t0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    prctl(PR_SET_TIMERSLACK, 1UL);            /* no wake-up slack for SCHED_OTHER */
    if (sched_setaffinity(0, sizeof set, &set) || set_policy(policy, j, unit)) {
        j->err = errno;
        __atomic_store_n(&j->ready, 1, __ATOMIC_SEQ_CST);
        _exit(1);
    }
    __atomic_store_n(&j->ready, 1, __ATOMIC_SEQ_CST);

    j->release = (long long)(j->arrival * unit);
    long long wake = t0 + j->release;
    struct timespec ts = {wake / 1000000000LL, wake % 1000000000LL};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR)
        ;
    j->start = now(CLOCK_MONOTONIC) - t0;     /* first time on the CPU after arrival */
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    long long c0 = now(CLOCK_THREAD_CPUTIME_ID), need = (long long)(j->burst * unit);
    while (now(CLOCK_THREAD_CPUTIME_ID) - c0 < need)
        ;
    j->cpu = now(CLOCK_THREAD_CPUTIME_ID) - c0;
    j->finish = now(CLOCK_MONOTONIC) - t0;
    getrusage(RUSAGE_SELF, &r1);
    j->nvcsw = r1.ru_nvcsw - r0.ru_nvcsw;
    j->nivcsw = r1.ru_nivcsw - r0.ru_nivcsw;
    j->done = 1;
    _exit(0);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s WORKLOAD.csv POLICY CPU OUT.csv [JOBS] [UNIT_NS]\n", argv[0]);
        return 2;
    }
    const char *policy = argv[2];
    int cpu = atoi(argv[3]);
    int max_jobs = argc > 5 ? atoi(argv[5]) : 1 << 30;
    long long unit = argc > 6 ? atoll(argv[6]) : 100000;

    FILE *in = fopen(argv[1], "r");
    if (!in) { perror(argv[1]); return 2; }
    char line[512];
    if (!fgets(line, sizeof line, in) || strncmp(line, "Arrival_Time,Burst_Time,Priority,Deadline", 41)) {
        fprintf(stderr, "unexpected header in %s\n", argv[1]);
        return 2;
    }
    int n = 0, cap = 1024;
    struct job *jobs = mmap(NULL, cap * sizeof *jobs, PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (jobs == MAP_FAILED) { perror("mmap"); return 2; }
    while (n < max_jobs && fgets(line, sizeof line, in)) {
        struct job *j = &jobs[n];
        if (sscanf(line, "%lf,%lf,%ld,%lf", &j->arrival, &j->burst, &j->borg_prio, &j->deadline) != 4) {
            fprintf(stderr, "bad row %d: %s", n + 1, line);
            return 2;
        }
        if (++n == cap) { fprintf(stderr, "more than %d jobs\n", cap); return 2; }
    }
    fclose(in);

    /* fifo-prio: RT priority 1 + rank of the job's Borg Priority among the distinct values
       (larger Borg Priority = more important = higher RT priority). Stays below 50 so
       threaded IRQ handlers (SCHED_FIFO 50) still preempt the jobs. */
    long *distinct = malloc(n * sizeof *distinct);
    for (int i = 0; i < n; i++) distinct[i] = jobs[i].borg_prio;
    qsort(distinct, n, sizeof *distinct, cmp_long);
    int k = 0;
    for (int i = 0; i < n; i++)
        if (!k || distinct[i] != distinct[k - 1]) distinct[k++] = distinct[i];
    if (k > 48) { fprintf(stderr, "%d distinct Borg priorities, more than 48 RT levels\n", k); return 2; }
    for (int i = 0; i < n; i++)
        for (int r = 0; r < k; r++)
            if (distinct[r] == jobs[i].borg_prio) jobs[i].rt_prio = 1 + r;

    long long t0 = now(CLOCK_MONOTONIC) + 2000000000LL + n * 2000000LL;   /* 2 s + 2 ms per job */
    pid_t *pids = calloc(n, sizeof *pids);
    for (int i = 0; i < n; i++) {
        pids[i] = fork();
        if (pids[i] < 0) { perror("fork"); goto abort; }
        if (pids[i] == 0) run_job(&jobs[i], policy, cpu, unit, t0);
    }
    for (int i = 0; i < n; i++) {
        while (!__atomic_load_n(&jobs[i].ready, __ATOMIC_SEQ_CST)) {
            if (now(CLOCK_MONOTONIC) > t0 - 100000000LL) {
                fprintf(stderr, "job %d not ready 100 ms before T0\n", i);
                goto abort;
            }
            usleep(1000);
        }
        if (jobs[i].err) {
            fprintf(stderr, "job %d: cannot set %s on CPU %d: %s\n", i, policy, cpu, strerror(jobs[i].err));
            goto abort;
        }
    }

    int failed = 0;
    for (int i = 0; i < n; i++) {
        int st;
        waitpid(pids[i], &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) || !jobs[i].done) failed++;
    }
    if (failed) { fprintf(stderr, "%d jobs did not complete\n", failed); return 1; }

    FILE *out = fopen(argv[4], "w");
    if (!out) { perror(argv[4]); return 2; }
    fprintf(out, "job,arrival_units,burst_units,borg_priority,rel_deadline_units,rt_priority,"
                 "release_ns,start_ns,finish_ns,cpu_ns,nvcsw,nivcsw\n");
    for (int i = 0; i < n; i++) {
        struct job *j = &jobs[i];
        fprintf(out, "%d,%.0f,%.0f,%ld,%.3f,%d,%lld,%lld,%lld,%lld,%ld,%ld\n", i, j->arrival, j->burst,
                j->borg_prio, j->deadline - j->arrival, j->rt_prio, j->release, j->start, j->finish,
                j->cpu, j->nvcsw, j->nivcsw);
    }
    fclose(out);
    return 0;

abort:
    for (int i = 0; i < n; i++)
        if (pids[i] > 0) kill(pids[i], SIGKILL);
    while (wait(NULL) > 0)
        ;
    return 1;
}
