"""
Runs a replication with no time limit, and stops it only when it hangs.

A replication may take as long as the Parent Paper's method needs (user's
rule, 2026-11-18), so a slow run is never stopped. A hung one is: a process
that has done no work for STALL_SECONDS. Every POLL_SECONDS the watchdog
measures the work done since the last sign of progress -- the CPU time of the
process and all its children, their disk reads and writes, the GPU's use and
any new output -- and a run with less than MIN_WORK of it in a whole
STALL_SECONDS window is hung: its process tree is stopped and the result says
so (hung=True). A slow run that keeps working, however little, keeps going.

Every HEARTBEAT_SECONDS it prints a "Still running" line, so the console
(and the dashboard) shows that a long run is alive and what it is doing.

The measure comes from psutil. Without psutil there is no measure, so the
watchdog never stops a run (a false "hung" would kill real work); it still
prints the heartbeat. A Docker container is measured through `docker stats`
(docker_progress), and stopped with `docker rm -f` (on_kill).

run() returns the same dict shape as the Replication Agent's _run_bounded:
exit_code, stdout, stderr, duration_seconds, timed_out (always False), plus
hung and stalled_seconds.
"""
import shutil
import subprocess
import threading
import time

try:
    import psutil
except ImportError:  # no measure: never declare a hang
    psutil = None

POLL_SECONDS = 30
STALL_SECONDS = 15 * 60      # this long with (almost) no work at all is a hang
MIN_WORK = 5.0               # units of work (CPU seconds, MB of disk I/O, GPU-busy seconds) that count as progress
HEARTBEAT_SECONDS = 10 * 60
# A desktop's display keeps an idle GPU at a few percent; only use above this counts as work.
GPU_IDLE_PERCENT = 15.0


def _tree(proc):
    try:
        root = psutil.Process(proc.pid)
        return [root, *root.children(recursive=True)]
    except Exception:
        return []


def process_progress(proc) -> float | None:
    """CPU seconds plus MB of disk I/O of the process tree so far, or None
    when it cannot be measured."""
    if psutil is None:
        return None
    total, measured = 0.0, False
    for p in _tree(proc):
        try:
            cpu = p.cpu_times()
            total += cpu.user + cpu.system
            measured = True
        except Exception:
            continue
        try:
            io = p.io_counters()
            total += (io.read_bytes + io.write_bytes) / 1e6
        except Exception:
            pass
    return total if measured else None


def _gpu_busy_percent() -> float:
    """The NVIDIA GPU's current use (0-100), 0 when there is none."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0.0
    try:
        out = subprocess.run([smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip().splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else 0.0
    except Exception:
        return 0.0


def docker_progress(docker: str, name: str):
    """A progress measure for a named container: its CPU use over the poll
    interval (docker stats gives a percentage), as CPU seconds."""
    state = {"total": 0.0}

    def measure(_proc) -> float | None:
        try:
            out = subprocess.run([docker, "stats", "--no-stream", "--format", "{{.CPUPerc}}", name],
                                 capture_output=True, text=True, timeout=30)
            if out.returncode != 0 or not out.stdout.strip():
                return None
            state["total"] += float(out.stdout.strip().rstrip("%")) / 100 * POLL_SECONDS
            return state["total"]
        except Exception:
            return None
    return measure


def _kill_tree(proc) -> None:
    for p in reversed(_tree(proc)) if psutil is not None else []:
        try:
            p.kill()
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _reader(stream, sink: list, seen: dict) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
        seen["at"] = time.monotonic()
    stream.close()


def run(cmd: list, cwd=None, env: dict = None, label: str = "Run", progress=None, on_kill=None,
        stall_seconds: float = STALL_SECONDS, poll_seconds: float = POLL_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_SECONDS, min_work: float = MIN_WORK, popen_kwargs: dict = None) -> dict:
    """Run cmd to the end, however long; stop it only when it hangs (see the
    module docstring). Never raises."""
    start = time.monotonic()
    measure = progress or process_progress
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                **(popen_kwargs or {}))
    except Exception as e:
        return {"exit_code": None, "stdout": "", "stderr": str(e), "duration_seconds": 0.0,
                "timed_out": False, "hung": False, "stalled_seconds": 0.0}
    out, err, seen = [], [], {"at": start}
    readers = [threading.Thread(target=_reader, args=(proc.stdout, out, seen), daemon=True),
               threading.Thread(target=_reader, args=(proc.stderr, err, seen), daemon=True)]
    for t in readers:
        t.start()

    last_progress, work_at_progress, last_heartbeat = start, measure(proc), start
    gpu_work, hung = 0.0, False
    while proc.poll() is None:
        try:
            proc.wait(timeout=poll_seconds)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        work = measure(proc)
        gpu_work += max(0.0, _gpu_busy_percent() - GPU_IDLE_PERCENT) / 100 * poll_seconds
        if work is None or work_at_progress is None:
            # Not measurable: never a hang, only output resets the clock.
            last_progress, work_at_progress = (now, work) if work is not None else (last_progress, None)
        elif (work + gpu_work) - work_at_progress >= min_work or seen["at"] > last_progress:
            last_progress, work_at_progress, gpu_work = now, work, 0.0
        if now - last_heartbeat >= heartbeat_seconds:
            last_heartbeat = now
            print(f"[{label}] Still running ({_minutes(now - start)}); last sign of work "
                  f"{_minutes(now - last_progress)} ago; {len(out)} line(s) of output so far.", flush=True)
        if psutil is not None and work is not None and now - last_progress >= stall_seconds:
            hung = True
            print(f"[{label}] No work for {_minutes(now - last_progress)} (no CPU, disk, GPU or output): "
                  "the run is hung and is stopped.", flush=True)
            if on_kill:
                try:
                    on_kill()
                except Exception:
                    pass
            _kill_tree(proc)
            break
    try:
        proc.wait(timeout=60)
    except Exception:
        pass
    for t in readers:
        t.join(timeout=10)
    return {"exit_code": None if hung else proc.returncode, "stdout": "".join(out), "stderr": "".join(err),
            "duration_seconds": time.monotonic() - start, "timed_out": False, "hung": hung,
            "stalled_seconds": time.monotonic() - last_progress if hung else 0.0}


def _minutes(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"
