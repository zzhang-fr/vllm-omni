"""Parse per-stage timings out of the vllm-omni Docker container logs.

This is a workaround for a vllm-omni bug: --enable-diffusion-pipeline-profiler
populates the pipeline's internal _stage_durations dict (visible in stderr
via DiffusionPipelineProfiler log lines), but does not forward it to
VideoGenerationArtifacts.stage_durations in the API response. We scrape the
logs until/unless we patch that plumbing upstream.

Each profiler line has the shape:
    [DiffusionPipelineProfiler] <PipelineClass>.<stage> took <duration>s

Usage:
    from _log_profile import read_profile_lines, group_runs
    lines = read_profile_lines("vllm-omni-wm-server")
    runs = group_runs(lines)           # list of dicts, newest last
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

LINE_RE = re.compile(
    r"INFO (?P<date>\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) "
    r"\[diffusion_pipeline_profiler\.py:\d+\] \[DiffusionPipelineProfiler\] "
    r"(?P<cls>\w+)\.(?P<stage>[\w.]+) took (?P<seconds>[\d.]+)s"
)


@dataclass
class ProfileLine:
    ts: datetime
    cls: str
    stage: str
    seconds: float


@dataclass
class ProfileRun:
    pipeline: str
    durations: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def add(self, p: ProfileLine) -> None:
        key = p.stage
        # Accumulate if same stage appears multiple times (e.g. text_encoder
        # for positive and negative CFG branches).
        if key in self.durations:
            self.durations[key] += p.seconds
            self.counts[key] += 1
        else:
            self.durations[key] = p.seconds
            self.counts[key] = 1
        if self.started_at is None or p.ts < self.started_at:
            self.started_at = p.ts
        if self.ended_at is None or p.ts > self.ended_at:
            self.ended_at = p.ts

    @property
    def wall_s(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds()


def read_profile_lines(container: str) -> list[ProfileLine]:
    out = subprocess.run(
        ["docker", "logs", container],
        capture_output=True, text=True, check=True,
    )
    # docker logs merges stdout+stderr; profiler lines are in stderr
    text = out.stdout + out.stderr
    year = datetime.now().year
    lines: list[ProfileLine] = []
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        ts = datetime.strptime(
            f"{year}-{m['date']} {m['time']}", "%Y-%m-%d %H:%M:%S"
        )
        lines.append(ProfileLine(
            ts=ts, cls=m["cls"], stage=m["stage"], seconds=float(m["seconds"])
        ))
    return lines


def group_runs(lines: Iterable[ProfileLine]) -> list[ProfileRun]:
    """Group lines into runs by logical stage sequence.

    Each run starts at the first text_encoder.forward seen after a
    previous run has emitted its final vae.decode. This is more robust
    than time-gap grouping because one inference can span >60s between
    vae.encode and vae.decode (the DiT denoising loop).
    """
    runs: list[ProfileRun] = []
    current: ProfileRun | None = None
    saw_decode = True  # start ready to open a new run
    for p in sorted(lines, key=lambda x: x.ts):
        if p.stage == "text_encoder.forward" and saw_decode:
            current = ProfileRun(pipeline=p.cls)
            runs.append(current)
            saw_decode = False
        if current is None:
            # tolerate missing opening text_encoder (e.g. warmup)
            current = ProfileRun(pipeline=p.cls)
            runs.append(current)
        current.add(p)
        if p.stage == "vae.decode":
            saw_decode = True
    return runs


def latest_run(container: str) -> ProfileRun | None:
    runs = group_runs(read_profile_lines(container))
    return runs[-1] if runs else None
