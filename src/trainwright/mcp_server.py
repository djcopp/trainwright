"""MCP server exposing the trainwright lifecycle as tools (stdio transport).

Each tool shells out to ``trainwright --json <command>`` in the project directory, so the MCP
surface behaves exactly like the CLI and a crash in training cannot take the server down.

Register with Claude Code::

    claude mcp add trainwright -- trainwright mcp
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any


def _run(args: list[str], cwd: str | None = None, timeout: float | None = None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "trainwright.cli", "--json", *args]
    try:
        p = subprocess.run(
            cmd, cwd=cwd or os.getcwd(), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s; use train.time_budget"}
    try:  # in --json mode stdout is exactly one JSON document (progress goes to stderr)
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": p.returncode == 0, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}


def build_server() -> Any:
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server  # type: ignore[no-redef]

    srv = Server("trainwright")

    @srv.tool()
    def status(project_dir: str) -> dict[str, Any]:
        """Lifecycle checklist for a trainwright project and the next commands to run. Start here."""
        return _run(["status"], project_dir)

    @srv.tool()
    def guide() -> str:
        """The agent playbook: rules, lifecycle, and what to try per modality."""
        return _run(["guide"]).get("result", {}).get("playbook", "")

    @srv.tool()
    def init(
        project_dir: str, data: str, target: str | None = None, text_column: str | None = None
    ) -> dict[str, Any]:
        """Create trainwright.yaml for a dataset (CSV/Parquet, image folder, or manifest)."""
        args = (
            ["init", "--data", data]
            + (["--target", target] if target else [])
            + (["--text-column", text_column] if text_column else [])
        )
        return _run(args, project_dir)

    @srv.tool()
    def profile(project_dir: str) -> dict[str, Any]:
        """Profile the data: types, leakage suspects, imbalance, recommended metric."""
        return _run(["profile"], project_dir)

    @srv.tool()
    def split(project_dir: str) -> dict[str, Any]:
        """Freeze the train/val/test split."""
        return _run(["split"], project_dir)

    @srv.tool()
    def baseline(project_dir: str) -> dict[str, Any]:
        """Fit reference baselines (recorded as runs)."""
        return _run(["baseline"], project_dir)

    @srv.tool()
    def train(
        project_dir: str,
        hypothesis: str,
        overrides: list[str] | None = None,
        from_run: str | None = None,
    ) -> dict[str, Any]:
        """Train one run. overrides like ["optim.lr=3e-4", "model.params.dropout=0.3"]."""
        args = ["train", "--hypothesis", hypothesis]
        for o in overrides or []:
            args += ["-o", o]
        if from_run:
            args += ["--from-run", from_run]
        return _run(args, project_dir)

    @srv.tool()
    def runs(project_dir: str) -> dict[str, Any]:
        """Validation leaderboard."""
        return _run(["runs"], project_dir)

    @srv.tool()
    def diagnose(project_dir: str, run: str = "latest") -> dict[str, Any]:
        """Curve diagnosis with suggested next experiments."""
        return _run(["diagnose", run], project_dir)

    @srv.tool()
    def evaluate(
        project_dir: str, run: str = "best", split: str = "val", reason: str | None = None
    ) -> dict[str, Any]:
        """Error analysis on val, or the final one-time test evaluation (split='test')."""
        return _run(
            ["evaluate", run, "--split", split] + (["--reason", reason] if reason else []),
            project_dir,
        )

    @srv.tool()
    def export(project_dir: str, run: str = "best") -> dict[str, Any]:
        """Export an ONNX bundle with server, Dockerfile and model card."""
        return _run(["export", run, "--force"], project_dir)

    @srv.tool()
    def benchmark(project_dir: str, run: str = "best") -> dict[str, Any]:
        """Latency/throughput and cost estimate vs the LLM call."""
        return _run(["benchmark", run], project_dir)

    @srv.tool()
    def report(project_dir: str) -> dict[str, Any]:
        """Write reports/REPORT.md."""
        return _run(["report"], project_dir)

    return srv


def main() -> None:
    build_server().run()
