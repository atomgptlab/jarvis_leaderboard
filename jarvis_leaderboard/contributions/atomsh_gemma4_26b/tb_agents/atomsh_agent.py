"""Terminal-Bench adapter for atomsh.

Installs atomsh into the task container and runs one non-interactive prompt.
`source="local"` embeds a wheel built from a working tree, so an unreleased
change can be measured; `source="pypi"` measures the published build.
"""

import base64
import json
import os
import shlex
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand


def _stored_token() -> str:
    """The credential atomsh itself would use."""
    env = os.environ.get("ATOMSH_API_KEY")
    if env:
        return env
    path = Path(
        os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    ) / "atomsh" / "auth.json"
    try:
        return json.loads(path.read_text()).get("access_token", "")
    except (OSError, ValueError):
        return ""


class AtomshAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "atomsh"

    def __init__(self, model_name: str | None = None, source: str = "local",
                 materials: bool = False, wheel_path: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._source = source
        self._materials = materials
        self._wheel_path = wheel_path or os.environ.get("ATOMSH_WHEEL", "")
        self._version = kwargs.get("version", "latest")

    @property
    def _env(self) -> dict[str, str]:
        token = _stored_token()
        if not token:
            raise RuntimeError(
                "No atomsh credential. Set ATOMSH_API_KEY or run `atomsh login`."
            )
        env = {"ATOMSH_API_KEY": token}
        if os.environ.get("ATOMSH_API_BASE"):
            env["ATOMSH_API_BASE"] = os.environ["ATOMSH_API_BASE"]
        # A benchmark task is one long turn; do not stop it mid-way.
        env["ATOMSH_MAX_STEPS"] = os.environ.get("ATOMSH_MAX_STEPS", "200")
        return env

    def _get_template_variables(self) -> dict[str, str]:
        variables = {"version": self._version or "latest", "wheel_b64": "",
                     "wheel_name": ""}
        if self._source == "local":
            wheel = Path(self._wheel_path)
            if not wheel.is_file():
                raise RuntimeError(
                    f"source='local' needs a wheel; {wheel!s} is not a file. "
                    "Build one with `uv build --wheel` and pass wheel_path= "
                    "or set ATOMSH_WHEEL."
                )
            variables["wheel_name"] = wheel.name
            blob = base64.b64encode(wheel.read_bytes()).decode()
            variables["wheel_b64"] = "\n".join(
                blob[i:i + 76] for i in range(0, len(blob), 76)
            )
        return variables

    @property
    def _install_agent_script_path(self) -> Path:
        return self._get_templated_script_path("atomsh-setup.sh.j2")

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        flags = ["--yolo"]
        if not self._materials:
            # Pure coding work: a narrower tool surface is easier to aim.
            flags.append("--no-materials")
        if self._model_name:
            flags += ["-m", self._model_name.split("/")[-1]]
        command = f"atomsh {' '.join(flags)} {shlex.quote(instruction)}"
        return [
            TerminalCommand(
                command=command,
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
