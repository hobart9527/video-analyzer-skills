"""Install skills to target AI assistant platforms."""

import shutil
import sys
from pathlib import Path
from typing import List, Optional


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _install_claude_code(
    dest: Optional[Path] = None,
    skills: Optional[List[str]] = None,
    project: bool = False,
) -> List[str]:
    """Install Claude Code skills. Returns list of installed paths."""
    results = []
    root = _get_project_root()
    source_dir = root / "claude-code"
    if not source_dir.exists():
        return results

    if dest is None:
        if project:
            dest = Path.cwd() / ".claude" / "skills"
        else:
            dest = Path.home() / ".claude" / "skills"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for md_file in sorted(source_dir.glob("*.md")):
        if skills and md_file.stem not in skills:
            continue
        target = dest / md_file.name
        shutil.copy2(md_file, target)
        results.append(str(target))
    return results


def _install_codex(
    dest: Optional[Path] = None,
    skills: Optional[List[str]] = None,
    project: bool = False,
) -> List[str]:
    """Install Codex skills. Returns list of installed paths."""
    results = []
    root = _get_project_root()
    source = root / "codex" / "skills.json"
    if not source.exists():
        return results

    if dest is None:
        if project:
            dest = Path.cwd() / ".codex"
        else:
            dest = Path.home() / ".codex"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "skills.json"
    shutil.copy2(source, target)
    results.append(str(target))
    return results


def _install_openclaw(
    dest: Optional[Path] = None,
    skills: Optional[List[str]] = None,
    project: bool = False,
) -> List[str]:
    """Install OpenClaw skills. Returns list of installed paths."""
    results = []
    root = _get_project_root()
    source = root / "openclaw" / "skills.md"
    if not source.exists():
        return results

    if dest is None:
        dest = Path.cwd() / ".openclaw"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "skills.md"
    shutil.copy2(source, target)
    results.append(str(target))
    return results


TARGETS = {
    "claude-code": _install_claude_code,
    "codex": _install_codex,
    "openclaw": _install_openclaw,
}


def install(
    target: str,
    dest: Optional[Path] = None,
    skills: Optional[List[str]] = None,
    project: bool = False,
) -> List[str]:
    """Install skills to the specified target.

    Args:
        target: One of "claude-code", "codex", "openclaw", or "all".
        dest: Optional destination directory. Uses default if None.
        skills: Optional list of skill names to filter. All if None.
        project: If True, install to project-level dirs instead of global.

    Returns:
        List of installed file paths.
    """
    installed = []
    kwargs = {"dest": dest, "skills": skills, "project": project}
    if target == "all":
        for t in TARGETS:
            installed.extend(TARGETS[t](**kwargs))
    elif target in TARGETS:
        installed.extend(TARGETS[target](**kwargs))
    else:
        raise ValueError(f"Unknown target: {target}. Choose from: {', '.join(TARGETS)} or all")
    return installed


def detect_targets() -> dict:
    """Detect which AI assistant tools are installed and their skill paths."""
    info = {}
    # Claude Code
    claude_global = Path.home() / ".claude" / "skills"
    info["claude-code"] = {
        "installed": claude_global.exists(),
        "global_path": str(claude_global),
        "project_path": str(Path.cwd() / ".claude" / "skills"),
    }
    # Codex
    codex_global = Path.home() / ".codex"
    info["codex"] = {
        "installed": codex_global.exists(),
        "global_path": str(codex_global),
        "project_path": str(Path.cwd() / ".codex"),
    }
    # OpenClaw
    openclaw_project = Path.cwd() / ".openclaw"
    info["openclaw"] = {
        "installed": openclaw_project.exists(),
        "global_path": "N/A (project-level only)",
        "project_path": str(openclaw_project),
    }
    return info
