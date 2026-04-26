"""Parse skill definitions from various source formats."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Skill:
    name: str
    description: str
    source_file: str
    platform: str


def _discover_claude_skills(root: Path) -> List[Skill]:
    """Discover Claude Code skills from markdown files."""
    skills = []
    skill_dir = root / "claude-code"
    if not skill_dir.exists():
        return skills
    for md_file in sorted(skill_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        title = md_file.stem
        description = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped.lstrip("# ").strip()
            elif stripped.startswith("## 概述"):
                # Description is typically the next non-empty line
                for desc_line in lines[lines.index(line) + 1 :]:
                    if desc_line.strip():
                        description = desc_line.strip()
                        break
                break
        skills.append(
            Skill(
                name=md_file.stem,
                description=description,
                source_file=str(md_file.relative_to(root)),
                platform="claude-code",
            )
        )
    return skills


def _discover_codex_skills(root: Path) -> List[Skill]:
    """Discover Codex skills from skills.json."""
    skills = []
    json_file = root / "codex" / "skills.json"
    if not json_file.exists():
        return skills
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for item in data.get("skills", []):
            # Extract first line of prompt as description if no explicit description
            desc = item.get("description", "")
            prompt = item.get("prompt", "")
            if not desc and prompt:
                desc = prompt.split("\n")[0][:120]
            skills.append(
                Skill(
                    name=item.get("name", ""),
                    description=desc,
                    source_file=str(json_file.relative_to(root)),
                    platform="codex",
                )
            )
    except (json.JSONDecodeError, OSError):
        pass
    return skills


def _discover_openclaw_skills(root: Path) -> List[Skill]:
    """Discover OpenClaw skills from skills.md."""
    skills = []
    md_file = root / "openclaw" / "skills.md"
    if not md_file.exists():
        return skills
    content = md_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## ") and not line.startswith("### "):
            name = line.lstrip("## ").strip()
            description = ""
            # Look for **Prompt** section
            for j in range(i + 1, min(i + 20, len(lines))):
                if lines[j].startswith("**Prompt**:"):
                    prompt_line = lines[j].strip()
                    # Prompt content may be on the same line or next line
                    if len(prompt_line) > len("**Prompt**:"):
                        description = prompt_line[len("**Prompt**:"):].strip()[:120]
                    else:
                        for k in range(j + 1, min(j + 5, len(lines))):
                            stripped = lines[k].strip()
                            if stripped and not stripped.startswith("-") and not stripped.startswith("**"):
                                description = stripped[:120]
                                break
                    break
            skills.append(
                Skill(
                    name=name,
                    description=description,
                    source_file=str(md_file.relative_to(root)),
                    platform="openclaw",
                )
            )
        i += 1
    return skills


def discover_all_skills(root: Optional[Path] = None) -> List[Skill]:
    """Discover all skills across all platforms."""
    if root is None:
        root = Path(__file__).resolve().parent.parent
    skills = []
    skills.extend(_discover_claude_skills(root))
    skills.extend(_discover_codex_skills(root))
    skills.extend(_discover_openclaw_skills(root))
    return skills


def find_skill_file(name: str, root: Optional[Path] = None) -> Optional[Path]:
    """Find the source file for a skill by name."""
    if root is None:
        root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "claude-code" / f"{name}.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
