"""CLI for video-analyzer-skills."""

import click

from video_analyzer_skills import __version__
from video_analyzer_skills.installer import detect_targets, install
from video_analyzer_skills.parser import discover_all_skills, find_skill_file


@click.group()
@click.version_option(version=__version__, prog_name="vas")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Video Analyzer Skills CLI.

    Manage and install optimization skills for Claude Code, Codex, and OpenClaw.

    Quick start:
      vas list              List all available skills
      vas show <skill>      View a skill's content
      vas install --target claude-code    Install to Claude Code
      vas doctor            Check environment
    """
    ctx.ensure_object(dict)


@cli.command("list")
@click.option("--platform", "plat", type=click.Choice(["claude-code", "codex", "openclaw"]),
              help="Filter by platform")
def list_skills(plat: str | None) -> None:
    """List all available skills."""
    skills = discover_all_skills()
    if not skills:
        click.echo("No skills found.")
        return

    by_platform: dict[str, list] = {}
    for skill in skills:
        if plat and skill.platform != plat:
            continue
        by_platform.setdefault(skill.platform, []).append(skill)

    for platform, skill_list in sorted(by_platform.items()):
        click.echo(click.style(f"\n[{platform}]", fg="green", bold=True))
        for skill in skill_list:
            click.echo(f"  {click.style(skill.name, fg='cyan')}")
            if skill.description:
                click.echo(f"    {skill.description[:80]}")

    total = sum(len(v) for v in by_platform.values())
    click.echo(f"\nTotal: {total} skill(s)")


@cli.command("show")
@click.argument("name")
@click.option("--platform", "plat", type=click.Choice(["claude-code", "codex", "openclaw"]),
              default="claude-code", show_default=True,
              help="Platform to look up the skill in")
def show_skill(name: str, plat: str) -> None:
    """Show the content of a specific skill."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if plat == "claude-code":
        source = root / "claude-code" / f"{name}.md"
    elif plat == "codex":
        source = root / "codex" / "skills.json"
    else:
        source = root / "openclaw" / "skills.md"

    if not source.exists():
        # Fallback: try to find by name
        found = find_skill_file(name)
        if found:
            source = found
        else:
            click.echo(click.style(f"Skill '{name}' not found.", fg="red"))
            click.echo(f"Run 'vas list' to see available skills.")
            raise click.Exit(1)

    content = source.read_text(encoding="utf-8")
    click.echo(content)


@cli.command("install")
@click.option("--target", "target", type=click.Choice(["claude-code", "codex", "openclaw", "all"]),
              default="all", show_default=True,
              help="Target platform to install skills to")
@click.option("--dest", type=click.Path(),
              help="Override default installation directory")
@click.option("--project", is_flag=True,
              help="Install to project-level directories (.claude/skills, .codex, .openclaw)")
@click.argument("skills", nargs=-1)
def install_skills(target: str, dest: str | None, project: bool, skills: tuple[str, ...]) -> None:
    """Install skills to the target platform.

    SKILLS are optional skill names to filter (e.g. 'gpu-auto-config').
    If omitted, all skills are installed.
    """
    dest_path = None
    if dest:
        from pathlib import Path
        dest_path = Path(dest)

    skill_list = list(skills) if skills else None

    try:
        installed = install(target=target, dest=dest_path, skills=skill_list, project=project)
    except ValueError as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        raise click.Exit(1)

    if not installed:
        click.echo(click.style("No files were installed.", fg="yellow"))
        return

    click.echo(click.style(f"Successfully installed {len(installed)} file(s):", fg="green"))
    for path in installed:
        click.echo(f"  {path}")


@cli.command("doctor")
def doctor() -> None:
    """Check environment and detect installed AI assistant tools."""
    click.echo(click.style("Environment Check\n", fg="green", bold=True))

    targets = detect_targets()
    for platform, info in targets.items():
        status = "installed" if info["installed"] else "not detected"
        color = "green" if info["installed"] else "yellow"
        click.echo(f"  {click.style(platform, fg='cyan')}: {click.style(status, fg=color)}")
        click.echo(f"    Global:  {info['global_path']}")
        click.echo(f"    Project: {info['project_path']}")

    click.echo(click.style("\nRecommendations:", fg="blue", bold=True))
    any_detected = any(info["installed"] for info in targets.values())
    if not any_detected:
        click.echo("  No AI assistant tools detected. Install Claude Code, Codex, or OpenClaw first.")
    else:
        click.echo("  Run 'vas install --target <platform>' to install skills.")


@cli.command("render")
@click.option("--check", is_flag=True, help="Check that all platforms have consistent skills")
def render(check: bool) -> None:
    """Render or verify skills across platforms."""
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parent.parent
    claude_dir = root / "claude-code"
    codex_file = root / "codex" / "skills.json"
    openclaw_file = root / "openclaw" / "skills.md"

    claude_skills = {f.stem for f in claude_dir.glob("*.md")} if claude_dir.exists() else set()

    codex_skills = set()
    if codex_file.exists():
        try:
            data = json.loads(codex_file.read_text())
            codex_skills = {s["name"] for s in data.get("skills", [])}
        except (json.JSONDecodeError, KeyError):
            pass

    openclaw_skills = set()
    if openclaw_file.exists():
        content = openclaw_file.read_text()
        for line in content.splitlines():
            if line.startswith("## ") and not line.startswith("### "):
                openclaw_skills.add(line.lstrip("## ").strip())

    click.echo(click.style("Skill coverage by platform:", fg="green", bold=True))
    all_names = sorted(claude_skills | codex_skills | openclaw_skills)
    for name in all_names:
        c = "yes" if name in claude_skills else "no"
        x = "yes" if name in codex_skills else "no"
        o = "yes" if name in openclaw_skills else "no"
        c_color = "green" if c == "yes" else "red"
        x_color = "green" if x == "yes" else "red"
        o_color = "green" if o == "yes" else "red"
        click.echo(
            f"  {name:40s} "
            f"claude={click.style(c, fg=c_color)} "
            f"codex={click.style(x, fg=x_color)} "
            f"openclaw={click.style(o, fg=o_color)}"
        )

    if check:
        missing = []
        for name in all_names:
            if name not in claude_skills or name not in codex_skills or name not in openclaw_skills:
                missing.append(name)
        if missing:
            click.echo(click.style(f"\nMissing in some platforms: {', '.join(missing)}", fg="red"))
            raise click.Exit(1)
        else:
            click.echo(click.style("\nAll platforms are in sync.", fg="green"))


def main() -> None:
    cli()
