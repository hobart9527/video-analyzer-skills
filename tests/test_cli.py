"""Tests for video_analyzer_skills CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from video_analyzer_skills.cli import cli


def test_list_command():
    """Test `vas list` shows all skills."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "[claude-code]" in result.output
    assert "Total:" in result.output
    assert "skill(s)" in result.output


def test_list_filter_platform():
    """Test `vas list --platform claude-code`."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--platform", "claude-code"])
    assert result.exit_code == 0
    assert "[claude-code]" in result.output
    assert "[codex]" not in result.output


def test_show_skill_exists():
    """Test `vas show gpu-auto-config`."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "gpu-auto-config"])
    assert result.exit_code == 0
    assert "gpu-auto-config" in result.output


def test_show_skill_not_found():
    """Test `vas show nonexistent` exits with error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_show_skill_codex():
    """Test `vas show gpu-auto-config --platform codex`."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "gpu-auto-config", "--platform", "codex"])
    assert result.exit_code == 0
    assert "gpu-auto-config" in result.output
    assert "Auto-detect" in result.output


def test_show_skill_openclaw():
    """Test `vas show gpu-auto-config --platform openclaw`."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "gpu-auto-config", "--platform", "openclaw"])
    assert result.exit_code == 0
    assert "gpu-auto-config" in result.output
    assert "Trigger" in result.output


def test_doctor_command():
    """Test `vas doctor` runs without error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Environment Check" in result.output


def test_render_check():
    """Test `vas render --check` passes when all platforms in sync."""
    runner = CliRunner()
    result = runner.invoke(cli, ["render", "--check"])
    assert result.exit_code == 0
    assert "All platforms are in sync." in result.output


def test_install_claude_code_global():
    """Test `vas install --target claude-code` to temp dir."""
    runner = CliRunner()
    with runner.isolated_filesystem() as fs:
        dest = Path(fs) / "skills"
        result = runner.invoke(cli, ["install", "--target", "claude-code", "--dest", str(dest)])
        assert result.exit_code == 0
        assert "Successfully installed" in result.output
        assert (dest / "gpu-auto-config.md").exists()


def test_install_project_mode():
    """Test `vas install --target claude-code --project`."""
    runner = CliRunner()
    with runner.isolated_filesystem() as fs:
        result = runner.invoke(
            cli, ["install", "--target", "claude-code", "--project"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (Path(fs) / ".claude" / "skills" / "gpu-auto-config.md").exists()


def test_install_with_skill_filter():
    """Test installing only specific skills."""
    runner = CliRunner()
    with runner.isolated_filesystem() as fs:
        dest = Path(fs) / "skills"
        result = runner.invoke(cli, [
            "install", "--target", "claude-code", "--dest", str(dest),
            "gpu-auto-config",
        ])
        assert result.exit_code == 0
        installed = list(dest.glob("*.md"))
        assert len(installed) == 1
        assert installed[0].name == "gpu-auto-config.md"
