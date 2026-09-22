"""The release workflow, checked without spending a Windows runner on it.

Every step in it runs under `pwsh`, and PowerShell's escape character is a
backtick.  A backslash is not one: a `\"` inside a double-quoted string ends
the string, and the rest of the line is parsed as code.  That cost a whole
build -- tests, PyInstaller, the self-installing update check, all of it
green -- to fail on the second-to-last step, writing release notes.

Nothing else here reads this file, so nothing else would notice.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    import yaml                                  # a dev dependency, on purpose:
                                                # importorskip here would make
                                                # the guard below silent, which
                                                # is the failure it exists for
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def scripts(workflow: dict):
    """Every ``run:`` block in the file, with the step name it belongs to."""
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                yield step.get("name", "<unnamed>"), step["run"]


class TestItIsValidPowerShell:
    def test_no_backslash_escaped_quotes(self, workflow):
        """The one that cost a build."""
        for name, script in scripts(workflow):
            assert '\\"' not in script, (
                f"{name!r} escapes a quote with a backslash. PowerShell does "
                "not: the string ends there and the rest is parsed as code.")

    def test_no_bash_style_variables(self, workflow):
        """``$env:NAME`` on Windows, not ``${NAME}``."""
        for name, script in scripts(workflow):
            assert not re.search(r"\$\{[A-Za-z_]", script), name

    def test_every_step_is_named(self, workflow):
        """A failing step is found by its name in a log of several hundred lines."""
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                assert "name" in step or "uses" in step, step


class TestItBuildsWhatItClaimsTo:
    def test_the_tests_run_before_the_build(self, workflow):
        """A bundle that fails its own tests is worse than no bundle: it works
        until somebody relies on it."""
        windows = [step.get("name", "") for step in
                   workflow["jobs"]["windows"]["steps"]]
        assert windows.index("Tests, on Windows") < windows.index("Build")

    def test_what_it_built_is_run_before_it_is_published(self, workflow):
        """PyInstaller reports success for bundles missing a module they only
        import at startup."""
        windows = [step.get("name", "") for step in
                   workflow["jobs"]["windows"]["steps"]]
        publish = windows.index("Publish")
        for smoke in ("The console build runs",
                      "The diagnosis runs on a real file",
                      "The vocoder survived the bundler",
                      "The update actually installs, on a throwaway copy"):
            assert windows.index(smoke) < publish, smoke

    def test_the_stamp_step_can_fail(self, workflow):
        """An unstamped bundle never offers an update, and nothing downstream
        would notice."""
        stamp = next(script for name, script in scripts(workflow)
                     if name == "Stamp the build")
        assert "throw" in stamp
