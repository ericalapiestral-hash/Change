"""The updater, which downloads and installs executable code.

Nothing here touches the network or the filesystem outside tmp_path.  The tests
that matter are the refusals: an updater that installs the wrong thing is worse
than one that does not work, and an updater that leaves no working program is
worse than either.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.error
import zipfile
from pathlib import Path

import pytest

from natvox.app import update


def release_json(*, digest=None, size=None, name="natvox-0.3.0-windows-x64.zip",
                 commit="a" * 40, assets=None):
    payload = {
        "tag_name": "desktop-build",
        "target_commitish": commit,
        "published_at": "2026-09-18T21:17:39Z",
        "body": "release notes",
        "assets": assets if assets is not None else [{
            "name": name,
            # The API URL is the one that works on a private repository; the
            # browser one answers a token with 404.  Both are present on a
            # real payload, which is how the wrong one got picked.
            "url": "https://api.github.com/repos/o/r/releases/assets/1",
            "browser_download_url":
                f"https://github.com/o/r/releases/download/desktop-build/{name}",
            "size": 93 * 1024 * 1024 if size is None else size,
            "digest": "sha256:" + "b" * 64 if digest is None else digest,
        }],
    }
    return payload


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def serving(monkeypatch, payload=None, body=None, error=None):
    """Point the module's opener at a canned response."""
    calls = []

    class Opener:
        def open(self, request, timeout=None):
            calls.append(request)
            if error is not None:
                raise error
            if body is not None and "releases/assets" in request.full_url:
                return FakeResponse(body)
            return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(update, "_opener", lambda: Opener())
    monkeypatch.delenv(update.TOKEN_ENV, raising=False)
    return calls


class TestChecking:
    def test_it_reads_the_release(self, monkeypatch):
        serving(monkeypatch, release_json())
        release = update.check()
        assert release.commit == "a" * 40
        assert release.url.startswith("https://api.github.com/"), \
            "browser_download_url answers a token with 404 on a private repo"
        assert release.short == "aaaaaaa"
        assert release.digest.startswith("sha256:")
        assert release.megabytes == pytest.approx(93.0, abs=0.5)

    def test_an_asset_with_no_checksum_is_refused(self, monkeypatch):
        """Refusing beats quietly lowering the standard if GitHub ever stops
        publishing them."""
        serving(monkeypatch, release_json(digest=""))
        with pytest.raises(update.UpdateError, match="SHA-256"):
            update.check()

    @pytest.mark.parametrize("size", [0, -1, update.MAX_ASSET_BYTES + 1])
    def test_an_impossible_size_is_refused(self, monkeypatch, size):
        serving(monkeypatch, release_json(size=size))
        with pytest.raises(update.UpdateError, match="not a build"):
            update.check()

    def test_a_malformed_payload_is_a_sentence_not_a_traceback(self, monkeypatch):
        # Passes every explicit check and then has no url: a KeyError, which
        # is not an answer to "is there an update".
        serving(monkeypatch, release_json(assets=[{
            "name": "n.zip", "size": 1000, "digest": "sha256:" + "b" * 64}]))
        with pytest.raises(update.UpdateError, match="not shaped like a release"):
            update.check()

    def test_a_login_page_where_json_should_be_says_so(self, monkeypatch):
        class Opener:
            def open(self, request, timeout=None):
                return FakeResponse(b"<html>sign in</html>")

        monkeypatch.setattr(update, "_opener", lambda: Opener())
        monkeypatch.delenv(update.TOKEN_ENV, raising=False)
        with pytest.raises(update.UpdateError, match="login page"):
            update.check()

    def test_a_release_with_no_zip_says_so(self, monkeypatch):
        serving(monkeypatch, release_json(assets=[{"name": "notes.txt"}]))
        with pytest.raises(update.UpdateError, match="no .zip"):
            update.check()

    def test_a_404_explains_that_private_looks_the_same(self, monkeypatch):
        """GitHub reports "private" and "missing" identically, and only one of
        them has a remedy."""
        serving(monkeypatch, error=urllib.error.HTTPError(
            "u", 404, "Not Found", {}, None))
        with pytest.raises(update.UpdateError) as raised:
            update.check()
        assert "no way to tell the two apart" in str(raised.value)
        assert update.HOW_TO_GET_A_TOKEN in str(raised.value)

    def test_a_404_with_a_working_token_does_not_blame_the_token(self, monkeypatch):
        """The token got an answer, so sending somebody to make another one is
        an hour of the wrong work."""
        serving(monkeypatch, error=urllib.error.HTTPError(
            "u", 404, "Not Found", {}, None))
        with pytest.raises(update.UpdateError) as raised:
            update.check(token="github_pat_x")
        assert "not an access problem" in str(raised.value)
        assert update.TOKEN_ENV not in str(raised.value)

    def test_being_refused_names_the_token(self, monkeypatch):
        serving(monkeypatch, error=urllib.error.HTTPError(
            "u", 403, "Forbidden", {}, None))
        with pytest.raises(update.UpdateError, match=update.TOKEN_ENV):
            update.check()

    def test_the_token_instructions_are_steps_not_a_noun(self):
        """"Set an environment variable to a token with read access" is a
        complete answer and a useless one: it assumes knowing what a token is,
        which of the several kinds to make, and how Windows sets one."""
        how = update.HOW_TO_GET_A_TOKEN
        assert "github.com/settings" in how, "where to go"
        assert update.REPO in how, "which repository to scope it to"
        assert "Read-only" in how, "how little to grant"
        assert "setx" in how, "how to set it on the platform this ships to"
        assert "new one" in how, "setx does not affect the window you type it in"

    def test_no_network_is_a_sentence(self, monkeypatch):
        serving(monkeypatch, error=urllib.error.URLError("no route"))
        with pytest.raises(update.UpdateError, match="could not reach"):
            update.check()

    def test_plain_http_is_refused_before_it_is_fetched(self):
        with pytest.raises(update.UpdateError, match="plain HTTP"):
            update._request("http://example.com/x", None)


class TestNewness:
    def test_it_compares_commits_not_versions(self, monkeypatch):
        """Every build is 0.3.0, so a version comparison would offer nothing
        ever or everything always."""
        monkeypatch.setattr(update, "COMMIT", "a" * 40)
        assert not update.is_newer(_release(commit="a" * 40))
        assert update.is_newer(_release(commit="c" * 40))

    def test_a_release_with_no_commit_is_not_newer(self, monkeypatch):
        monkeypatch.setattr(update, "COMMIT", "a" * 40)
        assert not update.is_newer(_release(commit=""))

    def test_going_backwards_is_allowed(self, monkeypatch):
        """The tag is recreated in place and there is no ordering, so identity
        is the only question -- which also makes rolling back work."""
        monkeypatch.setattr(update, "COMMIT", "z" * 40)
        assert update.is_newer(_release(commit="a" * 40))


def _release(commit="c" * 40, **kw):
    payload = release_json(commit=commit)
    asset = payload["assets"][0]
    return update.Release(
        tag=payload["tag_name"], commit=payload["target_commitish"],
        asset=asset["name"], url=asset["browser_download_url"],
        size=asset["size"], digest=asset["digest"],
        published=payload["published_at"], **kw)


class TestDownloading:
    def _release_for(self, body):
        digest = hashlib.sha256(body).hexdigest()
        return update.Release(
            tag="desktop-build", commit="c" * 40, asset="natvox.zip",
            url="https://api.github.com/repos/o/r/releases/assets/1",
            size=len(body), digest="sha256:" + digest, published="2026-09-18")

    def test_a_good_download_is_kept(self, monkeypatch, tmp_path):
        body = b"a bundle" * 100
        serving(monkeypatch, release_json(), body=body)
        got = update.download(self._release_for(body), into=tmp_path)
        assert got.read_bytes() == body

    def test_a_bad_checksum_leaves_nothing_behind(self, monkeypatch, tmp_path):
        body = b"a bundle" * 100
        serving(monkeypatch, release_json(), body=b"something else" * 60)
        release = self._release_for(body)
        release = update.Release(**{**release.__dict__, "size": len(b"something else" * 60)})
        with pytest.raises(update.UpdateError, match="checksum"):
            update.download(release, into=tmp_path)
        assert list(tmp_path.iterdir()) == [], "a failed download must not linger"

    def test_a_short_download_is_refused(self, monkeypatch, tmp_path):
        body = b"a bundle" * 100
        serving(monkeypatch, release_json(), body=body[:50])
        with pytest.raises(update.UpdateError, match="bytes where GitHub said"):
            update.download(self._release_for(body), into=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_progress_is_reported(self, monkeypatch, tmp_path):
        body = b"x" * 200_000
        serving(monkeypatch, release_json(), body=body)
        seen = []
        update.download(self._release_for(body), into=tmp_path,
                        progress=lambda a, b: seen.append((a, b)))
        assert seen and seen[-1][0] == len(body)


class TestTheRedirect:
    """An asset download redirects to a signed store on another host.

    Real urllib objects, because what is being checked is what urllib's own
    redirect handler copies across -- a stand-in for it would be testing the
    stand-in.
    """

    def _redirect(self, to, frm="https://api.github.com/x"):
        import urllib.request

        req = urllib.request.Request(frm)
        req.add_header("Authorization", "Bearer secret")
        return update._DropAuthOnRedirect().redirect_request(
            req, None, 302, "Found", {}, to)

    def test_urllib_would_have_carried_it(self):
        """The premise.  If urllib ever stops copying the header, the handler
        below is dead code and this test says so first."""
        import urllib.request

        req = urllib.request.Request("https://api.github.com/x")
        req.add_header("Authorization", "Bearer secret")
        plain = urllib.request.HTTPRedirectHandler().redirect_request(
            req, None, 302, "Found", {}, "https://objects.example.com/x")
        assert plain.has_header("Authorization")

    def test_the_token_does_not_follow_to_another_host(self):
        """It hands the credential to a third party, and the signed store
        rejects a request carrying an Authorization it did not ask for -- so
        this is a correctness fix as much as a careful one."""
        new = self._redirect("https://objects.githubusercontent.com/x")
        assert not new.has_header("Authorization")

    def test_the_token_survives_a_same_host_redirect(self):
        new = self._redirect("https://api.github.com/y")
        assert new.has_header("Authorization")

    def test_a_redirect_off_https_is_refused(self):
        with pytest.raises(update.UpdateError, match="HTTPS"):
            self._redirect("http://objects.example.com/x")


class TestStaging:
    #: Both launcher names, always.  stage() looks for the one this platform
    #: uses, so a zip carrying only the other passes on Linux and fails on
    #: Windows -- which is exactly what happened.
    LAUNCHERS = ("natvox.exe", "natvox")

    def _zip(self, path, names=("_internal/base.dll",), launchers=True):
        with zipfile.ZipFile(path, "w") as bundle:
            for name in (self.LAUNCHERS if launchers else ()) + tuple(names):
                bundle.writestr(name, b"x" * 16)
        return path

    def test_it_unpacks_beside_the_install(self, tmp_path):
        install = tmp_path / "natvox"
        install.mkdir()
        staged = update.stage(self._zip(tmp_path / "b.zip"), install)
        assert staged.parent == install.parent
        assert staged.name == update.STAGING
        assert (staged / "_internal" / "base.dll").exists()

    def test_an_archive_that_writes_outside_itself_is_refused(self, tmp_path):
        install = tmp_path / "natvox"
        install.mkdir()
        bad = self._zip(tmp_path / "bad.zip", names=("../../escaped.txt",))
        with pytest.raises(update.UpdateError, match="outside itself"):
            update.stage(bad, install)
        assert not (tmp_path.parent / "escaped.txt").exists()
        assert not (install.parent / update.STAGING).exists(), \
            "and it cleans up after refusing"

    @pytest.mark.parametrize("launcher", ["natvox.exe", "natvox"])
    def test_either_launcher_makes_it_a_build(self, tmp_path, launcher):
        """The question here is "is this a build of this program", and both
        names answer it.  Asking for the *running* platform's launcher
        conflates that with "could this run here" -- which is the swap
        script's question, checked there, at the moment it matters.

        It also meant a Windows archive could not be verified anywhere but
        Windows, which is exactly how the end-to-end path went untested: the
        download was broken for months and the only machine that could have
        noticed was not the one checking.
        """
        install = tmp_path / "natvox"
        install.mkdir()
        archive = self._zip(tmp_path / "b.zip", names=("_internal/lib",),
                            launchers=False)
        with zipfile.ZipFile(archive, "a") as bundle:
            bundle.writestr(launcher, b"x" * 16)
        assert update.stage(archive, install).exists()

    def test_an_archive_with_no_program_in_it_is_refused(self, tmp_path):
        install = tmp_path / "natvox"
        install.mkdir()
        bad = self._zip(tmp_path / "bad.zip", names=("readme.txt",),
                        launchers=False)
        with pytest.raises(update.UpdateError, match="has no natvox"):
            update.stage(bad, install)
        assert "natvox.exe" in str(update.LAUNCHERS)

    def test_something_that_is_not_a_zip_is_refused(self, tmp_path):
        install = tmp_path / "natvox"
        install.mkdir()
        rubbish = tmp_path / "b.zip"
        rubbish.write_bytes(b"not a zip at all")
        with pytest.raises(update.UpdateError, match="not a usable zip"):
            update.stage(rubbish, install)

    def test_staging_twice_replaces_rather_than_merges(self, tmp_path):
        install = tmp_path / "natvox"
        install.mkdir()
        first = update.stage(self._zip(tmp_path / "a.zip",
                                       names=("gone.txt",)), install)
        assert (first / "gone.txt").exists()
        second = update.stage(self._zip(tmp_path / "b.zip"), install)
        assert not (second / "gone.txt").exists()


class TestApplying:
    def test_the_paths_go_to_the_script_as_arguments(self, tmp_path):
        """Interpolating a Windows path into a batch file is a quoting problem
        with no good answer -- &, ^, % and ! are all legal in a profile name --
        and an updater that mangles a path deletes the wrong directory."""
        install = tmp_path / "nat & vox (1)"
        install.mkdir()
        staged = install.parent / update.STAGING
        staged.mkdir()
        spawned = []
        script = update.apply(staged, install, relaunch=False,
                              spawn=lambda cmd, **kw: spawned.append((cmd, kw)))
        body = script.read_text()
        assert str(install) not in body, "the path must not be in the script"
        assert str(install) in spawned[0][0], "it must be an argument"
        assert str(staged) in spawned[0][0]
        assert str(os.getpid()) in spawned[0][0]

    def test_it_refuses_to_swap_across_directories(self, tmp_path):
        """A rename is atomic and a copy is not, and the whole design rests on
        the old copy staying intact until the new one is in place."""
        install = tmp_path / "a" / "natvox"
        install.mkdir(parents=True)
        elsewhere = tmp_path / "b" / update.STAGING
        elsewhere.mkdir(parents=True)
        with pytest.raises(update.UpdateError, match="beside the install"):
            update.apply(elsewhere, install, spawn=lambda *a, **k: None)

    def test_nothing_staged_is_a_sentence(self, tmp_path):
        install = tmp_path / "natvox"
        install.mkdir()
        with pytest.raises(update.UpdateError, match="nothing has been staged"):
            update.apply(install.parent / update.STAGING, install,
                         spawn=lambda *a, **k: None)

    def test_a_checkout_has_nothing_to_replace(self, tmp_path):
        staged = tmp_path / update.STAGING
        staged.mkdir()
        with pytest.raises(update.UpdateError, match="use git"):
            update.apply(staged, None, spawn=lambda *a, **k: None)

    def test_the_windows_script_does_not_pipe_between_console_programs(self):
        """The first version polled with `tasklist | find` and hung forever.

        This script runs detached: no console, no standard handles, and a pipe
        between two console programs in that state does not complete.  CI
        caught it as `Terminate orphan process: pid (7224) (find)`; on a real
        machine it would simply never have installed anything, and never have
        said why.

        Nothing was lost by deleting it -- Windows refuses to rename a
        directory containing a running executable, so the rename already
        waits for exactly the thing the poll was waiting for.
        """
        assert "|" not in update._WINDOWS_SWAP
        assert "tasklist" not in update._WINDOWS_SWAP

    def test_the_rename_is_what_waits(self):
        """Not a sleep and not a poll: the operation itself.  That is also the
        only version that waits out a virus scanner or a second copy of the
        program holding a file."""
        assert "goto aside" in update._WINDOWS_SWAP
        assert 'move "%INSTALL%" "%OLD%"' in update._WINDOWS_SWAP

    def test_the_script_cleans_itself_up(self):
        script, _ = update._swap_script(relaunch=False)
        body = script.read_text()
        assert "rm -rf" in body or "rmdir" in body


class TestTheSwapForReal:
    """The swap script, run.

    Reading a shell script and asserting on its text proves the text. This is
    the one piece of the program that can leave somebody with nothing that
    starts, so it gets executed instead -- the batch file on Windows and the
    shell script elsewhere, each on the platform that runs it.
    """

    LAUNCHER = "natvox.exe" if os.name == "nt" else "natvox"

    def _tree(self, tmp_path, staged_is_a_build=True, stale_old=False):
        install = tmp_path / "natvox"
        install.mkdir()
        (install / self.LAUNCHER).write_text("OLD BUILD")
        (install / "_internal").mkdir()
        (install / "_internal" / "lib").write_text("old")
        staged = tmp_path / update.STAGING
        staged.mkdir()
        if staged_is_a_build:
            (staged / self.LAUNCHER).write_text("NEW BUILD")
            (staged / "_internal").mkdir()
            (staged / "_internal" / "lib").write_text("new")
        if stale_old:
            (tmp_path / "natvox.old").mkdir()
            (tmp_path / "natvox.old" / "precious.txt").write_text("not ours")
        return install, staged

    def _run(self, install, staged):
        import subprocess

        script, command = update._swap_script(relaunch=False)
        # A PID that is not running, so the wait loop falls straight through.
        # Two retries rather than the shipped sixty: the retry exists for a
        # file another copy of the program is still holding, and a test that
        # waits a minute to prove it gives up is a test nobody runs.
        return subprocess.run(
            command + ["999999", str(install), str(staged), "2"],
            capture_output=True, text=True, timeout=120)

    def _log(self, tmp_path):
        path = tmp_path / "natvox.update.log"
        return path.read_text() if path.exists() else ""

    def test_it_keeps_a_log_beside_the_install(self, tmp_path):
        """It runs detached with no console. Without this there is nothing at
        all to read when it goes wrong, and this is the one component that can
        leave somebody with no program."""
        install, staged = self._tree(tmp_path)
        self._run(install, staged)
        log = self._log(tmp_path)
        assert "waiting for pid" in log
        assert "installed" in log
        assert str(install) in log and str(staged) in log

    def test_the_log_says_why_it_refused(self, tmp_path):
        install, staged = self._tree(tmp_path, staged_is_a_build=False)
        self._run(install, staged)
        assert "not a natvox build" in self._log(tmp_path)

    def test_it_swaps(self, tmp_path):
        install, staged = self._tree(tmp_path)
        result = self._run(install, staged)
        assert result.returncode == 0, result.stderr
        assert (install / self.LAUNCHER).read_text() == "NEW BUILD"
        assert (install / "_internal" / "lib").read_text() == "new"
        assert not staged.exists(), "and takes the staging directory with it"
        assert not (tmp_path / "natvox.old").exists(), "and the old copy"

    def test_a_staged_directory_with_no_program_in_it_is_refused(self, tmp_path):
        """Better to install nothing than to install nothing that runs."""
        install, staged = self._tree(tmp_path, staged_is_a_build=False)
        result = self._run(install, staged)
        assert result.returncode == 1
        assert (install / self.LAUNCHER).read_text() == "OLD BUILD"
        assert (install / "_internal" / "lib").read_text() == "old"

    def test_it_will_not_delete_a_directory_it_did_not_make(self, tmp_path):
        """`natvox.old` is rmdir /s /q-ed. If somebody else's directory is
        sitting at that name, that is not a thing to delete quietly."""
        install, staged = self._tree(tmp_path, stale_old=True)
        result = self._run(install, staged)
        assert result.returncode == 1
        assert (tmp_path / "natvox.old" / "precious.txt").exists()
        assert (install / self.LAUNCHER).read_text() == "OLD BUILD"

    def test_a_missing_install_leaves_the_staged_copy_alone(self, tmp_path):
        """It gives up rather than half-installing: there is nothing to move
        aside, so there is nothing to put back if the rest goes wrong."""
        install, staged = self._tree(tmp_path)
        import shutil
        shutil.rmtree(install)
        result = self._run(install, staged)
        assert result.returncode == 1
        assert (staged / self.LAUNCHER).read_text() == "NEW BUILD"


class TestWhatItTellsYou:
    def test_running_a_checkout_says_so(self, monkeypatch):
        monkeypatch.setattr(update, "COMMIT", "")
        assert update.install_dir() is None

    def test_the_summary_reads_as_a_sentence(self, monkeypatch):
        serving(monkeypatch, release_json(commit="c" * 40))
        monkeypatch.setattr(update, "COMMIT", "a" * 40)
        state = update.state()
        assert state.available
        assert "an update is available" in state.summary()

    def test_being_up_to_date_reads_as_one_too(self, monkeypatch):
        serving(monkeypatch, release_json(commit="a" * 40))
        monkeypatch.setattr(update, "COMMIT", "a" * 40)
        state = update.state()
        assert not state.available
        assert "published build" in state.summary()

    def test_a_failure_becomes_the_summary(self, monkeypatch):
        serving(monkeypatch, error=urllib.error.URLError("no route"))
        state = update.state()
        assert not state.available
        assert "could not reach" in state.summary()
