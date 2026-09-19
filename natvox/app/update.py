"""Replacing this copy with a newer one, without downloading it by hand.

Three rules shape everything here, and they are worth stating because each one
costs something and each one is deliberate.

**It never applies an update on its own.**  This program is not code-signed --
Windows already says so when you first run it -- and a thing that is not signed
and also silently replaces itself is a thing nobody should be asked to trust.
So the check is automatic, the telling is automatic, and the applying happens
when somebody clicks.

**It verifies what it downloaded before it does anything with it.**  GitHub
publishes a SHA-256 for every release asset; the download is hashed as it
arrives and a mismatch is a hard failure, not a warning.  An asset with no
digest published is also a hard failure rather than an unverified install: if
GitHub ever stops publishing them, that should surface as an error message and
not as this module quietly lowering its standards.

**It cannot overwrite a running program, so it does not try.**  On Windows the
executable and every DLL beside it are locked while the process lives.  The new
copy is unpacked next to the old one and a small script waits for this process
to exit before swapping the two directories -- which also means a failure
halfway through leaves the old copy in place and working, rather than a
half-written install that starts neither.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from .._build import COMMIT, is_frozen_build

#: Where releases come from.  Hard-coded rather than configurable: a setting
#: that points an auto-updater at a different server is a setting that turns a
#: config file into remote code execution.
REPO = "ericalapiestral-hash/Change"
TAG = "desktop-build"
API_HOST = "api.github.com"

#: Refused above this, so a wrong URL cannot fill the disk.  The bundle is
#: about 93 MB.
MAX_ASSET_BYTES = 400 * 1024 * 1024

#: And refused above this once unpacked, which the compressed size does not
#: bound: a 93 MB zip can name terabytes.  The bundle unpacks to about 250 MB.
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024

#: Where a token is read from, when the repository is private.
#:
#: Read, never written: this module does not store credentials, so there is
#: nothing here to leak into a settings file or a log.  A public repository
#: needs none of it.
TOKEN_ENV = "NATVOX_GITHUB_TOKEN"

#: Name of the directory the new copy is unpacked into, beside the old one.
STAGING = ".natvox-staged"

#: What a bundle is called, on any platform.
LAUNCHERS = ("natvox.exe", "natvox")

CONNECT_TIMEOUT = 20.0

#: What to do about a private repository, in steps somebody can follow.
#:
#: "Set an environment variable to a token with read access" is a complete
#: answer and a useless one: it assumes knowing what a token is, where GitHub
#: keeps them, which of the several kinds to make, and how Windows sets an
#: environment variable.  Every one of those is a place to give up.
HOW_TO_GET_A_TOKEN = f"""This repository is private, so the update needs a token.

  1. Open https://github.com/settings/personal-access-tokens/new
  2. Repository access -> Only select repositories -> {REPO}
  3. Permissions -> Repository permissions -> Contents -> Read-only
  4. Generate, copy it, and in PowerShell:

       setx {TOKEN_ENV} "github_pat_...."

     then close that window and open a new one -- setx only affects
     windows opened after it.

Or make the repository public and none of this is needed."""

#: Whole-download budget.  A per-read timeout bounds nothing on its own: a
#: socket delivering one byte a minute never times out and never finishes.
DOWNLOAD_DEADLINE = 900.0

#: Swaps two sibling directories by renaming, once the old one is free.
#:
#: **The rename is the wait.**  The first version polled with
#: ``tasklist | find`` until the process was gone, and that hung forever: this
#: script runs detached, with no console and no standard handles, and a pipe
#: between two console programs in that state does not complete.  CI caught it
#: with `Terminate orphan process: pid (7224) (find)` -- on a real machine it
#: would simply never have installed anything, silently.
#:
#: Nothing was lost by deleting it.  Windows refuses to rename a directory
#: containing a running executable, so the rename fails while the program is
#: alive and succeeds the moment it is not.  The thing being waited for *is*
#: the operation, which is both simpler and the only version that can also
#: wait out a virus scanner or a second copy of the program holding a file.
#:
#: Every step is logged beside the install, because a detached process with no
#: console has nowhere else to say anything, and this is the one component
#: that can leave somebody with no program at all.
#:
#: The checks that matter:
#:
#:  * `move a b` on Windows puts `a` *inside* `b` when `b` is an existing
#:    directory, so both destinations are proved gone before either move.
#:  * The rollback is checked too.  An unchecked rollback that fails is the one
#:    path that ends with nothing installed at all.
#:  * The old copy is kept until the new one has been shown to contain a
#:    program, and is only ever removed after being confirmed to be one.
_WINDOWS_SWAP = """@echo off
setlocal
set "PID=%~1"
set "INSTALL=%~2"
set "STAGED=%~3"
set "LEFT=%~4"
set "OLD=%INSTALL%.old"
set "LOG=%INSTALL%.update.log"

echo [%DATE% %TIME%] swapping after pid %PID% lets go>"%LOG%"
echo   install "%INSTALL%">>"%LOG%"
echo   staged  "%STAGED%">>"%LOG%"

if exist "%OLD%\\natvox.exe" rmdir /s /q "%OLD%"
if exist "%OLD%" (
  echo [%TIME%] cannot clear "%OLD%" -- leaving everything as it is>>"%LOG%"
  exit /b 1
)
if not exist "%STAGED%\\natvox.exe" (
  echo [%TIME%] "%STAGED%" is not a natvox build -- leaving everything as it is>>"%LOG%"
  exit /b 1
)

:aside
ping -n 2 127.0.0.1 >nul
move "%INSTALL%" "%OLD%" >>"%LOG%" 2>&1
if not exist "%OLD%\\natvox.exe" (
  set /a LEFT-=1
  if %LEFT% GTR 0 goto aside
  echo [%TIME%] could not move "%INSTALL%" aside -- leaving everything as it is>>"%LOG%"
  exit /b 1
)
echo [%TIME%] moved aside>>"%LOG%"

if exist "%INSTALL%" rmdir /q "%INSTALL%" 2>nul
if exist "%INSTALL%" goto rollback
move "%STAGED%" "%INSTALL%" >>"%LOG%" 2>&1
if not exist "%INSTALL%\\natvox.exe" goto rollback

echo [%TIME%] installed>>"%LOG%"
rmdir /s /q "%OLD%"
@RELAUNCH@
(goto) 2>nul & rmdir /s /q "%~dp0"
exit /b 0

:rollback
echo [%TIME%] putting the old copy back>>"%LOG%"
if exist "%INSTALL%" rmdir /s /q "%INSTALL%"
move "%OLD%" "%INSTALL%" >>"%LOG%" 2>&1
if exist "%INSTALL%\\natvox.exe" (
  echo [%TIME%] the update failed and the old copy is back>>"%LOG%"
  exit /b 1
)
echo [%TIME%] THE UPDATE FAILED AND SO DID PUTTING IT BACK.>>"%LOG%"
echo Your program is in "%OLD%" -- rename that to "%INSTALL%".>>"%LOG%"
exit /b 2
"""

_POSIX_SWAP = """#!/bin/sh
PID="$1"; INSTALL="$2"; STAGED="$3"; LEFT="$4"
OLD="$2.old"; LOG="$2.update.log"
say() { echo "$@" >>"$LOG"; }

: >"$LOG"
say "waiting for pid $PID"
say "  install $INSTALL"
say "  staged  $STAGED"
while kill -0 "$PID" 2>/dev/null; do sleep 1; done
say "it is gone"

if [ -e "$OLD/natvox" ]; then rm -rf "$OLD"; fi
if [ -e "$OLD" ]; then
  say "cannot clear $OLD -- leaving everything as it is"
  exit 1
fi
if [ ! -e "$STAGED/natvox" ]; then
  say "$STAGED is not a natvox build -- leaving everything as it is"
  exit 1
fi

while [ ! -e "$OLD/natvox" ]; do
  mv "$INSTALL" "$OLD" 2>>"$LOG"
  if [ -e "$OLD/natvox" ]; then break; fi
  LEFT=$((LEFT - 1))
  if [ "$LEFT" -le 0 ]; then
    say "could not move $INSTALL aside -- leaving everything as it is"
    exit 1
  fi
  sleep 1
done
say "moved aside"

rmdir "$INSTALL" 2>/dev/null
ok=1
if [ -e "$INSTALL" ]; then ok=0; fi
if [ "$ok" = 1 ]; then mv "$STAGED" "$INSTALL" 2>>"$LOG" || ok=0; fi
if [ ! -e "$INSTALL/natvox" ]; then ok=0; fi
if [ "$ok" = 0 ]; then
  say "putting the old copy back"
  rm -rf "$INSTALL"
  if mv "$OLD" "$INSTALL" 2>>"$LOG" && [ -e "$INSTALL/natvox" ]; then
    say "the update failed and the old copy is back"
    exit 1
  fi
  say "THE UPDATE FAILED AND SO DID PUTTING IT BACK."
  say "Your program is in $OLD -- rename that to $INSTALL."
  exit 2
fi

say "installed"
rm -rf "$OLD"
@RELAUNCH@
rm -rf "$(dirname "$0")"
exit 0
"""


class UpdateError(RuntimeError):
    """Something went wrong, with a sentence explaining what."""


@dataclass
class Release:
    """A published build, as the update check found it."""

    tag: str
    commit: str
    asset: str
    #: The *API* asset URL, not ``browser_download_url``.
    #:
    #: The browser URL needs a browser session: on a private repository it
    #: answers an authenticated API token with 404, so an updater built on it
    #: can never install anything.  The API URL takes the same token, answers
    #: with ``Accept: application/octet-stream``, and redirects to a signed
    #: store -- which is why the redirect handler above has to strip the
    #: credential rather than merely being careful.
    url: str
    size: int
    #: ``sha256:<hex>`` as GitHub publishes it.
    digest: str
    published: str
    notes: str = ""

    @property
    def short(self) -> str:
        return self.commit[:7] if self.commit else self.tag

    @property
    def megabytes(self) -> float:
        return self.size / (1024 * 1024)

    def summary(self) -> str:
        return (f"{self.short} published {self.published[:10]} "
                f"({self.megabytes:.0f} MB)")


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Do not carry the token to wherever GitHub sends us next.

    An asset download redirects to a signed object store on another host.
    urllib re-sends headers added to the Request across that hop, which both
    hands the credential to a third party and breaks the download -- the object
    store rejects a request carrying an Authorization header it did not ask
    for, so this is a correctness fix as much as a careful one.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise UpdateError("refusing a redirect away from HTTPS")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _host(newurl) != _host(req.full_url):
            new.remove_header("Authorization")
        return new


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_DropAuthOnRedirect())


def _request(url: str, token: str | None, accept: str = "application/vnd.github+json"):
    if not url.lower().startswith("https://"):
        raise UpdateError(f"refusing to fetch over plain HTTP: {url}")
    request = urllib.request.Request(url, headers={
        "Accept": accept,
        "User-Agent": f"natvox/{__version__}",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return request


def token_from_environment() -> str | None:
    value = os.environ.get(TOKEN_ENV, "").strip()
    return value or None


def check(repo: str = REPO, tag: str = TAG, token: str | None = None,
          timeout: float = CONNECT_TIMEOUT) -> Release:
    """Ask GitHub what the published build is.

    Raises rather than returning None, because every failure here has a
    different remedy -- no network, a private repository with no token, a
    release that has no zip attached -- and a bare None would lose which.
    """
    token = token or token_from_environment()
    url = f"https://{API_HOST}/repos/{repo}/releases/tags/{tag}"
    try:
        with _opener().open(_request(url, token), timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise UpdateError(
                "GitHub refused the request -- the token is missing, expired, "
                f"or has no access to {repo}.\n\n{HOW_TO_GET_A_TOKEN}") from exc
        if exc.code == 404:
            if token:
                raise UpdateError(
                    f"there is no release tagged {tag!r} in {repo}. The token "
                    "worked, so this is not an access problem.") from exc
            raise UpdateError(
                f"GitHub says there is no release tagged {tag!r} in {repo}. "
                "It says that both when there really is not one and when the "
                f"repository is private and nobody asked with a token, and "
                f"there is no way to tell the two apart from here.\n\n"
                f"{HOW_TO_GET_A_TOKEN}") from exc
        raise UpdateError(f"GitHub returned {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"could not reach GitHub: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UpdateError(
            "the reply was not JSON. If this machine is behind a network that "
            "shows a login page, that is what answered.") from exc

    try:
        return _release_from(payload, tag)
    except UpdateError:
        raise
    except Exception as exc:                     # noqa: BLE001 - see below
        # A malformed payload is a KeyError or a TypeError or a ValueError
        # depending on which field is wrong, and a traceback is not an answer
        # to "is there an update".
        raise UpdateError(f"GitHub's reply was not shaped like a release: "
                          f"{type(exc).__name__}") from exc


def _release_from(payload: dict, tag: str) -> Release:
    assets = [a for a in payload.get("assets", [])
              if str(a.get("name", "")).endswith(".zip")]
    if not assets:
        raise UpdateError(f"the {tag!r} release has no .zip attached to it")
    asset = assets[0]
    digest = str(asset.get("digest") or "")
    if not digest.startswith("sha256:"):
        raise UpdateError(
            "GitHub published no SHA-256 for that download. Refusing rather "
            "than installing something unverified; download it by hand if you "
            "are sure.")
    size = int(asset.get("size") or 0)
    if size <= 0 or size > MAX_ASSET_BYTES:
        raise UpdateError(f"that download is {size} bytes, which is not a build")

    return Release(
        tag=str(payload.get("tag_name") or tag),
        commit=str(payload.get("target_commitish") or ""),
        asset=str(asset["name"]),
        url=str(asset["url"]),
        size=size,
        digest=digest,
        published=str(payload.get("published_at") or ""),
        notes=str(payload.get("body") or ""),
    )


def is_newer(release: Release) -> bool:
    """Whether ``release`` is a different build from this one.

    Different, not greater.  These builds are all version 0.3.0 and the tag is
    recreated in place, so there is no ordering to compare -- only identity.
    That also makes rolling back work, which a strict greater-than would not.
    """
    return bool(release.commit) and release.commit != COMMIT


def install_dir() -> Path | None:
    """Where the frozen program lives, or None when running from a checkout."""
    if not is_frozen_build() or not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent


def download(release: Release, into: Path | None = None,
             token: str | None = None, progress=None,
             timeout: float = CONNECT_TIMEOUT,
             deadline_seconds: float = DOWNLOAD_DEADLINE) -> Path:
    """Fetch the asset and verify its digest.  Returns the file.

    The hash is computed while the bytes arrive rather than afterwards, so a
    file that fails never existed in a complete state on disk.
    """
    token = token or token_from_environment()
    folder = Path(into) if into else Path(tempfile.mkdtemp(prefix="natvox-update-"))
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / release.asset
    wanted = release.digest.split(":", 1)[1].lower()

    digest = hashlib.sha256()
    written = 0
    # A per-read timeout does not bound anything: a socket delivering one byte
    # a minute never times out and never finishes.
    deadline = time.monotonic() + max(deadline_seconds, timeout)
    try:
        with _opener().open(
                _request(release.url, token, accept="application/octet-stream"),
                timeout=timeout) as response:
            with open(target, "wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > release.size:
                        raise UpdateError("the download kept going past the "
                                          "size GitHub said it was")
                    if time.monotonic() > deadline:
                        raise UpdateError(
                            f"the download was still going after "
                            f"{deadline_seconds:.0f} seconds; giving up")
                    digest.update(chunk)
                    handle.write(chunk)
                    if progress:
                        progress(written, release.size)
        if written != release.size:
            raise UpdateError(f"got {written} bytes where GitHub said "
                              f"{release.size}")
        if digest.hexdigest().lower() != wanted:
            raise UpdateError(
                "the download does not match the checksum GitHub published. "
                "Nothing has been installed.")
    except UpdateError:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:                     # noqa: BLE001 - see below
        # A truncated response is an http.client.IncompleteRead, which is not
        # an OSError; a proxy closing the connection is something else again.
        # Every one of them has to leave no partial file behind.
        target.unlink(missing_ok=True)
        raise UpdateError(f"the download failed: {exc}") from exc
    return target


def stage(archive: Path, beside: Path) -> Path:
    """Unpack ``archive`` into a staging directory next to the install.

    Next to it rather than inside it, so that the swap later is a directory
    rename and not a file-by-file copy that can be interrupted halfway.
    """
    archive, beside = Path(archive), Path(beside)
    staged = beside.parent / STAGING
    if staged.exists():
        shutil.rmtree(staged, ignore_errors=True)
    if staged.exists():
        raise UpdateError(
            f"cannot clear {staged} -- something is holding a file in it. "
            "Close any other copy of the program and try again.")
    try:
        staged.mkdir(parents=True)
    except OSError as exc:
        raise UpdateError(f"cannot make room to unpack: {exc}") from exc
    root = staged.resolve()
    done = False
    try:
        with zipfile.ZipFile(archive) as bundle:
            total = sum(max(0, info.file_size) for info in bundle.infolist())
            if total > MAX_UNPACKED_BYTES:
                raise UpdateError(
                    f"that archive unpacks to {total / 1e9:.1f} GB, which is "
                    "not a build of this program")
            for entry in bundle.namelist():
                # A zip can name ../ or an absolute path.  Compared as paths
                # and not as strings: "/tmp/natvox-staged-evil" starts with
                # "/tmp/natvox-staged" and is not inside it.
                resolved = (staged / entry).resolve()
                if resolved != root and root not in resolved.parents:
                    raise UpdateError(f"the archive tries to write outside "
                                      f"itself: {entry!r}")
            bundle.extractall(staged)
        done = True
    except zipfile.BadZipFile as exc:
        raise UpdateError(f"the download is not a usable zip: {exc}") from exc
    except OSError as exc:
        # Disk full, permission denied and an antivirus holding a file all
        # arrive here, and none of them is the archive being bad.
        raise UpdateError(f"could not unpack it: {exc}") from exc
    finally:
        # Any failure at all, not just a bad zip: a refusal that leaves half an
        # archive behind would be installed by the next attempt.
        if not done:
            shutil.rmtree(staged, ignore_errors=True)

    # Either name.  The question here is "is this a build of this program",
    # and both answer it; asking for the *running* platform's launcher
    # conflates that with "could this run here", which is the swap script's
    # question and is checked there, at the moment it matters.  Insisting on
    # it here means an archive cannot be verified anywhere but the platform it
    # was built for -- which is exactly how the end-to-end path went untested.
    if not any((staged / name).exists() for name in LAUNCHERS):
        shutil.rmtree(staged, ignore_errors=True)
        raise UpdateError("the archive has no natvox in it: "
                          f"expected one of {', '.join(LAUNCHERS)}")
    return staged



def _swap_script(relaunch: bool) -> tuple[Path, list[str]]:
    """The script that does the swap once this process is gone.

    The paths are passed to it as arguments rather than written into its body.
    Interpolating a Windows path into a batch file is a quoting problem with no
    good answer -- ``&``, ``^``, ``%`` and ``!`` are all legal in a user profile
    name and all mean something to cmd.exe -- and an updater that mangles a path
    deletes the wrong directory.  As arguments they are the shell's problem, and
    the shell is good at it.
    """
    if os.name == "nt":
        body = _WINDOWS_SWAP.replace(
            "@RELAUNCH@", 'start "" "%INSTALL%\\natvox.exe"' if relaunch else "")
        script = Path(tempfile.mkdtemp(prefix="natvox-update-")) / "swap.bat"
        script.write_text(body, encoding="ascii")
        return script, ["cmd", "/c", str(script)]

    body = _POSIX_SWAP.replace("@RELAUNCH@",
                               '"$INSTALL/natvox" &' if relaunch else "")
    script = Path(tempfile.mkdtemp(prefix="natvox-update-")) / "swap.sh"
    script.write_text(body)
    script.chmod(0o755)
    return script, ["/bin/sh", str(script)]


#: How many times the swap retries a rename before giving up, at a second
#: apart.  Another copy of the program may still be holding a file when this
#: one has gone, and waiting is the right answer to a lock.
SWAP_RETRIES = 60


def apply(staged: Path, install: Path | None = None, relaunch: bool = True,
          spawn=subprocess.Popen, retries: int = SWAP_RETRIES) -> Path:
    """Hand the swap to a detached script and return it.

    The caller is expected to exit promptly after this: the script is waiting
    for exactly that, and until it happens nothing has changed on disk.

    If the swap fails halfway, the script puts the old directory back.  That is
    the reason for renaming whole directories rather than copying files over
    the top of a program somebody is going to run tomorrow.
    """
    staged = Path(staged)
    install = Path(install) if install else install_dir()
    if install is None:
        raise UpdateError(
            "this is running from a checkout, not a downloaded build, so "
            "there is nothing here to replace -- use git")
    if not staged.exists():
        raise UpdateError("nothing has been staged to install")

    if staged.resolve().parent != install.resolve().parent:
        raise UpdateError("the staged copy is not beside the install, so the "
                          "swap would be a copy rather than a rename")
    script, command = _swap_script(relaunch)
    command = command + [str(os.getpid()), str(install), str(staged),
                         str(max(1, int(retries)))]
    flags = {}
    if os.name == "nt":
        # Detached, or the script dies with the process it is waiting for.
        flags["creationflags"] = 0x00000008 | 0x08000000   # DETACHED | NO_WINDOW
    else:
        flags["start_new_session"] = True
    spawn(command, cwd=str(install.parent), **flags)
    return script


def fetch_and_stage(release: Release, install: Path | None = None,
                    token: str | None = None, progress=None) -> Path:
    """Download, verify, unpack, and leave nothing behind but the staged copy.

    The install directory is resolved *first*.  Finding out that there is
    nothing here to replace is a thing to learn before 93 MB, not after.
    """
    install = Path(install) if install else install_dir()
    if install is None:
        raise UpdateError(
            "this is running from a checkout, not a downloaded build, so "
            "there is nothing here to replace -- use git")
    folder = Path(tempfile.mkdtemp(prefix="natvox-update-"))
    try:
        archive = download(release, into=folder, token=token, progress=progress)
        return stage(archive, install)
    finally:
        # 93 MB in the temp directory either way.  Nobody else is going to.
        shutil.rmtree(folder, ignore_errors=True)


def discard(install: Path | None = None) -> None:
    """Throw away anything staged.  Safe to call when there is nothing."""
    install = Path(install) if install else install_dir()
    if install is None:
        return
    shutil.rmtree(install.parent / STAGING, ignore_errors=True)


@dataclass
class UpdateState:
    """What the check found, in a form a window or a terminal can show."""

    installed: str
    release: Release | None = None
    staged: Path | None = None
    error: str = ""

    @property
    def available(self) -> bool:
        return self.release is not None and is_newer(self.release)

    def summary(self) -> str:
        if self.error:
            return self.error
        if self.release is None:
            return f"running {self.installed}; no release found"
        if not self.available:
            return f"running {self.installed}, which is the published build"
        line = (f"an update is available: {self.release.summary()}, "
                f"against {self.installed}")
        if self.staged:
            line += "\ndownloaded and checked -- restart to finish installing"
        return line


def state(token: str | None = None) -> UpdateState:
    """Check, and turn any failure into something worth reading."""
    from .._build import installed

    try:
        return UpdateState(installed(), check(token=token))
    except UpdateError as exc:
        return UpdateState(installed(), error=str(exc))
