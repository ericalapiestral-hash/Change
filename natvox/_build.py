"""What this copy was built from.

Rewritten by the release workflow before PyInstaller runs, and left as it is
here so that running from a checkout says so rather than lying about a commit.

It exists because ``__version__`` cannot answer the question an updater asks.
Every build so far has been ``0.3.0``; comparing that against a release tells
you nothing, and an updater that compares version strings would either never
offer an update or offer one every time.  The commit is the thing that
actually differs.
"""
from __future__ import annotations

#: Full commit SHA this was built from, or "" when running from a checkout.
COMMIT = ""

#: RFC 3339 timestamp of the build, or "" when running from a checkout.
BUILT = ""


def installed() -> str:
    """A short label for this copy: the commit, or "source"."""
    return COMMIT[:7] if COMMIT else "source"


def is_frozen_build() -> bool:
    """Whether this copy came from a release rather than a checkout.

    An updater must not offer to replace a working tree with a zip.
    """
    return bool(COMMIT)
