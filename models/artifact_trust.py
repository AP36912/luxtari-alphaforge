"""Explicit trust policy for loading serialized model artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Iterator


class UntrustedModelArtifactError(RuntimeError):
    """Raised when a serialized model artifact is not explicitly trusted."""


@dataclass(frozen=True)
class ModelArtifactTrustPolicy:
    """Allow loading only artifacts whose location and digest are approved."""

    allow_serialized_load: bool = False
    allowed_sha256: frozenset[str] = frozenset()
    allowed_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_sha256",
            frozenset(digest.lower() for digest in self.allowed_sha256),
        )
        object.__setattr__(
            self,
            "allowed_roots",
            tuple(Path(root).resolve() for root in self.allowed_roots),
        )


_CURRENT_POLICY: ContextVar[ModelArtifactTrustPolicy | None] = ContextVar(
    "model_artifact_trust_policy",
    default=None,
)
_ACTIVE_SNAPSHOTS: ContextVar[frozenset[Path]] = ContextVar(
    "active_model_artifact_snapshots",
    default=frozenset(),
)


@dataclass
class TrustedModelArtifact:
    """Verified private snapshot of a serialized model directory."""

    digest: str
    snapshot_path: Path
    _active_token: Token[frozenset[Path]] | None = field(default=None, init=False)

    def __enter__(self) -> TrustedModelArtifact:
        active = _ACTIVE_SNAPSHOTS.get()
        self._active_token = _ACTIVE_SNAPSHOTS.set(active | {self.snapshot_path})
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self._active_token is not None:
            _ACTIVE_SNAPSHOTS.reset(self._active_token)
            self._active_token = None
        self.cleanup()

    def cleanup(self) -> None:
        """Delete the snapshot when it is not managed by a ``with`` block."""
        _remove_snapshot(self.snapshot_path)


def _remove_snapshot(path: Path) -> None:
    if not path.exists():
        return
    for directory, dirs, files in os.walk(path):
        for name in dirs:
            os.chmod(Path(directory) / name, stat.S_IRWXU)
        for name in files:
            os.chmod(Path(directory) / name, stat.S_IRUSR | stat.S_IWUSR)
    os.chmod(path, stat.S_IRWXU)
    shutil.rmtree(path)


def _iter_regular_files(root: Path) -> list[Path]:
    """Return regular files while rejecting every symlink or special file."""
    files: list[Path] = []
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in dir_names:
            entry = current / name
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise UntrustedModelArtifactError(
                    f"Model artifact directory contains a symlink: {entry}"
                )
            if not stat.S_ISDIR(mode):
                raise UntrustedModelArtifactError(
                    f"Model artifact directory contains a non-directory entry: {entry}"
                )
        for name in file_names:
            entry = current / name
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise UntrustedModelArtifactError(
                    f"Model artifact directory contains a symlink: {entry}"
                )
            if not stat.S_ISREG(mode):
                raise UntrustedModelArtifactError(
                    f"Model artifact directory contains a non-regular file: {entry}"
                )
            files.append(entry)
    return sorted(files, key=lambda entry: entry.relative_to(root).as_posix())


def _hash_regular_files(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file_path in files:
        relative_name = file_path.relative_to(root).as_posix().encode("utf-8")
        # Length prefixes prevent ambiguity between path and file byte boundaries.
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        with file_path.open("rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
        digest.update((0).to_bytes(8, "big"))
    return digest.hexdigest()


def compute_dir_sha256(path: Path) -> str:
    """Hash regular files in deterministic path order, rejecting symlinks."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model artifact directory does not exist: {root}")
    return _hash_regular_files(root, _iter_regular_files(root))


def prepare_trusted_model_dir(
    path: Path,
    policy: ModelArtifactTrustPolicy | None,
) -> TrustedModelArtifact:
    """Copy, hash, and approve a private snapshot of a model artifact directory."""
    if policy is None or not policy.allow_serialized_load:
        raise UntrustedModelArtifactError(
            "Serialized model loading is denied by default; configure an explicit "
            "ModelArtifactTrustPolicy with allow_serialized_load=True and an approved SHA-256."
        )
    if not policy.allowed_roots:
        raise UntrustedModelArtifactError(
            "Serialized model loading requires at least one allowed root."
        )
    if not policy.allowed_sha256:
        raise UntrustedModelArtifactError(
            "Serialized model loading requires at least one approved SHA-256 digest."
        )

    requested = Path(path).absolute()
    if requested.is_symlink():
        raise UntrustedModelArtifactError(
            f"Model artifact directory is a symlink: {requested}"
        )
    try:
        root = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise UntrustedModelArtifactError(
            f"Model artifact directory does not exist: {requested}"
        ) from exc
    if not root.is_dir():
        raise UntrustedModelArtifactError(f"Model artifact path is not a directory: {root}")
    if not any(
        root == allowed_root or root.is_relative_to(allowed_root)
        for allowed_root in policy.allowed_roots
    ):
        raise UntrustedModelArtifactError(
            f"Model artifact directory is outside the policy's allowed roots: {root}"
        )

    files = _iter_regular_files(root)
    snapshot = Path(tempfile.mkdtemp(prefix="trusted-model-"))
    try:
        for source in files:
            relative = source.relative_to(root)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise UntrustedModelArtifactError(
                        f"Model artifact changed to a non-regular file: {source}"
                    )
                with os.fdopen(descriptor, "rb") as source_file:
                    descriptor = -1
                    with destination.open("xb") as destination_file:
                        shutil.copyfileobj(source_file, destination_file)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

        snapshot_files = _iter_regular_files(snapshot)
        digest = _hash_regular_files(snapshot, snapshot_files)
        if digest not in policy.allowed_sha256:
            raise UntrustedModelArtifactError(
                f"Model artifact SHA-256 is not approved by the trust policy: {digest}"
            )

        for file_path in snapshot_files:
            file_path.chmod(stat.S_IRUSR)
        directories = sorted(
            (entry for entry in snapshot.rglob("*") if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(stat.S_IRUSR | stat.S_IXUSR)
        snapshot.chmod(stat.S_IRUSR | stat.S_IXUSR)
        return TrustedModelArtifact(digest=digest, snapshot_path=snapshot)
    except BaseException:
        _remove_snapshot(snapshot)
        raise


@contextmanager
def trusted_model_dir(
    path: str | Path,
    policy: ModelArtifactTrustPolicy | None,
) -> Iterator[Path]:
    """Yield an active verified snapshot, preparing one for direct class loads."""
    resolved = Path(path).resolve()
    if resolved in _ACTIVE_SNAPSHOTS.get():
        yield resolved
        return
    with prepare_trusted_model_dir(resolved, policy) as artifact:
        yield artifact.snapshot_path


def assert_trusted_model_dir(
    path: Path,
    policy: ModelArtifactTrustPolicy | None,
) -> str:
    """Backward-compatible assertion that verifies through a private snapshot."""
    with prepare_trusted_model_dir(path, policy) as artifact:
        return artifact.digest


def set_model_artifact_trust_policy(policy: ModelArtifactTrustPolicy) -> None:
    """Set the trust policy for the current context."""
    _CURRENT_POLICY.set(policy)


def get_model_artifact_trust_policy() -> ModelArtifactTrustPolicy | None:
    """Return the trust policy for the current context, if any."""
    return _CURRENT_POLICY.get()


def clear_model_artifact_trust_policy() -> None:
    """Restore default-deny model loading for the current context."""
    _CURRENT_POLICY.set(None)
