from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from pathlib import Path


class MemoryStore(dict[str, bytes]):
    def __init__(self, *args, read_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_only = read_only

    def __setitem__(self, key: str, value: bytes) -> None:
        if self.read_only:
            raise PermissionError("store is read-only")
        super().__setitem__(key, bytes(value))

    def __delitem__(self, key: str) -> None:
        if self.read_only:
            raise PermissionError("store is read-only")
        super().__delitem__(key)


class LocalStore(MutableMapping[str, bytes]):
    def __init__(self, root: str | os.PathLike[str], *, read_only: bool = False):
        self.root = Path(root)
        self.read_only = read_only
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        clean = key.strip("/")
        path = self.root.joinpath(*clean.split("/")) if clean else self.root
        if self.root.resolve() not in (path.resolve(), *path.resolve().parents):
            raise ValueError("store key escapes the root")
        return path

    def __getitem__(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: bytes) -> None:
        if self.read_only:
            raise PermissionError("store is read-only")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".mojo-zarr.tmp")
        temporary.write_bytes(bytes(value))
        os.replace(temporary, path)

    def __delitem__(self, key: str) -> None:
        if self.read_only:
            raise PermissionError("store is read-only")
        path = self._path(key)
        path.unlink()
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def __iter__(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for path in self.root.rglob("*"):
            if path.is_file():
                yield path.relative_to(self.root).as_posix()

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._path(key).is_file()

    def __repr__(self) -> str:
        return f"LocalStore({str(self.root)!r}, read_only={self.read_only})"


DirectoryStore = LocalStore


def normalize_store(
    store: str | os.PathLike[str] | MutableMapping[str, bytes] | None,
) -> MutableMapping[str, bytes]:
    if store is None:
        return MemoryStore()
    if isinstance(store, (str, os.PathLike)):
        return LocalStore(store)
    if isinstance(store, MutableMapping):
        return store
    raise TypeError("store must be a path or mutable mapping")
