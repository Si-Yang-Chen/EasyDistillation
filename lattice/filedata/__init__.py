"""File I/O backends.

``FileData`` is the lazy-handle protocol every backing format implements: it
exposes ``__getitem__`` and materialises only what is indexed. Consumers write
``loader.load(key)[...]`` and never touch the underlying storage directly.

Backends, by on-disk format:

- ``abstract``  — ``FileData`` / ``File`` / ``FileMetaData`` protocol definitions
- ``binary``    — raw binary blobs with an element description
- ``ildg``      — ILDG gauge-field containers
- ``ndarray``   — ``.npy`` files, memory-mapped; single-file and per-timeslice
- ``timeslice`` — QDP lazy disk-map time slices

This package previously had no ``__init__.py`` and relied on PEP 420 implicit
namespace packaging. Every other ``lattice`` sub-package has one, and explicit
``packages = [...]`` or ``find_packages(namespaces=False)`` configurations would
have silently dropped the directory, breaking ``import lattice`` (``lattice``'s
own ``__init__`` pulls in ``preset``, which imports from here).

Only ``abstract`` is re-exported: it is the root of the dependency graph, so
importing it cannot introduce a load-order cycle.
"""

from .abstract import File, FileData, FileMetaData

__all__ = ["File", "FileData", "FileMetaData"]
