# Owner(s): ["module: inductor"]
import contextlib
import sys
import threading
from types import TracebackType
from typing import Generator, Optional, Type, Union
from typing_extensions import override, Self

import torch
from torch._inductor import config
from torch._inductor.remote_cache import RemoteCacheBackend

# The cache state is thread-local so if we're running multiple tests at once
# they won't cross contaminate. However - it needs to be "global" because we
# allow code to create new cache clients which refer to the same cache (because
# it's a remote cache).
class _MockCacheState(threading.local):
    def __init__(self, name: str):
        self.reset()
        self._name = name
        self._cache = {}
        self._clients = {}  # Used for Manifold

    def reset(self):
        self.num_init = 0
        self.num_put = 0
        self.num_get_hit = 0
        self.num_get_miss = 0

    def report(self):
        print(
            "".join(
                [
                    f"{self._name} cache: ",
                    f"init: {self.num_init}, ",
                    f"puts: {self.num_put}, ",
                    f"misses: {self.num_get_miss}, ",
                    f"hits: {self.num_get_hit}, ",
                ]
            ),
            file=sys.stderr,
        )


class _MockLocalAutotuneCacheBackend(RemoteCacheBackend):
    _state = _MockCacheState("Local")

    def __init__(self):
        state = self._state
        state.num_init += 1

    @override
    def get(self, key: str) -> Optional[bytes]:
        assert isinstance(key, str)

        state = self._state
        if key in state._cache:
            state.num_get_hit += 1
            return state._cache[key]
        else:
            state.num_get_miss += 1

    @override
    def put(self, key: str, data: bytes) -> None:
        assert isinstance(key, str)
        assert isinstance(data, bytes)

        state = self._state
        state.num_put += 1
        state._cache[key] = data


class _MockRedisRemoteCache:
    _state = _MockCacheState("Redis")

    def __init__(self, *args, **kwargs):
        state = self._state
        state.num_init += 1

    def get(self, key: Union[bytes, str]) -> Optional[Union[bytes, str, int, float]]:
        assert isinstance(key, (bytes, str))

        state = self._state

        if key in state._cache:
            state.num_get_hit += 1
        else:
            state.num_get_miss += 1
        return state._cache.get(key)

    def set(self, key: Union[bytes, str], data: Union[bytes, str, int, float]) -> None:
        assert isinstance(key, (bytes, str))
        assert isinstance(data, (bytes, str, int, float))

        state = self._state

        # According to https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.set
        # redis accepts Union[bytes, memoryview, str, int, float]
        state.num_put += 1
        state._cache[key] = data


_CACHES = (
    (
        "torch._inductor.runtime.triton_heuristics._LocalAutotuneCacheBackend",
        _MockLocalAutotuneCacheBackend,
        None,
    ),
    ("redis.Redis", _MockRedisRemoteCache, None),
)


class PatchCaches(contextlib.AbstractContextManager):
    num_init = 0
    num_put = 0
    num_get_miss = 0
    num_get_hit = 0

    @staticmethod
    def get_caches():
        if config.is_fbcode():
            from .fb.mock_cache import FB_CACHES

            return _CACHES + FB_CACHES
        else:
            return _CACHES

    def __init__(self):
        from unittest.mock import patch

        self._contexts = []
        for name, ty, fn in self.get_caches():
            self._contexts.append(patch(name, fn or ty, create=True))

    @classmethod
    def reset(cls):
        """
        Reset the patched cache states as well as the PatchCaches
        aggregation.
        """
        cls.num_init = 0
        cls.num_put = 0
        cls.num_get_miss = 0
        cls.num_get_hit = 0

        for _, ty, fn in cls.get_caches():
            ty._state.reset()

    @classmethod
    def update(cls):
        """
        Update PatchCaches' state with the values from all the patched caches.
        """
        cls.num_init = sum(ty._state.num_init for _, ty, fn in cls.get_caches())
        cls.num_put = sum(ty._state.num_put for _, ty, fn in cls.get_caches())
        cls.num_get_miss = sum(ty._state.num_get_miss for _, ty, fn in cls.get_caches())
        cls.num_get_hit = sum(ty._state.num_get_hit for _, ty, fn in cls.get_caches())

    @classmethod
    def setUp(cls):
        for _, ty, _ in cls.get_caches():
            if hasattr(ty, "setUp"):
                ty.setUp()

    @classmethod
    def tearDown(cls):
        for _, ty, _ in cls.get_caches()[::-1]:
            if hasattr(ty, "tearDown"):
                ty.tearDown()

    @classmethod
    def report(cls):
        """
        Report cache state for all patched caches.
        """
        for _, ty, fn in cls.get_caches():
            ty._state.report()
        print(
            "".join(
                [
                    "All caches: ",
                    f"init: {cls.num_init}, ",
                    f"puts: {cls.num_put}, ",
                    f"misses: {cls.num_get_miss}, ",
                    f"hits: {cls.num_get_hit}",
                ]
            ),
            file=sys.stderr,
        )

    def __enter__(self) -> Self:
        """
        Start mocking the patched caches.
        """
        self.reset()

        for ctx in self._contexts:
            ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Stop mocking the patched caches.
        """
        for ctx in self._contexts[::-1]:
            ctx.__exit__(exc_type, exc_value, traceback)

        self.update()


@contextlib.contextmanager
def patch_fbcode(state: bool) -> Generator[None, None, None]:
    if hasattr(torch.version, "git_version"):
        # Currently non-fbcode
        if state:
            old = torch.version.git_version
            delattr(torch.version, "git_version")
            try:
                yield
            finally:
                torch.version.git_version = old
        else:
            yield
    else:
        # Currently fbcode
        if state:
            yield
        else:
            torch.version.git_version = "12345+"
            try:
                yield
            finally:
                delattr(torch.version, "git_version")
