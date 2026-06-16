import io, json, os


class TarSlicePath(str):
    def __new__(cls, rel_path, *, shard_path, offset, size, start_frame=0):
        obj = super().__new__(cls, rel_path)
        obj._shard_path = shard_path
        obj._offset = int(offset)
        obj._size = int(size)
        obj._start_frame = int(start_frame)
        return obj

    def with_start_frame(self, start_frame):
        return TarSlicePath(
            str(self),
            shard_path=self._shard_path,
            offset=self._offset,
            size=self._size,
            start_frame=int(start_frame),
        )


class ShardIndex:
    _instances = {}

    def __init__(self, index_path):
        self.index_path = os.path.abspath(index_path)
        self.shard_dir = os.path.dirname(self.index_path)
        self._index = None
        self._fds = {}

    @classmethod
    def get(cls, index_path):
        key = os.path.abspath(index_path)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(key)
            cls._instances[key] = inst
        return inst

    def _ensure_loaded(self):
        if self._index is not None:
            return
        with open(self.index_path, "r") as f:
            self._index = json.load(f)

    def __contains__(self, rel_path):
        self._ensure_loaded()
        return rel_path in self._index

    def __len__(self):
        self._ensure_loaded()
        return len(self._index)

    def lookup(self, rel_path):
        self._ensure_loaded()
        entry = self._index.get(rel_path)
        if entry is None:
            return None
        shard_path = os.path.join(self.shard_dir, entry["shard"])
        return TarSlicePath(
            rel_path,
            shard_path=shard_path,
            offset=int(entry["offset"]),
            size=int(entry["size"]),
        )

    def _get_fd(self, shard_path):
        pid = os.getpid()
        per_pid = self._fds.get(pid)
        if per_pid is None:
            per_pid = {}
            self._fds[pid] = per_pid
        fd = per_pid.get(shard_path)
        if fd is None:
            fd = os.open(shard_path, os.O_RDONLY)
            per_pid[shard_path] = fd
        return fd

    def read(self, slice_path):
        fd = self._get_fd(slice_path._shard_path)
        os.lseek(fd, slice_path._offset, os.SEEK_SET)
        remaining = slice_path._size
        chunks = []
        while remaining > 0:
            buf = os.read(fd, remaining)
            if not buf:
                break
            chunks.append(buf)
            remaining -= len(buf)
        return b"".join(chunks)

    def read_bytesio(self, slice_path):
        return io.BytesIO(self.read(slice_path))


_ACTIVE_INDEX = None


def set_active_shard_index(index_path):
    global _ACTIVE_INDEX
    _ACTIVE_INDEX = None if not index_path else ShardIndex.get(index_path)
    return _ACTIVE_INDEX


def get_active_shard_index():
    global _ACTIVE_INDEX
    if _ACTIVE_INDEX is None:
        env_path = os.environ.get("AURORA_TAR_SHARD_INDEX")
        if env_path:
            _ACTIVE_INDEX = ShardIndex.get(env_path)
    return _ACTIVE_INDEX
