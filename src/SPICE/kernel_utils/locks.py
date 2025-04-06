import os
import uuid
import errno
import time
import logging

from filelock import FileLock

from src.SPICE.config import (
    SPICE_KERNEL_LOCK_TIMEOUT,
    KERNEL_LOCK_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)


class SharedFileUseLock:
    """
    This class implements a lock for a given target file.
    It creates a lock file (inside a dedicated directory) whose content is the PID of the
    process that “owns” it. The lock file is created and manipulated atomically
    using FileLock. It also supports cleaning up stale locks.
    """

    def __init__(self, target_path: str, check_stale: bool = True):
        self.target_path = os.path.abspath(target_path)
        self.lock_dir = self.target_path + ".locks"
        os.makedirs(self.lock_dir, exist_ok=True)
        self.check_stale = check_stale
        # Create a unique lock file name.
        self.lock_path = os.path.join(self.lock_dir, f"{uuid.uuid4().hex}.lock")
        # Clean up any stale locks in the directory.
        self.cleanup_stale_locks()

    def register_use(self) -> str:
        """
        Atomically writes the current process’s PID into the lock file.
        This uses FileLock to guarantee atomicity.
        """
        # Use a FileLock (not the auto-cleanup one) so that the file remains after the block.
        with FileLock(self.lock_path, timeout=SPICE_KERNEL_LOCK_TIMEOUT, poll_interval=KERNEL_LOCK_POLL_INTERVAL):
            with open(self.lock_path, "w") as f:
                f.write(str(os.getpid()))
        return self.lock_path

    def release_use(self):
        """
        Removes the given lock file to indicate the end of usage.
        """
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError as e:
            return e.errno == errno.EPERM
        return True

    def cleanup_stale_locks(self):
        """
        Iterates over lock files and removes those whose PID no longer exists.
        """
        for name in os.listdir(self.lock_dir):
            if not name.endswith(".lock"):
                continue
            path = os.path.join(self.lock_dir, name)
            try:
                with open(path, "r") as f:
                    pid = int(f.read().strip())
                if not self._pid_exists(pid):
                    os.remove(path)
                    logger.debug("Removed stale lock file: %s", path)
            except Exception:
                try:
                    os.remove(path)
                    logger.debug("Removed problematic lock file: %s", path)
                except Exception:
                    pass

    def is_in_use(self, cleanup_stale: bool = None) -> bool:
        """
        Checks if any process is using the file.
        If cleanup_stale is True (defaulting to self.check_stale), stale locks are cleaned up.
        """
        if cleanup_stale is None:
            cleanup_stale = self.check_stale
        if cleanup_stale:
            self.cleanup_stale_locks()
        return any(name.endswith(".lock") for name in os.listdir(self.lock_dir))

    def wait_until_free(
        self, timeout: float = SPICE_KERNEL_LOCK_TIMEOUT, poll_interval: float = KERNEL_LOCK_POLL_INTERVAL
    ):
        start = time.time()
        logger.debug("Waiting for lock to be free: %s", self.target_path)
        while self.is_in_use():
            if (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout: File still in use after {timeout} seconds: {self.target_path}")
            time.sleep(poll_interval)
        logger.debug("Lock is now free: %s", self.target_path)

    def force_clear(self):
        """
        Deletes all lock files. Use with caution.
        """
        for name in os.listdir(self.lock_dir):
            if name.endswith(".lock"):
                try:
                    os.remove(os.path.join(self.lock_dir, name))
                except Exception:
                    pass

    def async_try_delete_file(self) -> bool:
        """
        Tries to delete the target file if it's not in use.
        Returns True if file was deleted, False if still in use or not found.
        """
        self.cleanup_stale_locks()
        if self.is_in_use(cleanup_stale=False):
            return False

        try:
            os.remove(self.target_path)
            logger.debug("Asynchronously deleted unused file: %s", self.target_path)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning("Failed async delete on %s: %s", self.target_path, e)
            return False
