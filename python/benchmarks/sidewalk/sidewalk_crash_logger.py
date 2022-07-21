from src.wrappers.crash_logger import AbstractCrashLoggingWrapper


class SideWalkCrashLogger(AbstractCrashLoggingWrapper):
    def is_done(self) -> bool:
        return self.env.done

    def is_crash(self) -> bool:
        return self.env.crash
