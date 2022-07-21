from src.wrappers.crash_logger import AbstractCrashLoggingWrapper


class GridWorldCrashLogger(AbstractCrashLoggingWrapper):
    def is_done(self) -> bool:
        return self.env.unwrapped.done

    def is_crash(self) -> bool:
        return self.env.unwrapped.latest_output_proposition.hit_wall() or self.env.unwrapped.latest_output_proposition.crash()
