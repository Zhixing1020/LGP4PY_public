import sys
from src.ec.util.log import Log, LogRestarter
from pathlib import Path
from typing import Optional, TextIO, Union, Any
from io import TextIOWrapper
import warnings
from dataclasses import dataclass

@dataclass
class Announcement:
    """Stores announcement messages for potential reposting"""
    text: str

class OutputExitException(Exception):
    """Custom exception for output errors that should exit the program"""
    pass

class Output:

    V_VERBOSE = 0
    V_NO_MESSAGES = 1000
    V_NO_WARNINGS = 2000
    V_NO_GENERAL = 3000
    V_NO_ERRORS = 4000
    V_TOTALLY_SILENT = 5000
    
    # Special log identifiers
    ALL_MESSAGE_LOGS = -1
    NO_LOGS = -2
    
    def __init__(self, store_announcements_in_memory: bool = True):
        self.errors = False
        self.logs: list[Log] = []
        self.announcements: list[Announcement] = []
        self.store = store_announcements_in_memory
        self.file_prefix = ""
        self.throws_errors = False
        self.one_time_warnings = set()

    def error(self, message: str):
        raise Exception(message)  # or a custom exception type

    def warning(self, message: str):
        print(f"Warning: {message}", file=sys.stderr)

    def fatal(self, message: str, *args):
        error_msg = f"Fatal error: {message}. "
        if args:
            error_msg += ' Params: '.join(str(arg) for arg in args)
        raise SystemExit(error_msg)
    
    def message(self, message: str):
        print(f"{message}")
    

    def set_file_prefix(self, prefix: str) -> None:
        """Set prefix for log filenames"""
        self.file_prefix = prefix
    
    def set_throws_errors(self, val: bool) -> None:
        """Configure whether to throw exceptions instead of exiting"""
        self.throws_errors = val
    
    def close(self) -> None:
        """Close all log files"""
        self.flush()
        for log in self.logs:
            if not log.is_logging_to_system_out and log.writer:
                log.writer.close()
    
    def flush(self) -> None:
        """Flush all log writers"""
        for log in self.logs:
            if log.writer:
                log.writer.flush()
        sys.stdout.flush()
        sys.stderr.flush()
    
    def addLog(self, 
               file: Optional[Union[str, Path]] = None,
               post_announcements: bool = False,
               append_on_restart: bool = False,
               gzip: bool = False,
               descriptor: Optional[int] = None,
               writer: Optional[TextIO] = None,
               restarter: Optional[LogRestarter] = None,
               repost_announcements: Optional[bool] = None) -> int:
        """Add a new log with various configuration options"""
        if sum(x is not None for x in [file, descriptor, writer]) != 1:
            raise ValueError("Must specify exactly one of file, descriptor, or writer")
        
        if file is not None:
            if self.file_prefix:
                file = Path(file).parent / f"{self.file_prefix}{file.name}"
            log = Log(filename=file, 
                     post_announcements=post_announcements,
                     append_on_restart=append_on_restart,
                     gzip=gzip)
        elif descriptor is not None:
            log = Log(descriptor=descriptor, 
                     post_announcements=post_announcements)
        else:  # writer is not None
            if restarter is None:
                raise ValueError("Must provide restarter with custom writer")
            log = Log(writer=writer,
                     restarter=restarter,
                     post_announcements=post_announcements,
                     repost_announcements_on_restart=repost_announcements or not append_on_restart)
        
        self.logs.append(log)
        return len(self.logs) - 1
    
    def num_logs(self) -> int:
        """Get number of logs"""
        return len(self.logs)
    
    def get_log(self, index: int) -> 'Log':
        """Get log by index"""
        return self.logs[index]
    
    def remove_log(self, index: int) -> 'Log':
        """Remove and return log by index"""
        return self.logs.pop(index)

    def warnOnce(self, s: str, p1: Any = None, p2: Any = None) -> None:
        """Post a warning message only once"""
        if s not in self.one_time_warnings:
            self.one_time_warnings.add(s)
            self._build_message("ONCE-ONLY WARNING", s, p1, p2, False)

    def println(self, s: str, log: int, announcement: bool = False) -> None:
        """Print a line to specified log(s)"""
        if log == self.NO_LOGS:
            return
        
        if log == self.ALL_MESSAGE_LOGS:
            for l in self.logs:
                self._print_to_log(s + "\n", l, announcement, False)
        else:
            self._print_to_log(s + "\n", self.logs[log], announcement, False)
    
    def print(self, s: str, log: int) -> None:
        """Print without newline to specified log(s)"""
        if log == self.NO_LOGS:
            return
        
        if log == self.ALL_MESSAGE_LOGS:
            for l in self.logs:
                self._print_to_log(s, l, False, False)
        else:
            self._print_to_log(s, self.logs[log], False, False)
    
    def _print_to_log(self, s: str, log: 'Log', announcement: bool, reposting: bool) -> None:
        """Internal method to print to a single log"""
        if not log or not log.writer:
            return
        if not log.post_announcements and announcement:
            return
        if log.silent:
            return
        
        log.writer.write(s)
        log.writer.flush()
        
        if self.store and announcement and not reposting:
            self.announcements.append(Announcement(s))
    
    def _build_error_message(self, prefix: str, s: str, p1: Any, p2: Any) -> None:
        """Build error message with optional parameters"""
        self.error_message = f"{prefix}:\n{s}\n"
        if p1 is not None:
            self.error_message += f"PARAMETER: {p1}\n"
        if p2 is not None:
            if p1 is not None:
                self.error_message += f"     ALSO: {p2}\n"
            else:
                self.error_message += f"PARAMETER: {p2}\n"
        
        self.println(self.error_message, self.ALL_MESSAGE_LOGS, True)
    
    def _build_message(self, prefix: str, s: str, p1: Any, p2: Any, is_error: bool) -> None:
        """Build a message with optional parameters"""
        msg = f"{prefix}:\n{s}\n"
        if p1 is not None:
            msg += f"PARAMETER: {p1}\n"
        if p2 is not None:
            if p1 is not None:
                msg += f"     ALSO: {p2}\n"
            else:
                msg += f"PARAMETER: {p2}\n"
        
        if is_error:
            self.error(msg)
        else:
            self.println(msg, self.ALL_MESSAGE_LOGS, True)
    
    def _exit_with_error(self, message: str) -> None:
        """Exit with error message"""
        self.close()
        sys.stdout.flush()
        sys.stderr.flush()
        
        if self.throws_errors:
            raise OutputExitException(message)
        else:
            sys.exit(1)
    
    def exit_if_errors(self) -> None:
        """Exit if errors have been recorded"""
        if self.errors:
            self.println("SYSTEM EXITING FROM ERRORS\n", self.ALL_MESSAGE_LOGS, True)
            self._exit_with_error(self.error_message)
    
    def clear_errors(self) -> None:
        """Clear error flag"""
        self.errors = False
    
    def clear_announcements(self) -> None:
        """Clear stored announcements"""
        self.announcements = []
    
    def restart(self) -> None:
        """Restart all logs and repost announcements if needed"""
        for i, log in enumerate(self.logs):
            new_log = log.restarter.restart(log)
            self.logs[i] = new_log
            
            if new_log.repost_announcements_on_restart and self.store:
                for announcement in self.announcements:
                    self._print_to_log(announcement.text + "\n", new_log, True, True)
        
        self.exit_if_errors()

    # Static methods for initial messages
    @staticmethod
    def initial_warning(s: str, p1: Any = None, p2: Any = None) -> None:
        """Static method for startup warnings"""
        msg = ["STARTUP WARNING:", s]
        if p1 is not None:
            msg.append(f"PARAMETER: {p1}")
        if p2 is not None:
            msg.append(f"     ALSO: {p2}" if p1 else f"PARAMETER: {p2}")
        warnings.warn("\n".join(msg))
    
    @staticmethod
    def initial_error(s: str, p1: Any = None, p2: Any = None) -> None:
        """Static method for startup errors"""
        msg = ["STARTUP ERROR:", s]
        if p1 is not None:
            msg.append(f"PARAMETER: {p1}")
        if p2 is not None:
            msg.append(f"     ALSO: {p2}" if p1 else f"PARAMETER: {p2}")
        print("\n".join(msg), file=sys.stderr)
        sys.exit(1)
    
    @staticmethod
    def initial_message(s: str) -> None:
        """Static method for startup messages"""
        print(s, file=sys.stderr)
        sys.stderr.flush()