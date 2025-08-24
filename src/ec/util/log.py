

import gzip
import io
import sys
from pathlib import Path
from typing import Optional, TextIO, Union

class LogRestarter:
    def restart(self, log: 'Log') -> 'Log':
        raise NotImplementedError
        
    def reopen(self, log: 'Log') -> 'Log':
        raise NotImplementedError

class Log:
    D_STDOUT = 0
    D_STDERR = 1
    
    def __init__(self,
                 filename: Optional[Union[str, Path]] = None,
                 post_announcements: bool = False,
                 append_on_restart: bool = False,
                 gzip: bool = False,
                 descriptor: Optional[int] = None,
                 writer: Optional[TextIO] = None,
                 restarter: Optional[LogRestarter] = None,
                 repost_announcements_on_restart: Optional[bool] = None):
        """
        Creates a log with various configuration options.
        
        Args:
            filename: File path for file-based logging
            post_announcements: Whether to post announcements
            append_on_restart: Append to file on restart
            gzip: Use gzip compression
            descriptor: D_STDOUT or D_STDERR for system streams
            writer: Custom writer object
            restarter: Custom log restarter
            repost_announcements_on_restart: Whether to repost announcements on restart
        """
        self.silent = False
        self.writer: Optional[TextIO] = None
        self.filename: Optional[Path] = Path(filename) if filename else None
        self.post_announcements = post_announcements
        self.restarter: LogRestarter
        self.repost_announcements_on_restart = (
            repost_announcements_on_restart 
            if repost_announcements_on_restart is not None 
            else not append_on_restart
        )
        self.append_on_restart = append_on_restart
        self.is_logging_to_system_out = False

        if filename is not None and descriptor is not None:
            raise ValueError("Cannot specify both filename and descriptor")
            
        if filename is not None:
            self._init_file_log(filename, post_announcements, append_on_restart, gzip)
        elif descriptor is not None:
            self._init_system_log(descriptor, post_announcements)
        elif writer is not None and restarter is not None:
            self._init_custom_log(writer, restarter, post_announcements, 
                                repost_announcements_on_restart)
        else:
            raise ValueError("Must specify either filename, descriptor, or writer+restarter")

    def _init_file_log(self, filename: Union[str, Path], 
                      post_announcements: bool, 
                      append_on_restart: bool,
                      gzip: bool):
        """Initialize a file-based log"""
        self.filename = Path(filename)
        self.post_announcements = post_announcements
        self.repost_announcements_on_restart = not append_on_restart
        self.append_on_restart = append_on_restart
        self.is_logging_to_system_out = False
        
        # if gzip:
        #     if append_on_restart:
        #         raise IOError("Cannot gzip and appendOnRestart at the same time")
            
        #     self.filename = self.filename.with_suffix(self.filename.suffix + '.gz')
        #     self.writer = io.TextIOWrapper(
        #         gzip.GzipFile(self.filename, 'w'),
        #         encoding='utf-8'
        #     )
            
        #     class GzipLogRestarter(LogRestarter):
        #         def restart(self, log: Log) -> Log:
        #             return log.reopen()
                    
        #         def reopen(self, log: Log) -> Log:
        #             if log.writer and not log.is_logging_to_system_out:
        #                 log.writer.close()
        #             log.writer = io.TextIOWrapper(
        #                 gzip.GzipFile(log.filename, 'w'),
        #                 encoding='utf-8'
        #             )
        #             return log
                    
        #     self.restarter = GzipLogRestarter()
        # else:
        #     self.writer = open(self.filename, 
        #                      'a' if append_on_restart else 'w', 
        #                      encoding='utf-8')
            
        #     class FileLogRestarter(LogRestarter):
        #         def __init__(self, append: bool):
        #             self.append = append
                
        #         def restart(self, log: Log) -> Log:
        #             log.writer = open(log.filename, 
        #                             'a' if self.append else 'w',
        #                             encoding='utf-8')
        #             return log
                    
        #         def reopen(self, log: Log) -> Log:
        #             if log.writer and not log.is_logging_to_system_out:
        #                 log.writer.close()
        #             log.writer = open(log.filename, 'w', encoding='utf-8')
        #             return log
                    
        #     self.restarter = FileLogRestarter(append_on_restart)
        
        self.writer = open(self.filename, 
                             'a' if append_on_restart else 'w', 
                             encoding='utf-8')
            
        class FileLogRestarter(LogRestarter):
            def __init__(self, append: bool):
                self.append = append
            
            def restart(self, log: Log) -> Log:
                log.writer = open(log.filename, 
                                'a' if self.append else 'w',
                                encoding='utf-8')
                return log
                
            def reopen(self, log: Log) -> Log:
                if log.writer and not log.is_logging_to_system_out:
                    log.writer.close()
                log.writer = open(log.filename, 'w', encoding='utf-8')
                return log
                
        self.restarter = FileLogRestarter(append_on_restart)

    def _init_system_log(self, descriptor: int, post_announcements: bool):
        """Initialize a system stream log (stdout/stderr)"""
        self.filename = None
        self.post_announcements = post_announcements
        self.repost_announcements_on_restart = True
        self.append_on_restart = True  # doesn't matter
        self.is_logging_to_system_out = True
        
        if descriptor == self.D_STDOUT:
            self.writer = sys.stdout
        else:  # D_STDERR
            self.writer = sys.stderr
            
        class SystemLogRestarter(LogRestarter):
            def restart(self, log: Log) -> Log:
                return log
                
            def reopen(self, log: Log) -> Log:
                return log  # makes no sense for system streams
                
        self.restarter = SystemLogRestarter()

    def _init_custom_log(self, 
                        writer: TextIO,
                        restarter: LogRestarter,
                        post_announcements: bool,
                        repost_announcements_on_restart: bool):
        """Initialize with custom writer and restarter"""
        self.filename = None
        self.post_announcements = post_announcements
        self.repost_announcements_on_restart = repost_announcements_on_restart
        self.append_on_restart = True  # doesn't matter
        self.is_logging_to_system_out = False
        self.writer = writer
        self.restarter = restarter

    def __del__(self):
        """Destructor to ensure writer is closed"""
        if self.writer and not self.is_logging_to_system_out:
            self.writer.close()

    def restart(self) -> 'Log':
        """Restarts the log after a system restart from checkpoint"""
        return self.restarter.restart(self)

    def reopen(self) -> 'Log':
        """Forces a file-based log to reopen, erasing its previous contents"""
        return self.restarter.reopen(self)

    def write(self, message: str):
        """Write a message to the log"""
        if not self.silent and self.writer:
            self.writer.write(message)
            self.writer.flush()

    def flush(self):
        """Flush the log output"""
        if self.writer:
            self.writer.flush()