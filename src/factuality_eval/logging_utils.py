"""Utility functions related to logging."""

import datetime as dt
import logging
import sys
from pathlib import Path
from typing import IO, Iterable

import tqdm.std as _tqdm_std
from termcolor import colored
from tqdm.auto import tqdm

logging.basicConfig(level=logging.DEBUG)


logger = logging.getLogger()
logging.getLogger().setLevel(logging.DEBUG)


_TQDM_ASCII = "—▰"
_tqdm_original_init = _tqdm_std.tqdm.__init__


def _tqdm_patched_init(self: _tqdm_std.tqdm, *args, **kwargs) -> None:
    # Force the nice unicode square style for every tqdm instance — including
    # ones created inside third-party libraries (datasets, transformers,
    # lettucedetect). Libraries that explicitly request ascii=True (because
    # they detected a non-TTY stream) get the squares too instead of '#'.
    if kwargs.get("ascii") in (None, True):
        kwargs["ascii"] = _TQDM_ASCII
    _tqdm_original_init(self, *args, **kwargs)


_tqdm_std.tqdm.__init__ = _tqdm_patched_init


class _Tee:
    """File-like object that mirrors writes to multiple underlying streams."""

    def __init__(self, streams: Iterable[IO[str]]) -> None:
        self._streams = list(streams)

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        # Report TTY status from the first stream (usually the real terminal),
        # so libraries like tqdm keep their interactive rendering behaviour.
        return (
            bool(self._streams) and getattr(self._streams[0], "isatty", lambda: False)()
        )

    def fileno(self) -> int:
        return self._streams[0].fileno()


def capture_stdio_to_file(log_path: Path) -> None:
    """Tee stdout and stderr so everything written to them also lands in ``log_path``.

    This captures output that bypasses the ``logging`` module — ``print`` calls,
    tqdm progress bars (which write to stderr), tracebacks, and any third-party
    library that writes directly to the standard streams. Combined with Hydra's
    file handler this yields a single log file containing the full job output.

    Idempotent: calling more than once has no additional effect.

    Args:
        log_path: Destination file. Parent directories must already exist.
    """
    if getattr(sys.stdout, "_factuality_eval_tee", False):
        return

    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    tee_out = _Tee([sys.stdout, log_file])
    tee_err = _Tee([sys.stderr, log_file])
    tee_out._factuality_eval_tee = True  # type: ignore[attr-defined]
    tee_err._factuality_eval_tee = True  # type: ignore[attr-defined]
    sys.stdout = tee_out  # type: ignore[assignment]
    sys.stderr = tee_err  # type: ignore[assignment]


BOLD = "\033[1m"
RESET = "\033[0m"


def get_pbar(*tqdm_args, **tqdm_kwargs) -> tqdm:
    """Get a progress bar for vLLM with custom hard-coded arguments.

    Args:
        *tqdm_args:
            Positional arguments to pass to tqdm.
        **tqdm_kwargs:
            Additional keyword arguments to pass to tqdm.

    Returns:
        A tqdm progress bar.
    """
    tqdm_kwargs = (
        dict(colour="yellow", ascii="—▰", leave=False, dynamic_ncols=True) | tqdm_kwargs
    )
    tqdm_kwargs["desc"] = colored(
        text=tqdm_kwargs.get("desc", "Processing"), color="light_yellow"
    )
    return tqdm(*tqdm_args, **tqdm_kwargs)


def log(message: str, level: int, color: str | None = None) -> None:
    """Log a message.

    Args:
        message:
            The message to log.
        level:
            The logging level. Defaults to logging.INFO.
        color:
            The color to use for the message. If None, a default color will be used
            based on the logging level.

    Raises:
        ValueError:
            If the logging level is invalid.
    """
    match level:
        case logging.DEBUG:
            message = colored(
                text=(
                    "[DEBUG] "
                    + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + f" · {message}"
                ),
                color=color or "light_blue",
            )
            logger.debug(message)
        case logging.INFO:
            if color is not None:
                message = colored(text=message, color=color)
            logger.info(message)
        case logging.WARNING:
            message = colored(text=message, color=color or "light_red")
            logger.warning(message)
        case logging.ERROR:
            message = colored(text=message, color=color or "red")
            logger.error(message)
        case logging.CRITICAL:
            message = colored(text=message, color=color or "red")
            logger.critical(message)
        case _:
            raise ValueError(f"Invalid logging level: {level}")


def header(message: str, level: int, color: str | None = None) -> None:
    """Log a fancy header with decorative borders.

    Args:
        message:
            The message to display in the header.
        level:
            The logging level.
        color:
            The color to use for the header. If None, a default color will be used
            based on the logging level.
    """
    # Determine color based on level if not provided
    if color is None:
        match level:
            case logging.DEBUG:
                color = "light_blue"
            case logging.INFO:
                color = "white"
            case logging.WARNING:
                color = "light_red"
            case logging.ERROR | logging.CRITICAL:
                color = "red"
            case _:
                color = "white"

    # Create header with borders
    middle_line = f"{BOLD} {message} {RESET}"

    # Color and log the header
    colored_header = colored(text=middle_line, color=color)

    logger.log(level, "\n" + colored_header)
