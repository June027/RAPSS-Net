import logging
import traceback
from typing import Optional, Callable, Any


class ProjectError(Exception):
    pass


def handle_error(
    error: Exception,
    context: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    reraise: bool = False,
) -> None:
    """处理错误并记录详细的 Traceback 日志"""
    msg = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
    if logger:
        logger.error(msg)
        logger.debug(traceback.format_exc())
    else:
        print(f"❌ {msg}\n{traceback.format_exc()}")

    if reraise:
        raise error


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    context: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Any:
    """安全执行包装器"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, context=context, logger=logger)
        return default
