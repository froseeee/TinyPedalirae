"""iRacing SDK connector for TinyPedal

Handles connection and data retrieval from iRacing simulator via pyirsdk
"""

import logging
import threading
import time

try:
    import pyirsdk
except ImportError:
    pyirsdk = None

logger = logging.getLogger(__name__)


class IRacingConnector:
    """iRacing SDK data connector"""

    __slots__ = (
        "_ir",
        "_updating",
        "_update_thread",
        "_stop_event",
        "_connected",
        "data",
        "last_update",
    )

    def __init__(self) -> None:
        """Initialize iRacing connector"""
        if pyirsdk is None:
            logger.warning("pyirsdk not installed: pip install pyirsdk")
            self._ir = None
        else:
            self._ir = pyirsdk.IRSDK()
        
        self._updating = False
        self._update_thread = None
        self._stop_event = threading.Event()
        self._connected = False
        self.data = {}
        self.last_update = 0

    def start(self) -> bool:
        """Start iRacing connection thread"""
        if self._ir is None:
            logger.error("iRacing: pyirsdk not available")
            return False
            
        if self._updating:
            return True

        try:
            self._updating = True
            self._stop_event.clear()
            self._update_thread = threading.Thread(
                target=self._update_loop,
                name="iRacing-Connector",
                daemon=True
            )
            self._update_thread.start()
            logger.info("iRacing: connector started")
            return True
        except Exception as e:
            logger.error(f"iRacing: failed to start connector: {e}")
            self._updating = False
            return False

    def stop(self) -> None:
        """Stop iRacing connection thread"""
        if not self._updating:
            return

        try:
            self._stop_event.set()
            self._updating = False
            
            if self._update_thread:
                self._update_thread.join(timeout=2.0)
            
            if self._ir:
                self._ir.shutdown()
            
            self._connected = False
            logger.info("iRacing: connector stopped")
        except Exception as e:
            logger.error(f"iRacing: error stopping connector: {e}")

    def _update_loop(self) -> None:
        """Main update loop for iRacing data"""
        retry_count = 0
        max_retries = 10
        
        while not self._stop_event.is_set():
            try:
                if not self._connected:
                    if self._ir.startup():
                        self._connected = True
                        retry_count = 0
                        logger.info("iRacing: connected to simulator")
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            time.sleep(1.0)  # Back off if repeated failures
                            retry_count = 0
                        else:
                            time.sleep(0.1)
                        continue

                if self._connected and self._ir.is_connected:
                    try:
                        self._ir.freeze_var_buffer_latest()
                        self.data = self._ir.get_all_data_dict()
                        self.last_update = time.time()
                    except Exception as e:
                        logger.debug(f"iRacing: data read error: {e}")
                        self._connected = False
                else:
                    self._connected = False
                
                time.sleep(0.01)  # 10ms update cycle
                
            except Exception as e:
                logger.error(f"iRacing: update loop error: {e}")
                self._connected = False
                time.sleep(0.5)

    @property
    def is_connected(self) -> bool:
        """Check if connected to iRacing"""
        return self._connected and self._ir is not None

    @property
    def is_active(self) -> bool:
        """Check if iRacing is running and not paused"""
        if not self.is_connected:
            return False
        try:
            return bool(self._ir.is_active)
        except Exception:
            return False

    def get_var(self, var_name: str, default=None):
        """Get variable value from iRacing data"""
        try:
            return self.data.get(var_name, default)
        except (AttributeError, TypeError):
            return default

    def get_float(self, var_name: str, default: float = 0.0) -> float:
        """Get float variable from iRacing"""
        try:
            val = self.data.get(var_name, default)
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def get_int(self, var_name: str, default: int = 0) -> int:
        """Get integer variable from iRacing"""
        try:
            val = self.data.get(var_name, default)
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default
