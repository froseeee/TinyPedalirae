#  TinyPedal is an open-source overlay application for racing simulation.
#  Copyright (C) 2022-2026 TinyPedal developers, see contributors.md file
#
#  This file is part of TinyPedal.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
iRacing API connector
"""

from __future__ import annotations

import logging
import threading
from time import monotonic, sleep
from typing import Any

try:
    import irsdk
except ImportError:
    irsdk = None

logger = logging.getLogger(__name__)


class SyncData:
    """Synchronize iRacing data

    Attributes:
        paused: Data update state (boolean).
        ir: iRacing SDK instance.
        session_data: Session info data (YAML parsed).
        telemetry_data: Live telemetry data.
    """

    __slots__ = (
        "_updating",
        "_update_thread",
        "_event",
        "paused",
        "ir",
        "session_data",
        "telemetry_data",
        "_state_override",
        "_active_state",
    )

    def __init__(self) -> None:
        self._updating = False
        self._update_thread = None
        self._event = threading.Event()
        
        self.paused = False
        self.ir = None
        self.session_data = {}
        self.telemetry_data = {}
        self._state_override = False
        self._active_state = False

    def __del__(self):
        logger.info("iRacing: GC: SyncData")

    def start(self) -> None:
        """Start data updating thread"""
        if self._updating:
            logger.warning("iRacing: UPDATING: already started")
        else:
            self._updating = True
            # Initialize iRacing SDK
            if irsdk is None:
                logger.error("iRacing: pyirsdk not installed")
                return
            
            self.ir = irsdk.IRSDK()
            # Setup updating thread
            self._event.clear()
            self._update_thread = threading.Thread(target=self.__update, daemon=True)
            self._update_thread.start()
            logger.info("iRacing: UPDATING: thread started")

    def stop(self) -> None:
        """Join and stop updating thread"""
        if self._updating:
            self._event.set()
            self._updating = False
            self._update_thread.join()
            if self.ir:
                self.ir.shutdown()
                self.ir = None
        else:
            logger.warning("iRacing: UPDATING: already stopped")

    def __update(self) -> None:
        """Update iRacing data in separate thread"""
        self.paused = False
        _event_wait = self._event.wait
        last_update_time = 0.0
        data_freezed = True
        update_delay = 0.5  # longer delay while inactive
        connection_retry_delay = 1.0
        is_connected = False

        while not _event_wait(update_delay):
            # Try to connect if not connected
            if not is_connected:
                if self.ir.startup():
                    is_connected = True
                    update_delay = 0.01
                    self.paused = data_freezed = False
                    logger.info("iRacing: UPDATING: connected")
                else:
                    update_delay = connection_retry_delay
                    continue

            # Check if still connected
            if not self.ir.is_connected:
                is_connected = False
                self.paused = data_freezed = True
                update_delay = connection_retry_delay
                logger.info("iRacing: UPDATING: disconnected, retrying...")
                continue

            # Update data
            if self.ir.is_initialized:
                # Get fresh telemetry data
                self.telemetry_data = self.ir.freeze_var_buffer_latest()
                
                # Update session info when it changes
                if self.ir.last_tick_count > 0:
                    session_info = self.ir['SessionInfo']
                    if session_info:
                        self.session_data = session_info
                
                last_update_time = monotonic()

                # Update pause state
                if data_freezed:
                    update_delay = 0.01
                    self.paused = data_freezed = False
                    logger.info("iRacing: UPDATING: resumed")
            else:
                # Check for freeze state
                if not data_freezed and monotonic() - last_update_time > 2:
                    update_delay = 0.5
                    self.paused = data_freezed = True
                    logger.info("iRacing: UPDATING: paused")

        logger.info("iRacing: UPDATING: thread stopped")


class iRacingInfo:
    """iRacing shared memory data output"""

    __slots__ = (
        "_sync",
        "_state_override",
        "_active_state",
    )

    def __init__(self) -> None:
        self._sync = SyncData()
        self._state_override = False
        self._active_state = False

    def __del__(self):
        logger.info("iRacing: GC: iRacingInfo")

    def start(self) -> None:
        """Start data updating thread"""
        self._sync.start()

    def stop(self) -> None:
        """Stop data updating thread"""
        self._sync.stop()

    def setStateOverride(self, state: bool = False) -> None:
        """Enable state override"""
        self._state_override = state
        self._sync._state_override = state

    def setActiveState(self, state: bool = False) -> None:
        """Set state override"""
        self._active_state = state
        self._sync._active_state = state

    def get(self, key: str, default: Any = None) -> Any:
        """Get telemetry value by key
        
        Args:
            key: Telemetry key name
            default: Default value if key not found
            
        Returns:
            Telemetry value or default
        """
        if not self._sync.telemetry_data:
            return default
        return self._sync.telemetry_data.get(key, default)

    def session_info(self, *keys) -> Any:
        """Get session info value by keys path
        
        Args:
            *keys: Path to nested value in session data
            
        Returns:
            Session info value or None
        """
        if not self._sync.session_data:
            return None
        
        data = self._sync.session_data
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
                if data is None:
                    return None
            elif isinstance(data, list) and isinstance(key, int):
                if 0 <= key < len(data):
                    data = data[key]
                else:
                    return None
            else:
                return None
        return data

    @property
    def isPaused(self) -> bool:
        """Check whether data stopped updating"""
        return self._sync.paused

    @property
    def isActive(self) -> bool:
        """Check whether in active (driving or overriding) state"""
        if self._state_override:
            return self._active_state
        
        if self._sync.paused or not self._sync.ir:
            return False
        
        # Check if player is on track and car is running
        is_on_track = self.get('IsOnTrack', False)
        engine_running = self.get('EngineWarnings', 0) >= 0
        
        return is_on_track and engine_running


def test_api():
    """API test run"""
    # Add logger
    test_handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(test_handler)

    # Test run
    SEPARATOR = "=" * 50
    print("Test API - Start")
    info = iRacingInfo()
    info.start()
    sleep(2)

    print(SEPARATOR)
    print("Test API - Read")
    driver = info.get('DriverInfo', {}).get('DriverUserName', 'N/A')
    track = info.session_info('WeekendInfo', 'TrackDisplayName')
    speed = info.get('Speed', 0)
    print(f"driver name: {driver}")
    print(f"track name : {track if track else 'not running'}")
    print(f"speed      : {speed:.2f} m/s")

    print(SEPARATOR)
    print("Test API - Close")
    info.stop()


if __name__ == "__main__":
    test_api()
