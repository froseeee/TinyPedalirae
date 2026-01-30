"""iRacing data reader for TinyPedal

Provides structured data access and conversion for iRacing telemetry
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class IRacingReader:
    """Process iRacing telemetry data"""

    # iRacing telemetry variable mapping
    MAPPING = {
        # Speed and motion
        "speed": "Speed",  # m/s
        "throttle": "Throttle",  # 0-1
        "brake": "Brake",  # 0-1
        "clutch": "Clutch",  # 0-1
        "gear": "Gear",  # -1=R, 0=N, 1+=gears
        "rpm": "RPM",
        "max_rpm": "EngineMaxRPM",
        
        # Steering and suspension
        "steering_angle": "SteeringWheelAngle",  # radians
        "steering_wheel_torque": "SteeringWheelTorque",  # Nm
        "suspension_travel_fl": "SuspensionTravel_0",  # front left
        "suspension_travel_fr": "SuspensionTravel_1",  # front right
        "suspension_travel_rl": "SuspensionTravel_2",  # rear left
        "suspension_travel_rr": "SuspensionTravel_3",  # rear right
        
        # Tires
        "tire_temp_fl": "TireTemp_0",  # celsius
        "tire_temp_fr": "TireTemp_1",
        "tire_temp_rl": "TireTemp_2",
        "tire_temp_rr": "TireTemp_3",
        "tire_wear_fl": "TireWear_0",  # 0-1
        "tire_wear_fr": "TireWear_1",
        "tire_wear_rl": "TireWear_2",
        "tire_wear_rr": "TireWear_3",
        
        # Fuel
        "fuel_level": "FuelLevel",  # liters
        "fuel_pressure": "FuelPressure",  # bar
        
        # Engine
        "engine_temp": "WaterTemp",  # celsius
        "oil_temp": "OilTemp",
        "oil_pressure": "OilPressure",  # bar
        
        # G-forces
        "accel_x": "LongAccel",
        "accel_y": "LatAccel",
        "accel_z": "VertAccel",
        
        # Lap info
        "lap": "Lap",
        "lap_dist_pct": "LapDistPct",  # 0-1
        "current_lap_time": "LapCurrentLapTime",  # seconds
        "last_lap_time": "LastLapTime",
        "best_lap_time": "BestLapTime",
        
        # Session info
        "session_time": "SessionTime",  # seconds
        "session_state": "SessionState",
        "session_type": "SessionType",
        
        # Driver info
        "is_on_track": "IsOnTrack",
        "is_paused": "IsPaused",
    }

    def __init__(self, connector) -> None:
        """Initialize iRacing reader
        
        Args:
            connector: IRacingConnector instance
        """
        self.connector = connector
        self._last_data = {}

    def read(self) -> Dict[str, Any]:
        """Read and process current iRacing data
        
        Returns:
            Dictionary with processed telemetry data
        """
        data = {}
        
        if not self.connector.is_connected:
            return data
        
        try:
            # Read mapped variables
            for key, var_name in self.MAPPING.items():
                value = self.connector.get_var(var_name)
                if value is not None:
                    data[key] = value
            
            # Additional computed values
            data["connected"] = True
            data["active"] = self.connector.is_active
            data["timestamp"] = self.connector.last_update
            
            self._last_data = data
            return data
            
        except Exception as e:
            logger.error(f"iRacing: read error: {e}")
            return {"connected": False}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from last read data
        
        Args:
            key: Data key
            default: Default value if not found
            
        Returns:
            Data value or default
        """
        return self._last_data.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value from last read data"""
        try:
            val = self._last_data.get(key, default)
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value from last read data"""
        try:
            val = self._last_data.get(key, default)
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default
