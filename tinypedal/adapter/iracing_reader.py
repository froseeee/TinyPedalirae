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
iRacing API data reader

Notes:
    iRacing uses different coordinate system and units compared to RF2.
    Temperatures are already in Celsius.
    Pressures are in kPa.
    Most measurements follow standard conventions.
"""

from __future__ import annotations

import math

from ..calculation import (
    lap_progress_distance,
    mean,
    min_nonzero,
    oriyaw2rad,
    slip_angle,
    vel2speed,
)
from ..const_common import MAX_SECONDS, STINT_USAGE_DEFAULT
from ..formatter import strip_invalid_char
from ..process.weather import WeatherNode
from ..validator import infnan_to_zero as rmnan
from . import _reader, iracing_connector


class DataAdapter:
    """Read & sort data into groups"""

    __slots__ = (
        "ir",
    )

    def __init__(self, ir: iracing_connector.iRacingInfo) -> None:
        """Initialize API setting

        Args:
            ir: iRacing API connector.
        """
        self.ir = ir


class State(_reader.State, DataAdapter):
    """State"""

    __slots__ = ()

    def active(self) -> bool:
        """Is active (driving or overriding)"""
        return self.ir.isActive

    def paused(self) -> bool:
        """Is paused"""
        return self.ir.isPaused

    def desynced(self, index: int | None = None) -> bool:
        """Is player data desynced from others"""
        # iRacing typically keeps all data in sync
        return False

    def version(self) -> str:
        """Identify API version"""
        version = self.ir.session_info('WeekendInfo', 'TrackVersion')
        return str(version) if version else "unknown"


class Brake(_reader.Brake, DataAdapter):
    """Brake"""

    __slots__ = ()

    def bias_front(self, index: int | None = None) -> float:
        """Brake bias front (fraction)"""
        bias = self.ir.get('dcBrakeBias', 0.5)
        return rmnan(bias)

    def pressure(self, index: int | None = None, scale: float = 1) -> tuple[float, ...]:
        """Brake pressure (fraction)"""
        # iRacing reports brake as a single value, replicate to all wheels
        brake = self.ir.get('Brake', 0.0) * scale
        return (brake, brake, brake, brake)

    def temperature(self, index: int | None = None) -> tuple[float, ...]:
        """Brake temperature (Celsius)"""
        # iRacing provides brake temperatures
        lf_temp = rmnan(self.ir.get('LFbrakeLinePress', 0.0))  # Using line pressure as proxy
        rf_temp = rmnan(self.ir.get('RFbrakeLinePress', 0.0))
        lr_temp = rmnan(self.ir.get('LRbrakeLinePress', 0.0))
        rr_temp = rmnan(self.ir.get('RRbrakeLinePress', 0.0))
        # Estimate temperature from pressure (rough approximation)
        return (lf_temp * 100, rf_temp * 100, lr_temp * 100, rr_temp * 100)

    def wear(self, index: int | None = None) -> tuple[float, ...]:
        """Brake remaining thickness (meters)"""
        # iRacing doesn't provide brake wear directly, return nominal values
        return (0.02, 0.02, 0.02, 0.02)


class ElectricMotor(_reader.ElectricMotor, DataAdapter):
    """Electric motor"""

    __slots__ = ()

    def state(self, index: int | None = None) -> int:
        """Motor state, 0 = n/a, 1 = off, 2 = drain, 3 = regen"""
        # iRacing has ERS data for hybrid cars
        ers_deployed = self.ir.get('EnergyERSBatteryDeployed', 0.0)
        if ers_deployed > 0:
            return 2  # drain
        elif ers_deployed < 0:
            return 3  # regen
        return 1  # off

    def battery_charge(self, index: int | None = None) -> float:
        """Battery charge (fraction)"""
        battery_pct = self.ir.get('EnergyBatteryToMGU_pct', 0.0)
        return rmnan(battery_pct)

    def rpm(self, index: int | None = None) -> float:
        """Motor RPM (rev per minute)"""
        # iRacing typically doesn't provide separate motor RPM
        return 0.0

    def torque(self, index: int | None = None) -> float:
        """Motor torque (Nm)"""
        ers_torque = self.ir.get('EnergyERSBatteryDeployed', 0.0)
        return rmnan(ers_torque)

    def motor_temperature(self, index: int | None = None) -> float:
        """Motor temperature (Celsius)"""
        # Not typically available in iRacing
        return 0.0

    def water_temperature(self, index: int | None = None) -> float:
        """Motor water temperature (Celsius)"""
        # Not typically available in iRacing
        return 0.0


class Engine(_reader.Engine, DataAdapter):
    """Engine"""

    __slots__ = ()

    def gear(self, index: int | None = None) -> int:
        """Gear"""
        return self.ir.get('Gear', 0)

    def gear_max(self, index: int | None = None) -> int:
        """Max gear"""
        # Get from car info
        driver_car_idx = self.ir.get('PlayerCarIdx', 0)
        car_info = self.ir.session_info('DriverInfo', 'Drivers', driver_car_idx)
        if car_info:
            # Estimate from car name or default to 6
            return 6
        return 6

    def rpm(self, index: int | None = None) -> float:
        """RPM (rev per minute)"""
        return rmnan(self.ir.get('RPM', 0.0))

    def rpm_max(self, index: int | None = None) -> float:
        """Max RPM (rev per minute)"""
        return rmnan(self.ir.get('EngineMaxRPM', 10000.0))

    def torque(self, index: int | None = None) -> float:
        """Torque (Nm)"""
        # iRacing doesn't directly provide torque, estimate from power
        rpm = self.rpm(index)
        if rpm > 0:
            power = self.ir.get('Power', 0.0)  # in watts
            torque = power / (rpm * 2 * math.pi / 60)
            return rmnan(torque)
        return 0.0

    def turbo(self, index: int | None = None) -> float:
        """Turbo pressure (Pa)"""
        manifold_press = self.ir.get('ManifoldPress', 0.0)  # in bar
        return rmnan(manifold_press * 100000)  # convert bar to Pa

    def oil_temperature(self, index: int | None = None) -> float:
        """Oil temperature (Celsius)"""
        return rmnan(self.ir.get('OilTemp', 0.0))

    def water_temperature(self, index: int | None = None) -> float:
        """Water temperature (Celsius)"""
        return rmnan(self.ir.get('WaterTemp', 0.0))


class Inputs(_reader.Inputs, DataAdapter):
    """Inputs"""

    __slots__ = ()

    def throttle(self, index: int | None = None) -> float:
        """Throttle filtered (fraction)"""
        return rmnan(self.ir.get('Throttle', 0.0))

    def throttle_raw(self, index: int | None = None) -> float:
        """Throttle raw (fraction)"""
        return rmnan(self.ir.get('ThrottleRaw', self.ir.get('Throttle', 0.0)))

    def brake(self, index: int | None = None) -> float:
        """Brake filtered (fraction)"""
        return rmnan(self.ir.get('Brake', 0.0))

    def brake_raw(self, index: int | None = None) -> float:
        """Brake raw (fraction)"""
        return rmnan(self.ir.get('BrakeRaw', self.ir.get('Brake', 0.0)))

    def clutch(self, index: int | None = None) -> float:
        """Clutch filtered (fraction)"""
        return rmnan(self.ir.get('Clutch', 0.0))

    def clutch_raw(self, index: int | None = None) -> float:
        """Clutch raw (fraction)"""
        return rmnan(self.ir.get('ClutchRaw', self.ir.get('Clutch', 0.0)))

    def steering(self, index: int | None = None) -> float:
        """Steering filtered (fraction)"""
        steering_angle = self.ir.get('SteeringWheelAngle', 0.0)  # in radians
        steering_max = self.ir.get('SteeringWheelAngleMax', math.pi)
        if steering_max > 0:
            return rmnan(steering_angle / steering_max)
        return 0.0

    def steering_raw(self, index: int | None = None) -> float:
        """Steering raw (fraction)"""
        # iRacing doesn't separate raw/filtered steering
        return self.steering(index)

    def steering_shaft_torque(self, index: int | None = None) -> float:
        """Steering shaft torque (Nm)"""
        return rmnan(self.ir.get('SteeringWheelTorque', 0.0))

    def steering_range_physical(self, index: int | None = None) -> float:
        """Steering physical rotation range (degrees)"""
        angle_max = self.ir.get('SteeringWheelAngleMax', math.pi)  # in radians
        return rmnan(math.degrees(angle_max * 2))  # full range left to right

    def steering_range_visual(self, index: int | None = None) -> float:
        """Steering visual rotation range (degrees)"""
        # Same as physical for iRacing
        return self.steering_range_physical(index)

    def force_feedback(self) -> float:
        """Steering force feedback (fraction)"""
        torque = self.ir.get('SteeringWheelTorque', 0.0)
        max_torque = self.ir.get('SteeringWheelTorque_ST', [30.0])
        if isinstance(max_torque, list) and len(max_torque) > 0:
            max_torque = max_torque[0]
        if max_torque > 0:
            return rmnan(torque / max_torque)
        return 0.0


class Lap(_reader.Lap, DataAdapter):
    """Lap"""

    __slots__ = ()

    def number(self, index: int | None = None) -> int:
        """Current lap number"""
        return self.ir.get('Lap', 0)

    def completed_laps(self, index: int | None = None) -> int:
        """Total completed laps"""
        return self.ir.get('LapsComplete', 0)

    def track_length(self) -> float:
        """Full lap or track length (meters)"""
        track_length = self.ir.session_info('WeekendInfo', 'TrackLength')
        if track_length:
            # Convert from km to meters if needed
            length_str = str(track_length).split()[0]
            try:
                length = float(length_str)
                # If it's in km (small number), convert to meters
                if length < 100:  # assume km
                    return length * 1000
                return length
            except:
                pass
        return 1000.0  # default fallback

    def distance(self, index: int | None = None) -> float:
        """Distance into lap (meters)"""
        lap_dist = self.ir.get('LapDist', 0.0)
        track_length = self.track_length()
        # LapDist is percentage, convert to meters
        return rmnan(lap_dist * track_length)

    def progress(self, index: int | None = None) -> float:
        """Lap progress (fraction), distance into lap"""
        return rmnan(self.ir.get('LapDistPct', 0.0))

    def maximum(self) -> int:
        """Maximum lap"""
        session_laps = self.ir.session_info('SessionInfo', 'Sessions')
        if session_laps and isinstance(session_laps, list):
            for session in session_laps:
                if isinstance(session, dict) and session.get('ResultsOfficial') == 0:
                    laps = session.get('SessionLaps', 0)
                    if isinstance(laps, str):
                        if laps == 'unlimited':
                            return 999
                        try:
                            return int(laps)
                        except:
                            pass
        return 999

    def sector_index(self, index: int | None = None) -> int:
        """Sector index, 0 = S1, 1 = S2, 2 = S3"""
        # iRacing doesn't directly provide sector, calculate from lap progress
        progress = self.progress(index)
        if progress < 0.333:
            return 0
        elif progress < 0.666:
            return 1
        return 2

    def behind_leader(self, index: int | None = None) -> int:
        """Laps behind leader"""
        player_lap = self.number(index)
        leader_lap = self.ir.get('LeaderLap', player_lap)
        return leader_lap - player_lap

    def behind_next(self, index: int | None = None) -> int:
        """Laps behind next place"""
        # iRacing doesn't directly provide this, estimate from positions
        return 0


class Session(_reader.Session, DataAdapter):
    """Session"""

    __slots__ = ()

    def combo_name(self) -> str:
        """Track & vehicle combo name, strip off invalid char"""
        track_name = self.ir.session_info('WeekendInfo', 'TrackName')
        car_name = self.ir.session_info('DriverInfo', 'Drivers', 0, 'CarScreenName')
        return strip_invalid_char(f"{track_name} - {car_name}")

    def track_name(self) -> str:
        """Track name, strip off invalid char"""
        track = self.ir.session_info('WeekendInfo', 'TrackDisplayName')
        if not track:
            track = self.ir.session_info('WeekendInfo', 'TrackName')
        return strip_invalid_char(str(track) if track else "Unknown Track")

    def identifier(self) -> tuple[int, int, int]:
        """Identify session"""
        session_num = self.ir.get('SessionNum', 0)
        session_time = int(self.elapsed())
        session_laps = self.ir.get('LapsComplete', 0)
        return session_num, session_time, session_laps

    def elapsed(self) -> float:
        """Session elapsed time (seconds)"""
        return rmnan(self.ir.get('SessionTime', 0.0))

    def start(self) -> float:
        """Session start time (seconds)"""
        # iRacing doesn't provide session start time directly
        return 0.0

    def end(self) -> float:
        """Session end time (seconds)"""
        remaining = self.remaining()
        elapsed = self.elapsed()
        return rmnan(elapsed + remaining)

    def remaining(self) -> float:
        """Session time remaining (seconds)"""
        return rmnan(self.ir.get('SessionTimeRemain', 999999.0))

    def session_type(self) -> int:
        """Session type, 0 = TESTDAY, 1 = PRACTICE, 2 = QUALIFY, 3 = WARMUP, 4 = RACE"""
        session_info = self.ir.session_info('SessionInfo', 'Sessions')
        if session_info and isinstance(session_info, list):
            session_num = self.ir.get('SessionNum', 0)
            if 0 <= session_num < len(session_info):
                session = session_info[session_num]
                if isinstance(session, dict):
                    session_type = session.get('SessionType', '')
                    if 'Race' in session_type:
                        return 4
                    if 'Qualify' in session_type:
                        return 2
                    if 'Practice' in session_type:
                        return 1
                    if 'Warmup' in session_type:
                        return 3
        return 0  # default to test day

    def lap_type(self) -> bool:
        """Is lap type session, false for time type"""
        max_laps = self.maximum()
        return max_laps < 999

    def in_race(self) -> bool:
        """Is in race session"""
        return self.session_type() == 4

    def private_qualifying(self) -> bool:
        """Is private qualifying"""
        # iRacing doesn't have private qualifying in the same sense
        return False

    def in_countdown(self) -> bool:
        """Is in countdown phase before race"""
        session_state = self.ir.get('SessionState', 0)
        return session_state == 3  # Getting ready

    def in_formation(self) -> bool:
        """Is in formation phase before race"""
        session_state = self.ir.get('SessionState', 0)
        return session_state == 7  # Parade laps

    def pit_open(self) -> bool:
        """Is pit lane open"""
        # iRacing pits are generally always open except in specific phases
        session_state = self.ir.get('SessionState', 0)
        return session_state >= 4  # Racing state or later

    def pre_race(self) -> bool:
        """Before race starts (green flag)"""
        session_state = self.ir.get('SessionState', 0)
        return session_state < 4  # Before green flag

    def green_flag(self) -> bool:
        """Green flag (race starts)"""
        session_flags = self.ir.get('SessionFlags', 0)
        return bool(session_flags & 0x00008000)  # Green flag bit

    def blue_flag(self, index: int | None = None) -> bool:
        """Is under blue flag"""
        car_idx = self.ir.get('PlayerCarIdx', 0) if index is None else index
        car_idx_flags = self.ir.get('CarIdxLapDistPct', [0] * 64)
        if car_idx < len(car_idx_flags):
            # Check if being lapped
            player_progress = self.progress()
            leader_progress = max(car_idx_flags)
            return leader_progress - player_progress > 0.5
        return False

    def yellow_flag(self) -> bool:
        """Is there yellow flag in any sectors"""
        session_flags = self.ir.get('SessionFlags', 0)
        # Check for caution, yellow, or caution waving flags
        return bool(session_flags & (0x00000040 | 0x00000080 | 0x00000100))

    def start_lights(self) -> int:
        """Start lights countdown sequence"""
        # iRacing doesn't have visible start lights like RF2
        session_state = self.ir.get('SessionState', 0)
        if session_state == 3:  # Getting ready
            return 5
        return 0

    def track_temperature(self) -> float:
        """Track temperature (Celsius)"""
        return rmnan(self.ir.get('TrackTempCrew', 20.0))

    def ambient_temperature(self) -> float:
        """Ambient temperature (Celsius)"""
        return rmnan(self.ir.get('AirTemp', 20.0))

    def raininess(self) -> float:
        """Rain severity (fraction)"""
        # iRacing weather info
        skies = self.ir.get('Skies', 0)
        if skies >= 3:  # Overcast or worse
            return 0.5
        return 0.0

    def wetness_minimum(self) -> float:
        """Road minimum wetness (fraction)"""
        return 0.0  # iRacing doesn't provide detailed wetness data

    def wetness_maximum(self) -> float:
        """Road maximum wetness (fraction)"""
        return 0.0

    def wetness_average(self) -> float:
        """Road average wetness (fraction)"""
        return 0.0

    def wetness(self) -> tuple[float, float, float]:
        """Road wetness set (fraction)"""
        return (0.0, 0.0, 0.0)

    def weather_forecast(self) -> tuple[WeatherNode, ...]:
        """Weather forecast nodes"""
        # iRacing doesn't provide weather forecast in the same detail
        return tuple()

    def time_scale(self) -> int:
        """Time scale"""
        # iRacing uses real time
        return 1


class Switch(_reader.Switch, DataAdapter):
    """Switch"""

    __slots__ = ()

    def headlights(self, index: int | None = None) -> int:
        """Headlights"""
        # iRacing doesn't expose headlights state directly
        return 0

    def ignition_starter(self, index: int | None = None) -> int:
        """Ignition"""
        engine_warnings = self.ir.get('EngineWarnings', 0)
        # If engine is running
        if engine_warnings >= 0:
            return 1
        return 0

    def speed_limiter(self, index: int | None = None) -> int:
        """Speed limiter"""
        on_pit_road = self.ir.get('OnPitRoad', False)
        return 1 if on_pit_road else 0

    def drs_status(self, index: int | None = None) -> int:
        """DRS status, 0 not_available, 1 available, 2 allowed(not activated), 3 activated"""
        # iRacing doesn't have DRS in most cars
        return 0

    def auto_clutch(self) -> bool:
        """Auto clutch"""
        # Check if using auto clutch
        driver_info = self.ir.session_info('DriverInfo', 'DriverSetupName')
        # This is an approximation
        return True


class Timing(_reader.Timing, DataAdapter):
    """Timing"""

    __slots__ = ()

    def start(self, index: int | None = None) -> float:
        """Current lap start time (seconds)"""
        elapsed = self.elapsed(index)
        current_lap_time = self.current_laptime(index)
        return rmnan(elapsed - current_lap_time)

    def elapsed(self, index: int | None = None) -> float:
        """Current lap elapsed time (seconds)"""
        return rmnan(self.ir.get('SessionTime', 0.0))

    def current_laptime(self, index: int | None = None) -> float:
        """Current lap time (seconds)"""
        return rmnan(self.ir.get('LapCurrentLapTime', 0.0))

    def last_laptime(self, index: int | None = None) -> float:
        """Last lap time (seconds)"""
        return rmnan(self.ir.get('LapLastLapTime', 0.0))

    def best_laptime(self, index: int | None = None) -> float:
        """Best lap time (seconds)"""
        return rmnan(self.ir.get('LapBestLapTime', 0.0))

    def reference_laptime(self, index: int | None = None):
        """Reference lap time (seconds)"""
        init_time = min_nonzero((
            self.best_laptime(index),
            self.last_laptime(index),
            MAX_SECONDS,
        ))
        if 0 < init_time < MAX_SECONDS:
            return init_time
        return MAX_SECONDS

    def estimated_laptime(self, index: int | None = None) -> float:
        """Estimated lap time (seconds)"""
        # Use best lap as estimate
        best = self.best_laptime(index)
        if best > 0:
            return best
        return MAX_SECONDS

    def estimated_time_into(self, index: int | None = None) -> float:
        """Estimated time into lap (seconds)"""
        return self.current_laptime(index)

    def current_sector1(self, index: int | None = None) -> float:
        """Current lap sector 1 time (seconds)"""
        # iRacing doesn't provide sector times in the same way
        return 0.0

    def current_sector2(self, index: int | None = None) -> float:
        """Current lap sector 1+2 time (seconds)"""
        return 0.0

    def last_sector1(self, index: int | None = None) -> float:
        """Last lap sector 1 time (seconds)"""
        return 0.0

    def last_sector2(self, index: int | None = None) -> float:
        """Last lap sector 1+2 time (seconds)"""
        return 0.0

    def best_sector1(self, index: int | None = None) -> float:
        """Best lap sector 1 time (seconds)"""
        return 0.0

    def best_sector2(self, index: int | None = None) -> float:
        """Best lap sector 1+2 time (seconds)"""
        return 0.0

    def behind_leader(self, index: int | None = None) -> float:
        """Time behind leader (seconds)"""
        lap_behind = self.ir.get('LapBehindLeader', 0)
        if lap_behind < 0:
            # Ahead of leader
            return rmnan(lap_behind)
        # Convert laps to time estimate
        best_lap = self.best_laptime()
        return rmnan(lap_behind * best_lap)

    def behind_next(self, index: int | None = None) -> float:
        """Time behind next place (seconds)"""
        # iRacing provides delta to car ahead
        car_idx_lap_dist = self.ir.get('CarIdxLapDistPct', [0] * 64)
        player_idx = self.ir.get('PlayerCarIdx', 0)
        if player_idx < len(car_idx_lap_dist):
            player_dist = car_idx_lap_dist[player_idx]
            # Find car just ahead
            ahead_dist = player_dist
            for dist in car_idx_lap_dist:
                if player_dist < dist < ahead_dist + 1:
                    ahead_dist = dist
            if ahead_dist > player_dist:
                # Convert distance to time
                track_length = self.track_length()
                speed = self.ir.get('Speed', 1.0)
                if speed > 0:
                    dist_diff = (ahead_dist - player_dist) * track_length
                    return rmnan(dist_diff / speed)
        return 0.0


class Tyre(_reader.Tyre, DataAdapter):
    """Tyre"""

    __slots__ = ()

    def compound_front(self, index: int | None = None) -> int:
        """Tyre compound (front)"""
        # iRacing doesn't separate front/rear compounds in the same way
        return 0

    def compound_rear(self, index: int | None = None) -> int:
        """Tyre compound (rear)"""
        return 0

    def compound(self, index: int | None = None) -> tuple[int, int]:
        """Tyre compound set (front, rear)"""
        return (0, 0)

    def compound_name_front(self, index: int | None = None) -> str:
        """Tyre compound name (front)"""
        return "Unknown"

    def compound_name_rear(self, index: int | None = None) -> str:
        """Tyre compound name (rear)"""
        return "Unknown"

    def compound_name(self, index: int | None = None) -> tuple[str, str]:
        """Tyre compound name set (front, rear)"""
        return ("Unknown", "Unknown")

    def surface_temperature_avg(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre surface temperature set (Celsius) average"""
        lf_temp = mean([self.ir.get('LFtempCL', 0.0), 
                       self.ir.get('LFtempCM', 0.0), 
                       self.ir.get('LFtempCR', 0.0)])
        rf_temp = mean([self.ir.get('RFtempCL', 0.0), 
                       self.ir.get('RFtempCM', 0.0), 
                       self.ir.get('RFtempCR', 0.0)])
        lr_temp = mean([self.ir.get('LRtempCL', 0.0), 
                       self.ir.get('LRtempCM', 0.0), 
                       self.ir.get('LRtempCR', 0.0)])
        rr_temp = mean([self.ir.get('RRtempCL', 0.0), 
                       self.ir.get('RRtempCM', 0.0), 
                       self.ir.get('RRtempCR', 0.0)])
        return (rmnan(lf_temp), rmnan(rf_temp), rmnan(lr_temp), rmnan(rr_temp))

    def surface_temperature_ico(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre surface temperature set (Celsius) inner,center,outer"""
        return (
            rmnan(self.ir.get('LFtempCL', 0.0)),
            rmnan(self.ir.get('LFtempCM', 0.0)),
            rmnan(self.ir.get('LFtempCR', 0.0)),
            rmnan(self.ir.get('RFtempCL', 0.0)),
            rmnan(self.ir.get('RFtempCM', 0.0)),
            rmnan(self.ir.get('RFtempCR', 0.0)),
            rmnan(self.ir.get('LRtempCL', 0.0)),
            rmnan(self.ir.get('LRtempCM', 0.0)),
            rmnan(self.ir.get('LRtempCR', 0.0)),
            rmnan(self.ir.get('RRtempCL', 0.0)),
            rmnan(self.ir.get('RRtempCM', 0.0)),
            rmnan(self.ir.get('RRtempCR', 0.0)),
        )

    def inner_temperature_avg(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre inner temperature set (Celsius) average"""
        # iRacing doesn't separate inner temps, use surface temps
        return self.surface_temperature_avg(index)

    def inner_temperature_ico(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre inner temperature set (Celsius) inner,center,outer"""
        return self.surface_temperature_ico(index)

    def pressure(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre pressure (kPa)"""
        return (
            rmnan(self.ir.get('LFpressure', 0.0)),
            rmnan(self.ir.get('RFpressure', 0.0)),
            rmnan(self.ir.get('LRpressure', 0.0)),
            rmnan(self.ir.get('RRpressure', 0.0)),
        )

    def load(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre load (Newtons)"""
        # iRacing provides tire load
        return (
            rmnan(self.ir.get('LFshockForce', 0.0)),
            rmnan(self.ir.get('RFshockForce', 0.0)),
            rmnan(self.ir.get('LRshockForce', 0.0)),
            rmnan(self.ir.get('RRshockForce', 0.0)),
        )

    def wear(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre wear (fraction)"""
        return (
            rmnan(self.ir.get('LFwearL', 0.0) + self.ir.get('LFwearM', 0.0) + self.ir.get('LFwearR', 0.0)) / 3,
            rmnan(self.ir.get('RFwearL', 0.0) + self.ir.get('RFwearM', 0.0) + self.ir.get('RFwearR', 0.0)) / 3,
            rmnan(self.ir.get('LRwearL', 0.0) + self.ir.get('LRwearM', 0.0) + self.ir.get('LRwearR', 0.0)) / 3,
            rmnan(self.ir.get('RRwearL', 0.0) + self.ir.get('RRwearM', 0.0) + self.ir.get('RRwearR', 0.0)) / 3,
        )

    def carcass_temperature(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre carcass temperature (Celsius)"""
        # Use surface temps as approximation
        return self.surface_temperature_avg(index)

    def vertical_deflection(self, index: int | None = None) -> tuple[float, ...]:
        """Tyre vertical deflection (millimeters)"""
        # iRacing doesn't provide this directly
        return (0.0, 0.0, 0.0, 0.0)


class Vehicle(_reader.Vehicle, DataAdapter):
    """Vehicle"""

    __slots__ = ()

    def is_player(self, index: int=0) -> bool:
        """Is local player"""
        player_idx = self.ir.get('PlayerCarIdx', -1)
        return player_idx == index

    def is_driving(self) -> bool:
        """Is local player driving or in monitor"""
        return self.ir.get('IsOnTrack', False)

    def player_index(self) -> int:
        """Get Local player index"""
        return self.ir.get('PlayerCarIdx', 0)

    def slot_id(self, index: int | None = None) -> int:
        """Vehicle slot id"""
        if index is None:
            index = self.player_index()
        return index

    def driver_name(self, index: int | None = None) -> str:
        """Driver name"""
        if index is None:
            index = self.player_index()
        drivers = self.ir.session_info('DriverInfo', 'Drivers')
        if drivers and isinstance(drivers, list) and 0 <= index < len(drivers):
            driver = drivers[index]
            if isinstance(driver, dict):
                return str(driver.get('UserName', 'Unknown'))
        return "Unknown"

    def vehicle_name(self, index: int | None = None) -> str:
        """Vehicle name"""
        if index is None:
            index = self.player_index()
        drivers = self.ir.session_info('DriverInfo', 'Drivers')
        if drivers and isinstance(drivers, list) and 0 <= index < len(drivers):
            driver = drivers[index]
            if isinstance(driver, dict):
                return str(driver.get('CarScreenName', 'Unknown'))
        return "Unknown"

    def class_name(self, index: int | None = None) -> str:
        """Vehicle class name"""
        if index is None:
            index = self.player_index()
        drivers = self.ir.session_info('DriverInfo', 'Drivers')
        if drivers and isinstance(drivers, list) and 0 <= index < len(drivers):
            driver = drivers[index]
            if isinstance(driver, dict):
                return str(driver.get('CarClassShortName', 'Unknown'))
        return "Unknown"

    def same_class(self, index: int | None = None) -> bool:
        """Is same vehicle class"""
        player_class = self.class_name()
        other_class = self.class_name(index)
        return player_class == other_class

    def total_vehicles(self) -> int:
        """Total vehicles"""
        drivers = self.ir.session_info('DriverInfo', 'Drivers')
        if drivers and isinstance(drivers, list):
            return len(drivers)
        return 0

    def place(self, index: int | None = None) -> int:
        """Vehicle overall place"""
        if index is None:
            return self.ir.get('PlayerCarPosition', 0)
        car_idx_position = self.ir.get('CarIdxPosition', [0] * 64)
        if index < len(car_idx_position):
            return car_idx_position[index]
        return 0

    def qualification(self, index: int | None = None) -> int:
        """Vehicle qualification place"""
        # Use driver info from session
        if index is None:
            index = self.player_index()
        drivers = self.ir.session_info('DriverInfo', 'Drivers')
        if drivers and isinstance(drivers, list) and 0 <= index < len(drivers):
            driver = drivers[index]
            if isinstance(driver, dict):
                return driver.get('CarIdxQualifyPosition', 0) + 1
        return 0

    def in_pits(self, index: int | None = None) -> bool:
        """Is in pits"""
        return self.ir.get('OnPitRoad', False)

    def in_garage(self, index: int | None = None) -> bool:
        """Is in garage"""
        # iRacing doesn't distinguish garage from pits in same way
        return not self.ir.get('IsOnTrack', True) and self.ir.get('OnPitRoad', False)

    def in_paddock(self, index: int | None = None) -> int:
        """Is in paddock (either pit lane or garage), 0 = on track, 1 = pit lane, 2 = garage"""
        if self.in_garage(index):
            return 2
        if self.in_pits(index):
            return 1
        return 0

    def number_pitstops(self, index: int | None = None, penalty: int = 0) -> int:
        """Number of pit stops"""
        return self.ir.get('PlayerCarPitSvLap', 0)

    def number_penalties(self, index: int | None = None) -> int:
        """Number of penalties"""
        # iRacing doesn't expose penalty count directly
        return 0

    def pit_request(self, index: int | None = None) -> bool:
        """Is requested pit"""
        pit_sv_status = self.ir.get('PitSvStatus', 0)
        return pit_sv_status > 0

    def pit_stop_time(self) -> float:
        """Estimated pit stop time (seconds)"""
        # Estimate typical pit stop time
        return 30.0

    def absolute_refill(self) -> float:
        """Absolute refill fuel (liter) or virtual energy (percent)"""
        return 0.0

    def stint_usage(self, driver_name: str) -> tuple[float, float, float, float, int]:
        """Stint usage data"""
        return STINT_USAGE_DEFAULT

    def finish_state(self, index: int | None = None) -> int:
        """Finish state, 0 = none, 1 = finished, 2 = DNF, 3 = DQ"""
        car_idx_track_surface = self.ir.get('CarIdxTrackSurface', [0] * 64)
        if index is None:
            index = self.player_index()
        if index < len(car_idx_track_surface):
            surface = car_idx_track_surface[index]
            if surface == -1:  # Not in world
                return 2  # DNF
        return 0

    def fuel(self, index: int | None = None) -> float:
        """Remaining fuel (liters)"""
        fuel_level = self.ir.get('FuelLevel', 0.0)
        # iRacing fuel is in liters or gallons depending on settings
        return rmnan(fuel_level)

    def tank_capacity(self, index: int | None = None) -> float:
        """Fuel tank capacity (liters)"""
        return rmnan(self.ir.get('FuelLevelMax', 100.0))

    def virtual_energy(self, index: int | None = None) -> float:
        """Remaining virtual energy (joule)"""
        return 0.0

    def max_virtual_energy(self, index: int | None = None) -> float:
        """Max virtual energy (joule)"""
        return 0.0

    def orientation_yaw_radians(self, index: int | None = None) -> float:
        """Orientation yaw (radians)"""
        yaw = self.ir.get('Yaw', 0.0)
        return rmnan(yaw)

    def position_xyz(self, index: int | None = None) -> tuple[float, float, float]:
        """Raw x,y,z position (meters)"""
        x = self.ir.get('CarIdxX', [0] * 64)
        y = self.ir.get('CarIdxY', [0] * 64)
        z = self.ir.get('CarIdxZ', [0] * 64)
        if index is None:
            index = self.player_index()
        if index < len(x):
            return (rmnan(x[index]), rmnan(y[index]), rmnan(z[index]))
        return (0.0, 0.0, 0.0)

    def position_longitudinal(self, index: int | None = None) -> float:
        """Longitudinal axis position (meters) related to world plane"""
        return self.position_xyz(index)[0]

    def position_lateral(self, index: int | None = None) -> float:
        """Lateral axis position (meters) related to world plane"""
        return self.position_xyz(index)[1]

    def position_vertical(self, index: int | None = None) -> float:
        """Vertical axis position (meters) related to world plane"""
        return self.position_xyz(index)[2]

    def accel_lateral(self, index: int | None = None) -> float:
        """Lateral acceleration (m/s^2)"""
        return rmnan(self.ir.get('LatAccel', 0.0))

    def accel_longitudinal(self, index: int | None = None) -> float:
        """Longitudinal acceleration (m/s^2)"""
        return rmnan(self.ir.get('LongAccel', 0.0))

    def accel_vertical(self, index: int | None = None) -> float:
        """Vertical acceleration (m/s^2)"""
        return rmnan(self.ir.get('VertAccel', 0.0))

    def velocity_lateral(self, index: int | None = None) -> float:
        """Lateral velocity (m/s) x"""
        return rmnan(self.ir.get('VelocityX', 0.0))

    def velocity_longitudinal(self, index: int | None = None) -> float:
        """Longitudinal velocity (m/s) y"""
        return rmnan(self.ir.get('VelocityY', 0.0))

    def velocity_vertical(self, index: int | None = None) -> float:
        """Vertical velocity (m/s) z"""
        return rmnan(self.ir.get('VelocityZ', 0.0))

    def speed(self, index: int | None = None) -> float:
        """Speed (m/s)"""
        return rmnan(self.ir.get('Speed', 0.0))

    def downforce_front(self, index: int | None = None) -> float:
        """Downforce front (Newtons)"""
        # iRacing doesn't provide downforce directly
        return 0.0

    def downforce_rear(self, index: int | None = None) -> float:
        """Downforce rear (Newtons)"""
        return 0.0

    def damage_severity(self, index: int | None = None) -> tuple[int, int, int, int, int, int, int, int]:
        """Damage severity, sort row by row from left to right, top to bottom"""
        # iRacing doesn't provide detailed damage model
        return (0, 0, 0, 0, 0, 0, 0, 0)

    def aero_damage(self, index: int | None = None) -> float:
        """Aerodynamic damage (fraction), 0.0 no damage, 1.0 totaled"""
        return 0.0

    def integrity(self, index: int | None = None) -> float:
        """Vehicle integrity"""
        player_car_tow_time = self.ir.get('PlayerCarTowTime', 0.0)
        if player_car_tow_time > 0:
            return 0.0  # Being towed = totaled
        return 1.0

    def is_detached(self, index: int | None = None) -> bool:
        """Whether any vehicle parts are detached"""
        return False

    def impact_time(self, index: int | None = None) -> float:
        """Last impact time stamp (seconds)"""
        return 0.0

    def impact_magnitude(self, index: int | None = None) -> float:
        """Last impact magnitude"""
        return 0.0

    def impact_position(self, index: int | None = None) -> tuple[float, float]:
        """Last impact position x,y coordinates"""
        return (0.0, 0.0)


class Wheel(_reader.Wheel, DataAdapter):
    """Wheel & suspension"""

    __slots__ = ()

    def camber(self, index: int | None = None) -> tuple[float, ...]:
        """Wheel camber (radians)"""
        return (
            rmnan(self.ir.get('LFcamber', 0.0)),
            rmnan(self.ir.get('RFcamber', 0.0)),
            rmnan(self.ir.get('LRcamber', 0.0)),
            rmnan(self.ir.get('RRcamber', 0.0)),
        )

    def toe(self, index: int | None = None) -> tuple[float, ...]:
        """Wheel toe (radians)"""
        # iRacing doesn't provide toe data
        return (0.0, 0.0, 0.0, 0.0)

    def toe_symmetric(self, index: int | None = None) -> tuple[float, ...]:
        """Wheel toe symmetric (radians)"""
        return (0.0, 0.0, 0.0, 0.0)

    def rotation(self, index: int | None = None) -> tuple[float, ...]:
        """Wheel rotation (radians per second)"""
        rpm = self.ir.get('RPM', 0.0)
        gear = self.ir.get('Gear', 0)
        if gear > 0 and rpm > 0:
            # Rough approximation
            wheel_speed = rpm * 2 * math.pi / 60
            return (wheel_speed, wheel_speed, wheel_speed, wheel_speed)
        return (0.0, 0.0, 0.0, 0.0)

    def velocity_lateral(self, index: int | None = None) -> tuple[float, ...]:
        """Lateral velocity (m/s) x"""
        vel = self.ir.get('VelocityX', 0.0)
        return (vel, vel, vel, vel)

    def velocity_longitudinal(self, index: int | None = None) -> tuple[float, ...]:
        """Longitudinal velocity (m/s) y"""
        vel = self.ir.get('VelocityY', 0.0)
        return (vel, vel, vel, vel)

    def slip_angle_fl(self, index: int | None = None) -> float:
        """Slip angle (radians) front left"""
        # iRacing doesn't provide slip angle directly
        return 0.0

    def slip_angle_fr(self, index: int | None = None) -> float:
        """Slip angle (radians) front right"""
        return 0.0

    def slip_angle_rl(self, index: int | None = None) -> float:
        """Slip angle (radians) rear left"""
        return 0.0

    def slip_angle_rr(self, index: int | None = None) -> float:
        """Slip angle (radians) rear right"""
        return 0.0

    def ride_height(self, index: int | None = None) -> tuple[float, ...]:
        """Ride height (convert meters to millimeters)"""
        return (
            rmnan(self.ir.get('LFrideHeight', 0.0)) * 1000,
            rmnan(self.ir.get('RFrideHeight', 0.0)) * 1000,
            rmnan(self.ir.get('LRrideHeight', 0.0)) * 1000,
            rmnan(self.ir.get('RRrideHeight', 0.0)) * 1000,
        )

    def third_spring_deflection(self, index: int | None = None) -> tuple[float, ...]:
        """Third spring deflection front & rear (convert meters to millimeters)"""
        # iRacing doesn't provide third spring data
        return (0.0, 0.0, 0.0, 0.0)

    def suspension_deflection(self, index: int | None = None) -> tuple[float, ...]:
        """Suspension deflection (convert meters to millimeters)"""
        return (
            rmnan(self.ir.get('LFshockDefl', 0.0)) * 1000,
            rmnan(self.ir.get('RFshockDefl', 0.0)) * 1000,
            rmnan(self.ir.get('LRshockDefl', 0.0)) * 1000,
            rmnan(self.ir.get('RRshockDefl', 0.0)) * 1000,
        )

    def suspension_force(self, index: int | None = None) -> tuple[float, ...]:
        """Suspension force (Newtons)"""
        return (
            rmnan(self.ir.get('LFshockForce', 0.0)),
            rmnan(self.ir.get('RFshockForce', 0.0)),
            rmnan(self.ir.get('LRshockForce', 0.0)),
            rmnan(self.ir.get('RRshockForce', 0.0)),
        )

    def suspension_damage(self, index: int | None = None) -> tuple[float, ...]:
        """Suspension damage (fraction), 0.0 no damage, 1.0 totaled"""
        # iRacing doesn't provide suspension damage
        return (0.0, 0.0, 0.0, 0.0)

    def position_vertical(self, index: int | None = None) -> tuple[float, ...]:
        """Vertical wheel position (convert meters to millimeters) related to vehicle"""
        # Use ride height as approximation
        return self.ride_height(index)

    def is_detached(self, index: int | None = None) -> tuple[bool, ...]:
        """Whether wheel is detached"""
        return (False, False, False, False)

    def offroad(self, index: int | None = None) -> int:
        """Number of wheels currently off the road"""
        track_surface = self.ir.get('PlayerTrackSurface', 0)
        # 0 = NotInWorld, 1 = OffTrack, 2 = InPitStall, 3 = AproachingPits, 
        # 4 = OnTrack
        if track_surface == 1:  # OffTrack
            return 4  # All wheels off
        return 0
