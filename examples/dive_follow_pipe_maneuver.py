#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import imcpy
import imcpy.coordinates
from imcpy.actors import DynamicActor
from imcpy.decorators import Periodic, Subscribe

logger = logging.getLogger("examples.DiveFollowPipe")

EARTH_RADIUS_M = 6378137.0

# ---------------------------------------------------------------------------
# Default mission configuration (replaces terminal arguments)
# ---------------------------------------------------------------------------
DEFAULT_TARGET = "lauv-simulator-1"
DEFAULT_PIPE_POINTS = [
    (63.4405751198, 10.3500859633,  30.22),  # idx=57   chainage=57m
    (63.4405833333, 10.3519140583,  59.93),  # idx=148  chainage=148m
    (63.4413382504, 10.3545055410,  85.06),  # idx=302  chainage=302m
    (63.4415903606, 10.3551316102,  90.07),  # idx=344  chainage=344m
    (63.4421819609, 10.3556644799, 100.02),  # idx=415  chainage=415m
    (63.4427734651, 10.3559132434, 108.98),  # idx=482  chainage=482m
    (63.4436376715, 10.3554451060, 114.99),  # idx=581  chainage=581m
    (63.4443672078, 10.3549410453, 120.00),  # idx=666  chainage=666m
    (63.4449501803, 10.3542697744, 124.94),  # idx=739  chainage=739m
    (63.4457471642, 10.3528641150, 110.00),  # idx=852  chainage=852m
    (63.4465093560, 10.3521499432, 110.01),  # idx=944  chainage=944m
    (63.4475061785, 10.3520945642, 122.97),  # idx=1055 chainage=1055m
    (63.4487719116, 10.3519883311, 127.04),  # idx=1196 chainage=1196m
    (63.4506472715, 10.3517888866, 142.00),  # idx=1405 chainage=1405m
    (63.4523536937, 10.3517086820, 161.91),  # idx=1595 chainage=1595m
    (63.4535805556, 10.3517194444, 200.00),  # idx=1732 chainage=1732m
]
DEFAULT_SIDE             = "right"
DEFAULT_OFFSET_M         = 20.0
DEFAULT_ALTITUDE_M       = 5.0       # altitude above seafloor for pipe legs
DEFAULT_MAX_PITCH_DEG    = 10.0
DEFAULT_ACCEPT_RADIUS    = 0.0
DEFAULT_SPEED_MPS        = 1.6
DEFAULT_POPUP_DURATION   = 30
DEFAULT_POPUP_TIMEOUT    = 120
DEFAULT_POPUP_RADIUS     = 10.0
DEFAULT_PLOT_ON_EXIT     = True
DEFAULT_ENABLE_SIDESCAN  = True      # activate sidescan sonar on pipe legs
DEFAULT_SIM_MODE         = True      # use depth (pipe_depth - altitude) instead of ZUnits.ALTITUDE



def parse_pipe_point(raw: str, input_in_radians: bool) -> Tuple[float, float, Optional[float]]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid --pipe-point '{raw}'. Use lat,lon or lat,lon,depth.")

    lat = float(parts[0])
    lon = float(parts[1])
    depth = float(parts[2]) if len(parts) == 3 else None

    if not input_in_radians:
        lat = math.radians(lat)
        lon = math.radians(lon)

    return lat, lon, depth


class MissionLogger:
    """
    Writes mission telemetry to two files on close:
      - <stem>_track.csv       : one row per EstimatedState sample
      - <stem>_events.csv      : one row per discrete mission event
      - <stem>.geojson         : GeoJSON FeatureCollection with all layers

    Call  logger.log_track(...)   from recv_estate
          logger.log_event(...)   from any state-change / reference send
          logger.save(actor)      once the mission exits (in the finally block)
    """

    # CSV column definitions
    TRACK_COLS = [
        "timestamp_utc", "elapsed_s",
        "lat_deg", "lon_deg", "depth_m",
        "ref_idx", "followref_state", "z_mode",
    ]
    EVENT_COLS = [
        "timestamp_utc", "elapsed_s",
        "event_type", "ref_idx", "ref_name",
        "lat_deg", "lon_deg", "depth_m",
        "followref_state", "detail",
    ]

    def __init__(self, stem: Optional[str] = None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.stem = stem or f"mission_{ts}"
        self.t0   = time.time()

        self._track_rows: List[Dict[str, Any]] = []
        self._event_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_track(
        self,
        lat_deg: float,
        lon_deg: float,
        depth_m: float,
        ref_idx: int,
        followref_state: str,
        z_mode: str = "depth",
    ) -> None:
        self._track_rows.append({
            "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
            "elapsed_s":       round(time.time() - self.t0, 2),
            "lat_deg":         round(lat_deg, 8),
            "lon_deg":         round(lon_deg, 8),
            "depth_m":         round(depth_m, 3),
            "ref_idx":         ref_idx,
            "followref_state": followref_state,
            "z_mode":          z_mode,
        })

    def log_event(
        self,
        event_type: str,
        ref_idx: int     = -1,
        ref_name: str    = "",
        lat_deg: float   = float("nan"),
        lon_deg: float   = float("nan"),
        depth_m: float   = float("nan"),
        followref_state: str = "",
        detail: str      = "",
    ) -> None:
        self._event_rows.append({
            "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
            "elapsed_s":       round(time.time() - self.t0, 2),
            "event_type":      event_type,
            "ref_idx":         ref_idx,
            "ref_name":        ref_name,
            "lat_deg":         round(lat_deg, 8) if not math.isnan(lat_deg) else "",
            "lon_deg":         round(lon_deg, 8) if not math.isnan(lon_deg) else "",
            "depth_m":         round(depth_m, 3) if not math.isnan(depth_m) else "",
            "followref_state": followref_state,
            "detail":          detail,
        })

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, actor: "FollowRef") -> None:
        """Write CSV and GeoJSON files. Call once on mission exit."""
        self._write_track_csv()
        self._write_events_csv()
        self._write_geojson(actor)
        logger.info(
            "Logs saved: %s_track.csv, %s_events.csv, %s.geojson",
            self.stem, self.stem, self.stem,
        )

    def _write_track_csv(self) -> None:
        path = f"{self.stem}_track.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.TRACK_COLS)
            w.writeheader()
            w.writerows(self._track_rows)
        logger.info("Track CSV  → %s  (%d rows)", path, len(self._track_rows))

    def _write_events_csv(self) -> None:
        path = f"{self.stem}_events.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.EVENT_COLS)
            w.writeheader()
            w.writerows(self._event_rows)
        logger.info("Events CSV → %s  (%d rows)", path, len(self._event_rows))

    def _write_geojson(self, actor: "FollowRef") -> None:
        features = []

        # 1. Vehicle track (LineString with per-point depth in properties)
        if actor.est_track:
            coords   = [[math.degrees(lon), math.degrees(lat), -depth]
                        for lat, lon, depth in actor.est_track]
            depths   = [depth for _, _, depth in actor.est_track]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "layer":    "vehicle_track",
                    "n_points": len(coords),
                    "depth_min_m": round(min(depths), 2),
                    "depth_max_m": round(max(depths), 2),
                },
            })
            # Individual track points (for depth-coloured symbology in QGIS etc.)
            for lat, lon, depth in actor.est_track:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [math.degrees(lon), math.degrees(lat), -depth],
                    },
                    "properties": {"layer": "vehicle_track_pt", "depth_m": round(depth, 3)},
                })

        # 2. Pipe centreline
        if actor.pipe_points and actor.pipe_point_depths:
            coords = [
                [math.degrees(lon), math.degrees(lat), -depth]
                for (lat, lon), depth in zip(actor.pipe_points, actor.pipe_point_depths)
            ]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"layer": "pipe_centreline"},
            })
            for (lat, lon), depth in zip(actor.pipe_points, actor.pipe_point_depths):
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [math.degrees(lon), math.degrees(lat), -depth],
                    },
                    "properties": {"layer": "pipe_centreline_pt", "depth_m": round(depth, 3)},
                })

        # 3. Offset path
        if actor.offset_pipe_points and actor.mission_points:
            off_depths = [d for _, _, _, d, _ in actor.mission_points[1:]]
            n_off = len(actor.offset_pipe_points)
            if len(off_depths) < n_off:
                off_depths += [off_depths[-1]] * (n_off - len(off_depths))
            coords = [
                [math.degrees(lon), math.degrees(lat), -depth]
                for (lat, lon), depth in zip(actor.offset_pipe_points, off_depths)
            ]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "layer":     "offset_path",
                    "side":      actor.pipe_side,
                    "offset_m":  actor.pipe_offset_m,
                },
            })

        # 4. Sent reference waypoints
        if actor.sent_refs and actor.mission_points:
            mp_depths = [d for _, _, _, d, _ in actor.mission_points]
            for i, (lat, lon) in enumerate(actor.sent_refs):
                depth = mp_depths[i] if i < len(mp_depths) else mp_depths[-1]
                name  = actor.mission_points[i][0] if i < len(actor.mission_points) else f"REF{i}"
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [math.degrees(lon), math.degrees(lat), -depth],
                    },
                    "properties": {
                        "layer":   "sent_reference",
                        "ref_idx": i,
                        "name":    name,
                        "depth_m": round(depth, 3),
                    },
                })

        # 5. Mission events (from event log) that carry coordinates
        for row in self._event_rows:
            if row["lat_deg"] == "" or row["lon_deg"] == "":
                continue
            depth_val = row["depth_m"] if row["depth_m"] != "" else 0.0
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["lon_deg"]), float(row["lat_deg"]), -float(depth_val)],
                },
                "properties": {
                    "layer":           "mission_event",
                    "event_type":      row["event_type"],
                    "ref_idx":         row["ref_idx"],
                    "ref_name":        row["ref_name"],
                    "depth_m":         depth_val,
                    "elapsed_s":       row["elapsed_s"],
                    "timestamp_utc":   row["timestamp_utc"],
                    "followref_state": row["followref_state"],
                    "detail":          row["detail"],
                },
            })

        # Mission metadata in a non-geometric feature
        features.append({
            "type": "Feature",
            "geometry": None,
            "properties": {
                "layer":          "mission_metadata",
                "target":         actor.target,
                "pipe_side":      actor.pipe_side,
                "pipe_offset_m":  actor.pipe_offset_m,
                "depth_m":        actor.depth_m,
                "speed_mps":      actor.speed_mps,
                "mission_done":   actor.mission_done,
                "n_waypoints":    len(actor.mission_points) if actor.mission_points else 0,
                "n_track_pts":    len(actor.est_track),
                "n_sent_refs":    len(actor.sent_refs),
                "saved_utc":      datetime.now(timezone.utc).isoformat(),
            },
        })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        path = f"{self.stem}.geojson"
        with open(path, "w") as f:
            json.dump(geojson, f, indent=2)
        logger.info("GeoJSON    → %s  (%d features)", path, len(features))


class FollowRef(DynamicActor):
    def __init__(
        self,
        target: str,
        pipe_points: List[Tuple[float, float]],
        pipe_point_depths: List[float],
        depth_m: float,
        altitude_m: float = 5.0,
        pipe_side: str = "right",
        pipe_offset_m: float = 8.0,
        accept_radius_m: float = 0.0,
        max_pitch_deg: float = 10.0,
        dive_heading_offset_deg: float = 0.0,
        speed_mps: float = 1.6,
        popup_duration_s: int = 30,
        popup_timeout_s: int = 120,
        popup_radius_m: float = 10.0,
        popup_wait_surface: bool = True,
        popup_station_keep: bool = False,
        enable_sidescan: bool = True,
        sim_mode: bool = False,
    ):
        super().__init__()
        self.target = target
        self.heartbeat.append(target)

        self.state = None
        self.lat = 0.0
        self.lon = 0.0
        self.start_lat = None
        self.start_lon = None
        self.est_depth = 0.0

        self.pipe_points = pipe_points
        self.pipe_point_depths = pipe_point_depths  # per-waypoint depths (used for dive & plotting)
        self.depth_m = abs(float(depth_m))          # depth of the first pipe point (for plotting/logging)
        self.altitude_m = abs(float(altitude_m))    # altitude above seafloor for PIPE legs
        # Dive target: descend to just above the first pipe waypoint so
        # altitude control engages immediately at the right height.
        self.dive_depth = max(1.0, self.depth_m - self.altitude_m)
        self.pipe_side = self._validate_side(pipe_side)
        self.pipe_offset_m = abs(float(pipe_offset_m))
        self.enable_sidescan = bool(enable_sidescan)
        self._sidescan_active = False               # tracks whether sonar is currently on
        self.sim_mode = bool(sim_mode)              # True → depth mode in sim, False → altitude mode

        self.accept_radius_m = max(0.0, float(accept_radius_m))
        self.reach_radius_m = self.accept_radius_m if self.accept_radius_m > 0.0 else 3.0

        self.max_pitch_deg = float(max_pitch_deg)
        self.dive_heading_offset_deg = float(dive_heading_offset_deg)
        self.speed_mps = float(speed_mps)
        self.popup_duration_s = int(popup_duration_s)
        self.popup_timeout_s = int(popup_timeout_s)
        self.popup_radius_m = float(popup_radius_m)
        self.popup_wait_surface = bool(popup_wait_surface)
        self.popup_station_keep = bool(popup_station_keep)

        # Mission point tuple: (name, lat, lon, depth, final)
        self.mission_points: Optional[List[Tuple[str, float, float, float, bool]]] = None
        self.current_ref_idx = 0
        self.current_target = None
        self.last_ref = None
        self.mission_done = False
        self.popup_started = False
        self.current_ref_sent_t = None

        self.min_dive_leg_time_s = 8.0
        self.max_dive_leg_time_s = 25.0
        self.dive_depth_tolerance_m = 0.7
        self.min_wp_leg_time_s = 0.0

        self.sent_refs: List[Tuple[float, float]] = []
        self.est_track: List[Tuple[float, float, float]] = []   # (lat, lon, depth)
        self.offset_pipe_points: List[Tuple[float, float]] = []

        # Mission data logger (CSV + GeoJSON)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.mlog = MissionLogger(stem=f"mission_{target}_{ts}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_side(side: str) -> str:
        side_n = side.strip().lower()
        if side_n not in ("left", "right"):
            raise ValueError("pipe_side must be 'left' or 'right'")
        return side_n

    # ------------------------------------------------------------------
    # Sidescan sonar control
    # ------------------------------------------------------------------

    def _set_sidescan(self, active: bool) -> None:
        """
        Activate or deactivate the sidescan sonar via EntityParameter.
        The entity name 'Sidescan' and parameter 'Active' follow the
        standard DUNE/IMC convention used in the LSTS toolchain.
        """
        if not self.enable_sidescan:
            return
        if active == self._sidescan_active:
            return  # already in the desired state
        try:
            node = self.resolve_node_id(self.target)
            ep = imcpy.EntityParameter()
            ep.name  = "Active"
            ep.value = "true" if active else "false"

            ec = imcpy.SetEntityParameters()
            ec.name = "Sidescan"
            ec.params.append(ep)

            self.send(node, ec)
            self._sidescan_active = active
            state_str = "ON" if active else "OFF"
            logger.info("Sidescan sonar → %s", state_str)
            self.mlog.log_event(
                event_type = f"sidescan_{state_str.lower()}",
                ref_idx    = self.current_ref_idx,
                lat_deg    = math.degrees(self.lat),
                lon_deg    = math.degrees(self.lon),
                depth_m    = self.est_depth,
                detail     = f"active={active}",
            )
        except KeyError:
            logger.warning("Could not resolve target to set sidescan state")

    def _interpolate_depth(self, wp_index: int) -> float:
        """Return depth for offset waypoint wp_index from the per-point depths list."""
        depths = self.pipe_point_depths
        if not depths:
            return self.depth_m
        idx = min(wp_index, len(depths) - 1)
        return abs(float(depths[idx]))

    def compute_dive_offset(self) -> float:
        """Horizontal run-in distance needed to reach dive_depth at max_pitch_deg."""
        gamma = math.radians(abs(self.max_pitch_deg))
        tan_gamma = math.tan(gamma)
        if abs(tan_gamma) < 1e-6:
            raise ValueError("max_pitch_deg must be non-zero and not near 0")
        return self.dive_depth / tan_gamma

    @staticmethod
    def _line_intersection(
        p1: Tuple[float, float],
        v1: Tuple[float, float],
        p2: Tuple[float, float],
        v2: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        den = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(den) < 1e-9:
            return None
        dp_n = p2[0] - p1[0]
        dp_e = p2[1] - p1[1]
        t = (dp_n * v2[1] - dp_e * v2[0]) / den
        return p1[0] + t * v1[0], p1[1] + t * v1[1]

    @staticmethod
    def _wgs84_to_local_ne(points: List[Tuple[float, float]], lat0: float, lon0: float) -> List[Tuple[float, float]]:
        out = []
        cos_lat0 = math.cos(lat0)
        for lat, lon in points:
            dn = (lat - lat0) * EARTH_RADIUS_M
            de = (lon - lon0) * EARTH_RADIUS_M * cos_lat0
            out.append((dn, de))
        return out

    @staticmethod
    def _local_ne_to_wgs84(points: List[Tuple[float, float]], lat0: float, lon0: float) -> List[Tuple[float, float]]:
        out = []
        for n, e in points:
            lat, lon = imcpy.coordinates.WGS84.displace(lat0, lon0, n=n, e=e)
            out.append((lat, lon))
        return out

    def _offset_open_polyline(self, points_ne: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        n_pts = len(points_ne)
        if n_pts < 2 or self.pipe_offset_m <= 0.0:
            return points_ne

        sign = 1.0 if self.pipe_side == "right" else -1.0
        d = sign * self.pipe_offset_m
        out: List[Tuple[float, float]] = []

        seg_dirs = []
        seg_normals = []
        for i in range(n_pts - 1):
            p0 = points_ne[i]
            p1 = points_ne[i + 1]
            v = (p1[0] - p0[0], p1[1] - p0[1])
            norm = math.hypot(v[0], v[1])
            if norm < 1e-9:
                seg_dirs.append((0.0, 0.0))
                seg_normals.append((0.0, 0.0))
            else:
                seg_dirs.append((v[0] / norm, v[1] / norm))
                seg_normals.append((-v[1] / norm, v[0] / norm))

        p0 = points_ne[0]
        n0 = seg_normals[0]
        out.append((p0[0] + d * n0[0], p0[1] + d * n0[1]))

        for i in range(1, n_pts - 1):
            cur = points_ne[i]
            v_prev = seg_dirs[i - 1]
            v_next = seg_dirs[i]
            n_prev = seg_normals[i - 1]
            n_next = seg_normals[i]

            if (abs(v_prev[0]) + abs(v_prev[1]) < 1e-9) or (abs(v_next[0]) + abs(v_next[1]) < 1e-9):
                out.append(cur)
                continue

            p1 = (cur[0] + d * n_prev[0], cur[1] + d * n_prev[1])
            p2 = (cur[0] + d * n_next[0], cur[1] + d * n_next[1])
            x = self._line_intersection(p1, v_prev, p2, v_next)
            if x is None:
                n_avg = (n_prev[0] + n_next[0], n_prev[1] + n_next[1])
                n_avg_norm = math.hypot(n_avg[0], n_avg[1])
                if n_avg_norm < 1e-9:
                    out.append(p1)
                else:
                    out.append((cur[0] + d * n_avg[0] / n_avg_norm, cur[1] + d * n_avg[1] / n_avg_norm))
            else:
                out.append(x)

        p_last = points_ne[-1]
        n_last = seg_normals[-1]
        out.append((p_last[0] + d * n_last[0], p_last[1] + d * n_last[1]))
        return out

    def _distance_to_current_target_m(self):
        if self.current_target is None:
            return None
        t_lat, t_lon = self.current_target
        return imcpy.coordinates.WGS84.distance(self.lat, self.lon, 0.0, t_lat, t_lon, 0.0)

    def _leg_hold_time_elapsed(self, min_time_s: float) -> bool:
        if self.current_ref_sent_t is None:
            return False
        return (time.time() - self.current_ref_sent_t) >= min_time_s

    def _is_dive_leg_complete(self) -> bool:
        if self.current_ref_idx != 0:
            return True
        if not self._leg_hold_time_elapsed(self.min_dive_leg_time_s):
            return False
        elapsed = 0.0 if self.current_ref_sent_t is None else (time.time() - self.current_ref_sent_t)
        depth_ok = self.est_depth >= max(0.0, self.dive_depth - self.dive_depth_tolerance_m)
        if depth_ok:
            return True
        if elapsed >= self.max_dive_leg_time_s:
            logger.warning(
                "Advancing from DIVE by timeout (elapsed=%.1fs depth=%.2fm target=%.2fm)",
                elapsed, self.est_depth, self.dive_depth,
            )
            return True
        return False

    def _try_advance_mission(self, msg: imcpy.FollowRefState, allow_without_xy_near: bool = False) -> bool:
        xy_near = bool(msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR)
        z_near  = bool(msg.proximity & imcpy.FollowRefState.ProximityBits.Z_NEAR)
        dist = self._distance_to_current_target_m()
        thr  = self.accept_radius_m if self.accept_radius_m > 0.0 else self.reach_radius_m

        if self.current_ref_idx == 0:
            if not self._leg_hold_time_elapsed(self.min_dive_leg_time_s):
                return False
            depth_ok = self.est_depth >= max(0.0, self.dive_depth - self.dive_depth_tolerance_m)
            if not (depth_ok or z_near):
                elapsed = 0.0 if self.current_ref_sent_t is None else (time.time() - self.current_ref_sent_t)
                if elapsed < self.max_dive_leg_time_s:
                    return False
                logger.warning(
                    "Advancing from DIVE by timeout (elapsed=%.1fs depth=%.2fm target=%.2fm)",
                    elapsed, self.est_depth, self.dive_depth,
                )
            if not xy_near and not (allow_without_xy_near and dist is not None and dist <= thr):
                return False
        else:
            if not self._leg_hold_time_elapsed(self.min_wp_leg_time_s):
                return False
            if not xy_near and not (allow_without_xy_near and dist is not None and dist <= thr):
                return False

        if self.current_ref_idx >= len(self.mission_points) - 1:
            logger.info("Final waypoint reached, starting PopUp")
            self.start_popup_maneuver()
        else:
            self.send_next_reference()
        return True

    # ------------------------------------------------------------------
    # PopUp
    # ------------------------------------------------------------------

    def start_popup_maneuver(self):
        if self.popup_started:
            return
        try:
            node = self.resolve_node_id(self.target)

            man = imcpy.PopUp()
            man.timeout     = self.popup_timeout_s
            man.duration    = self.popup_duration_s
            man.z           = 0.0
            man.z_units     = imcpy.ZUnits.DEPTH
            man.radius      = self.popup_radius_m
            man.speed       = self.speed_mps
            man.speed_units = imcpy.SpeedUnits.METERS_PS

            flags = imcpy.PopUp.FlagsBits.CURR_POS
            if self.popup_wait_surface:
                flags = flags | imcpy.PopUp.FlagsBits.WAIT_AT_SURFACE
            if self.popup_station_keep:
                flags = flags | imcpy.PopUp.FlagsBits.STATION_KEEP
            man.flags = flags

            pman = imcpy.PlanManeuver()
            pman.data        = man
            pman.maneuver_id = "PopUpManeuver"

            spec = imcpy.PlanSpecification()
            spec.plan_id      = "PopUpPlan"
            spec.maneuvers.append(pman)
            spec.start_man_id = "PopUpManeuver"
            spec.description  = "End mission pop-up"

            pc = imcpy.PlanControl()
            pc.type    = imcpy.PlanControl.TypeEnum.REQUEST
            pc.op      = imcpy.PlanControl.OperationEnum.START
            pc.plan_id = "PopUpPlan"
            pc.arg     = spec

            self.send(node, pc)
            self.popup_started = True
            self.mission_done  = True
            self.last_ref      = None
            self._set_sidescan(False)   # stop sonar at surface
            logger.info("Started PopUp maneuver (duration=%ds)", self.popup_duration_s)
            self.mlog.log_event(
                event_type      = "popup_started",
                followref_state = str(self.state) if self.state is not None else "",
                detail          = f"duration={self.popup_duration_s}s timeout={self.popup_timeout_s}s",
            )
        except KeyError:
            pass

    # ------------------------------------------------------------------
    # Mission building
    # ------------------------------------------------------------------

    def build_mission_points(self):
        if self.mission_points is not None:
            return
        if self.start_lat is None or self.start_lon is None:
            return
        if len(self.pipe_points) < 2:
            return

        origin_lat, origin_lon = self.pipe_points[0]
        center_ne  = self._wgs84_to_local_ne(self.pipe_points, origin_lat, origin_lon)
        offset_ne  = self._offset_open_polyline(center_ne)
        self.offset_pipe_points = self._local_ne_to_wgs84(offset_ne, origin_lat, origin_lon)

        if not self.offset_pipe_points:
            return

        first_wp_lat, first_wp_lon = self.offset_pipe_points[0]
        dn = (first_wp_lat - self.start_lat) * EARTH_RADIUS_M
        de = (first_wp_lon - self.start_lon) * EARTH_RADIUS_M * math.cos(self.start_lat)
        base_norm = math.hypot(dn, de)

        if base_norm < 1e-6 and len(self.offset_pipe_points) > 1:
            lat2, lon2 = self.offset_pipe_points[1]
            dn = (lat2 - first_wp_lat) * EARTH_RADIUS_M
            de = (lon2 - first_wp_lon) * EARTH_RADIUS_M * math.cos(first_wp_lat)
            base_norm = math.hypot(dn, de)

        if base_norm < 1e-6:
            dive_n = dive_e = 0.0
        else:
            L            = self.compute_dive_offset()
            base_heading = math.atan2(de, dn)
            dive_heading = base_heading + math.radians(self.dive_heading_offset_deg)
            dive_n       = math.cos(dive_heading) * L
            dive_e       = math.sin(dive_heading) * L

        dive_lat, dive_lon = imcpy.coordinates.WGS84.displace(
            self.start_lat, self.start_lon, n=dive_n, e=dive_e
        )

        # Tuple layout: (name, lat, lon, depth, final)
        mission = [("DIVE", dive_lat, dive_lon, self.dive_depth, False)]
        for i, (lat, lon) in enumerate(self.offset_pipe_points):
            final        = i == (len(self.offset_pipe_points) - 1)
            depth_at_wp  = self._interpolate_depth(i)
            mission.append((f"PIPE{i}", lat, lon, depth_at_wp, final))

        self.mission_points = mission
        logger.info(
            "Mission built: DIVE(%.1fm = pipe_depth %.1fm - alt %.1fm) + %d offset pipe waypoints (%s %.1fm)",
            self.dive_depth, self.depth_m, self.altitude_m,
            len(self.offset_pipe_points), self.pipe_side, self.pipe_offset_m,
        )
        for name, lat, lon, depth, final in mission:
            logger.info(
                "  %s  lat=%.7f lon=%.7f depth=%.1fm final=%s",
                name, math.degrees(lat), math.degrees(lon), depth, final,
            )
        self.mlog.log_event(
            event_type = "mission_built",
            detail     = (
                f"n_waypoints={len(mission)} "
                f"dive_depth={self.dive_depth:.1f}m (pipe={self.depth_m:.1f}m - alt={self.altitude_m:.1f}m) "
                f"side={self.pipe_side} offset={self.pipe_offset_m:.1f}m"
            ),
        )

    # ------------------------------------------------------------------
    # Reference sending
    # ------------------------------------------------------------------

    def send_current_reference(self):
        if self.mission_points is None or self.mission_done:
            return

        if self.current_ref_idx >= len(self.mission_points):
            self.mission_done = True
            self.last_ref     = None
            logger.info("Mission finished: all references sent")
            return

        ref_name, lat, lon, wp_depth, final = self.mission_points[self.current_ref_idx]
        try:
            node = self.resolve_node_id(self.target)
        except KeyError:
            return

        r     = imcpy.Reference()
        r.lat = lat
        r.lon = lon

        dz = imcpy.DesiredZ()
        is_dive = (self.current_ref_idx == 0)
        if is_dive:
            # DIVE leg: descend to (first_pipe_depth - altitude) so the vehicle
            # enters altitude control at exactly the right height above the pipe.
            dz.value   = self.dive_depth
            dz.z_units = imcpy.ZUnits.DEPTH
            z_label    = f"depth={self.dive_depth:.1f}m"
            self._set_sidescan(False)   # sonar off during dive
        else:
            # PIPE legs: altitude above seafloor (real vehicle) or
            # pipe_depth - altitude (simulator, no DVL seabed model).
            if self.sim_mode:
                dz.value   = max(1.0, wp_depth - self.altitude_m)
                dz.z_units = imcpy.ZUnits.DEPTH
                z_label    = f"sim_depth={dz.value:.1f}m (pipe={wp_depth:.1f}m - alt={self.altitude_m:.1f}m)"
            else:
                dz.value   = self.altitude_m
                dz.z_units = imcpy.ZUnits.ALTITUDE
                z_label    = f"alt={self.altitude_m:.1f}m"
            self._set_sidescan(True)    # sonar on while scanning pipe
        r.z = dz

        ds              = imcpy.DesiredSpeed()
        ds.value        = self.speed_mps
        ds.speed_units  = imcpy.SpeedUnits.METERS_PS
        r.speed         = ds

        r.flags = (
            imcpy.Reference.FlagsBits.LOCATION
            | imcpy.Reference.FlagsBits.SPEED
            | imcpy.Reference.FlagsBits.Z
        )

        logger.info(
            "Sending reference %s idx=%d lat=%.7f lon=%.7f %s final=%s",
            ref_name, self.current_ref_idx,
            math.degrees(lat), math.degrees(lon), z_label, str(final),
        )
        self.current_target     = (lat, lon)
        self.current_ref_sent_t = time.time()
        self.last_ref           = r
        self.sent_refs.append((lat, lon))
        self.send(node, r)
        self.mlog.log_event(
            event_type      = "reference_sent",
            ref_idx         = self.current_ref_idx,
            ref_name        = ref_name,
            lat_deg         = math.degrees(lat),
            lon_deg         = math.degrees(lon),
            depth_m         = wp_depth,           # planned pipe depth (for logging/plotting)
            followref_state = str(self.state) if self.state is not None else "",
            detail          = f"final={final} z_mode={'depth' if is_dive else ('sim_depth' if self.sim_mode else 'altitude')} {z_label}",
        )

    def send_next_reference(self):
        self.current_ref_idx += 1
        self.send_current_reference()

    def is_from_target(self, msg):
        try:
            node = self.resolve_node_id(msg)
            return node.sys_name == self.target
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _mission_point_depths(self) -> List[float]:
        """Return the depth value for each mission point (DIVE + PIPE waypoints)."""
        if not self.mission_points:
            return []
        return [depth for _, _, _, depth, _ in self.mission_points]

    @staticmethod
    def _points_to_local_m(
        lats_rad: List[float],
        lons_rad: List[float],
        lat0: float,
        lon0: float,
    ) -> Tuple[List[float], List[float]]:
        """Convert lists of WGS-84 (rad) coordinates to local East/North offsets in metres."""
        cos_lat0 = math.cos(lat0)
        east  = [(lon - lon0) * EARTH_RADIUS_M * cos_lat0 for lon in lons_rad]
        north = [(lat - lat0) * EARTH_RADIUS_M            for lat in lats_rad]
        return east, north

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_mission(self):
        """Render a side-by-side 2-D top-down overview and a 3-D depth plot."""
        if not self.pipe_points:
            logger.info("No pipe points available for plotting")
            return
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            import matplotlib.cm as cm
            import numpy as np
        except ImportError as exc:
            logger.warning("Plotting requires matplotlib + numpy: %s", exc)
            return

        # Use the first pipe-point as the local origin so axes are in metres.
        lat0, lon0 = self.pipe_points[0]

        def to_enu(lats_rad, lons_rad):
            return self._points_to_local_m(lats_rad, lons_rad, lat0, lon0)

        # ---- collect series ------------------------------------------------
        c_lats = [p[0] for p in self.pipe_points]
        c_lons = [p[1] for p in self.pipe_points]
        c_depths = [abs(float(d)) for d in self.pipe_point_depths]
        ce, cn   = to_enu(c_lats, c_lons)

        off_e = off_n = off_depths = []
        if self.offset_pipe_points:
            o_lats   = [p[0] for p in self.offset_pipe_points]
            o_lons   = [p[1] for p in self.offset_pipe_points]
            off_depths = self._mission_point_depths()[1:]   # skip DIVE point
            # Pad / trim to match offset point count
            n_off = len(self.offset_pipe_points)
            if len(off_depths) < n_off:
                off_depths += [off_depths[-1]] * (n_off - len(off_depths))
            else:
                off_depths = off_depths[:n_off]
            off_e, off_n = to_enu(o_lats, o_lons)

        sent_e = sent_n = []
        sent_depths = []
        if self.sent_refs and self.mission_points:
            sr_lats = [p[0] for p in self.sent_refs]
            sr_lons = [p[1] for p in self.sent_refs]
            sent_e, sent_n = to_enu(sr_lats, sr_lons)
            mp_depths = self._mission_point_depths()
            sent_depths = mp_depths[:len(self.sent_refs)]
            if len(sent_depths) < len(self.sent_refs):
                sent_depths += [sent_depths[-1]] * (len(self.sent_refs) - len(sent_depths))

        track_e = track_n = []
        track_depths = []
        if self.est_track:
            tr_lats  = [p[0] for p in self.est_track]
            tr_lons  = [p[1] for p in self.est_track]
            track_depths = [p[2] for p in self.est_track]   # real per-point depths
            track_e, track_n = to_enu(tr_lats, tr_lons)

        # ---- figure layout -------------------------------------------------
        fig = plt.figure(figsize=(18, 8))
        fig.patch.set_facecolor("#0d1117")

        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")

        for ax in (ax2d,):
            ax.set_facecolor("#161b22")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.tick_params(colors="#8b949e")
            ax.xaxis.label.set_color("#8b949e")
            ax.yaxis.label.set_color("#8b949e")
            ax.title.set_color("#e6edf3")

        ax3d.set_facecolor("#161b22")
        ax3d.tick_params(colors="#8b949e")
        ax3d.xaxis.label.set_color("#8b949e")
        ax3d.yaxis.label.set_color("#8b949e")
        ax3d.zaxis.label.set_color("#8b949e")
        ax3d.title.set_color("#e6edf3")

        # colour map for depth (deeper = darker blue)
        depth_cmap = cm.get_cmap("cool")

        all_depths = c_depths + (list(off_depths) if off_depths else [])
        d_min = min(all_depths) if all_depths else 0.0
        d_max = max(all_depths) if all_depths else 1.0
        if d_max == d_min:
            d_max = d_min + 1.0

        def dnorm(d):
            return (d - d_min) / (d_max - d_min)

        # ===== 2-D top-down =================================================
        # Pipe centerline
        ax2d.plot(ce, cn, "--", color="#e6a817", lw=1.5, label="Pipe centreline", zorder=2)
        # Centreline waypoints coloured by depth
        sc = ax2d.scatter(ce, cn, c=c_depths, cmap="cool", vmin=d_min, vmax=d_max,
                          s=50, zorder=4, edgecolors="#0d1117", linewidths=0.5)

        if off_e:
            ax2d.plot(off_e, off_n, "-", color="#58a6ff", lw=2,
                      label=f"Offset path ({self.pipe_side}, {self.pipe_offset_m:.0f} m)", zorder=3)
            ax2d.scatter(off_e, off_n, c=off_depths, cmap="cool", vmin=d_min, vmax=d_max,
                         s=40, zorder=5, edgecolors="#0d1117", linewidths=0.5)

        if sent_e:
            ax2d.scatter(sent_e, sent_n, s=12, color="#3fb950", alpha=0.8,
                         label="Sent references", zorder=6)

        if track_e:
            ax2d.plot(track_e, track_n, "-", color="#f85149", lw=1.2,
                      alpha=0.85, label="Vehicle track", zorder=3)

        cbar = fig.colorbar(sc, ax=ax2d, pad=0.02, fraction=0.046)
        cbar.set_label("Depth [m]", color="#8b949e")
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

        ax2d.set_xlabel("East [m]")
        ax2d.set_ylabel("North [m]")
        ax2d.set_title("Top-down overview", fontsize=12, fontweight="bold")
        ax2d.set_aspect("equal")
        ax2d.grid(True, color="#21262d", linewidth=0.6)
        ax2d.legend(loc="best", facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#e6edf3", fontsize=8)

        # ===== 3-D ==========================================================
        # Helper: draw a coloured line in 3-D by breaking it into segments
        def plot3d_colored(xs, ys, zs_depth, lw=2, alpha=1.0):
            """Draw a polyline in 3-D with each segment coloured by depth."""
            if len(xs) < 2:
                return
            pts = np.array([xs, ys, [-d for d in zs_depth]]).T  # depth → negative Z
            segs = [pts[i:i+2] for i in range(len(pts) - 1)]
            seg_depths = [(zs_depth[i] + zs_depth[i+1]) / 2 for i in range(len(zs_depth) - 1)]
            colors = [depth_cmap(dnorm(d)) for d in seg_depths]
            col = Line3DCollection(segs, colors=colors, linewidths=lw, alpha=alpha)
            ax3d.add_collection3d(col)

        # Pipe centreline
        plot3d_colored(ce, cn, c_depths, lw=1.5, alpha=0.6)
        ax3d.scatter(ce, cn, [-d for d in c_depths],
                     c=c_depths, cmap="cool", vmin=d_min, vmax=d_max,
                     s=40, depthshade=True, edgecolors="none", label="Pipe centreline")

        # Offset path
        if off_e:
            plot3d_colored(list(off_e), list(off_n), list(off_depths), lw=2.5)
            ax3d.scatter(off_e, off_n, [-d for d in off_depths],
                         c=off_depths, cmap="cool", vmin=d_min, vmax=d_max,
                         s=50, depthshade=True, edgecolors="none",
                         label=f"Offset path ({self.pipe_side})")

        # Vehicle track
        if track_e:
            ax3d.plot(track_e, track_n, [-d for d in track_depths],
                      "-", color="#f85149", lw=1.5, alpha=0.85, label="Vehicle track")

        # Sent references
        if sent_e:
            ax3d.scatter(sent_e, sent_n, [-d for d in sent_depths],
                         s=18, color="#3fb950", alpha=0.85,
                         depthshade=False, label="Sent references")

        # Vertical "dive" line from surface to first offset waypoint
        if off_e:
            ax3d.plot([off_e[0], off_e[0]], [off_n[0], off_n[0]],
                      [0, -off_depths[0]],
                      "--", color="#8b949e", lw=1, alpha=0.5)

        ax3d.set_xlabel("East [m]",  labelpad=6)
        ax3d.set_ylabel("North [m]", labelpad=6)
        ax3d.set_zlabel("Depth [m]", labelpad=6)
        ax3d.set_title("3-D mission profile", fontsize=12, fontweight="bold")

        # Flip Z axis so depth increases downward visually
        z_lo = -(d_max * 1.05)
        z_hi = max(0.0, -(d_min * 0.95)) + 5
        ax3d.set_zlim(z_lo, z_hi)
        # Relabel Z ticks as positive depths
        z_ticks = np.linspace(z_lo, 0, 6)
        ax3d.set_zticks(z_ticks)
        ax3d.set_zticklabels([f"{abs(z):.0f}" for z in z_ticks])

        ax3d.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#e6edf3", fontsize=8)
        ax3d.grid(True, color="#21262d", linewidth=0.4)
        ax3d.view_init(elev=25, azim=-60)

        fig.suptitle(
            f"Dive + Pipe Follow Mission  |  depth {d_min:.0f}–{d_max:.0f} m  |  "
            f"{self.pipe_side} offset {self.pipe_offset_m:.0f} m",
            color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"dive_follow_pipe_plot_{ts}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved mission plot to %s", out_file)
        plt.show()

    # ------------------------------------------------------------------
    # IMC periodic / subscribers
    # ------------------------------------------------------------------

    @Periodic(10)
    def init_followref(self):
        if self.state is not None:
            return
        try:
            node = self.resolve_node_id(self.target)

            fr                  = imcpy.FollowReference()
            fr.control_src      = 0xFFFF
            fr.control_ent      = 0xFF
            fr.timeout          = 10.0
            fr.loiter_radius    = 0
            fr.altitude_interval = 0

            pman             = imcpy.PlanManeuver()
            pman.data        = fr
            pman.maneuver_id = "FollowReferenceManeuver"

            spec              = imcpy.PlanSpecification()
            spec.plan_id      = "FollowReference"
            spec.maneuvers.append(pman)
            spec.start_man_id = "FollowReferenceManeuver"
            spec.description  = "Dive first, then single-pass offset pipe tracking"

            pc         = imcpy.PlanControl()
            pc.type    = imcpy.PlanControl.TypeEnum.REQUEST
            pc.op      = imcpy.PlanControl.OperationEnum.START
            pc.plan_id = "FollowReference"
            pc.arg     = spec

            self.send(node, pc)
            logger.info("Started FollowRef command")
        except KeyError:
            pass

    @Subscribe(imcpy.EstimatedState)
    def recv_estate(self, msg):
        if not self.is_from_target(msg):
            return
        self.lat, self.lon, _ = imcpy.coordinates.toWGS84(msg)
        self.est_depth = float(msg.depth)
        self.est_track.append((self.lat, self.lon, self.est_depth))
        self.mlog.log_track(
            lat_deg         = math.degrees(self.lat),
            lon_deg         = math.degrees(self.lon),
            depth_m         = self.est_depth,
            ref_idx         = self.current_ref_idx,
            followref_state = str(self.state) if self.state is not None else "UNKNOWN",
            z_mode          = "depth" if self.current_ref_idx == 0 else ("sim_depth" if self.sim_mode else "altitude"),
        )
        if self.start_lat is None:
            self.start_lat = self.lat
            self.start_lon = self.lon
            logger.info("Locked start position")
            self.mlog.log_event(
                event_type = "start_position_locked",
                lat_deg    = math.degrees(self.lat),
                lon_deg    = math.degrees(self.lon),
                depth_m    = self.est_depth,
            )
        self.build_mission_points()

    @Subscribe(imcpy.FollowRefState)
    def recv_followrefstate(self, msg: imcpy.FollowRefState):
        if not self.is_from_target(msg):
            return

        prev_state = self.state
        self.state = msg.state

        # Log every state transition
        state_str = str(msg.state)
        if msg.state != prev_state:
            self.mlog.log_event(
                event_type      = "followref_state_change",
                ref_idx         = self.current_ref_idx,
                lat_deg         = math.degrees(self.lat),
                lon_deg         = math.degrees(self.lon),
                depth_m         = self.est_depth,
                followref_state = state_str,
                detail          = f"prev={prev_state} -> now={msg.state} proximity={msg.proximity}",
            )

        if self.mission_points is None:
            return
        if self.mission_done or self.popup_started:
            return

        if self.last_ref is None and not self.mission_done:
            self.current_ref_idx = 0
            self.send_current_reference()
            return

        if msg.state == imcpy.FollowRefState.StateEnum.GOTO:
            self._try_advance_mission(msg, allow_without_xy_near=False)
        elif msg.state in (
            imcpy.FollowRefState.StateEnum.LOITER,
            imcpy.FollowRefState.StateEnum.HOVER,
            imcpy.FollowRefState.StateEnum.WAIT,
        ):
            self._try_advance_mission(msg, allow_without_xy_near=True)
        elif msg.state == imcpy.FollowRefState.StateEnum.TIMEOUT:
            logger.info("FollowRef timeout")
            self.mlog.log_event(
                event_type      = "followref_timeout",
                ref_idx         = self.current_ref_idx,
                lat_deg         = math.degrees(self.lat),
                lon_deg         = math.degrees(self.lon),
                depth_m         = self.est_depth,
                followref_state = state_str,
            )

    @Periodic(1.0)
    def periodic_ref(self):
        if self.last_ref and not self.mission_done and not self.popup_started:
            try:
                node = self.resolve_node_id(self.target)
                self.send(node, self.last_ref)
            except KeyError:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Dive to depth then follow pipe at fixed altitude with optional sidescan sonar."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--pipe-point", action="append", default=None,
                        help='Override pipe points: "lat,lon,depth". Repeat in order.')
    parser.add_argument("--input-radians", action="store_true")
    parser.add_argument("--pipe-depth", type=float, default=None)
    parser.add_argument("--altitude", type=float, default=DEFAULT_ALTITUDE_M,
                        help="Altitude above seafloor for pipe following (default: 5.0 m)")
    parser.add_argument("--side", choices=("left", "right"), default=DEFAULT_SIDE)
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET_M)
    parser.add_argument("--accept-radius", type=float, default=DEFAULT_ACCEPT_RADIUS)
    parser.add_argument("--max-pitch-deg", type=float, default=DEFAULT_MAX_PITCH_DEG)
    parser.add_argument("--dive-heading-offset-deg", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED_MPS)
    parser.add_argument("--popup-duration", type=int, default=DEFAULT_POPUP_DURATION)
    parser.add_argument("--popup-timeout", type=int, default=DEFAULT_POPUP_TIMEOUT)
    parser.add_argument("--popup-radius", type=float, default=DEFAULT_POPUP_RADIUS)
    parser.add_argument("--no-popup-wait-surface", action="store_true")
    parser.add_argument("--popup-station-keep", action="store_true")
    parser.add_argument("--no-sidescan", action="store_true",
                        help="Disable sidescan sonar activation during pipe following")
    parser.add_argument("--sim-mode", action="store_true", default=DEFAULT_SIM_MODE,
                        help=(
                            "Simulator mode: command depth=(pipe_depth - altitude) per waypoint "
                            "instead of ZUnits.ALTITUDE. Use this when the simulator has no "
                            "bathymetry model. Remove for real vehicle deployment."
                        ))
    parser.add_argument("--plot-on-exit", action="store_true", default=DEFAULT_PLOT_ON_EXIT)
    args = parser.parse_args()

    # Use CLI pipe-points if provided, otherwise fall back to hardcoded defaults
    if args.pipe_point:
        parsed = [parse_pipe_point(p, input_in_radians=args.input_radians) for p in args.pipe_point]
    else:
        parsed = [
            (math.radians(lat), math.radians(lon), depth)
            for lat, lon, depth in DEFAULT_PIPE_POINTS
        ]

    if len(parsed) < 2:
        raise ValueError("Provide at least two pipe points.")

    point_depths = [d for _, _, d in parsed if d is not None]
    if args.pipe_depth is not None:
        depth_m = args.pipe_depth
    elif point_depths:
        depth_m = point_depths[0]
    else:
        depth_m = 10.0

    pipe_points       = [(lat, lon) for lat, lon, _ in parsed]
    pipe_point_depths = [d if d is not None else depth_m for _, _, d in parsed]

    actor = FollowRef(
        target                  = args.target,
        pipe_points             = pipe_points,
        pipe_point_depths       = pipe_point_depths,
        depth_m                 = depth_m,
        altitude_m              = args.altitude,
        pipe_side               = args.side,
        pipe_offset_m           = args.offset,
        accept_radius_m         = args.accept_radius,
        max_pitch_deg           = args.max_pitch_deg,
        dive_heading_offset_deg = args.dive_heading_offset_deg,
        speed_mps               = args.speed,
        popup_duration_s        = args.popup_duration,
        popup_timeout_s         = args.popup_timeout,
        popup_radius_m          = args.popup_radius,
        popup_wait_surface      = not args.no_popup_wait_surface,
        popup_station_keep      = args.popup_station_keep,
        enable_sidescan         = not args.no_sidescan,
        sim_mode                = args.sim_mode,
    )
    try:
        actor.run()
    finally:
        actor.mlog.save(actor)
        if args.plot_on_exit:
            actor.plot_mission()