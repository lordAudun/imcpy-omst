#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import imcpy
import imcpy.coordinates
from imcpy.actors import DynamicActor
from imcpy.decorators import Periodic, Subscribe

logger = logging.getLogger("examples.DiveFollowPipe")

EARTH_RADIUS_M = 6378137.0


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


class FollowRef(DynamicActor):
    def __init__(
        self,
        target: str,
        pipe_points: List[Tuple[float, float]],
        depth_m: float,
        pipe_depth_m: float,
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
        self.depth_m = abs(float(depth_m))
        self.pipe_depth_m = abs(float(pipe_depth_m))
        self.pipe_side = self._validate_side(pipe_side)
        self.pipe_offset_m = abs(float(pipe_offset_m))

        # If user sets accept_radius_m = 0, use a conservative fallback.
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

        # Mission point tuple: (name, lat, lon, final)
        self.mission_points: Optional[List[Tuple[str, float, float, bool]]] = None
        self.current_ref_idx = 0
        self.current_target = None
        self.last_ref = None
        self.mission_done = False
        self.popup_started = False
        self.current_ref_sent_t = None

        # Mission timing gates
        self.min_dive_leg_time_s = 8.0
        self.max_dive_leg_time_s = 25.0
        self.dive_depth_tolerance_m = 0.7
        self.min_wp_leg_time_s = 0.0

        self.sent_refs: List[Tuple[float, float]] = []
        self.est_track: List[Tuple[float, float]] = []
        self.offset_pipe_points: List[Tuple[float, float]] = []

    @staticmethod
    def _validate_side(side: str) -> str:
        side_n = side.strip().lower()
        if side_n not in ("left", "right"):
            raise ValueError("pipe_side must be 'left' or 'right'")
        return side_n

    def compute_dive_offset(self) -> float:
        gamma = math.radians(abs(self.max_pitch_deg))
        tan_gamma = math.tan(gamma)
        if abs(tan_gamma) < 1e-6:
            raise ValueError("max_pitch_deg must be non-zero and not near 0")
        return self.depth_m / tan_gamma

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
                # Right normal in N/E frame.
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
        # Only applies to idx=0 (the DIVE point)
        if self.current_ref_idx != 0:
            return True

        # Dive leg is about reaching depth with enough run-in distance/time,
        # not necessarily passing exactly over the dive XY point.
        if not self._leg_hold_time_elapsed(self.min_dive_leg_time_s):
            return False

        elapsed = 0.0 if self.current_ref_sent_t is None else (time.time() - self.current_ref_sent_t)
        depth_ok = self.est_depth >= max(0.0, self.depth_m - self.dive_depth_tolerance_m)
        if depth_ok:
            return True

        # Failsafe: do not stall forever on DIVE if depth estimate is noisy.
        if elapsed >= self.max_dive_leg_time_s:
            logger.warning(
                "Advancing from DIVE by timeout (elapsed=%.1fs depth=%.2fm target=%.2fm)",
                elapsed,
                self.est_depth,
                self.depth_m,
            )
            return True
        return False

    def _try_advance_mission(self, msg: imcpy.FollowRefState, allow_without_xy_near: bool = False) -> bool:
        xy_near = bool(msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR)
        z_near = bool(msg.proximity & imcpy.FollowRefState.ProximityBits.Z_NEAR)
        dist = self._distance_to_current_target_m()
        thr = self.accept_radius_m if self.accept_radius_m > 0.0 else self.reach_radius_m

        if self.current_ref_idx == 0:
            # Dive leg: require a minimum hold time and depth convergence.
            if not self._leg_hold_time_elapsed(self.min_dive_leg_time_s):
                return False
            depth_ok = self.est_depth >= max(0.0, self.depth_m - self.dive_depth_tolerance_m)
            if not (depth_ok or z_near):
                # Keep fallback timeout to avoid permanent stall on noisy depth.
                elapsed = 0.0 if self.current_ref_sent_t is None else (time.time() - self.current_ref_sent_t)
                if elapsed < self.max_dive_leg_time_s:
                    return False
                logger.warning(
                    "Advancing from DIVE by timeout (elapsed=%.1fs depth=%.2fm target=%.2fm)",
                    elapsed,
                    self.est_depth,
                    self.depth_m,
                )
            if not xy_near and not (allow_without_xy_near and dist is not None and dist <= thr):
                return False
        else:
            # Pipe legs: usually XY_NEAR is the primary trigger.
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

    def start_popup_maneuver(self):
        if self.popup_started:
            return
        try:
            node = self.resolve_node_id(self.target)

            man = imcpy.PopUp()
            man.timeout = self.popup_timeout_s
            man.duration = self.popup_duration_s
            man.z = 0.0
            man.z_units = imcpy.ZUnits.DEPTH
            man.radius = self.popup_radius_m
            man.speed = self.speed_mps
            man.speed_units = imcpy.SpeedUnits.METERS_PS

            flags = imcpy.PopUp.FlagsBits.CURR_POS
            if self.popup_wait_surface:
                flags = flags | imcpy.PopUp.FlagsBits.WAIT_AT_SURFACE
            if self.popup_station_keep:
                flags = flags | imcpy.PopUp.FlagsBits.STATION_KEEP
            man.flags = flags

            pman = imcpy.PlanManeuver()
            pman.data = man
            pman.maneuver_id = "PopUpManeuver"

            spec = imcpy.PlanSpecification()
            spec.plan_id = "PopUpPlan"
            spec.maneuvers.append(pman)
            spec.start_man_id = "PopUpManeuver"
            spec.description = "End mission pop-up"

            pc = imcpy.PlanControl()
            pc.type = imcpy.PlanControl.TypeEnum.REQUEST
            pc.op = imcpy.PlanControl.OperationEnum.START
            pc.plan_id = "PopUpPlan"
            pc.arg = spec

            self.send(node, pc)
            self.popup_started = True
            self.mission_done = True
            self.last_ref = None
            logger.info("Started PopUp maneuver (duration=%ds)", self.popup_duration_s)
        except KeyError:
            pass

    def build_mission_points(self):
        if self.mission_points is not None:
            return
        if self.start_lat is None or self.start_lon is None:
            return
        if len(self.pipe_points) < 2:
            return

        origin_lat, origin_lon = self.pipe_points[0]
        center_ne = self._wgs84_to_local_ne(self.pipe_points, origin_lat, origin_lon)
        offset_ne = self._offset_open_polyline(center_ne)
        self.offset_pipe_points = self._local_ne_to_wgs84(offset_ne, origin_lat, origin_lon)

        if not self.offset_pipe_points:
            return

        # Dive point from current vehicle start, along heading to first offset waypoint.
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
            dive_n = 0.0
            dive_e = 0.0
        else:
            L = self.compute_dive_offset()
            base_heading = math.atan2(de, dn)
            dive_heading = base_heading + math.radians(self.dive_heading_offset_deg)
            dive_n = math.cos(dive_heading) * L
            dive_e = math.sin(dive_heading) * L

        dive_lat, dive_lon = imcpy.coordinates.WGS84.displace(self.start_lat, self.start_lon, n=dive_n, e=dive_e)

        mission = [("DIVE", dive_lat, dive_lon, False)]
        for i, (lat, lon) in enumerate(self.offset_pipe_points):
            final = i == (len(self.offset_pipe_points) - 1)
            mission.append((f"PIPE{i}", lat, lon, final))

        self.mission_points = mission
        logger.info(
            "Mission built: DIVE + %d offset pipe waypoints (%s %.1fm, depth %.1fm)",
            len(self.offset_pipe_points),
            self.pipe_side,
            self.pipe_offset_m,
            self.depth_m,
        )

    def send_current_reference(self):
        if self.mission_points is None or self.mission_done:
            return

        if self.current_ref_idx >= len(self.mission_points):
            self.mission_done = True
            self.last_ref = None
            logger.info("Mission finished: all references sent")
            return

        ref_name, lat, lon, final = self.mission_points[self.current_ref_idx]
        try:
            node = self.resolve_node_id(self.target)
        except KeyError:
            return

        r = imcpy.Reference()
        r.lat = lat
        r.lon = lon

        dz = imcpy.DesiredZ()
        dz.value = self.pipe_depth_m if self.current_ref_idx > 0 else self.depth_m
        dz.z_units = imcpy.ZUnits.DEPTH
        r.z = dz

        ds = imcpy.DesiredSpeed()
        ds.value = self.speed_mps
        ds.speed_units = imcpy.SpeedUnits.METERS_PS
        r.speed = ds

        # Do not use MANDONE here: in FollowReference it may complete the plan
        # before the vehicle actually transits the final segment.
        r.flags = imcpy.Reference.FlagsBits.LOCATION | imcpy.Reference.FlagsBits.SPEED | imcpy.Reference.FlagsBits.Z

        logger.info(
            "Sending reference %s idx=%d lat=%.7f lon=%.7f final=%s",
            ref_name,
            self.current_ref_idx,
            lat,
            lon,
            str(final),
        )
        self.current_target = (lat, lon)
        self.current_ref_sent_t = time.time()
        self.last_ref = r
        self.sent_refs.append((lat, lon))
        self.send(node, r)

    def send_next_reference(self):
        self.current_ref_idx += 1
        self.send_current_reference()

    def is_from_target(self, msg):
        try:
            node = self.resolve_node_id(msg)
            return node.sys_name == self.target
        except KeyError:
            return False

    def plot_mission(self):
        if not self.pipe_points:
            logger.info("No pipe points available for plotting")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed, skipping plot")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        def to_deg(points):
            return [(math.degrees(lat), math.degrees(lon)) for lat, lon in points]

        center_deg = to_deg(self.pipe_points)
        center_lats = [p[0] for p in center_deg]
        center_lons = [p[1] for p in center_deg]
        ax.plot(center_lons, center_lats, "--", color="tab:orange", label="Pipe centerline")

        if self.offset_pipe_points:
            offset_deg = to_deg(self.offset_pipe_points)
            off_lats = [p[0] for p in offset_deg]
            off_lons = [p[1] for p in offset_deg]
            ax.plot(
                off_lons,
                off_lats,
                "-",
                color="tab:blue",
                label=f"Offset path ({self.pipe_side}, {self.pipe_offset_m:.1f} m)",
            )

        if self.sent_refs:
            sent_deg = to_deg(self.sent_refs)
            s_lats = [p[0] for p in sent_deg]
            s_lons = [p[1] for p in sent_deg]
            ax.scatter(s_lons, s_lats, s=16, color="tab:green", alpha=0.7, label="Sent references")

        if self.est_track:
            track_deg = to_deg(self.est_track)
            t_lats = [p[0] for p in track_deg]
            t_lons = [p[1] for p in track_deg]
            ax.plot(t_lons, t_lats, "-", color="tab:red", alpha=0.8, label="Vehicle track")

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title("Dive + Pipe Follow Mission Plot")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best")
        fig.tight_layout()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"dive_follow_pipe_plot_{ts}.png"
        fig.savefig(out_file, dpi=150)
        logger.info("Saved mission plot to %s", out_file)
        plt.show()

    @Periodic(10)
    def init_followref(self):
        if self.state is not None:
            return
        try:
            node = self.resolve_node_id(self.target)

            fr = imcpy.FollowReference()
            fr.control_src = 0xFFFF
            fr.control_ent = 0xFF
            fr.timeout = 10.0
            fr.loiter_radius = 0
            fr.altitude_interval = 0

            pman = imcpy.PlanManeuver()
            pman.data = fr
            pman.maneuver_id = "FollowReferenceManeuver"

            spec = imcpy.PlanSpecification()
            spec.plan_id = "FollowReference"
            spec.maneuvers.append(pman)
            spec.start_man_id = "FollowReferenceManeuver"
            spec.description = "Dive first, then single-pass offset pipe tracking"

            pc = imcpy.PlanControl()
            pc.type = imcpy.PlanControl.TypeEnum.REQUEST
            pc.op = imcpy.PlanControl.OperationEnum.START
            pc.plan_id = "FollowReference"
            pc.arg = spec

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
        self.est_track.append((self.lat, self.lon))
        if self.start_lat is None:
            self.start_lat = self.lat
            self.start_lon = self.lon
            logger.info("Locked start position")
        self.build_mission_points()

    @Subscribe(imcpy.FollowRefState)
    def recv_followrefstate(self, msg: imcpy.FollowRefState):
        if not self.is_from_target(msg):
            return

        self.state = msg.state
        if self.mission_points is None:
            return

        if self.mission_done or self.popup_started:
            return

        # Ensure at least one reference is sent once mission is built
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
            # Same pattern as working examples: allow progress while in
            # loiter-like states if we're near enough to the current target.
            self._try_advance_mission(msg, allow_without_xy_near=True)
        elif msg.state == imcpy.FollowRefState.StateEnum.TIMEOUT:
            logger.info("FollowRef timeout")

    @Periodic(1.0)
    def periodic_ref(self):
        # Fix: send to resolved node, not string name
        if self.last_ref and not self.mission_done and not self.popup_started:
            try:
                node = self.resolve_node_id(self.target)
                self.send(node, self.last_ref)
            except KeyError:
                pass

            # Keep-alive only; mission progression is handled by FollowRefState.


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Dive to depth then follow pipe in parallel offset path (global pipe coordinates)."
    )
    parser.add_argument("--target", default="lauv-simulator-1", help="Target IMC system name")
    parser.add_argument(
        "--pipe-point",
        action="append",
        required=True,
        help='Pipe point "lat,lon[,depth]". Repeat in order along the pipe.',
    )
    parser.add_argument(
        "--input-radians",
        action="store_true",
        help="Interpret --pipe-point lat/lon as radians (default is degrees).",
    )
    parser.add_argument(
        "--pipe-depth",
        type=float,
        default=None,
        help="Pipe depth in meters. Overrides per-point depth if provided.",
    )
    parser.add_argument(
        "--pipe-follow-depth",
        type=float,
        default=5.0,
        help="Depth in meters used for waypoint references while following the pipe (excluding DIVE).",
    )
    parser.add_argument("--side", choices=("left", "right"), default="right", help="Offset side of the pipe path")
    parser.add_argument("--offset", type=float, default=8.0, help="Lateral offset from pipe in meters")
    parser.add_argument(
        "--accept-radius",
        type=float,
        default=0.0,
        help="Meters required before advancing (0 => use fallback reach radius).",
    )
    parser.add_argument("--max-pitch-deg", type=float, default=10.0, help="Max pitch used to compute dive offset")
    parser.add_argument("--dive-heading-offset-deg", type=float, default=0.0, help="Extra heading offset for dive leg")
    parser.add_argument("--speed", type=float, default=1.6, help="Reference speed in m/s")
    parser.add_argument("--popup-duration", type=int, default=30, help="Pop-up duration at surface in seconds")
    parser.add_argument("--popup-timeout", type=int, default=120, help="Pop-up maneuver timeout in seconds")
    parser.add_argument("--popup-radius", type=float, default=10.0, help="Pop-up station radius in meters")
    parser.add_argument(
        "--no-popup-wait-surface",
        action="store_true",
        help="Disable WAIT_AT_SURFACE flag on pop-up maneuver",
    )
    parser.add_argument(
        "--popup-station-keep",
        action="store_true",
        help="Enable STATION_KEEP flag on pop-up maneuver",
    )
    parser.add_argument("--plot-on-exit", action="store_true", help="Plot paths and track when mission exits")
    args = parser.parse_args()

    parsed = [parse_pipe_point(p, input_in_radians=args.input_radians) for p in args.pipe_point]
    if len(parsed) < 2:
        raise ValueError("Provide at least two --pipe-point entries.")

    point_depths = [d for _, _, d in parsed if d is not None]
    if args.pipe_depth is not None:
        depth_m = args.pipe_depth
    elif point_depths:
        depth_m = point_depths[0]
    else:
        depth_m = 10.0

    pipe_points = [(lat, lon) for lat, lon, _ in parsed]

    actor = FollowRef(
        target=args.target,
        pipe_points=pipe_points,
        depth_m=depth_m,
        pipe_depth_m=args.pipe_follow_depth,
        pipe_side=args.side,
        pipe_offset_m=args.offset,
        accept_radius_m=args.accept_radius,
        max_pitch_deg=args.max_pitch_deg,
        dive_heading_offset_deg=args.dive_heading_offset_deg,
        speed_mps=args.speed,
        popup_duration_s=args.popup_duration,
        popup_timeout_s=args.popup_timeout,
        popup_radius_m=args.popup_radius,
        popup_wait_surface=not args.no_popup_wait_surface,
        popup_station_keep=args.popup_station_keep,
    )
    try:
        actor.run()
    finally:
        if args.plot_on_exit:
            actor.plot_mission()
