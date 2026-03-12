#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dive + single-pass pipe scan using FollowReference.

  1. Load pipe centreline from CSV (lat, lon, depth_m).
  2. Offset the path laterally (left / right).
  3. Compute a dive waypoint back-projected along the pipe heading so
     the vehicle reaches operating depth at the first offset waypoint.
  4. Send one waypoint at a time with a fixed commanded depth:
       - dive leg   : depth = first_pipe_depth − altitude_m
       - pipe legs  : sim_mode  → depth  = pipe_depth − altitude_m
                      real mode → altitude = altitude_m (ZUnits.ALTITUDE)
  5. Advance when XY_NEAR or vehicle enters LOITER / HOVER / WAIT.
  6. PopUp at the final waypoint.
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import imcpy
import imcpy.coordinates
from imcpy.actors import DynamicActor
from imcpy.decorators import Periodic, Subscribe

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_378_137.0

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TARGET        = "lauv-simulator-1"
DEFAULT_ALTITUDE_M    = 5.0
DEFAULT_OFFSET_M      = 20.0
DEFAULT_SIDE          = "right"
DEFAULT_SPEED_MPS     = 2.5
DEFAULT_MAX_PITCH_DEG = 12.0
DEFAULT_SIM_MODE      = True   # if True, use depth=(pipe_depth-altitude) for pipe legs instead of ZUnits.ALTITUDE


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_csv(path: str) -> List[Tuple[float, float, float]]:
    """Return [(lat_rad, lon_rad, depth_m), ...] from a pipe-centreline CSV."""
    pts = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pts.append((
                math.radians(float(row["lat"])),
                math.radians(float(row["lon"])),
                abs(float(row["depth_m"])),
            ))
    if len(pts) < 2:
        raise ValueError(f"Need at least 2 points in '{path}'.")
    return pts


# ── actor ─────────────────────────────────────────────────────────────────────
class FollowPipe(DynamicActor):
    """FollowReference actor for a single-pass offset pipe survey."""

    def __init__(
        self,
        target: str,
        pipe_pts: List[Tuple[float, float, float]],   # (lat_rad, lon_rad, depth_m)
        altitude_m: float    = DEFAULT_ALTITUDE_M,
        side: str            = DEFAULT_SIDE,
        offset_m: float      = DEFAULT_OFFSET_M,
        speed_mps: float     = DEFAULT_SPEED_MPS,
        max_pitch_deg: float = DEFAULT_MAX_PITCH_DEG,
        sim_mode: bool       = DEFAULT_SIM_MODE,
        popup_duration: int  = 30,
        popup_timeout: int   = 120,
    ):
        super().__init__()
        self.target        = target
        self.heartbeat.append(target)

        self.pipe_pts      = pipe_pts
        self.altitude_m    = float(altitude_m)
        self.side          = side.strip().lower()
        self.offset_m      = float(offset_m)
        self.speed_mps     = float(speed_mps)
        self.max_pitch_deg = float(max_pitch_deg)
        self.sim_mode      = bool(sim_mode)
        self.popup_duration = int(popup_duration)
        self.popup_timeout  = int(popup_timeout)

        # mission state
        self.waypoints: Optional[List[Tuple[float, float, float]]] = None
        self.wp_idx         = 0
        self.last_ref       = None
        self.fr_state       = None
        self.mission_done   = False
        self.popup_started  = False
        self._just_advanced = False   # blocks one periodic_ref cycle after each advance

        # vehicle position
        self.lat       = 0.0
        self.lon       = 0.0
        self.depth     = 0.0
        self.start_lat: Optional[float] = None
        self.start_lon: Optional[float] = None

        # recording for plot
        self.track:          List[Tuple[float, float, float]] = []  # (lat, lon, depth)
        self.track_desired:  List[float]                      = []  # commanded depth
        self.track_elapsed:  List[float]                      = []  # seconds since t0
        self.t0 = time.time()

    # ── geometry ──────────────────────────────────────────────────────────────
    def _build_waypoints(self) -> List[Tuple[float, float, float]]:
        """
        Returns [(lat_rad, lon_rad, commanded_depth_m), ...]:
          index 0   = DIVE waypoint
          index 1…N = lateral-offset pipe waypoints
        """
        ll     = [(la, lo) for la, lo, _  in self.pipe_pts]
        depths = [d        for _,  _,  d  in self.pipe_pts]
        lat0, lon0 = ll[0]
        cos0 = math.cos(lat0)

        # convert centreline to local NE (metres)
        ne = [((la - lat0) * EARTH_RADIUS_M,
               (lo - lon0) * EARTH_RADIUS_M * cos0) for la, lo in ll]

        # lateral offset
        ne_off = self._offset_ne(ne)

        # back to WGS-84
        ll_off = [imcpy.coordinates.WGS84.displace(lat0, lon0, n=n, e=e)
                  for n, e in ne_off]

        # dive waypoint: back-project along the approach heading
        first_lat, first_lon = ll_off[0]
        dn = (first_lat - self.start_lat) * EARTH_RADIUS_M
        de = (first_lon - self.start_lon) * EARTH_RADIUS_M * math.cos(self.start_lat)
        if math.hypot(dn, de) < 1e-6 and len(ll_off) > 1:
            dn = (ll_off[1][0] - first_lat) * EARTH_RADIUS_M
            de = (ll_off[1][1] - first_lon) * EARTH_RADIUS_M * math.cos(first_lat)

        dive_depth = max(1.0, depths[0] - self.altitude_m)
        heading    = math.atan2(de, dn)
        run        = dive_depth / math.tan(math.radians(abs(self.max_pitch_deg)))
        dive_lat, dive_lon = imcpy.coordinates.WGS84.displace(
            self.start_lat, self.start_lon,
            n=math.cos(heading) * run,
            e=math.sin(heading) * run,
        )

        wps = [(dive_lat, dive_lon, dive_depth)]
        for i, (la, lo) in enumerate(ll_off):
            cmd = max(1.0, depths[i] - self.altitude_m) if self.sim_mode else self.altitude_m
            wps.append((la, lo, cmd))
        return wps

    def _offset_ne(self, ne: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Offset a polyline laterally by self.offset_m."""
        n = len(ne)
        if n < 2 or self.offset_m <= 0:
            return ne
        sign = 1.0 if self.side == "right" else -1.0
        d    = sign * self.offset_m

        dirs, norms = [], []
        for i in range(n - 1):
            dx = ne[i+1][0] - ne[i][0]
            dy = ne[i+1][1] - ne[i][1]
            L  = math.hypot(dx, dy)
            if L < 1e-9:
                dirs.append((0., 0.)); norms.append((0., 0.))
            else:
                dirs.append((dx/L, dy/L)); norms.append((-dy/L, dx/L))

        def isect(p1, v1, p2, v2):
            den = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(den) < 1e-9:
                return None
            t = ((p2[0]-p1[0])*v2[1] - (p2[1]-p1[1])*v2[0]) / den
            return p1[0]+t*v1[0], p1[1]+t*v1[1]

        out = [(ne[0][0] + d*norms[0][0], ne[0][1] + d*norms[0][1])]
        for i in range(1, n-1):
            if abs(dirs[i-1][0])+abs(dirs[i-1][1]) < 1e-9 or \
               abs(dirs[i][0])  +abs(dirs[i][1])   < 1e-9:
                out.append(ne[i]); continue
            p1 = (ne[i][0]+d*norms[i-1][0], ne[i][1]+d*norms[i-1][1])
            p2 = (ne[i][0]+d*norms[i][0],   ne[i][1]+d*norms[i][1])
            x  = isect(p1, dirs[i-1], p2, dirs[i])
            if x:
                out.append(x)
            else:
                avg = (norms[i-1][0]+norms[i][0], norms[i-1][1]+norms[i][1])
                L   = math.hypot(*avg)
                out.append((ne[i][0]+d*avg[0]/L, ne[i][1]+d*avg[1]/L) if L > 1e-9 else p1)
        out.append((ne[-1][0]+d*norms[-1][0], ne[-1][1]+d*norms[-1][1]))
        return out

    # ── reference / popup ─────────────────────────────────────────────────────
    def _send_wp(self, idx: int) -> None:
        """Build and send a Reference message for waypoint idx."""
        if self.waypoints is None or idx >= len(self.waypoints):
            return
        try:
            node = self.resolve_node_id(self.target)
        except KeyError:
            return

        lat, lon, depth = self.waypoints[idx]

        r     = imcpy.Reference()
        r.lat = lat
        r.lon = lon

        dz = imcpy.DesiredZ()
        if idx == 0 or self.sim_mode:
            dz.value   = depth
            dz.z_units = imcpy.ZUnits.DEPTH
        else:
            dz.value   = self.altitude_m
            dz.z_units = imcpy.ZUnits.ALTITUDE
        r.z = dz

        ds             = imcpy.DesiredSpeed()
        ds.value       = self.speed_mps
        ds.speed_units = imcpy.SpeedUnits.METERS_PS
        r.speed        = ds

        r.flags = (imcpy.Reference.FlagsBits.LOCATION
                   | imcpy.Reference.FlagsBits.SPEED
                   | imcpy.Reference.FlagsBits.Z)

        self.last_ref = r
        self.send(node, r)

        tag = "DIVE" if idx == 0 else f"PIPE{idx-1}"
        logger.info("WP %d (%s)  lat=%.6f lon=%.6f depth/alt=%.1f",
                    idx, tag, math.degrees(lat), math.degrees(lon), depth)

    def _advance(self) -> None:
        """Move to the next waypoint, or start PopUp at the last one."""
        if self.waypoints is None:
            return
        if self.wp_idx >= len(self.waypoints) - 1:
            self._start_popup()
        else:
            self.wp_idx        += 1
            self._just_advanced = True
            self._send_wp(self.wp_idx)

    def _start_popup(self) -> None:
        if self.popup_started:
            return
        try:
            node = self.resolve_node_id(self.target)
        except KeyError:
            return

        man             = imcpy.PopUp()
        man.timeout     = self.popup_timeout
        man.duration    = self.popup_duration
        man.z           = 0.0
        man.z_units     = imcpy.ZUnits.DEPTH
        man.radius      = 10.0
        man.speed       = self.speed_mps
        man.speed_units = imcpy.SpeedUnits.METERS_PS
        man.flags       = imcpy.PopUp.FlagsBits.CURR_POS | imcpy.PopUp.FlagsBits.WAIT_AT_SURFACE

        pman             = imcpy.PlanManeuver()
        pman.data        = man
        pman.maneuver_id = "PopUp"

        spec              = imcpy.PlanSpecification()
        spec.plan_id      = "PopUpPlan"
        spec.maneuvers.append(pman)
        spec.start_man_id = "PopUp"

        pc         = imcpy.PlanControl()
        pc.type    = imcpy.PlanControl.TypeEnum.REQUEST
        pc.op      = imcpy.PlanControl.OperationEnum.START
        pc.plan_id = "PopUpPlan"
        pc.arg     = spec

        self.send(node, pc)
        self.popup_started = True
        self.mission_done  = True
        self.last_ref      = None
        logger.info("PopUp maneuver started")

    # ── IMC handlers ──────────────────────────────────────────────────────────
    @Periodic(10)
    def _start_followref(self):
        """Send the FollowReference plan every 10 s until the vehicle responds."""
        if self.fr_state is not None:
            return
        try:
            node = self.resolve_node_id(self.target)
        except KeyError:
            return

        fr                   = imcpy.FollowReference()
        fr.control_src       = 0xFFFF
        fr.control_ent       = 0xFF
        fr.timeout           = 10.0
        fr.loiter_radius     = 0
        fr.altitude_interval = 0

        pman             = imcpy.PlanManeuver()
        pman.data        = fr
        pman.maneuver_id = "FollowRef"

        spec              = imcpy.PlanSpecification()
        spec.plan_id      = "FollowReference"
        spec.maneuvers.append(pman)
        spec.start_man_id = "FollowRef"

        pc         = imcpy.PlanControl()
        pc.type    = imcpy.PlanControl.TypeEnum.REQUEST
        pc.op      = imcpy.PlanControl.OperationEnum.START
        pc.plan_id = "FollowReference"
        pc.arg     = spec

        self.send(node, pc)
        logger.info("FollowReference sent to %s", self.target)

    @Subscribe(imcpy.EstimatedState)
    def _recv_estate(self, msg: imcpy.EstimatedState):
        if not self._from_target(msg):
            return
        self.lat, self.lon, _ = imcpy.coordinates.toWGS84(msg)
        self.depth = float(msg.depth)

        desired = self.waypoints[self.wp_idx][2] if self.waypoints else 0.0
        self.track.append((self.lat, self.lon, self.depth))
        self.track_desired.append(desired)
        self.track_elapsed.append(time.time() - self.t0)

        if self.start_lat is None:
            self.start_lat = self.lat
            self.start_lon = self.lon
            logger.info("Start locked: %.6f %.6f",
                        math.degrees(self.lat), math.degrees(self.lon))
            self.waypoints = self._build_waypoints()
            logger.info("Mission built: %d waypoints", len(self.waypoints))
            for i, (la, lo, d) in enumerate(self.waypoints):
                tag = "DIVE" if i == 0 else f"PIPE{i-1}"
                logger.info("  %s  lat=%.6f lon=%.6f depth=%.1f",
                            tag, math.degrees(la), math.degrees(lo), d)

    @Subscribe(imcpy.FollowRefState)
    def _recv_frs(self, msg: imcpy.FollowRefState):
        if not self._from_target(msg):
            return
        if self.mission_done or self.popup_started or self.waypoints is None:
            return

        self.fr_state = msg.state

        # Send the first waypoint as soon as FollowReference is active
        if self.last_ref is None:
            self.wp_idx = 0
            self._send_wp(0)
            return

        # After advancing, skip processing for one periodic cycle (1 s) so the
        # vehicle has time to process the new reference before we can advance again.
        #if self._just_advanced:
        #    return

        xy_near   = bool(msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR)
        z_near    = bool(msg.proximity & imcpy.FollowRefState.ProximityBits.Z_NEAR)
        loitering = msg.state in (
            imcpy.FollowRefState.StateEnum.LOITER,
            imcpy.FollowRefState.StateEnum.HOVER,
            imcpy.FollowRefState.StateEnum.WAIT,
        )
        if (xy_near and z_near) or loitering:
            self._advance()

    @Periodic(1.0)
    def _periodic_ref(self):
        """Resend the current reference every second to keep FollowRef alive."""
        if self.mission_done or self.popup_started or self.last_ref is None:
            return
        # Clear the advance-guard and skip this cycle so the new reference
        # has time to reach the vehicle before we start re-sending it.
        if self._just_advanced:
            self._just_advanced = False
            #return 
        try:
            self.send(self.resolve_node_id(self.target), self.last_ref)
        except KeyError:
            pass

    def _from_target(self, msg) -> bool:
        try:
            return self.resolve_node_id(msg).sys_name == self.target
        except KeyError:
            return False

    # ── plot ──────────────────────────────────────────────────────────────────
    def plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            logger.warning("matplotlib not available — skipping plot")
            return

        if not self.track or self.waypoints is None:
            logger.info("Nothing to plot")
            return

        lat0, lon0 = self.pipe_pts[0][:2]
        cos0 = math.cos(lat0)

        def enu(lats, lons):
            return ([(lo - lon0) * EARTH_RADIUS_M * cos0 for lo in lons],
                    [(la - lat0) * EARTH_RADIUS_M          for la in lats])

        ce,  cn  = enu([p[0] for p in self.pipe_pts], [p[1] for p in self.pipe_pts])
        we,  wn  = enu([w[0] for w in self.waypoints[1:]], [w[1] for w in self.waypoints[1:]])
        dve, dvn = enu([self.waypoints[0][0]], [self.waypoints[0][1]])
        tre, trn = enu([p[0] for p in self.track], [p[1] for p in self.track])
        tr_depths = [p[2] for p in self.track]

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#0d1117")
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.6, 1], hspace=0.35, wspace=0.3)
        ax2d = fig.add_subplot(gs[0, 0])
        axdp = fig.add_subplot(gs[1, :])

        for ax in (ax2d, axdp):
            ax.set_facecolor("#161b22")
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            ax.tick_params(colors="#8b949e")
            ax.xaxis.label.set_color("#8b949e")
            ax.yaxis.label.set_color("#8b949e")
            ax.title.set_color("#e6edf3")

        # 2-D top-down
        ax2d.plot(ce, cn, "--", color="#e6a817", lw=1.5, label="Pipe centreline")
        ax2d.plot(we, wn, "-o", color="#58a6ff", lw=2, ms=5,
                  label=f"Offset path ({self.side}, {self.offset_m:.0f} m)")
        ax2d.scatter(dve, dvn, s=80, color="#3fb950", zorder=5, label="Dive point")
        if tre:
            ax2d.plot(tre, trn, "-", color="#f85149", lw=1.2, alpha=0.85, label="Vehicle track")
        ax2d.set_xlabel("East [m]"); ax2d.set_ylabel("North [m]")
        ax2d.set_title("Top-down overview", fontsize=11, fontweight="bold")
        ax2d.set_aspect("equal")
        ax2d.grid(True, color="#21262d", lw=0.6)
        ax2d.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

        # Depth vs time  —  actual depth + one commanded depth per waypoint
        axdp.plot(self.track_elapsed, tr_depths,
                  "-",  color="#f85149", lw=1.2, label="Actual depth")
        axdp.plot(self.track_elapsed, self.track_desired,
                  "--", color="#58a6ff", lw=1.5, label="Commanded depth (per waypoint)")
        axdp.invert_yaxis()
        axdp.set_xlabel("Elapsed time [s]"); axdp.set_ylabel("Depth [m]")
        axdp.set_title("Actual vs Commanded depth", fontsize=11, fontweight="bold")
        axdp.grid(True, color="#21262d", lw=0.6)
        axdp.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

        fig.suptitle(
            f"Pipe Follow Mission  |  {self.side} offset {self.offset_m:.0f} m"
            f"  |  alt {self.altitude_m:.1f} m",
            color="#e6edf3", fontsize=13, fontweight="bold")
        fig.tight_layout()

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"pipe_mission_{ts}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Plot saved: %s", out)
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    ap = argparse.ArgumentParser(description="Dive + pipe-scan via FollowReference")
    ap.add_argument("--target",         default=DEFAULT_TARGET)
    ap.add_argument("--csv",            default="pipe_centerline_optimized.csv",
                    help="CSV with columns: idx,lat,lon,depth_m")
    ap.add_argument("--start-idx",      type=int, default=0,
                    help="Skip CSV rows before this idx value (default: 0)")
    ap.add_argument("--altitude",       type=float, default=DEFAULT_ALTITUDE_M,
                    help="Altitude above seafloor for pipe legs (m)")
    ap.add_argument("--side",           choices=("left", "right"), default=DEFAULT_SIDE)
    ap.add_argument("--offset",         type=float, default=DEFAULT_OFFSET_M,
                    help="Lateral offset from pipe centreline (m)")
    ap.add_argument("--speed",          type=float, default=DEFAULT_SPEED_MPS)
    ap.add_argument("--max-pitch-deg",  type=float, default=DEFAULT_MAX_PITCH_DEG)
    ap.add_argument("--sim-mode",       action="store_true", default=DEFAULT_SIM_MODE,
                    help="Use depth=(pipe_depth-altitude) instead of ZUnits.ALTITUDE")
    ap.add_argument("--popup-duration", type=int,   default=30)
    ap.add_argument("--popup-timeout",  type=int,   default=120)
    ap.add_argument("--no-plot",        action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)

    pipe_pts = load_csv(args.csv)
    if args.start_idx > 0:
        pipe_pts = pipe_pts[args.start_idx:]
        if len(pipe_pts) < 2:
            logger.error("--start-idx %d leaves fewer than 2 points", args.start_idx)
            sys.exit(1)
        logger.info("Starting from CSV index %d (%d points remaining)", args.start_idx, len(pipe_pts))

    actor = FollowPipe(
        target        = args.target,
        pipe_pts      = pipe_pts,
        altitude_m    = args.altitude,
        side          = args.side,
        offset_m      = args.offset,
        speed_mps     = args.speed,
        max_pitch_deg = args.max_pitch_deg,
        sim_mode      = args.sim_mode,
        popup_duration = args.popup_duration,
        popup_timeout  = args.popup_timeout,
    )
    try:
        actor.run()
    finally:
        if not args.no_plot:
            actor.plot()
