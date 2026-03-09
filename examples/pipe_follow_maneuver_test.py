import argparse
import logging
import math
import sys
from datetime import datetime

import imcpy
import imcpy.coordinates
from imcpy.actors import DynamicActor
from imcpy.decorators import Periodic, Subscribe

logger = logging.getLogger('examples.PipeFollowRef')


class FollowRef(DynamicActor):
    def __init__(self, target, pipe_side='right', pipe_offset_m=8.0, accept_radius_m=0.0):
        super().__init__()
        self.target = target
        self.heartbeat.append(target)

        self.state = None
        self.lat = 0.0
        self.lon = 0.0
        self.last_ref = False

        # Pipe centerline increments in local North/East meters
        # (same semantics as follow_reference_example).
        self.wp = [
            (50.0, 0.0),
            (0.0, 50.0),
            (-50.0, 0.0),
            (0.0, -50.0),
        ]
        self.wp_next = 0

        self.pipe_side = self._validate_side(pipe_side)
        self.pipe_offset_m = abs(float(pipe_offset_m))
        self.accept_radius_m = max(0.0, float(accept_radius_m))
        self.path_origin = None
        self.sent_refs = []  # list[(lat, lon)]
        self.est_track = []  # list[(lat, lon)]
        self.current_target = None  # (lat, lon)

    @staticmethod
    def _validate_side(side: str) -> str:
        side_n = side.strip().lower()
        if side_n not in ('left', 'right'):
            raise ValueError("pipe_side must be 'left' or 'right'")
        return side_n

    def _offset_pipe_point(self, idx):
        """
        Return point idx from a proper closed-polyline offset path.
        """
        pts = self._offset_path_points_ne()
        return pts[idx]

    def _centerline_points_ne(self):
        """
        Build centerline corner points from incremental N/E legs.
        """
        pts = []
        n_acc = 0.0
        e_acc = 0.0
        for dn, de in self.wp:
            n_acc += dn
            e_acc += de
            pts.append((n_acc, e_acc))
        return pts

    @staticmethod
    def _line_intersection(p1, v1, p2, v2):
        """
        Intersect two 2D lines: p1 + t*v1 and p2 + s*v2. Returns None if parallel.
        """
        den = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(den) < 1e-9:
            return None
        dp_n = p2[0] - p1[0]
        dp_e = p2[1] - p1[1]
        t = (dp_n * v2[1] - dp_e * v2[0]) / den
        return (p1[0] + t * v1[0], p1[1] + t * v1[1])

    def _offset_path_points_ne(self):
        """
        Build a closed polyline offset path so each edge stays parallel to centerline.
        """
        center = self._centerline_points_ne()
        n_pts = len(center)
        if n_pts < 2 or self.pipe_offset_m <= 0.0:
            return center

        sign = 1.0 if self.pipe_side == 'right' else -1.0
        d = sign * self.pipe_offset_m
        out = []

        for i in range(n_pts):
            prev_pt = center[(i - 1) % n_pts]
            cur_pt = center[i]
            next_pt = center[(i + 1) % n_pts]

            v_prev = (cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1])
            v_next = (next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1])

            norm_prev = math.hypot(v_prev[0], v_prev[1])
            norm_next = math.hypot(v_next[0], v_next[1])
            if norm_prev < 1e-9 or norm_next < 1e-9:
                out.append(cur_pt)
                continue

            # Right normals.
            n_prev = (-v_prev[1] / norm_prev, v_prev[0] / norm_prev)
            n_next = (-v_next[1] / norm_next, v_next[0] / norm_next)

            # Two offset lines, one for each adjacent segment.
            p1 = (cur_pt[0] + d * n_prev[0], cur_pt[1] + d * n_prev[1])
            p2 = (cur_pt[0] + d * n_next[0], cur_pt[1] + d * n_next[1])

            x = self._line_intersection(p1, v_prev, p2, v_next)
            if x is None:
                # Parallel fallback near straight corners.
                n_avg = (n_prev[0] + n_next[0], n_prev[1] + n_next[1])
                n_avg_norm = math.hypot(n_avg[0], n_avg[1])
                if n_avg_norm < 1e-9:
                    out.append(p1)
                else:
                    out.append((cur_pt[0] + d * n_avg[0] / n_avg_norm, cur_pt[1] + d * n_avg[1] / n_avg_norm))
            else:
                out.append(x)

        return out

    def _absolute_waypoints(self):
        """
        Convert local centerline and offset waypoints into absolute lat/lon.
        """
        if self.path_origin is None:
            return [], []

        o_lat, o_lon = self.path_origin
        center = []
        offset = []
        center_ne = self._centerline_points_ne()
        offset_ne = self._offset_path_points_ne()
        for i, (n, e) in enumerate(center_ne):
            c_lat, c_lon = imcpy.coordinates.WGS84.displace(o_lat, o_lon, n=n, e=e)
            center.append((c_lat, c_lon))

            n_off, e_off = offset_ne[i]
            o2_lat, o2_lon = imcpy.coordinates.WGS84.displace(o_lat, o_lon, n=n_off, e=e_off)
            offset.append((o2_lat, o2_lon))
        return center, offset

    def _distance_to_current_target_m(self):
        if self.current_target is None:
            return None
        t_lat, t_lon = self.current_target
        return imcpy.coordinates.WGS84.distance(self.lat, self.lon, 0.0, t_lat, t_lon, 0.0)

    def plot_mission(self):
        """
        Plot centerline, offset path, sent references, and estimated track after mission.
        """
        if self.path_origin is None:
            logger.info('No origin locked; nothing to plot')
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning('matplotlib is not installed, skipping plot')
            return

        center, offset = self._absolute_waypoints()
        if not center:
            logger.info('No waypoints available for plotting')
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        def plot_line(points, label, style, color):
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]
            lats.append(points[0][0])
            lons.append(points[0][1])
            ax.plot(lons, lats, style, color=color, label=label)

        plot_line(center, 'Pipe centerline', '--', 'tab:orange')
        plot_line(offset, f'Offset path ({self.pipe_side}, {self.pipe_offset_m:.1f} m)', '-', 'tab:blue')

        if self.sent_refs:
            ref_lats = [p[0] for p in self.sent_refs]
            ref_lons = [p[1] for p in self.sent_refs]
            ax.scatter(ref_lons, ref_lats, s=12, color='tab:green', alpha=0.7, label='Sent references')

        if self.est_track:
            tr_lats = [p[0] for p in self.est_track]
            tr_lons = [p[1] for p in self.est_track]
            ax.plot(tr_lons, tr_lats, '-', color='tab:red', alpha=0.8, label='Vehicle track')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Pipe Follow - Mission Plot')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='best')
        fig.tight_layout()

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = f'pipe_follow_plot_{ts}.png'
        fig.savefig(out_file, dpi=150)
        logger.info('Saved mission plot to %s', out_file)
        plt.show()

    def send_reference(self, node_id, final=False):
        """
        After FollowReferenceManeuver starts, references must be sent continuously.
        """
        try:
            offset_ne = self._offset_path_points_ne()
            idx = self.wp_next % len(offset_ne)
            n, e = offset_ne[idx]
            if self.path_origin is None:
                self.path_origin = (self.lat, self.lon)
            origin_lat, origin_lon = self.path_origin
            lat, lon = imcpy.coordinates.WGS84.displace(origin_lat, origin_lon, n=n, e=e)
            self.wp_next += 1

            node = self.resolve_node_id(node_id)
            r = imcpy.Reference()
            r.lat = lat
            r.lon = lon

            dz = imcpy.DesiredZ()
            dz.value = 0.0
            dz.z_units = imcpy.ZUnits.DEPTH
            r.z = dz

            ds = imcpy.DesiredSpeed()
            ds.value = 1.6
            ds.speed_units = imcpy.SpeedUnits.METERS_PS
            r.speed = ds

            flags = imcpy.Reference.FlagsBits.LOCATION | imcpy.Reference.FlagsBits.SPEED | imcpy.Reference.FlagsBits.Z
            flags = flags | imcpy.Reference.FlagsBits.MANDONE if final else flags
            r.flags = flags

            logger.info(
                'Sending reference idx=%d side=%s offset=%.1fm lat=%.7f lon=%.7f',
                idx,
                self.pipe_side,
                self.pipe_offset_m,
                lat,
                lon,
            )
            self.current_target = (lat, lon)
            self.sent_refs.append((lat, lon))
            self.last_ref = r
            self.send(node, r)
        except KeyError:
            pass

    def is_from_target(self, msg):
        """
        Check that incoming message is from the target system.
        """
        try:
            node = self.resolve_node_id(msg)
            return node.sys_name == self.target
        except KeyError:
            return False

    @Periodic(10)
    def init_followref(self):
        """
        If target is connected, start the FollowReferenceManeuver.
        """
        if not self.state:
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
                pman.maneuver_id = 'FollowReferenceManeuver'

                spec = imcpy.PlanSpecification()
                spec.plan_id = 'FollowReference'
                spec.maneuvers.append(pman)
                spec.start_man_id = 'FollowReferenceManeuver'
                spec.description = 'Pipe-follow test with side offset references'

                pc = imcpy.PlanControl()
                pc.type = imcpy.PlanControl.TypeEnum.REQUEST
                pc.op = imcpy.PlanControl.OperationEnum.START
                pc.plan_id = 'FollowReference'
                pc.arg = spec

                self.send(node, pc)
                logger.info('Started FollowRef command')
            except KeyError:
                pass

    @Subscribe(imcpy.EstimatedState)
    def recv_estate(self, msg):
        if self.is_from_target(msg):
            self.lat, self.lon, _ = imcpy.coordinates.toWGS84(msg)
            self.est_track.append((self.lat, self.lon))
            if self.path_origin is None:
                self.path_origin = (self.lat, self.lon)

    @Subscribe(imcpy.FollowRefState)
    def recv_followrefstate(self, msg: imcpy.FollowRefState):
        if not self.is_from_target(msg):
            return

        logger.info('Received FollowRefState')
        self.state = msg.state

        if msg.state == imcpy.FollowRefState.StateEnum.GOTO:
            logger.info('Goto')
            if msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR:
                dist = self._distance_to_current_target_m()
                if dist is None or dist <= self.accept_radius_m:
                    logger.info('-- Near XY and within %.1fm', self.accept_radius_m)
                    self.send_reference(node_id=self.target)
                else:
                    logger.info('-- Near XY but %.2fm from target (threshold %.1fm)', dist, self.accept_radius_m)
        elif msg.state in (
            imcpy.FollowRefState.StateEnum.LOITER,
            imcpy.FollowRefState.StateEnum.HOVER,
            imcpy.FollowRefState.StateEnum.WAIT,
        ):
            logger.info('Waiting')
            self.send_reference(node_id=self.target)
        elif msg.state == imcpy.FollowRefState.StateEnum.ELEVATOR:
            logger.info('Elevator')
        elif msg.state == imcpy.FollowRefState.StateEnum.TIMEOUT:
            logger.info('Timeout')

    @Periodic(1.0)
    def periodic_ref(self):
        if self.last_ref:
            try:
                self.send(self.target, self.last_ref)
            except KeyError:
                pass


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(description='Pipe follow test with lateral waypoint offset.')
    parser.add_argument('--target', default='lauv-simulator-1', help='Target IMC system name')
    parser.add_argument('--side', choices=('left', 'right'), default='right', help='Offset side relative to pipe direction')
    parser.add_argument('--offset', type=float, default=8.0, help='Lateral offset in meters')
    parser.add_argument('--accept-radius', type=float, default=0.0, help='Meters required before switching to next reference')
    parser.add_argument('--plot-on-exit', action='store_true', help='Plot paths and track when mission exits')
    args = parser.parse_args()

    x = FollowRef(
        args.target,
        pipe_side=args.side,
        pipe_offset_m=args.offset,
        accept_radius_m=args.accept_radius,
    )
    try:
        x.run()
    finally:
        if args.plot_on_exit:
            x.plot_mission()