#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import sys
import time

import imcpy
import imcpy.coordinates
from imcpy.actors import DynamicActor
from imcpy.decorators import Periodic, Subscribe

logger = logging.getLogger('examples.FollowRef')


class FollowRef(DynamicActor):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.heartbeat.append(target)

        self.state = None
        self.lat = 0.0
        self.lon = 0.0
        self.last_ref = None

        # Actual commanded depth in references (kept at surface for simulation stability).
        self.depth = 10.0
        self.max_pitch_deg = 15.0
        self.speed_mps = 1.6

        # Rotation of dive direction relative to direction of first square point.
        self.dive_heading_offset_deg = 0.0

        # Incremental square edges from start (clear square, not diamond).
        self.square_edge_increments = [
            (50.0, 0.0),
            (0.0, 50.0),
            (-50.0, 0.0),
            (0.0, -50.0),
        ]

        self.start_lat = None
        self.start_lon = None

        # Mission queue: [('DIVE', lat, lon), ('WP0', lat, lon), ...]
        self.mission_points = None
        self.current_ref_idx = 0
        self.last_ref_sent_t = None
        self.min_dive_leg_time_s = 8.0

    def compute_dive_offset(self) -> float:
        gamma = math.radians(abs(self.max_pitch_deg))
        tan_gamma = math.tan(gamma)
        if abs(tan_gamma) < 1e-6:
            raise ValueError('max_pitch_deg must be non-zero and not near 0')
        return abs(self.depth) / tan_gamma

    def build_mission_points(self):
        if self.start_lat is None or self.start_lon is None:
            return
        if self.mission_points is not None:
            return

        # Build absolute square waypoints from start by cumulative increments.
        square_points = []
        n_acc = 0.0
        e_acc = 0.0
        for dn, de in self.square_edge_increments:
            n_acc += dn
            e_acc += de
            lat, lon = imcpy.coordinates.WGS84.displace(self.start_lat, self.start_lon, n=n_acc, e=e_acc)
            square_points.append((lat, lon))

        # Dive point direction is based on vector from start to first square waypoint.
        n_first = n_acc = self.square_edge_increments[0][0]
        e_first = e_acc = self.square_edge_increments[0][1]
        norm = math.hypot(n_first, e_first)
        if norm < 1e-6:
            dive_n, dive_e = 0.0, 0.0
        else:
            L = self.compute_dive_offset()
            base_heading = math.atan2(e_first, n_first)
            dive_heading = base_heading + math.radians(self.dive_heading_offset_deg)
            dive_n = math.cos(dive_heading) * L
            dive_e = math.sin(dive_heading) * L

        dive_lat, dive_lon = imcpy.coordinates.WGS84.displace(self.start_lat, self.start_lon, n=dive_n, e=dive_e)

        self.mission_points = [('DIVE', dive_lat, dive_lon)]
        for i, (lat, lon) in enumerate(square_points):
            self.mission_points.append((f'WP{i}', lat, lon))

        logger.info('Mission built: DIVE + %d square waypoints', len(square_points))

    def send_current_reference(self, node_id):
        if self.mission_points is None:
            return
        if self.current_ref_idx >= len(self.mission_points):
            # After first cycle, keep looping square only (skip DIVE).
            self.current_ref_idx = 1

        ref_name, lat, lon = self.mission_points[self.current_ref_idx]

        try:
            node = self.resolve_node_id(node_id)
        except KeyError:
            return

        r = imcpy.Reference()
        r.lat = lat
        r.lon = lon

        dz = imcpy.DesiredZ()
        dz.value = self.depth
        dz.z_units = imcpy.ZUnits.DEPTH
        r.z = dz

        ds = imcpy.DesiredSpeed()
        ds.value = self.speed_mps
        ds.speed_units = imcpy.SpeedUnits.METERS_PS
        r.speed = ds

        r.flags = (
            imcpy.Reference.FlagsBits.LOCATION
            | imcpy.Reference.FlagsBits.SPEED
            | imcpy.Reference.FlagsBits.Z
        )

        logger.info('Sending reference %s (idx=%d) depth=%.1f', ref_name, self.current_ref_idx, self.depth)
        self.last_ref = r
        self.last_ref_sent_t = time.time()
        self.send(node, r)

    def send_next_reference(self, node_id):
        self.current_ref_idx += 1
        self.send_current_reference(node_id)

    def is_from_target(self, msg):
        try:
            node = self.resolve_node_id(msg)
            return node.sys_name == self.target
        except KeyError:
            return False

    @Periodic(10)
    def init_followref(self):
        # Same behavior as follow_reference_example: keep trying until we
        # actually receive FollowRefState.
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
            pman.maneuver_id = 'FollowReferenceManeuver'

            spec = imcpy.PlanSpecification()
            spec.plan_id = 'FollowReference'
            spec.maneuvers.append(pman)
            spec.start_man_id = 'FollowReferenceManeuver'
            spec.description = 'Dive point first, then square waypoints'

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
        if not self.is_from_target(msg):
            return

        self.lat, self.lon, _ = imcpy.coordinates.toWGS84(msg)
        if self.start_lat is None:
            self.start_lat = self.lat
            self.start_lon = self.lon
            logger.info('Locked start position')

        self.build_mission_points()

        # First reference must be DIVE, once maneuver is confirmed alive.
        if self.state is not None and self.last_ref is None and self.mission_points is not None:
            self.current_ref_idx = 0
            self.send_current_reference(self.target)

    @Subscribe(imcpy.FollowRefState)
    def recv_followrefstate(self, msg: imcpy.FollowRefState):
        if not self.is_from_target(msg):
            return

        self.state = msg.state

        # If FollowReference is running and no reference has been sent yet,
        # force DIVE as first reference immediately.
        if self.last_ref is None and self.mission_points is not None:
            self.current_ref_idx = 0
            self.send_current_reference(self.target)
            return

        if msg.state == imcpy.FollowRefState.StateEnum.GOTO:
            if msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR:
                # Hold DIVE point briefly so it is visibly the first active leg.
                if self.current_ref_idx == 0 and self.last_ref_sent_t is not None:
                    dt = time.time() - self.last_ref_sent_t
                    if dt < self.min_dive_leg_time_s:
                        logger.info('DIVE hold active (%.1fs / %.1fs)', dt, self.min_dive_leg_time_s)
                        return
                self.send_next_reference(self.target)

        elif msg.state in (
            imcpy.FollowRefState.StateEnum.LOITER,
            imcpy.FollowRefState.StateEnum.HOVER,
            imcpy.FollowRefState.StateEnum.WAIT,
        ):
            self.send_next_reference(self.target)

    @Periodic(1.0)
    def periodic_ref(self):
        if self.last_ref:
            try:
                self.send(self.target, self.last_ref)
            except KeyError:
                pass


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    x = FollowRef('lauv-simulator-1')
    x.run()
