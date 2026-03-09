import logging
import sys
import threading

import imcpy
import imcpy.coordinates
from imcpy.actors import DynamicActor
from imcpy.decorators import Periodic, Subscribe

logger = logging.getLogger("examples.FollowRef")


class FollowRef(DynamicActor):
    def __init__(self, target: str):
        super().__init__()

        self.target = target
        self.heartbeat.append(target)

        # Vehicle state
        self.state = None
        self.lat = 0.0
        self.lon = 0.0

        # Local plan: N/E offsets (incremental square)
        # These are increments, not absolute positions.
        self.ne_increments = [
            (50.0, 0.0),
            (0.0, 50.0),
            (-50.0, 0.0),
            (0.0, -50.0),
        ]

        # Global WGS84 plan (filled once we know home position)
        self.plan_wgs84 = None  # list[(lat, lon)]

        # Waypoint indices + state (CheckPoint-style)
        self.target_idx = 0
        self.previous_idx = None
        self.in_go_back = False
        self.resume_idx = None
        self.finished = False

        # Last Reference message sent (for keep-alive)
        self.last_ref = None

        # Start a keyboard listener for 'g' go-back
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    # ==========================================================
    # KEYBOARD LISTENER (manual go-back)
    # ==========================================================
    def keyboard_listener(self):
        while True:
            try:
                key = input()
            except EOFError:
                # No stdin available, just stop listening
                break

            if key.lower() == "g":
                ok = self.request_go_back()
                if ok:
                    logger.info("Manual GO BACK requested")
                else:
                    logger.info(
                        "GO BACK requested but no previous waypoint exists yet"
                    )

    # ==========================================================
    # PLAN INITIALIZATION (once we know home lat/lon)
    # ==========================================================
    def initialize_plan(self):
        """Create absolute WGS84 waypoints once, from initial position."""
        if self.plan_wgs84 is not None:
            return
        # We need a valid home position
        if self.lat == 0.0 and self.lon == 0.0:
            # Very naive check; real-world: check a flag
            return

        logger.info("Initializing WGS84 waypoint plan from home position")
        self.plan_wgs84 = []

        # Build closed square by cumulative N/E increments from home
        home_lat = self.lat
        home_lon = self.lon
        n_accum = 0.0
        e_accum = 0.0
        for dn, de in self.ne_increments:
            n_accum += dn
            e_accum += de
            wp_lat, wp_lon = imcpy.coordinates.WGS84.displace(
                home_lat, home_lon, n=n_accum, e=e_accum
            )
            self.plan_wgs84.append((wp_lat, wp_lon))

        logger.info("Plan has %d waypoints", len(self.plan_wgs84))

    # ==========================================================
    # CHECKPOINT-LIKE LOGIC
    # ==========================================================
    def request_go_back(self) -> bool:
        """
        Interrupt current target and command vehicle to previous waypoint.
        Returns False if no previous waypoint exists yet.
        """
        if self.plan_wgs84 is None:
            return False
        if self.previous_idx is None:
            return False
        if self.finished:
            return False
        if self.in_go_back:
            # already in go-back mode
            return True

        self.resume_idx = self.target_idx
        self.target_idx = self.previous_idx
        self.in_go_back = True

        logger.info(
            "GO BACK: now targeting previous waypoint index %d (resume %d)",
            self.target_idx,
            self.resume_idx if self.resume_idx is not None else -1,
        )

        # Immediately send new target reference
        self.send_current_target_reference()
        return True

    def advance_plan_after_reach(self):
        """Advance target index when a target is reached."""
        if self.plan_wgs84 is None:
            return

        if self.in_go_back:
            # We just reached the previous waypoint.
            logger.info("Reached previous waypoint in GO BACK mode")
            self.in_go_back = False
            if self.resume_idx is not None:
                self.target_idx = self.resume_idx
                logger.info("Resuming forward plan at index %d", self.target_idx)
                self.resume_idx = None
                self.send_current_target_reference()
            return

        # Normal forward progression in a circular plan
        self.previous_idx = self.target_idx
        self.target_idx = (self.target_idx + 1) % len(self.plan_wgs84)
        logger.info(
            "Reached waypoint, advancing: previous=%d, new target=%d",
            self.previous_idx,
            self.target_idx,
        )
        self.send_current_target_reference()

    # ==========================================================
    # SEND REFERENCE FOR CURRENT TARGET
    # ==========================================================
    def send_current_target_reference(self):
        if self.plan_wgs84 is None:
            return

        if self.finished:
            return

        try:
            node = self.resolve_node_id(self.target)
        except KeyError:
            return

        lat, lon = self.plan_wgs84[self.target_idx]

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

        # We don't end the maneuver here, so no MANDONE flag.
        r.flags = (
            imcpy.Reference.FlagsBits.LOCATION
            | imcpy.Reference.FlagsBits.SPEED
            | imcpy.Reference.FlagsBits.Z
        )

        logger.info(
            "Sending reference: target_idx=%d lat=%.7f lon=%.7f",
            self.target_idx,
            lat,
            lon,
        )
        self.last_ref = r
        self.send(node, r)

    # ==========================================================
    # CHECK IF MESSAGE IS FROM TARGET
    # ==========================================================
    def is_from_target(self, msg) -> bool:
        try:
            node = self.resolve_node_id(msg)
            return node.sys_name == self.target
        except KeyError:
            return False

    # ==========================================================
    # START FOLLOW REFERENCE MANEUVER
    # ==========================================================
    @Periodic(10)
    def init_followref(self):
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
                pman.maneuver_id = "FollowReferenceManeuver"

                spec = imcpy.PlanSpecification()
                spec.plan_id = "FollowReference"
                spec.maneuvers.append(pman)
                spec.start_man_id = "FollowReferenceManeuver"
                spec.description = "Square path with go-back behavior"

                pc = imcpy.PlanControl()
                pc.type = imcpy.PlanControl.TypeEnum.REQUEST
                pc.op = imcpy.PlanControl.OperationEnum.START
                pc.plan_id = "FollowReference"
                pc.arg = spec

                self.send(node, pc)
                logger.info("Started FollowRef command")
            except KeyError:
                pass

    # ==========================================================
    # RECEIVE POSITION
    # ==========================================================
    @Subscribe(imcpy.EstimatedState)
    def recv_estate(self, msg: imcpy.EstimatedState):
        if self.is_from_target(msg):
            self.lat, self.lon, _ = imcpy.coordinates.toWGS84(msg)

            # Initialize plan once we know where we are
            self.initialize_plan()

    # ==========================================================
    # FOLLOWREF STATE MACHINE
    # ==========================================================
    @Subscribe(imcpy.FollowRefState)
    def recv_followrefstate(self, msg: imcpy.FollowRefState):
        if not self.is_from_target(msg):
            return

        self.state = msg.state

        if self.plan_wgs84 is None:
            # We don't know waypoints yet
            return

        if msg.state == imcpy.FollowRefState.StateEnum.GOTO:
            # When near XY, treat current target as "reached"
            if msg.proximity & imcpy.FollowRefState.ProximityBits.XY_NEAR:
                self.advance_plan_after_reach()

        elif msg.state in (
            imcpy.FollowRefState.StateEnum.LOITER,
            imcpy.FollowRefState.StateEnum.HOVER,
            imcpy.FollowRefState.StateEnum.WAIT,
        ):
            # Typically first state after starting – kick off the first target
            logger.info("FollowRef in WAIT/LOITER/HOVER, sending first reference")
            self.send_current_target_reference()

        # Other states (ELEVATOR, TIMEOUT, etc.) are just logged
        elif msg.state == imcpy.FollowRefState.StateEnum.ELEVATOR:
            logger.info("Elevator state")
        elif msg.state == imcpy.FollowRefState.StateEnum.TIMEOUT:
            logger.info("FollowRef timeout")

    # ==========================================================
    # RESEND LAST REF (KEEP ALIVE)
    # ==========================================================
    @Periodic(1.0)
    def periodic_ref(self):
        if self.last_ref is not None:
            try:
                self.send(self.target, self.last_ref)
            except KeyError:
                pass


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    actor = FollowRef("lauv-simulator-1")
    actor.run()