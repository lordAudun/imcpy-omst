"""
@file: path_optimization.py
@brief: Waypoint reduction based on path curvature
"""
import csv
from pathlib import Path
import math


R = 6378137.0


def load_pipe_centerline(file_path):
    """Load pipe_centerline.csv, return (xy_points, origin, depths).

    xy_points : list of (x, y) in local metres
    origin    : (lat0_rad, lon0_rad) reference for inverse projection
    depths    : list of depth_m values matching xy_points
    """
    rows = []
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["lat"]), float(row["lon"]), float(row["depth_m"])))

    lat0 = math.radians(rows[0][0])
    lon0 = math.radians(rows[0][1])

    points, depths = [], []
    for lat_deg, lon_deg, depth in rows:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x = (lon - lon0) * math.cos(lat0) * R
        y = (lat - lat0) * R
        points.append((x, y))
        depths.append(depth)

    return points, (lat0, lon0), depths


def xy_to_latlon(x, y, lat0, lon0):
    """Convert local x/y (metres) back to lat/lon degrees."""
    lat = lat0 + y / R
    lon = lon0 + x / (R * math.cos(lat0))
    return math.degrees(lat), math.degrees(lon)


def nearest_depth(wx, wy, orig_points, orig_depths):
    """Return depth of the closest original point to (wx, wy)."""
    best_idx = min(
        range(len(orig_points)),
        key=lambda i: (orig_points[i][0] - wx) ** 2 + (orig_points[i][1] - wy) ** 2,
    )
    return orig_depths[best_idx]


def save_waypoints(file_path, waypoints, origin, orig_points, orig_depths):
    """Save optimized waypoints as CSV with lat, lon, depth_m."""
    lat0, lon0 = origin
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "lat", "lon", "depth_m"])
        for i, (x, y) in enumerate(waypoints):
            lat, lon = xy_to_latlon(x, y, lat0, lon0)
            depth = nearest_depth(x, y, orig_points, orig_depths)
            writer.writerow([i, lat, lon, depth])


class PathOptimization:

    def __init__(self, points, sampling_rate=5.0):
        self.points = points
        self.points_new = []
        self.sampling_rate = sampling_rate
        self.path_points = []

    def get_path(self):
        return self.path_points

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def triangle_angle(self, previous_point, next_point, point):
        a = self.euclidean_distance(previous_point[0], previous_point[1], point[0], point[1])
        b = self.euclidean_distance(point[0], point[1], next_point[0], next_point[1])
        c = self.euclidean_distance(previous_point[0], previous_point[1], next_point[0], next_point[1])
        try:
            angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        except (ValueError, ZeroDivisionError):
            angle = math.pi
        return math.degrees(angle)

    def optimize_path(self):
        # First pass: keep points at curves or at the sampling interval
        for i, point in enumerate(self.points):
            if i == 0:
                self.points_new.append(point)
                continue
            if i >= len(self.points) - self.sampling_rate:
                break
            angle = self.triangle_angle(self.points[i - 1], self.points[i + 1], point)
            if angle < 150 or i % self.sampling_rate == 0:
                self.points_new.append(point)
        self.points_new.append(self.points[-1])

        # Iteratively smooth sharp angles by averaging with neighbours
        has_acute_angle = True
        while has_acute_angle:
            has_acute_angle = False
            for i in range(1, len(self.points_new) - 1):
                angle = self.triangle_angle(self.points_new[i - 1], self.points_new[i + 1], self.points_new[i])
                if angle == 180 or angle == 0:
                    continue
                if angle < 150:
                    new_x = (self.points_new[i - 1][0] + self.points_new[i + 1][0]) / 2
                    new_y = (self.points_new[i - 1][1] + self.points_new[i + 1][1]) / 2
                    self.points_new[i] = (new_x, new_y)
                    has_acute_angle = True

        self.path_points = [(p[0], p[1]) for p in self.points_new]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    csv_path = Path(__file__).parent / "pipe_centerline.csv"
    points, origin, depths = load_pipe_centerline(csv_path)

    path_optimization = PathOptimization(points, sampling_rate=5.0)
    path_optimization.optimize_path()
    waypoints = path_optimization.get_path()
    print(f"Input points: {len(points)}, Output waypoints: {len(waypoints)}")

    out_path = Path(__file__).parent / "pipe_centerline_optimized.csv"
    save_waypoints(out_path, waypoints, origin, points, depths)
    print(f"Saved to {out_path}")

    orig_x = [p[0] for p in points]
    orig_y = [p[1] for p in points]
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].plot(orig_x, orig_y, "b-", linewidth=1)
    axes[0].set_title(f"Original ({len(points)} points)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")

    axes[1].plot(orig_x, orig_y, "b-", linewidth=1, alpha=0.3, label="original")
    axes[1].plot(wp_x, wp_y, "r-o", linewidth=1.5, markersize=3, label=f"waypoints ({len(waypoints)})")
    axes[1].set_title(f"Optimized ({len(waypoints)} waypoints)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_aspect("equal")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "path_optimization.png", dpi=150)
    plt.show()
