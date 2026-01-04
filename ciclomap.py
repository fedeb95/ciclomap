import time
import gpxpy
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import osmnx as ox
from pyproj import Transformer
from scipy.signal import savgol_filter
from shapely.geometry import Point
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

# -----------------------------
# Logging helper
# -----------------------------
def log(msg, t0):
    elapsed = time.perf_counter() - t0
    print(f"[{elapsed:.3f}s] {msg}")
    return time.perf_counter()

# -----------------------------
# Load GPX
# -----------------------------
def load_gpx(path):
    t0 = time.perf_counter()
    with open(path) as f:
        gpx = gpxpy.parse(f)
    pts = []
    for t in gpx.tracks:
        for s in t.segments:
            for p in s.points:
                if p.elevation is not None:
                    pts.append((p.latitude, p.longitude, p.elevation))
    log("GPX loaded", t0)
    return np.array(pts)

# -----------------------------
# Project to Web Mercator
# -----------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def project(latlon):
    t0 = time.perf_counter()
    x, y = transformer.transform(latlon[:,1], latlon[:,0])
    result = np.column_stack((x, y, latlon[:,2]))
    log("Projected GPX to EPSG:3857", t0)
    return result

# -----------------------------
# Slope calculation
# -----------------------------
def cumulative_distance(xy):
    dxy = np.diff(xy[:,:2], axis=0)
    return np.insert(np.sqrt((dxy**2).sum(axis=1)).cumsum(), 0, 0)

def compute_slope(dist, ele):
    t0 = time.perf_counter()
    raw = np.gradient(ele, dist)
    smooth = savgol_filter(raw, 21, 2)
    log("Slope computed & smoothed", t0)
    return smooth

def slope_to_width(slope, min_w=1.5, max_w=8):
    s = np.clip(np.abs(slope), 0, 0.15)
    return min_w + (max_w - min_w) * (s / s.max())

# -----------------------------
# Load OSM Graph (bounding box con buffer)
# -----------------------------
def load_osm_graph(track_latlon, buffer=0.01):
    t0 = time.perf_counter()

    # calcola bounding box in lat/lon
    lats = track_latlon[:,0]
    lons = track_latlon[:,1]
    north = lats.max() + buffer
    south = lats.min() - buffer
    east  = lons.max() + buffer
    west  = lons.min() - buffer

    # bounding box come singola tupla (west, south, east, north)
    bbox = (west, south, east, north)

    G = ox.graph_from_bbox(bbox, network_type='all')  # correttamente con tupla
    log("OSM graph downloaded", t0)
    return G

# -----------------------------
# Road type → color map
# -----------------------------
TYPE_COLORS = {
    'cycleway':'green',
    'residential':'gray',
    'tertiary':'blue',
    'secondary':'orange',
    'primary':'red',
    'trunk':'darkred',
    'unclassified':'lightgray',
    'path':'brown'
}

def edge_color(edge):
    hw = edge.get('highway')
    if isinstance(hw, list):
        hw = hw[0]
    return TYPE_COLORS.get(hw, 'black')

# -----------------------------
# Build KD-Tree from OSM edges
# -----------------------------
def build_osm_kdtree(G):
    t0 = time.perf_counter()
    edge_midpoints = []
    edge_colors = []

    for u, v, key, data in G.edges(keys=True, data=True):
        geom = data.get('geometry')
        if geom:
            x, y = geom.xy
            mx, my = np.mean(x), np.mean(y)
        else:
            mx = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
            my = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
        edge_midpoints.append((mx, my))
        edge_colors.append(edge_color(data))

    tree = cKDTree(edge_midpoints)
    log("KD-Tree built for OSM edges", t0)
    return tree, edge_colors

# -----------------------------
# Assign color to GPX points via KD-Tree
# -----------------------------
def get_colors_gpx_kdtree(xy, tree, edge_colors):
    t0 = time.perf_counter()
    _, idx = tree.query(xy)
    colors = [edge_colors[i] for i in idx]
    log("Assigned colors to GPX points via KD-Tree", t0)
    return colors

# -----------------------------
# Plot map
# -----------------------------
def plot_gpx_osm(x, y, slope, colors):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(10,10))

    widths = slope_to_width(slope)

    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2],
                linewidth=widths[i],
                color=colors[i],
                solid_capstyle='round',
                alpha=0.9)

    pad = 200
    ax.set_xlim(x.min()-pad, x.max()+pad)
    ax.set_ylim(y.min()-pad, y.max()+pad)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                    crs='EPSG:3857', zorder=0)

    ax.set_aspect('equal')
    ax.set_axis_off()

    legend_lines = [Line2D([0],[0], color=c, lw=3) for c in TYPE_COLORS.values()]
    ax.legend(legend_lines, TYPE_COLORS.keys(),
              title="Tipo strada", loc='lower right')
    log("Map plotted", t0)
    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
def main():
    t0 = time.perf_counter()

    gpx_file = "track.gpx"
    pts = load_gpx(gpx_file)

    proj = project(pts)
    x, y, ele = proj[:,0], proj[:,1], proj[:,2]

    dist = cumulative_distance(proj)
    slope = compute_slope(dist, ele)

    # Scarica la rete OSM solo nell’area della traccia + buffer
    G = load_osm_graph(pts, buffer=0.01)
    G_proj = ox.project_graph(G, to_crs="EPSG:3857")
    tree, edge_colors = build_osm_kdtree(G_proj)

    xy_points = np.column_stack((x, y))
    colors = get_colors_gpx_kdtree(xy_points, tree, edge_colors)

    plot_gpx_osm(x, y, slope, colors)

    log("Total execution", t0)

if __name__ == "__main__":
    main()

