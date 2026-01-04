import time
import os
import hashlib
import gpxpy
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import osmnx as ox
from pyproj import Transformer
from scipy.signal import savgol_filter
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

def slope_to_width(slope, min_w=2.5, max_w=9, max_slope=None):
    """
    Converte pendenza in spessore linea.
    slope: array o float
    max_slope: massimo valore di pendenza (default: massimo della traccia)
    """
    if max_slope is None:
        max_slope = np.max(np.abs(slope)) if np.any(slope) else 0.01
    s = np.clip(np.abs(slope), 0, max_slope)
    return min_w + (max_w - min_w) * (s / max_slope)

# -----------------------------
# OSM Cache helper
# -----------------------------
def bbox_hash(bbox):
    """Crea un hash della bounding box per il nome file cache"""
    s = "_".join([f"{b:.6f}" for b in bbox])
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def load_osm_graph_cached(track_latlon, buffer=0.01, cache_dir="cache_osm"):
    t0 = time.perf_counter()
    os.makedirs(cache_dir, exist_ok=True)

    lats = track_latlon[:,0]
    lons = track_latlon[:,1]
    north = lats.max() + buffer
    south = lats.min() - buffer
    east  = lons.max() + buffer
    west  = lons.min() - buffer
    bbox = (west, south, east, north)

    filename = os.path.join(cache_dir, f"osm_{bbox_hash(bbox)}.graphml")

    if os.path.exists(filename):
        log(f"Loading OSM graph from cache: {filename}", t0)
        G = ox.load_graphml(filename)
    else:
        log("Downloading OSM graph…", t0)
        G = ox.graph_from_bbox(bbox, network_type='all')  # tupla singola
        ox.save_graphml(G, filename)
        log(f"OSM graph saved to cache: {filename}", t0)

    return G

# -----------------------------
# Road type → color map
# -----------------------------
TYPE_COLORS = {
    'cycleway':'green',
    'residential':'pink',
    'tertiary':'yellow',
    'secondary':'orange',
    'primary':'red',
    'trunk':'magenta',
    'unclassified':'lightgrey',
    'path':'saddlebrown'
}

def edge_color(edge):
    hw = edge.get('highway')
    if isinstance(hw, list):
        hw = hw[0]
    return TYPE_COLORS.get(hw, 'black')

# -----------------------------
# KD-Tree for OSM edges
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

def get_colors_gpx_kdtree(xy, tree, edge_colors):
    t0 = time.perf_counter()
    _, idx = tree.query(xy)
    colors = [edge_colors[i] for i in idx]
    log("Assigned colors to GPX points via KD-Tree", t0)
    return colors

# -----------------------------
# Plot map
# -----------------------------
def plot_gpx_osm(x, y, slope, colors, base_interval=10, label_offset=30):
    """
    x, y: coordinate del percorso
    slope: array di pendenze
    colors: array colori per tipo strada
    base_interval: intervallo base tra label per tratti pianeggianti
    label_offset: spostamento delle label dal percorso
    """
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(10,10))

    widths = slope_to_width(slope)

    # Traccia percorso con bordo nero
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2],
                linewidth=widths[i]+2, color='black', solid_capstyle='round', alpha=0.9)
        ax.plot(x[i:i+2], y[i:i+2],
                linewidth=widths[i], color=colors[i], solid_capstyle='round', alpha=0.9)

    slope_abs = np.abs(slope)
    slope_max = np.max(slope_abs)
    slope_mean = np.mean(slope_abs)

    # Aggiungi label pendenza in modo più frequente sui tratti ripidi
    for i in range(len(x)):
        # Determina intervallo adattivo: lineare tra base_interval e base_interval/3
        freq = int(base_interval * (1 - slope_abs[i]/slope_max * 2/3))
        freq = max(3, freq)  # non troppo piccolo per evitare sovrapposizioni

        if i % freq != 0:
            continue

        px, py = x[i], y[i]
        p_slope = slope[i]
        label = f"{p_slope:.1%}"

        # Spostamento semplice orizzontale e verticale
        ox = label_offset
        oy = label_offset if slope[i] >= 0 else -label_offset

        ax.text(px + ox, py + oy, label,
                color='black', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    pad = 200
    ax.set_xlim(x.min()-pad, x.max()+pad)
    ax.set_ylim(y.min()-pad, y.max()+pad)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                    crs='EPSG:3857', zorder=0)

    ax.set_aspect('equal')
    ax.set_axis_off()

    # --------------------------
    # Legenda tipi strada (colori)
    # --------------------------
    legend_lines_type = [
        Line2D([0],[0], color=c, lw=3, solid_capstyle='round')
        for c in TYPE_COLORS.values()
    ]
    legend1 = ax.legend(legend_lines_type, TYPE_COLORS.keys(),
                        title="Tipo strada", loc='lower right')
    ax.add_artist(legend1)

    # --------------------------
    # Legenda pendenza (spessore)
    # --------------------------
    slope_min = np.min(slope)
    slope_max_trace = np.max(slope)
    slope_values = np.linspace(slope_min, slope_max_trace, 5)
    lines_slope = [
        Line2D([0],[0], color='black', lw=slope_to_width(s, max_slope=slope_max_trace))
        for s in slope_values
    ]
    labels_slope = [f"{s:.1%}" for s in slope_values]
    legend2 = ax.legend(lines_slope, labels_slope,
                        title="Pendenza", loc='upper right')
    ax.add_artist(legend2)

    log("Map plotted with adaptive slope labels", t0)
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

    # OSM con cache
    G = load_osm_graph_cached(pts, buffer=0.01)
    G_proj = ox.project_graph(G, to_crs="EPSG:3857")
    tree, edge_colors = build_osm_kdtree(G_proj)

    xy_points = np.column_stack((x, y))
    colors = get_colors_gpx_kdtree(xy_points, tree, edge_colors)

    plot_gpx_osm(x, y, slope, colors)

    log("Total execution", t0)

if __name__ == "__main__":
    main()

