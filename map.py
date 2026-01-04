import gpxpy
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D

# -----------------------------
# 1. GPX
# -----------------------------
def load_gpx(path):
    with open(path) as f:
        gpx = gpxpy.parse(f)
    pts = []
    for t in gpx.tracks:
        for s in t.segments:
            for p in s.points:
                if p.elevation is not None:
                    pts.append((p.latitude, p.longitude, p.elevation))
    return np.array(pts)

# -----------------------------
# 2. Proiezione
# -----------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def project(latlon):
    x, y = transformer.transform(latlon[:,1], latlon[:,0])
    return np.column_stack((x, y, latlon[:,2]))

# -----------------------------
# 3. Pendenza
# -----------------------------
def cumulative_distance(xy):
    dxy = np.diff(xy[:,:2], axis=0)
    return np.insert(np.sqrt((dxy**2).sum(axis=1)).cumsum(), 0, 0)

def compute_slope(dist, ele):
    slope = np.gradient(ele, dist)
    return savgol_filter(slope, 21, 2)

def slope_to_width(slope, min_w=1.5, max_w=8):
    s = np.clip(np.abs(slope), 0, 0.15)
    return min_w + (max_w - min_w) * (s / s.max())

# -----------------------------
# 4. Plot con legenda
# -----------------------------
def plot_map_with_legend(x, y, slope):
    fig, ax = plt.subplots(figsize=(10,10))
    widths = slope_to_width(slope)

    # Traccia GPX
    for i in range(len(x)-1):
        ax.plot(
            x[i:i+2], y[i:i+2],
            linewidth=widths[i],
            color='crimson',
            alpha=0.8,
            solid_capstyle='round',
            zorder=10
        )

    # Padding
    pad = 200
    ax.set_xlim(x.min()-pad, x.max()+pad)
    ax.set_ylim(y.min()-pad, y.max()+pad)

    # Basemap OSM
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857", zorder=0)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # --- Legenda manuale ---
    # scegli 3 valori rappresentativi di pendenza
    slopes_legend = [0.02, 0.07, 0.12]  # valori in frazione (2%, 7%, 12%)
    labels = [f"{int(s*100)}%" for s in slopes_legend]
    widths_legend = slope_to_width(np.array(slopes_legend))

    legend_lines = [Line2D([0],[0], color='crimson', lw=w) for w in widths_legend]
    ax.legend(legend_lines, labels, title="Pendenza", loc='lower right', framealpha=0.8)

    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
def main():
    gpx_file = "track.gpx"
    points = load_gpx(gpx_file)
    proj = project(points)
    x, y, ele = proj[:,0], proj[:,1], proj[:,2]
    dist = cumulative_distance(proj)
    slope = compute_slope(dist, ele)
    plot_map_with_legend(x, y, slope)

if __name__ == "__main__":
    main()

