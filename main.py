"""
app-coreg-v2: Automated MEG/EEG coregistration to MRI.

Inputs : raw or epochs FIF (sensor positions + digitization),
         FreeSurfer subject directory.
Outputs: trans.fif (MRI-to-head transform), quality report.
"""

import os
import sys
import numpy as np

# Headless 3D rendering — must be set BEFORE vtk/pyvista/mne.viz is imported.
# QT_QPA_PLATFORM=offscreen lets Qt init without X11 (bypasses MNE's _display_is_valid check).
# VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1 tells VTK to render offscreen (uses OSMesa if available).
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

# Set up FreeSurfer environment (needed for make_scalp_surfaces)
if not os.environ.get('FREESURFER_HOME'):
    for _candidate in ['/usr/local/freesurfer', '/opt/freesurfer', '/usr/share/freesurfer']:
        if os.path.isdir(os.path.join(_candidate, 'bin')):
            os.environ['FREESURFER_HOME'] = _candidate
            break
fs_home = os.environ.get('FREESURFER_HOME', '')
if fs_home:
    os.environ['PATH'] = os.path.join(fs_home, 'bin') + ':' + os.environ.get('PATH', '')

# Resolve brainlife_utils — try local copy first, then parent monorepo
app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
for search_path in [app_dir, parent_dir]:
    if os.path.isdir(os.path.join(search_path, 'brainlife_utils')):
        sys.path.insert(0, search_path)
        break

from brainlife_utils import (
    setup_matplotlib_backend,
    load_config,
    ensure_output_dirs,
    add_info_to_product,
    add_image_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import matplotlib.pyplot as plt

import mne
from mne.io.constants import FIFF

# == SETUP ==
ensure_output_dirs('out_dir', 'out_figs', 'out_dir_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()
config = {k.strip(): v for k, v in config.items()}  # strip tabs/spaces from keys (Brainlife UI bug)

# == LOAD SENSOR DATA (for info + digitization) ==
epochs_file = config.get('epochs') or config.get('epo') or None
raw_file    = config.get('raw') or None

try:
    if epochs_file and os.path.isfile(epochs_file):
        data = mne.read_epochs(epochs_file, preload=False)
    elif raw_file and os.path.isfile(raw_file):
        data = mne.io.read_raw_fif(raw_file, preload=False)
    else:
        add_info_to_product(
            report_items,
            "FATAL: No sensor data found. Set 'epochs' or 'raw' in config.json.",
            "error"
        )
        create_product_json(report_items)
        sys.exit(1)
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not load sensor data: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

info = data.info

# Detect modality
ch_types  = info.get_channel_types()
meg_types = {'mag', 'grad', 'ref_meg'}
eeg_count = sum(1 for t in ch_types if t == 'eeg')
meg_count = sum(1 for t in ch_types if t in meg_types)

if meg_count > 0 and eeg_count > 0:
    modality = 'meeg'
elif meg_count > 0:
    modality = 'meg'
elif eeg_count > 0:
    modality = 'eeg'
else:
    add_info_to_product(
        report_items,
        f"FATAL: No MEG or EEG channels found. Types: {set(ch_types)}",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

add_info_to_product(
    report_items,
    f"Modality: {modality} | EEG: {eeg_count} | MEG: {meg_count}",
    "info"
)

# == RESOLVE FREESURFER DIRECTORY ==
# Brainlife passes the FreeSurfer subject directory as 'freesurfer'.
# We derive subjects_dir (parent) and subject (basename) from it.
fs_path      = config.get('freesurfer') or config.get('output')
subjects_dir = config.get('subjects_dir')
subject      = config.get('subject')

if fs_path and os.path.isdir(fs_path):
    fs_path = os.path.abspath(fs_path)
    if not subjects_dir:
        subjects_dir = os.path.dirname(fs_path)
    if not subject:
        subject = os.path.basename(fs_path)

if not subjects_dir or not subject:
    add_info_to_product(
        report_items,
        "FATAL: No FreeSurfer directory found. "
        "Set 'freesurfer' in config.json to the subject's FreeSurfer directory.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

if not os.path.isdir(os.path.join(subjects_dir, subject)):
    add_info_to_product(
        report_items,
        f"FATAL: FreeSurfer subject directory not found: "
        f"{os.path.join(subjects_dir, subject)}",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

add_info_to_product(report_items, f"Subject: {subject}", "info")

# == MAKE SCALP SURFACES (required for 3D alignment plots) ==
# Creates head-dense.fif in the FreeSurfer subject's bem/ directory.
# Requires FreeSurfer binaries (mkheadsurf). Skipped gracefully if unavailable.
# mkheadsurf needs SUBJECTS_DIR in env (not just as a CLI arg).
os.environ["SUBJECTS_DIR"] = subjects_dir
try:
    mne.bem.make_scalp_surfaces(subject, subjects_dir=subjects_dir,
                                force=True, overwrite=True, no_decimate=True,
                                verbose=True)
    add_info_to_product(report_items, "Scalp surface created (head-dense)", "info")
except Exception as e:
    add_info_to_product(report_items,
                        f"Scalp surface generation skipped (no FreeSurfer?): {e}", "warning")

# == CHECK DIGITIZATION POINTS ==
if not info['dig']:
    add_info_to_product(
        report_items,
        "FATAL: No digitization points found. "
        "Coregistration requires digitized fiducials (nasion, LPA, RPA).",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

fid_count = sum(1 for d in info['dig'] if d['kind'] == FIFF.FIFFV_POINT_CARDINAL)
hsp_count = sum(1 for d in info['dig']
                if d['kind'] in (FIFF.FIFFV_POINT_HPI, FIFF.FIFFV_POINT_EXTRA))

add_info_to_product(
    report_items,
    f"Digitization: {fid_count} fiducials, {hsp_count} head shape points",
    "info"
)

if fid_count < 3:
    add_info_to_product(
        report_items,
        f"FATAL: Only {fid_count} fiducial point(s) found (need 3: nasion, LPA, RPA).",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

if hsp_count == 0:
    add_info_to_product(
        report_items,
        "No head shape points — running fiducials-only fit (less accurate). "
        "For better results, digitize head shape points.",
        "warning"
    )

# == PREPARE 3D BACKEND (offscreen via QT_QPA_PLATFORM=offscreen + VTK offscreen) ==
# Pre-create QApplication so MNE's _display_is_valid() check is bypassed.
use_3d = False
use_meg = modality in ('meg', 'meeg')
use_eeg = modality in ('eeg', 'meeg')

try:
    # Pre-create QApplication so MNE's _display_is_valid() check is bypassed
    from qtpy.QtWidgets import QApplication
    _qapp = QApplication.instance() or QApplication(sys.argv)

    import pyvista as pv
    pv.OFF_SCREEN = True
    mne.viz.set_3d_backend('pyvistaqt')

    # Monkey-patch: always create pyvista.Plotter(off_screen=True) instead of
    # BackgroundPlotter — plot_alignment() doesn't pass off_screen so the original
    # patch misses it and BackgroundPlotter renders black without hardware GL.
    from mne.viz.backends._pyvista import (
        PyVistaFigure, Plotter as PVPlotter, _PyVistaRenderer, _ALL_PLOTTERS,
    )
    import mne.viz.backends.renderer as renderer_mod

    def _patched_build(self):
        if self._plotter is None:
            store_filtered = {k: v for k, v in self.store.items()
                              if k in ('window_size', 'shape', 'border', 'multi_samples')}
            plotter = PVPlotter(off_screen=True, **store_filtered)
            plotter.background_color = self.background_color
            self._plotter = plotter
            try:
                _ALL_PLOTTERS[plotter._id_name] = plotter
            except AttributeError:
                pass
        if self.plotter.iren is not None:
            self.plotter.iren.initialize()
            def safe_update(stime=1, force_redraw=True):
                self.plotter.render()
            self.plotter.update = safe_update
        return self.plotter

    PyVistaFigure._build = _patched_build

    class _OffscreenRenderer(_PyVistaRenderer):
        _kind = 'pyvistaqt'
        def _window_initialize(self, **kwargs): pass
        def _window_close_connect(self, func, *, after=True): pass
        def _window_close_disconnect(self, func): pass
        def _window_set_theme(self, theme): pass

    renderer_mod.backend._Renderer = _OffscreenRenderer
    use_3d = True
except Exception as e:
    add_info_to_product(report_items,
                        f"3D alignment plots unavailable: {e}", "warning")

# Shared kwargs for all mne.viz.plot_alignment calls
# Use head-dense if available (requires make_scalp_surfaces), else fall back to white
_head_dense_fif = os.path.join(subjects_dir, subject, 'bem', f'{subject}-head-dense.fif')
# For EEG, a scalp surface is required for electrode projection — use 'auto' as fallback.
# 'auto' lets MNE pick the best available surface. For MEG-only, 'white' is acceptable.
if os.path.isfile(_head_dense_fif):
    _plot_surface = 'head-dense'
elif use_eeg:
    _plot_surface = 'auto'
    add_info_to_product(report_items,
                        "head-dense surface not found — using surfaces='auto' for EEG projection",
                        "warning")
else:
    _plot_surface = 'white'
    add_info_to_product(report_items,
                        "head-dense surface not found — alignment plots will show white matter surface",
                        "warning")
plot_kwargs = dict(
    subject=subject, subjects_dir=subjects_dir,
    surfaces=_plot_surface,
    dig=True,
    meg='sensors' if use_meg else [],
    eeg='projected' if use_eeg else [],
    coord_frame='head',
    show_axes=True,
)


def _save_alignment_fig(step_name, label, add_to_product=False):
    """Save 4-view (front/left/right/top) tiled screenshot. Skips if 3D unavailable."""
    if not use_3d:
        return
    _views = [
        ("Front",  (0,   90)),
        ("Right",  (90,  90)),
        ("Left",   (270, 90)),
        ("Top",    (0,   180)),
    ]
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
        imgs = []
        for view_label, (azimuth, elevation) in _views:
            fig.plotter.camera_position = "xy"
            fig.plotter.camera.azimuth = azimuth
            fig.plotter.camera.elevation = elevation
            fig.plotter.camera.reset_clipping_range()
            fig.plotter.render()
            img = fig.plotter.screenshot(return_img=True)
            imgs.append((view_label, img))
        try:
            fig.plotter.close()
        except Exception:
            pass
        # tile into 2x2 grid
        fig_mpl, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig_mpl.suptitle(label, fontsize=13, fontweight="bold")
        for ax, (view_label, img) in zip(axes.flat, imgs):
            ax.imshow(img)
            ax.set_title(view_label, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        path = os.path.join("out_figs", f"{step_name}.png")
        fig_mpl.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig_mpl)
        if add_to_product:
            add_image_to_product(report_items, label, filepath=path)
    except Exception as e:
        add_info_to_product(report_items, f"Could not save {label} plot: {e}", "warning")


# == COREGISTRATION ==
# fiducials: 'auto' | 'estimated' | JSON dict string e.g. '{"nasion":[0,0.1,0],"lpa":[-0.07,0,0],"rpa":[0.07,0,0]}'
fiducials_raw = config.get('fiducials') or 'auto'
import json
try:
    fiducials = json.loads(fiducials_raw)  # parse dict if JSON string provided
except (TypeError, ValueError, json.JSONDecodeError):
    fiducials = fiducials_raw  # keep as string ('auto', 'estimated')

try:
    coreg = mne.coreg.Coregistration(
        info, subject, subjects_dir,
        fiducials=fiducials,
        on_defects='warn'  # don't crash on minor surface defects
    )
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not initialise coregistration: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

try:
    # Step 1: initial state (saved to file only, not product.json)
    _save_alignment_fig('coreg_01_initial', '1. Initial (before fit)')

    # Step 2: coarse fit using fiducials
    coreg.fit_fiducials(verbose=True)
    add_info_to_product(report_items, "Fiducials fit complete", "info")
    _save_alignment_fig('coreg_02_fiducials', '2. After fiducials fit')

    # Step 3: ICP refinement — only if head shape points available
    if hsp_count > 0:
        icp_iter_1   = int(config.get('icp_iterations_1') or 6)
        icp_iter_2   = int(config.get('icp_iterations_2') or 20)
        nasion_w1    = float(config.get('nasion_weight_1') or 2.0)
        nasion_w2    = float(config.get('nasion_weight_2') or 10.0)
        omit_dist_mm = float(config.get('omit_distance_mm') or 5.0)

        coreg.fit_icp(n_iterations=icp_iter_1, nasion_weight=nasion_w1, verbose=True)
        _save_alignment_fig('coreg_03_icp1', f'3. After ICP ({icp_iter_1} iterations)')  # file only

        coreg.omit_head_shape_points(distance=omit_dist_mm / 1000)
        coreg.fit_icp(n_iterations=icp_iter_2, nasion_weight=nasion_w2, verbose=True)
        add_info_to_product(
            report_items,
            f"ICP refinement complete ({icp_iter_1} + {icp_iter_2} iterations, "
            f"omit > {omit_dist_mm} mm)",
            "info"
        )

    # Step 4: final result — only this one goes into product.json
    _save_alignment_fig('coreg_04_final', 'Final alignment', add_to_product=True)

except Exception as e:
    add_info_to_product(report_items, f"FATAL: Coregistration fitting failed: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == QUALITY CHECK ==
dists = np.array([])
try:
    dists  = coreg.compute_dig_mri_distances() * 1e3  # mm
    mean_d = np.mean(dists)
    min_d  = np.min(dists)
    max_d  = np.max(dists)

    quality = "warning" if mean_d > 5.0 else "info"
    add_info_to_product(
        report_items,
        f"Fit quality — HSP ↔ MRI surface distances:\n"
        f"  mean: {mean_d:.2f} mm | min: {min_d:.2f} mm | max: {max_d:.2f} mm",
        quality
    )
    if mean_d > 5.0:
        add_info_to_product(
            report_items,
            f"Mean distance {mean_d:.1f} mm > 5 mm. "
            "Check digitization quality and inspect the alignment plots.",
            "warning"
        )
except Exception as e:
    add_info_to_product(report_items, f"Could not compute fit distances: {e}", "warning")

# == SAVE trans.fif ==
trans_path = os.path.join('out_dir', 'trans.fif')
try:
    mne.write_trans(trans_path, coreg.trans, overwrite=True)
    add_info_to_product(report_items, "trans.fif saved successfully", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not save trans.fif: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == FIGURES (always-available matplotlib plots) ==

# Distance histogram
if len(dists) > 0:
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dists, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(dists), color='red', linestyle='--',
                   label=f'Mean: {np.mean(dists):.2f} mm')
        ax.axvline(5.0, color='orange', linestyle=':', alpha=0.8, label='5 mm threshold')
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Count')
        ax.set_title('Head Shape Point to MRI Surface Distances')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join('out_figs', 'coreg_distances.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        add_image_to_product(report_items, 'Fit Distances Histogram', filepath=fig_path)
    except Exception as e:
        add_info_to_product(report_items, f"Could not plot distances: {e}", "warning")

# Sensor topomap (saved to file for report only)
try:
    fig_sensors = mne.viz.plot_sensors(info, show_names=False, show=False)
    fig_path = os.path.join('out_figs', 'coreg_sensors.png')
    fig_sensors.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig_sensors)
except Exception as e:
    add_info_to_product(report_items, f"Could not plot sensors: {e}", "warning")

# == SAVE REPORT ==
report = mne.Report(title='Coregistration Report')
_fig_entries = [
    ('coreg_01_initial',   '1. Initial (before fit)'),
    ('coreg_02_fiducials', '2. After fiducials fit'),
    ('coreg_03_icp1',      '3. After ICP pass 1'),
    ('coreg_04_final',     'Final alignment'),
    ('coreg_distances',    'Fit distances histogram'),
    ('coreg_sensors',      'Sensor topomap'),
]
for fname, title in _fig_entries:
    fpath = os.path.join('out_figs', fname + '.png')
    if os.path.isfile(fpath):
        report.add_image(fpath, title=title)
report.save(os.path.join('out_dir_report', 'report.html'), overwrite=True)

add_info_to_product(report_items, "Coregistration completed successfully", "success")
create_product_json(report_items)
print("Done.")
