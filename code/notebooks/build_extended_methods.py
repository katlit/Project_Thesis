"""Build the learned-NVS, VGGT-X, and cross-method comparison notebooks."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    stem = Path(name).stem.lower().replace("_", "-")
    for index, cell in enumerate(cells):
        cell["id"] = f"{stem}-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    nbf.write(notebook, ROOT / name)


common_setup = r'''
import shutil, subprocess, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
CODE_ROOT = Path("/content/Project_Thesis_code")
REPOSITORY = "https://github.com/katlit/Project_Thesis.git"
BRANCH = "codex/hq200-example-notebook"
if CODE_ROOT.exists() and not (CODE_ROOT / ".git").is_dir(): shutil.rmtree(CODE_ROOT)
command = (["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)]
           if not CODE_ROOT.exists() else ["git", "-C", str(CODE_ROOT), "pull", "--ff-only", "origin", BRANCH])
subprocess.run(command, check=True)
sys.path.insert(0, str(CODE_ROOT / "code"))
PROJECT_ROOT = Path("/content/drive/MyDrive/ITU/3D/Thesis")
'''


save("06_LagerNVS_confidence_3dgs.ipynb", [
    md('''# 06 - Learned LagerNVS views and confidence-weighted 3DGS

This notebook tests learned Plucker-ray NVS. LagerNVS generates dense RGB views. VGGT geometry supplies a separate validity mask and confidence weight. The learned RGB is never treated as ground truth without geometric support.

Run notebook 05 first because this notebook loads its `vggt_geometry.npz`. The LagerNVS checkpoint is gated; request access on Hugging Face and provide a token when prompted.'''),
    code(r'''# Start from LagerNVS's official dependency file. Open3D and websockets
# are used only by its interactive viewers, not by this notebook. Open3D has no
# wheel for the current Colab Python, and would abort the whole installation.
import subprocess
from pathlib import Path

LAGERNVS_ROOT = Path("/content/lagernvs")
if (LAGERNVS_ROOT / ".git").is_dir():
    subprocess.run(["git", "-C", str(LAGERNVS_ROOT), "pull", "--ff-only"], check=True)
else:
    subprocess.run([
        "git", "clone", "-q",
        "https://github.com/facebookresearch/lagernvs.git",
        str(LAGERNVS_ROOT),
    ], check=True)
%pip -q install --index-url https://download.pytorch.org/whl/cu126 "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"

official_requirements = (LAGERNVS_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
viewer_only = ("open3d", "websockets")
colab_requirements = [
    line for line in official_requirements
    if not line.strip().lower().startswith(viewer_only)
]
COLAB_REQUIREMENTS = Path("/content/lagernvs_colab_requirements.txt")
COLAB_REQUIREMENTS.write_text("\n".join(colab_requirements) + "\n", encoding="utf-8")
print("Skipped optional viewer packages: open3d, websockets")

%pip -q install -r /content/lagernvs_colab_requirements.txt
%pip -q install "gsplat==1.3.0" matplotlib imageio imageio-ffmpeg

# Verify the dependency that caused the previous hidden import failure.
import torch, xformers
print("Torch:", torch.__version__, "| xFormers:", xformers.__version__)
print("Restart the runtime now only if Colab asks you to do so.")'''),
    code(common_setup + r'''
import gc, getpass, json
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, login
from IPython.display import Video, display

sys.path.insert(0, "/content/lagernvs")
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vis import compute_plucker_coordinates, create_360_camera_trajectory_from_c2w_and_intrinsics, render_chunked
from src.gaussian_full import initialize_full_gaussians, render_full, train_adaptive
from src.lagernvs_bridge import normalize_lagernvs_cameras, restore_lagernvs_trajectory, save_pseudo_view_set
from src.render_comparison import foreground_metrics
from src.vggt_teacher import build_teacher_gaussians, render_teacher

if not torch.cuda.is_available(): raise RuntimeError("Connect a GPU runtime.")
DATASET = "3DRealCar"
SCENE = None
TARGET_FRAMES = 80
STEPS = 30_000
SYNTHETIC_WEIGHT = 0.25
SYNTHETIC_PROBABILITY = 0.25
MODEL_REPO = "facebook/lagernvs_general_512"
manifest = pd.read_csv(PROJECT_ROOT / "data_processed/method_inputs/manifest.csv")
scene_rows = manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET").copy()
SCENE = SCENE or sorted(scene_rows.scene.unique())[0]
scene_rows = scene_rows[scene_rows.scene.eq(SCENE)].sort_values("view_order")
assert len(scene_rows) == 8
GEOMETRY_ROOT = PROJECT_ROOT / "experiments/Geometry/VGGT" / DATASET / SCENE
PSEUDO_ROOT = PROJECT_ROOT / "experiments/SyntheticViews/LagerNVS" / DATASET / SCENE
RUN_ROOT = PROJECT_ROOT / "experiments/3DGS/LagerNVS_confidence_weighted" / DATASET / SCENE
PSEUDO_ROOT.mkdir(parents=True, exist_ok=True); RUN_ROOT.mkdir(parents=True, exist_ok=True)
print(DATASET, SCENE)'''),
    md('''## 1. Load the same eight selected inputs

The displayed order must be front, front-left, side-left, rear-left, rear, rear-right, side-right, front-right.'''),
    code(r'''fig, axes = plt.subplots(1, 8, figsize=(24, 3))
for axis, row in zip(axes, scene_rows.itertuples()):
    axis.imshow(Image.open(row.method_image).convert("RGB")); axis.set_title(str(row.view_order)); axis.axis("off")
plt.tight_layout(); plt.show()'''),
    md('''## 2. Build target Plucker rays and render learned RGB

The target cameras form a closed orbit. LagerNVS receives their Plücker rays and directly predicts dense RGB. This is learned NVS, not point splatting.'''),
    code(r'''token = getpass.getpass("Hugging Face token (input is hidden): ")
login(token=token, add_to_git_credential=False)
saved = np.load(GEOMETRY_ROOT / "vggt_geometry.npz")
extrinsics, intrinsics = saved["extrinsics"], saved["intrinsics"]
normalized_c2w, normalized_k, first_camera, scene_scale = normalize_lagernvs_cameras(extrinsics, intrinsics)

input_images = load_and_preprocess_images(scene_rows.method_image.tolist(), mode="resize", target_size=512, patch_size=8).to("cuda")[None]
height, width = input_images.shape[-2:]
normalized_k[:, 0, 0] = width; normalized_k[:, 1, 1] = width
normalized_k[:, 0, 2] = width / 2; normalized_k[:, 1, 2] = height / 2
_, target_c2w, target_fxfycxcy = create_360_camera_trajectory_from_c2w_and_intrinsics(
    normalized_c2w[None].cuda(), normalized_k[None].cuda(), TARGET_FRAMES, len(scene_rows)
)
target_rays = compute_plucker_coordinates(target_c2w, target_fxfycxcy, (height, width))
conditioning_rays = torch.zeros(1, len(scene_rows), 6, height, width, device="cuda")
rays = torch.cat([conditioning_rays, target_rays], dim=1)
camera_scale = torch.linalg.norm(normalized_c2w[:, :3, 3], dim=-1).max().item()
camera_tokens = torch.zeros(1, len(scene_rows) + TARGET_FRAMES, 11, device="cuda")
camera_tokens[:, :, 9] = camera_scale

model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
checkpoint = hf_hub_download(MODEL_REPO, filename="model.pt")
model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
model = model.cuda().eval()
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    learned_rgb = render_chunked(model, (input_images, rays, camera_tokens), num_cond_views=len(scene_rows))[0]
learned_rgb = learned_rgb.float().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
target_extrinsics = restore_lagernvs_trajectory(target_c2w[0].cpu(), first_camera, scene_scale).numpy()
target_intrinsics = np.zeros((TARGET_FRAMES, 3, 3), np.float32)
target_values = target_fxfycxcy[0].float().cpu().numpy()
target_intrinsics[:, 0, 0] = target_values[:, 0]; target_intrinsics[:, 1, 1] = target_values[:, 1]
target_intrinsics[:, 0, 2] = target_values[:, 2]; target_intrinsics[:, 1, 2] = target_values[:, 3]; target_intrinsics[:, 2, 2] = 1
del model, input_images, rays, camera_tokens; gc.collect(); torch.cuda.empty_cache()
print("Learned views:", learned_rgb.shape)'''),
    md('''## 3. Add geometric trust maps

LagerNVS does not provide calibrated per-pixel confidence. We therefore render VGGT geometry at every target camera. Pixels receive supervision only where this geometry provides sufficient opacity and confidence.'''),
    code(r'''points, confidence, masks = saved["points"], saved["confidence"], saved["masks"].astype(bool)
source_rgb = np.stack([np.asarray(Image.open(p).convert("RGB").resize((points.shape[2], points.shape[1]))) / 255 for p in scene_rows.method_image])
support = saved["support"] if "support" in saved.files else np.ones_like(confidence)
keep = masks & np.isfinite(points).all(-1) & (confidence >= 0.05) & (support >= 2)
teacher = build_teacher_gaussians(points[keep], source_rgb[keep], confidence[keep])
validity, trust = [], []
for camera_e, camera_k in zip(target_extrinsics, target_intrinsics):
    _, valid, conf, _ = render_teacher(teacher, camera_e, camera_k, height, width, alpha_min=0.40, confidence_min=0.10)
    validity.append(valid); trust.append(conf * valid)
validity, trust = np.stack(validity), np.stack(trust)
save_pseudo_view_set(PSEUDO_ROOT, learned_rgb, validity, trust, target_extrinsics, target_intrinsics)
del teacher; torch.cuda.empty_cache()
fig, axes = plt.subplots(3, 12, figsize=(28, 7))
for column, index in enumerate(np.linspace(0, TARGET_FRAMES - 1, 12, dtype=int)):
    axes[0,column].imshow(learned_rgb[index]); axes[1,column].imshow(validity[index], cmap="gray"); axes[2,column].imshow(trust[index], vmin=0,vmax=1,cmap="viridis")
    for axis in axes[:,column]: axis.axis("off")
plt.tight_layout(); plt.show()'''),
    md('''## 4. Train our confidence-weighted adaptive 3DGS

Real images use foreground RGB, masked SSIM, and silhouette loss. Learned LagerNVS pixels use a binary geometric validity mask multiplied by continuous VGGT confidence.'''),
    code(r'''real_rgb = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255 for p in scene_rows.method_image])
real_masks = np.stack([np.asarray(Image.open(p).convert("L")) >= 128 for p in scene_rows.method_mask])
real_views = [{"rgb": real_rgb[i], "mask": real_masks[i], "extrinsic": extrinsics[i], "intrinsic": intrinsics[i],
               "height": real_rgb.shape[1], "width": real_rgb.shape[2]} for i in range(8)]
synthetic_views = [{"rgb": learned_rgb[i], "validity": validity[i], "confidence": trust[i],
                    "extrinsic": target_extrinsics[i], "intrinsic": target_intrinsics[i],
                    "height": height, "width": width} for i in range(TARGET_FRAMES)]
params, gs_scale = initialize_full_gaussians(points[keep], source_rgb[keep], max_initial=50_000, sh_degree=3)
params, history = train_adaptive(params, gs_scale, real_views, synthetic_views, RUN_ROOT,
    steps=STEPS, synthetic_probability=SYNTHETIC_PROBABILITY, synthetic_weight=SYNTHETIC_WEIGHT,
    sh_degree=3, checkpoint_every=5_000)
pd.DataFrame(history).to_csv(RUN_ROOT / "training_history.csv", index=False)'''),
    md('''## 5. Check the fit on the eight real input views

These are training-view diagnostics. They show whether 3DGS can reproduce its known inputs, but they are not a novel-view test.'''),
    code(r'''metric_rows = []
for index, sample in enumerate(real_views):
    with torch.inference_mode():
        rendered, alpha, _ = render_full(
            params,
            torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device="cuda"),
            torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device="cuda"),
            sample["height"], sample["width"], 3,
        )
    prediction_rgb = rendered[0].permute(1, 2, 0).cpu().numpy()
    prediction_alpha = alpha[0].permute(1, 2, 0).cpu().numpy()
    metric_rows.append({"view": index, **foreground_metrics(
        prediction_rgb, sample["rgb"], sample["mask"], prediction_alpha
    )})
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(RUN_ROOT / "metrics.csv", index=False)
display(metrics)
print("Mean training-view metrics:")
display(metrics.drop(columns="view").mean().to_frame().T)'''),
    md('''## 6. Render the learned NVS and trained 3DGS side by side

The left side is direct LagerNVS. The right side is our trainable 3DGS supervised by real images and geometrically trusted LagerNVS pixels.'''),
    code(r'''video_path = RUN_ROOT / "lagernvs_vs_confidence_3dgs.mp4"
ours_frame_root = RUN_ROOT / "orbit_frames"; ours_frame_root.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(video_path, fps=15, codec="libx264", quality=8, macro_block_size=None)
try:
    for index in range(TARGET_FRAMES):
        with torch.inference_mode():
            rendered, alpha, _ = render_full(params, torch.tensor(target_extrinsics[index], device="cuda"),
                torch.tensor(target_intrinsics[index], device="cuda"), height, width, 3)
        rgb = rendered[0].permute(1,2,0).cpu().numpy(); a = alpha[0].permute(1,2,0).cpu().numpy()
        ours = np.clip(rgb + 1 - a, 0, 1)
        Image.fromarray(np.uint8(ours * 255)).save(ours_frame_root / f"view_{index:03d}.png")
        frame = np.concatenate([learned_rgb[index], np.ones((height,8,3)), ours], axis=1)
        writer.append_data(np.uint8(frame * 255))
finally: writer.close()
print(video_path); display(Video(str(video_path), embed=True, html_attributes="controls autoplay loop muted"))'''),
])


save("07_VGGT_X_MCMC_3DGS.ipynb", [
    md('''# 07 - VGGT-X MCMC-3DGS

This notebook runs the official baseline from beginning to end:

1. copy the same eight selected images used by the other experiments;
2. run VGGT-X with global alignment and export COLMAP geometry;
3. inspect the registered cameras and sparse points;
4. train CityGaussian with its MCMC-3DGS pose-optimization configuration;
5. evaluate the eight real views;
6. save a closed-orbit video and individual frames for notebook 08.

VGGT-X was designed for dense image collections. Eight images are used here for a fair sparse-view comparison, so a failure is also a meaningful experimental result.'''),
    md('''## Important environment design

The active Colab kernel is not modified. `uv` creates two isolated environments because the official projects require different Python and Torch versions. Installation can take several minutes and compile CUDA extensions. Start from a fresh GPU runtime.'''),
    code(common_setup + r'''
import json, os, shutil, subprocess, textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import Video, display

if subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    raise RuntimeError("Connect a GPU runtime before running notebook 07.")

DATASET = "3DRealCar"
SCENE = None
FRAMES_BETWEEN = 10
MAX_GAUSSIANS = 300_000
DOWN_SAMPLE_FACTOR = 1
RUN_INSTALL = True
RUN_VGGT_X = True
RUN_TRAINING = True
RUN_EVALUATION = True
RUN_ORBIT_RENDER = True

manifest = pd.read_csv(PROJECT_ROOT / "data_processed/method_inputs/manifest.csv")
rows = manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET").copy()
SCENE = SCENE or sorted(rows.scene.unique())[0]
rows = rows[rows.scene.eq(SCENE)].sort_values(["view_order", "source"])
assert len(rows) == 8, f"Expected eight selected images, found {len(rows)}"

STAGE = Path("/content/vggtx_data") / SCENE
VGGT_X_OUTPUT = STAGE.parent / f"{SCENE}_vggt_x"
FINAL_ROOT = PROJECT_ROOT / "experiments/3DGS/VGGT_X_MCMC" / DATASET / SCENE
FINAL_ROOT.mkdir(parents=True, exist_ok=True)
print("Scene:", SCENE)
print("Final Drive folder:", FINAL_ROOT)'''),
    md('''## 1. Install the official projects

The installation uses the repositories' own requirement files. If CUDA compilation fails, keep the error: do not silently replace the official renderer with our implementation.'''),
    code(r'''%pip -q install uv

VGGT_X_ROOT = Path("/content/VGGT-X")
CITY_ROOT = Path("/content/CityGaussian")
VGGT_ENV = Path("/content/envs/vggt_x")
CITY_ENV = Path("/content/envs/citygaussian")

def run(command, cwd=None, env=None):
    print("RUN:", " ".join(map(str, command)))
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)

if RUN_INSTALL:
    if not (VGGT_X_ROOT / ".git").is_dir():
        run(["git", "clone", "--recursive", "https://github.com/Linketic/VGGT-X.git", VGGT_X_ROOT])
    else:
        run(["git", "-C", VGGT_X_ROOT, "pull", "--ff-only"])
        run(["git", "-C", VGGT_X_ROOT, "submodule", "update", "--init", "--recursive"])

    if not (CITY_ROOT / ".git").is_dir():
        run(["git", "clone", "--recursive", "https://github.com/Linketic/CityGaussian.git", CITY_ROOT])
    else:
        run(["git", "-C", CITY_ROOT, "pull", "--ff-only"])
        run(["git", "-C", CITY_ROOT, "submodule", "update", "--init", "--recursive"])

    run(["uv", "venv", "--python", "3.10", VGGT_ENV])
    run(["uv", "pip", "install", "--python", VGGT_ENV / "bin/python", "-r", VGGT_X_ROOT / "requirements.txt"])

    run(["uv", "venv", "--python", "3.9", CITY_ENV])
    city_python = CITY_ENV / "bin/python"
    run(["uv", "pip", "install", "--python", city_python, "-r", CITY_ROOT / "requirements/pyt201_cu118.txt"])
    run(["uv", "pip", "install", "--python", city_python, "-r", CITY_ROOT / "requirements.txt"])
    run(["uv", "pip", "install", "--python", city_python, "-r", CITY_ROOT / "requirements/gsplat.txt"])

print("VGGT-X Python:", VGGT_ENV / "bin/python")
print("CityGaussian Python:", CITY_ENV / "bin/python")'''),
    md('''## 2. Stage and verify the eight inputs

Only the annotated training selections are copied. Numeric filenames preserve canonical rotational order.'''),
    code(r'''if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / "images").mkdir(parents=True)
fig, axes = plt.subplots(1, 8, figsize=(24, 3))
for index, row in enumerate(rows.itertuples(index=False)):
    image = Image.open(row.method_image).convert("RGB")
    image.save(STAGE / "images" / f"{index:02d}.png")
    axes[index].imshow(image); axes[index].set_title(f"view {int(row.view_order)}"); axes[index].axis("off")
plt.tight_layout(); plt.show()
print("Staged:", STAGE)'''),
    md('''## 3. Run VGGT-X global alignment

This creates the `_vggt_x` folder containing images, COLMAP cameras, points, and `matches.pt`. The geometry must register all eight images before 3DGS training starts.'''),
    code(r'''if RUN_VGGT_X:
    run([
        VGGT_ENV / "bin/python", VGGT_X_ROOT / "demo_colmap.py",
        "--scene_dir", STAGE,
        "--shared_camera", "--use_ga", "--save_depth",
        "--total_frame_num", "8",
    ], cwd=VGGT_X_ROOT)

sparse_candidates = [VGGT_X_OUTPUT / "sparse/0", VGGT_X_OUTPUT / "sparse"]
SPARSE = next((path for path in sparse_candidates if (path / "cameras.bin").is_file()), None)
if SPARSE is None:
    raise FileNotFoundError(f"VGGT-X did not create a COLMAP model under {VGGT_X_OUTPUT}")
print("COLMAP model:", SPARSE)'''),
    md('''## 4. Inspect geometry before training

This cell reads the official COLMAP result inside the VGGT-X environment and saves a small diagnostic file. The notebook then plots camera centers and a sampled point cloud.'''),
    code(r'''INSPECT_SCRIPT = Path("/content/inspect_vggtx.py")
INSPECT_SCRIPT.write_text(textwrap.dedent(f"""
import numpy as np, pycolmap
r = pycolmap.Reconstruction(r'{SPARSE}')
images = sorted(r.images.values(), key=lambda x: x.name)
centers = []
for image in images:
    value = image.cam_from_world
    pose = value() if callable(value) else value
    centers.append(np.asarray(pose.inverse().translation))
points = np.asarray([point.xyz for point in r.points3D.values()])
colors = np.asarray([point.color for point in r.points3D.values()]) / 255.0
np.savez(r'/content/vggtx_inspection.npz', centers=centers, points=points, colors=colors,
         registered=len(images), cameras=len(r.cameras))
"""), encoding="utf-8")
run([VGGT_ENV / "bin/python", INSPECT_SCRIPT])
inspection = np.load("/content/vggtx_inspection.npz")
print("Registered images:", int(inspection["registered"]), "/ 8")
if int(inspection["registered"]) != 8:
    raise RuntimeError("Stop: VGGT-X did not register every selected input.")

centers, points, point_colors = inspection["centers"], inspection["points"], inspection["colors"]
rng = np.random.default_rng(42)
if len(points) > 50_000:
    chosen = rng.choice(len(points), 50_000, replace=False)
    points, point_colors = points[chosen], point_colors[chosen]
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d"); ax2 = fig.add_subplot(122, projection="3d")
closed = np.vstack([centers, centers[0]])
ax1.plot(*closed.T, "o-"); ax1.set_title("VGGT-X closed camera orbit")
ax2.scatter(*points.T, c=point_colors, s=.2); ax2.set_title(f"VGGT-X COLMAP points: {len(points):,} shown")
for axis in [ax1, ax2]: axis.set_box_aspect(np.ptp((closed if axis is ax1 else points), axis=0).clip(min=1e-6))
plt.tight_layout(); plt.show()'''),
    md('''## 5. Train official CityGaussian MCMC-3DGS

The official pose-optimization configuration jointly refines the imperfect VGGT-X cameras and the Gaussians. `MAX_GAUSSIANS` controls memory use.'''),
    code(r'''RUN_NAME = f"{SCENE}_vggtx_mcmc"
CITY_OUTPUT = CITY_ROOT / "outputs" / RUN_NAME
if RUN_TRAINING:
    run([
        CITY_ENV / "bin/python", CITY_ROOT / "main.py", "fit",
        "--config", CITY_ROOT / "configs/colmap_pose_opt_mcmc.yaml",
        "--data.path", VGGT_X_OUTPUT,
        "--data.parser.init_args.down_sample_factor", str(DOWN_SAMPLE_FACTOR),
        "--data.parser.init_args.down_sample_rounding_mode", "round",
        "--model.density.init_args.cap_max", str(MAX_GAUSSIANS),
        "-n", RUN_NAME,
    ], cwd=CITY_ROOT)
if not (CITY_OUTPUT / "config.yaml").is_file():
    raise FileNotFoundError(f"CityGaussian training output missing: {CITY_OUTPUT}")
print("Training output:", CITY_OUTPUT)'''),
    md('''## 6. Evaluate the real training views

These metrics measure input-view fit. They are useful diagnostics but are not held-out NVS scores.'''),
    code(r'''if RUN_EVALUATION:
    run([
        CITY_ENV / "bin/python", CITY_ROOT / "main.py", "test",
        "--config", CITY_OUTPUT / "config.yaml", "--save_val", "--val_train",
    ], cwd=CITY_ROOT)

# Keep the official files together on Drive. This includes the checkpoint and
# any metric/render files produced by the official test command.
DRIVE_MODEL = FINAL_ROOT / "citygaussian_output"
shutil.copytree(CITY_OUTPUT, DRIVE_MODEL, dirs_exist_ok=True)
metric_candidates = list(CITY_OUTPUT.rglob("*.csv")) + list(CITY_OUTPUT.rglob("*.json"))
print("Metric/result files found:")
for path in metric_candidates: print(" -", path.relative_to(CITY_OUTPUT))
csv_candidates = [path for path in metric_candidates if path.suffix.lower() == ".csv" and
                  any(word in path.name.lower() for word in ["metric", "result", "score"])]
if csv_candidates:
    shutil.copy2(csv_candidates[0], FINAL_ROOT / "metrics.csv")
    print("Standard metric table:", FINAL_ROOT / "metrics.csv")
else:
    print("The official test command did not expose a CSV. Its original logs remain in citygaussian_output.")'''),
    md('''## 7. Render the same closed orbit used for comparison

The path follows the eight VGGT-X COLMAP cameras and inserts ten frames between neighboring views. CityGaussian renders both an MP4 and individual PNG files.'''),
    code(r'''PATH_SCRIPT = Path("/content/make_city_path.py")
PATH_SCRIPT.write_text(textwrap.dedent(f"""
import sys, pycolmap
sys.path.insert(0, r'{CODE_ROOT / "code"}')
from src.citygaussian_bridge import closed_colmap_camera_path
r = pycolmap.Reconstruction(r'{SPARSE}')
print(closed_colmap_camera_path(r, r'{FINAL_ROOT / "camera_path.json"}', frames_between={FRAMES_BETWEEN}))
"""), encoding="utf-8")
run([VGGT_ENV / "bin/python", PATH_SCRIPT])

ORBIT_VIDEO = FINAL_ROOT / "closed_orbit.mp4"
if RUN_ORBIT_RENDER:
    run([
        CITY_ENV / "bin/python", CITY_ROOT / "render.py", CITY_OUTPUT,
        "--camera-path-filename", FINAL_ROOT / "camera_path.json",
        "--output-path", ORBIT_VIDEO, "--save-images", "--disable-transform",
    ], cwd=CITY_ROOT)

generated_frames = Path(str(ORBIT_VIDEO) + "_frames")
ORBIT_FRAMES = FINAL_ROOT / "orbit_frames"
ORBIT_FRAMES.mkdir(parents=True, exist_ok=True)
for index, source in enumerate(sorted(generated_frames.glob("*.png"))):
    shutil.copy2(source, ORBIT_FRAMES / f"view_{index:03d}.png")
print("Saved video:", ORBIT_VIDEO)
print("Saved frames:", len(list(ORBIT_FRAMES.glob("view_*.png"))))
display(Video(str(ORBIT_VIDEO), embed=True, html_attributes="controls autoplay loop muted"))'''),
    md('''## 8. Final output check

Notebook 08 needs `closed_orbit.mp4`, `orbit_frames/`, and the official CityGaussian output. Missing files are reported before you disconnect the runtime.'''),
    code(r'''checks = {
    "COLMAP cameras": SPARSE / "cameras.bin",
    "COLMAP images": SPARSE / "images.bin",
    "COLMAP points": SPARSE / "points3D.bin",
    "CityGaussian config": DRIVE_MODEL / "config.yaml",
    "closed orbit": ORBIT_VIDEO,
    "first comparison frame": ORBIT_FRAMES / "view_000.png",
}
for label, path in checks.items():
    print("OK     " if path.exists() else "MISSING", label, path)
if not all(path.exists() for path in checks.values()):
    raise RuntimeError("Notebook 07 is incomplete. Read the first missing item above.")'''),
])


save("08_compare_nvs_and_3dgs.ipynb", [
    md('''# 08 - Compare learned NVS and 3DGS methods

This notebook does not train anything. It loads finished artifacts and shows them together:

1. **LagerNVS** - learned Plucker-ray RGB prediction;
2. **VGGT-X MCMC-3DGS** - official optimized Gaussian baseline;
3. **Ours** - confidence-weighted adaptive 3DGS supervised by real views and trusted LagerNVS pixels.

Only methods with real output files are displayed. Missing experiments are reported clearly.'''),
    code(r'''%pip -q install pandas pillow matplotlib imageio imageio-ffmpeg
''' + common_setup + r'''
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from IPython.display import Video, display
DATASET="3DRealCar"; SCENE=None
manifest=pd.read_csv(PROJECT_ROOT/"data_processed/method_inputs/manifest.csv")
available=manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET")
SCENE=SCENE or sorted(available.scene.unique())[0]
paths={
 "LagerNVS": PROJECT_ROOT/"experiments/SyntheticViews/LagerNVS"/DATASET/SCENE/"images",
 "VGGT-X MCMC-3DGS": PROJECT_ROOT/"experiments/3DGS/VGGT_X_MCMC"/DATASET/SCENE/"orbit_frames",
 "Ours: confidence-weighted 3DGS": PROJECT_ROOT/"experiments/3DGS/LagerNVS_confidence_weighted"/DATASET/SCENE/"orbit_frames",
}
for name,path in paths.items(): print(("FOUND" if path.exists() else "MISSING"),name,path)'''),
    md('''## Quantitative interpretation

Training-view PSNR, SSIM and silhouette IoU measure how well a method fits its inputs. They do not prove novel-view quality. The main ranking should use held-out real images with known cameras. Runtime, peak GPU memory and Gaussian count should be reported beside image metrics.'''),
    code(r'''metric_files={
 "VGGT-X MCMC-3DGS": PROJECT_ROOT/"experiments/3DGS/VGGT_X_MCMC"/DATASET/SCENE/"metrics.csv",
 "Ours": PROJECT_ROOT/"experiments/3DGS/LagerNVS_confidence_weighted"/DATASET/SCENE/"metrics.csv",
}
tables=[]
for method,path in metric_files.items():
    if path.is_file():
        table=pd.read_csv(path); table.insert(0,"method",method); tables.append(table)
if tables: display(pd.concat(tables,ignore_index=True))
else: print("No comparable metric files yet. Finish notebooks 06 and 07 first.")'''),
    md('''## Side-by-side orbit

For a strict visual comparison, export the same number of frames from the same canonical front-to-front orbit. Do not compare unrelated camera paths as if they were pixel-aligned.'''),
    code(r'''from PIL import Image, ImageDraw
frame_sets={name: sorted(path.glob("view_*.png")) for name,path in paths.items() if path.is_dir()}
if len(frame_sets) != 3:
    print("Finish all three methods first. Available:", {name:len(files) for name,files in frame_sets.items()})
else:
    count=min(len(files) for files in frame_sets.values())
    output=PROJECT_ROOT/"experiments/Comparisons"/DATASET/SCENE; output.mkdir(parents=True,exist_ok=True)
    video=output/"three_method_closed_orbit.mp4"
    writer=imageio.get_writer(video,fps=15,codec="libx264",quality=8,macro_block_size=None)
    try:
        for index in range(count):
            panels=[(name,Image.open(files[index]).convert("RGB")) for name,files in frame_sets.items()]
            h=max(panel.height for _,panel in panels); w=sum(panel.width for _,panel in panels)+8*(len(panels)-1)
            canvas=Image.new("RGB",(w,h+32),"white"); draw=ImageDraw.Draw(canvas); x=0
            for name,panel in panels:
                canvas.paste(panel,(x,32)); draw.text((x+8,8),name,fill="black"); x+=panel.width+8
            writer.append_data(np.asarray(canvas))
    finally: writer.close()
    print("Saved:",video); display(Video(str(video),embed=True,width=1200,html_attributes="controls autoplay loop muted"))'''),
])

print("Built notebooks 06, 07, and 08")
