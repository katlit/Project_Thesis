"""Build the learned-NVS, VGGT-X, and cross-method comparison notebooks."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
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
    code(r'''# Use LagerNVS's official dependency file. It includes xformers, which
# is required while importing the renderer attention blocks.
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
%pip -q install -r /content/lagernvs/requirements.txt "gsplat==1.3.0" matplotlib imageio imageio-ffmpeg

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
    md('''# 07 - Official VGGT-X and MCMC-3DGS baseline

This notebook is an isolated baseline. It does not reuse our custom Gaussian trainer. It runs the released VGGT-X global-alignment export, then the CityGaussian MCMC-3DGS configuration recommended by the authors.

VGGT-X was designed for dense image collections. Using the same eight images is intentional here: it makes the comparison fair, but it is a sparse-input stress test.'''),
    md('''## Important environment note

VGGT-X pins Python 3.10-era Torch 2.3.1 and PyCOLMAP 3.10.0. Do not install it into the notebook 05/06 runtime. Use a fresh runtime and follow the official environment installation below. If Colab cannot build the CUDA extensions, run this notebook on a Linux CUDA machine.'''),
    code(common_setup + r'''
import pandas as pd
from PIL import Image
DATASET="3DRealCar"; SCENE=None
manifest=pd.read_csv(PROJECT_ROOT/"data_processed/method_inputs/manifest.csv")
rows=manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET")
SCENE=SCENE or sorted(rows.scene.unique())[0]; rows=rows[rows.scene.eq(SCENE)].sort_values("view_order")
STAGE=Path("/content/vggtx_input")/SCENE; (STAGE/"images").mkdir(parents=True,exist_ok=True)
for i,row in enumerate(rows.itertuples()):
    Image.open(row.method_image).convert("RGB").save(STAGE/"images"/f"{i:02d}.png")
print("Staged",len(rows),"images at",STAGE)'''),
    code(r'''# Run these commands in the dedicated Python 3.10 environment described by VGGT-X.
!test -d /content/VGGT-X/.git || git clone --recursive https://github.com/Linketic/VGGT-X.git /content/VGGT-X
print("Official geometry command:")
print(f"python /content/VGGT-X/demo_colmap.py --scene_dir {STAGE} --shared_camera --use_ga --total_frame_num 8")'''),
    md('''## Run the official commands

The first command must create a valid COLMAP model and `matches.pt`. Inspect the camera orbit before training. If global alignment destroys the orbit, report the failure rather than silently using it.

After that, follow the official CityGaussian `configs/colmap_pose_opt_mcmc.yaml` command. Save its rendered orbit and metrics under:

`experiments/3DGS/VGGT_X_MCMC/3DRealCar/<scene>/`

This notebook deliberately does not imitate VGGT-X with our trainer. A valid baseline must come from the released VGGT-X and CityGaussian implementations.'''),
    code(r'''OUTPUT_ROOT=PROJECT_ROOT/"experiments/3DGS/VGGT_X_MCMC"/DATASET/SCENE
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
print("Expected final video:",OUTPUT_ROOT/"closed_orbit.mp4")
print("Expected synchronized frames:",OUTPUT_ROOT/"orbit_frames/view_000.png ...")
print("Expected metrics:",OUTPUT_ROOT/"metrics.csv")'''),
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
