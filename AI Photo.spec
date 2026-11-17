# PyInstaller macOS application bundle, built for Apple Silicon.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

root = SPECPATH
datas = [
    (root + "/ai_photo_app", "ai_photo_app"),
    (root + "/resources/AI Foto logo.jpg", "resources"),
]
weight_file = root + "/resources/nonescape-mini-v0.safetensors"
if os.path.isfile(weight_file):
    datas.append((weight_file, "resources"))
binaries = []
hiddenimports = [
    "ai_photo_app.aidetector.model",
    "ai_photo_app.aidetector.nonescape_adapter",
    "ai_photo_app.aidetector.sentry_adapter",
    "transformers.models.auto",
]
for package in (
    "torch",
    "torchvision",
    "open_clip",
    "timm",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

analysis = Analysis(
    [root + "/main.py"],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pytest", "IPython", "jupyter", "gradio", "fastapi"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="AI Foto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    analysis.binaries,
    analysis.datas,
    name="AI Foto.app",
    bundle_identifier="tr.aifoto.desktop",
    info_plist={
        "CFBundleDisplayName": "AI Foto",
        "CFBundleName": "AI Foto",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
    },
    target_arch="arm64",
    icon=root + "/resources/ai-foto.icns",
)
