from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

APP_TITLE = "AI Foto"

LANGUAGE = "tr"
ACTIVE_WINDOW = None
TRANSLATIONS = {
    "Dosya": "File", "Görsel Ekle…": "Add Images…", "Word Raporunu Dışa Aktar…": "Export Word Report…",
    "Çıkış": "Quit", "Dil / Language": "Language / Dil", "AI Görsel Analizi": "AI Photo Analysis",
    "Üç algılayıcı ve bunlardan türetilen iki birleşik puan": "Three detectors and two derived ensemble scores",
    "Görsel Ekle": "Add Images", "Analizi Başlat": "Analyze Images",
    "Görsel ekleyin veya buraya sürükleyip bırakın": "Add images or drag and drop them here",
    "GÖRSELLER": "IMAGES", "İçe aktarılan görseller": "Imported images",
    "Görselleri buraya sürükleyin\nveya ‘Görsel Ekle’ düğmesine tıklayın": "Drop images here\nor click ‘Add Images’",
    "GÖRSEL ÖNİZLEME": "IMAGE PREVIEW", "Bir görsel seçin": "Select an image",
    "HF bölge duyarlılığı": "HF Region Sensitivity",
    "HF AI puanının, görüntünün 8×8 bölgeleri nötr griyle kapatıldığında nasıl değiştiğini hesaplar.": "Measures how the HF AI score changes when each of 64 image regions is covered with neutral gray.",
    "ALGILAYICI PUANLARI": "DETECTOR SCORES", "NORMALİZE AI PUANI · İNDEKS": "NORMALIZED AI SCORE · INDEX",
    "Ham ağırlıklı puan: —": "Raw weighted score: —", "Model": "Model", "AI puanı": "AI score",
    "Güvenilirlik ağırlığı": "Reliability weight", "AĞIRLIK KAYNAĞI": "WEIGHT SOURCE",
    "İlk üç modelin dengeli doğruluğu": "Balanced accuracy of the first three models",
    "Normalize ağırlık": "Normalized weight", "Word Raporunu Dışa Aktar": "Export Word Report",
    "Görsel seç": "Choose images", "Görsel dosyaları (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)": "Image files (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
    "Ham ağırlıklı puan: sonuçlar bekleniyor": "Raw weighted score: waiting for results",
    "Modeller yükleniyor…": "Loading models…", "Üç temel model sırayla çalışıyor…": "Running three base models sequentially…",
    "Harita hesaplanıyor…": "Calculating map…", "HF algılayıcısı ile 8×8 bölge duyarlılığı hesaplanıyor…": "Calculating 8×8 region sensitivity with the HF detector…",
    "Duyarlılık haritası oluşturulamadı": "Could not create sensitivity map", "Analiz tamamlandı": "Analysis complete",
    "Sonuç yok": "No results", "Word raporu oluşturmak için önce görselleri analiz edin.": "Analyze images before exporting a Word report.",
    "Word raporunu kaydet": "Save Word report", "Word belgesi (*.docx)": "Word document (*.docx)",
    "Dışa aktarma tamamlandı": "Export complete", "Word raporu kaydedildi:\n": "Word report saved:\n",
    "Dışa aktarma başarısız": "Export failed", "Görseller buraya sürükleyin": "Drop images here",
    "Görsel açma": "Opening image", "HF modelini yükleme": "Loading HF model",
    "Harita hesaplama": "Calculating map", "PNG kaydetme": "Saving PNG", "PNG doğrulama": "Validating PNG",
    "Hata": "Error", "HATA": "ERROR", "Aşama": "Stage",
    "Aşama:": "Stage:", "Görsel seçildi": "images selected",
 "Turuncu: kapatılınca AI puanı düştü": "Orange: covering this region lowered the AI score",
    "Mavi: kapatılınca AI puanı yükseldi": "Blue: covering this region raised the AI score",
    "Algılayıcı": "Detector", "Ağırlıklı AI puanı": "Raw weighted AI score",
    "Normalize AI puanı · indeks": "Normalized AI score · index",
    "AI puanları model çıktılarıdır. Kalibre edilmiş olasılık veya tek başına doğrulama kanıtı olarak yorumlanmamalıdır.": "AI scores are model outputs. They are not calibrated probabilities or standalone proof of authenticity.",
    "HF bölge duyarlılığı · turuncu: kapatılınca AI puanı düştü · mavi: kapatılınca AI puanı yükseldi": "HF region sensitivity · orange: covering lowered score · blue: covering raised score",
}
def tr(text: str) -> str:
    return TRANSLATIONS.get(text, text) if LANGUAGE == "en" else text

MODEL_ORDER = ["univfd", "hf", "nonescape-mini", "hybrid", "hybrid-plus"]
MODEL_LABELS = {
    "univfd": "UnivFD",
    "hf": "Hugging Face",
    "nonescape-mini": "Nonescape Mini",
    "hybrid": "Hybrid",
    "hybrid-plus": "Hybrid Plus",
}
RELIABILITY = {"univfd": 0.690, "hf": 0.580, "nonescape-mini": 0.772}
BASE_BACKENDS = ("univfd", "hf", "nonescape-mini")
HYBRID_UNIVFD_WEIGHT = 0.25
HYBRID_PLUS_PRIMARY_WEIGHT = 0.85

# Fixed monotonic display mapping anchored to the observed ends of the existing
# six-image smoke sample. These anchors change only the displayed index, never
# any detector output, raw weighted score, or threshold decision.
NORMALIZED_SCORE_ANCHORS = (
    (0.0, 0.0),
    (0.3670895411108459, 0.0670895411108459),
    (0.44861145599231667, 0.14861145599231667),
    (0.5, 0.5),
    (0.6735056723366756, 0.8735056723366756),
    (0.7497484895905132, 0.9497484895905132),
    (1.0, 1.0),
)


def calculate_weighted_ai_score(scores: dict[str, float]) -> float | None:
    if not all(key in scores for key in RELIABILITY):
        return None
    return sum(scores[key] * RELIABILITY[key] for key in RELIABILITY) / sum(RELIABILITY.values())


def normalize_score_index(score: float) -> float:
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    score = min(1.0, max(0.0, score))
    for (raw_left, index_left), (raw_right, index_right) in zip(
        NORMALIZED_SCORE_ANCHORS[:-1], NORMALIZED_SCORE_ANCHORS[1:], strict=True
    ):
        if score <= raw_right:
            fraction = (score - raw_left) / (raw_right - raw_left)
            return index_left + fraction * (index_right - index_left)
    return 1.0


def build_sensitivity_overlay(
    image: Any, detector: Any, path: Path
) -> tuple[Any, list[float], float]:
    """Return an 8×8 signed HF occlusion map and the raw score deltas."""
    from PIL import Image, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    baseline = detector.predict_image(image, path=path).probability_ai
    columns = rows = 8
    width, height = image.size
    probabilities = []
    # Keep only one inference batch of full-resolution images in memory.
    # Holding all 64 masks at once needs several GB for modern phone photos
    # and can corrupt image decoding under memory pressure.
    batch_size = 4
    for start in range(0, rows * columns, batch_size):
        occluded_batch = []
        for index in range(start, min(start + batch_size, rows * columns)):
            row, col = divmod(index, columns)
            altered = Image.new("RGB", image.size, (127, 127, 127))
            left, top = col * width // columns, row * height // rows
            right, bottom = (col + 1) * width // columns, (row + 1) * height // rows
            altered.paste(image.crop((left, top, right, bottom)), (left, top))
            occluded_batch.append(altered)
        probabilities.extend(
            result.probability_ai
            for result in detector.predict_images(occluded_batch)
        )
        occluded_batch.clear()

    changes = [baseline - value for value in probabilities]
    grid = Image.new("RGBA", (columns, rows), (0, 0, 0, 0))
    for index, change in enumerate(changes):
        intensity = min(1.0, abs(change) / 0.25)
        alpha = round(170 * intensity)
        color = (232, 92, 46, alpha) if change > 0 else (37, 119, 204, alpha)
        grid.putpixel((index % columns, index // columns), color)
    blur_radius = max(2, min(image.size) / 48)
    colors = grid.resize(image.size, Image.Resampling.BILINEAR).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    overlay = Image.alpha_composite(image.convert("RGBA"), colors)
    return overlay, changes, baseline

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "AI Photo"
MODEL_CACHE = APP_SUPPORT / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
# Reuse the standard per-user caches. The previous app-local cache forced this
# Mac to download several GB of model weights again after the CLI had fetched them.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))

if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    sys.path.insert(0, str(BUNDLE_DIR))
else:
    BUNDLE_DIR = Path(__file__).resolve().parent

from PySide6.QtCore import Qt, QThread, Signal, QSize, QCoreApplication, QSettings
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QDialog,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


class AnalysisWorker(QThread):
    progress = Signal(str, int, int)
    scores_ready = Signal(dict)
    backend_failed = Signal(str, str)
    completed = Signal()

    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths

    def run(self) -> None:
        try:
            from PIL import Image
            from ai_photo_app.aidetector.model import create_detector
            # Import the shared PyTorch/vision stack before entering workers.
            # Lazy import locks in torchvision can deadlock if several workers
            # import model modules at the same time.
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            import transformers  # noqa: F401
            import open_clip  # noqa: F401
        except Exception as exc:
            self.backend_failed.emit("AI Foto", str(exc))
            self.completed.emit()
            return

        collected: dict[str, dict[str, float]] = {p: {} for p in self.paths}
        progress_lock = threading.Lock()
        progress_done = 0
        progress_total = max(1, len(self.paths) * len(BASE_BACKENDS))
        self.progress.emit(tr("Üç temel model sırayla çalışıyor…"), 0, progress_total)

        def run_backend(backend: str):
            nonlocal progress_done
            if backend == "nonescape-mini":
                bundled = BUNDLE_DIR / "resources" / "nonescape-mini-v0.safetensors"
                cached = MODEL_CACHE / "nonescape" / bundled.name
                if bundled.is_file() and not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bundled, cached)
            detector = create_detector(backend, device="mps")
            values = {}
            try:
                for image_path in self.paths:
                    with Image.open(image_path) as source:
                        result = detector.predict_image(source.convert("RGB"), path=Path(image_path))
                    values[image_path] = float(result.probability_ai)
                    with progress_lock:
                        progress_done += 1
                        current = progress_done
                    self.progress.emit(f"{MODEL_LABELS[backend]} · {Path(image_path).name}", current, progress_total)
                return values
            finally:
                del detector
                gc.collect()

        # Limit CPU thread pools and run one backend at a time to reduce peak
        # contention and memory pressure on fanless Apple Silicon laptops.
        import torch
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # PyTorch may have initialized its inter-op pool already.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector") as pool:
            jobs = {pool.submit(run_backend, backend): backend for backend in BASE_BACKENDS}
            for future in as_completed(jobs):
                backend = jobs[future]
                try:
                    for image_path, value in future.result().items():
                        collected[image_path][backend] = value
                except Exception as exc:
                    self.backend_failed.emit(backend, str(exc))
        # These two displayed scores are defined as weighted combinations of the
        # three base scores. Reusing those exact outputs avoids loading the same
        # UnivFD, HF and Nonescape models again inside composite detectors.
        for image_path, values in collected.items():
            if all(key in values for key in BASE_BACKENDS):
                hybrid = HYBRID_UNIVFD_WEIGHT * values["univfd"] + (1 - HYBRID_UNIVFD_WEIGHT) * values["hf"]
                values["hybrid"] = hybrid
                values["hybrid-plus"] = HYBRID_PLUS_PRIMARY_WEIGHT * hybrid + (1 - HYBRID_PLUS_PRIMARY_WEIGHT) * values["nonescape-mini"]
        self.scores_ready.emit(collected)
        self.completed.emit()


class SensitivityWorker(QThread):
    """On-demand 8×8 occlusion sensitivity for one image using the HF backend."""
    ready = Signal(str, str)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def run(self) -> None:
        stage = "görsel açma"
        try:
            from PIL import Image
            import torch
            from ai_photo_app.aidetector.model import create_detector

            torch.set_num_threads(2)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

            path = Path(self.image_path)
            from PIL import ImageOps
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB").copy()
            stage = "HF modelini yükleme"
            detector = create_detector("hf", device="mps")
            try:
                stage = "duyarlılık haritasını hesaplama"
                overlay, _, _ = build_sensitivity_overlay(image, detector, path)
            finally:
                del detector
                gc.collect()

            stage = "haritayı kaydetme"
            out_dir = APP_SUPPORT / "sensitivity_maps"
            out_dir.mkdir(parents=True, exist_ok=True)
            path_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
            safe_name = f"{path.stem}-{path_id}-sensitivity.png"
            out_path = out_dir / safe_name
            temporary_path = out_path.with_name(f".{out_path.stem}-{os.getpid()}.tmp.png")
            rgb = overlay.convert("RGB")
            rgb.save(temporary_path, format="PNG", optimize=False)
            # Read the encoded file fully before exposing it to the GUI/export path.
            with Image.open(temporary_path) as check:
                check.load()
                if check.size != rgb.size:
                    raise OSError(f"harita boyutu uyuşmuyor: {check.size} != {rgb.size}")
            os.replace(temporary_path, out_path)
            self.ready.emit(str(path), str(out_path))
        except Exception as exc:
            self.failed.emit(f"{tr('Aşama:')} {tr(stage)}. {type(exc).__name__}: {exc}")
        finally:
            self.completed.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1240, 820)
        self.setMinimumSize(1080, 720)
        self.setAcceptDrops(True)
        self.paths: list[str] = []
        self.scores: dict[str, dict[str, float]] = {}
        self.sensitivity_maps: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self._worker: AnalysisWorker | None = None
        self._sensitivity_worker: SensitivityWorker | None = None
        self._make_menu()
        self._make_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(tr("Görsel ekleyin veya buraya sürükleyip bırakın"))

    def _make_menu(self):
        menu = self.menuBar().addMenu(tr("Dosya"))
        add = QAction(tr("Görsel Ekle…"), self)
        add.setShortcut(QKeySequence.Open)
        add.triggered.connect(self.choose_images)
        menu.addAction(add)
        export = QAction(tr("Word Raporunu Dışa Aktar…"), self)
        export.setShortcut(QKeySequence.SaveAs)
        export.triggered.connect(self.export_docx)
        menu.addAction(export)
        menu.addSeparator()
        quit_action = QAction(tr("Çıkış"), self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)
        language_menu = self.menuBar().addMenu(tr("Dil / Language"))
        for code, title_text in (("tr", "Türkçe"), ("en", "English")):
            action = QAction(title_text, self)
            action.setCheckable(True)
            action.setChecked(LANGUAGE == code)
            action.triggered.connect(lambda checked=False, selected=code: self.switch_language(selected))
            language_menu.addAction(action)

    def switch_language(self, language: str) -> None:
        global LANGUAGE, ACTIVE_WINDOW
        if language == LANGUAGE or self._worker is not None and self._worker.isRunning():
            return
        if self._sensitivity_worker is not None and self._sensitivity_worker.isRunning():
            return
        paths = self.paths.copy()
        scores = {key: value.copy() for key, value in self.scores.items()}
        maps = self.sensitivity_maps.copy()
        failures = self.failures.copy()
        row = self.file_list.currentRow()
        LANGUAGE = language
        QSettings("AIFoto", "AI Foto").setValue("ui_language", language)
        replacement = MainWindow()
        ACTIVE_WINDOW = replacement
        if paths:
            replacement.add_paths(paths)
            replacement.scores = scores
            replacement.sensitivity_maps = maps
            replacement.failures = failures
            replacement.export_button.setEnabled(bool(scores))
            replacement.file_list.setCurrentRow(max(0, min(row, len(paths) - 1)))
            replacement.show_selected(replacement.file_list.currentRow())
        replacement.show()
        self.close()

    def _make_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(16)

        top = QHBoxLayout()
        logo = QLabel()
        logo_path = BUNDLE_DIR / "resources" / "AI Foto logo.jpg"
        if logo_path.is_file():
            logo.setPixmap(QPixmap(str(logo_path)).scaled(52, 52, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        top.addWidget(logo)
        brand_box = QVBoxLayout()
        title = QLabel(tr("AI Görsel Analizi"))
        title.setObjectName("title")
        subtitle = QLabel(tr("Üç algılayıcı ve bunlardan türetilen iki birleşik puan"))
        subtitle.setObjectName("muted")
        brand_box.addWidget(title)
        brand_box.addWidget(subtitle)
        top.addLayout(brand_box, 1)
        self.add_button = QPushButton(tr("Görsel Ekle"))
        self.add_button.setObjectName("secondaryButton")
        self.add_button.clicked.connect(self.choose_images)
        top.addWidget(self.add_button)
        self.analyze_button = QPushButton(tr("Analizi Başlat"))
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self.start_analysis)
        top.addWidget(self.analyze_button)
        outer.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_image_list())
        splitter.addWidget(self._build_preview())
        splitter.addWidget(self._build_results())
        splitter.setSizes([240, 460, 440])
        outer.addWidget(splitter, 1)

        progress_row = QHBoxLayout()
        self.progress_label = QLabel(tr("Görsel ekleyin veya buraya sürükleyip bırakın"))
        self.progress_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(260)
        progress_row.addWidget(self.progress_label, 1)
        progress_row.addWidget(self.progress)
        outer.addLayout(progress_row)
        self.setCentralWidget(root)

    def _build_image_list(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        heading = QLabel(tr("GÖRSELLER"))
        heading.setObjectName("sectionLabel")
        self.file_list = QListWidget()
        self.file_list.setAccessibleName(tr("İçe aktarılan görseller"))
        self.file_list.currentRowChanged.connect(self.show_selected)
        self.empty_label = QLabel(tr("Görselleri buraya sürükleyin\nveya ‘Görsel Ekle’ düğmesine tıklayın"))
        self.empty_label.setObjectName("emptyHint")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(heading)
        layout.addWidget(self.file_list, 1)
        layout.addWidget(self.empty_label)
        return panel

    def _build_preview(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)
        header = QHBoxLayout()
        label = QLabel(tr("GÖRSEL ÖNİZLEME"))
        label.setObjectName("sectionLabel")
        self.dimensions = QLabel("")
        self.dimensions.setObjectName("muted")
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(self.dimensions)
        self.preview = QLabel(tr("Bir görsel seçin"))
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 380)
        self.preview.setScaledContents(False)
        self.filename = QLabel("")
        self.filename.setObjectName("fileName")
        self.sensitivity_button = QPushButton(tr("HF bölge duyarlılığı"))
        self.sensitivity_button.setObjectName("secondaryButton")
        self.sensitivity_button.setToolTip(
            tr("HF AI puanının, görüntünün 8×8 bölgeleri nötr griyle kapatıldığında nasıl değiştiğini hesaplar.")
        )
        self.sensitivity_button.setEnabled(False)
        self.sensitivity_button.clicked.connect(self.make_sensitivity_map)
        layout.addLayout(header)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.filename)
        layout.addWidget(self.sensitivity_button)
        return panel

    def _build_results(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)
        heading = QLabel(tr("ALGILAYICI PUANLARI"))
        heading.setObjectName("sectionLabel")
        self.score_card = QFrame()
        self.score_card.setObjectName("scoreCard")
        card_layout = QVBoxLayout(self.score_card)
        card_layout.setContentsMargins(16, 12, 16, 12)
        card_label = QLabel(tr("NORMALİZE AI PUANI · İNDEKS"))
        card_label.setObjectName("cardLabel")
        self.weighted_score = QLabel("—")
        self.weighted_score.setObjectName("scoreValue")
        self.weighted_state = QLabel(tr("Ham ağırlıklı puan: —"))
        self.weighted_state.setObjectName("cardSubtext")
        card_layout.addWidget(card_label)
        card_layout.addWidget(self.weighted_score)
        card_layout.addWidget(self.weighted_state)

        self.score_table = QTableWidget(5, 3)
        self.score_table.setHorizontalHeaderLabels(["Model", tr("AI puanı"), tr("Güvenilirlik ağırlığı")])
        self.score_table.verticalHeader().setVisible(False)
        self.score_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.score_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.score_table.setAlternatingRowColors(True)
        self.score_table.setMinimumHeight(280)
        self.score_table.horizontalHeader().setStretchLastSection(True)
        self.score_table.setColumnWidth(0, 130)
        self.score_table.setColumnWidth(1, 80)
        self.score_table.setColumnWidth(2, 160)
        for row, backend in enumerate(MODEL_ORDER):
            self._set_table_cell(row, 0, MODEL_LABELS[backend], left=True)
            self._set_table_cell(row, 1, "—")
            reliability = RELIABILITY.get(backend)
            self._set_table_cell(row, 2, f"{reliability:.3f}" if reliability else "—")

        weight_title = QLabel(tr("AĞIRLIK KAYNAĞI"))
        weight_title.setObjectName("sectionLabel")
        weight_hint = QLabel(tr("İlk üç modelin dengeli doğruluğu"))
        weight_hint.setObjectName("muted")
        self.weight_table = QTableWidget(3, 2)
        self.weight_table.setHorizontalHeaderLabels(["Model", tr("Normalize ağırlık")])
        self.weight_table.verticalHeader().setVisible(False)
        self.weight_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.weight_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.weight_table.setFixedHeight(144)
        self.weight_table.horizontalHeader().setStretchLastSection(True)
        for row, backend in enumerate(list(RELIABILITY)):
            self._set_table_cell(row, 0, MODEL_LABELS[backend], target=self.weight_table, left=True)
            normalized = RELIABILITY[backend] / sum(RELIABILITY.values())
            self._set_table_cell(row, 1, f"{normalized*100:.1f}%", target=self.weight_table)
        self.export_button = QPushButton(tr("Word Raporunu Dışa Aktar"))
        self.export_button.setObjectName("secondaryButton")
        self.export_button.clicked.connect(self.export_docx)
        self.export_button.setEnabled(False)
        layout.addWidget(heading)
        layout.addWidget(self.score_card)
        layout.addWidget(self.score_table, 1)
        layout.addWidget(weight_title)
        layout.addWidget(weight_hint)
        layout.addWidget(self.weight_table)
        layout.addWidget(self.export_button)
        return panel

    def _set_table_cell(self, row: int, col: int, text: str, target=None, left=False):
        table = target or self.score_table
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignCenter))
        table.setItem(row, col, item)

    def choose_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("Görsel seç"), str(Path.home()),
            tr("Görsel dosyaları (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"),
        )
        self.add_paths(files)

    def add_paths(self, paths):
        supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        added = False
        for path in paths:
            p = Path(path).expanduser()
            if p.is_file() and p.suffix.lower() in supported and str(p) not in self.paths:
                self.paths.append(str(p))
                self.file_list.addItem(QListWidgetItem(p.name))
                added = True
        self.empty_label.setVisible(not self.paths)
        if added:
            self.file_list.setCurrentRow(len(self.paths) - 1)
            self.export_button.setEnabled(bool(self.scores))
            self.progress_label.setText(f"{len(self.paths)} {tr('Görsel seçildi')}")

    def show_selected(self, row):
        if row < 0 or row >= len(self.paths):
            return
        path = Path(self.paths[row])
        pixmap = QPixmap(str(path))
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.filename.setText(path.name)
        self.dimensions.setText(f"{pixmap.width()} × {pixmap.height()} px")
        self._show_scores(str(path))
        self.sensitivity_button.setEnabled(bool(self.scores.get(str(path))) and self._sensitivity_worker is None)

    def _show_scores(self, path):
        data = self.scores.get(path, {})
        for row, backend in enumerate(MODEL_ORDER):
            value = data.get(backend)
            self._set_table_cell(row, 1, f"{value*100:.1f}%" if value is not None else ("Hata" if backend in self.failures else "—"))
        if all(backend in data for backend in RELIABILITY):
            score = calculate_weighted_ai_score(data)
            self.weighted_score.setText(f"{normalize_score_index(score)*100:.1f}%")
            self.weighted_state.setText(f"{tr('Ham ağırlıklı puan: —').split(':')[0]}: {score*100:.1f}%")
        else:
            self.weighted_score.setText("—")
            self.weighted_state.setText(tr("Ham ağırlıklı puan: sonuçlar bekleniyor"))

    def start_analysis(self):
        if not self.paths:
            self.choose_images()
        if not self.paths:
            return
        self.analyze_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.progress.setValue(0)
        self.failures.clear()
        self.progress_label.setText(tr("Modeller yükleniyor…"))
        self._worker = AnalysisWorker(self.paths.copy())
        self._worker.progress.connect(self._on_progress)
        self._worker.scores_ready.connect(self._on_scores)
        self._worker.backend_failed.connect(self._on_failure)
        self._worker.completed.connect(self._on_complete)
        self._worker.start()

    def _on_progress(self, message, current, total):
        self.progress_label.setText(message)
        self.progress.setValue(int(float(current) / max(1, total) * 100))

    def _on_scores(self, scores):
        self.scores = scores
        self.export_button.setEnabled(True)
        self.show_selected(self.file_list.currentRow())

    def make_sensitivity_map(self):
        row = self.file_list.currentRow()
        if not (0 <= row < len(self.paths)):
            return
        path = self.paths[row]
        existing = self.sensitivity_maps.get(path)
        if existing and Path(existing).is_file():
            self._show_sensitivity_dialog(path, existing)
            return
        self.sensitivity_button.setEnabled(False)
        self.sensitivity_button.setText(tr("Harita hesaplanıyor…"))
        self.analyze_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.progress_label.setText(tr("HF algılayıcısı ile 8×8 bölge duyarlılığı hesaplanıyor…"))
        self._sensitivity_worker = SensitivityWorker(path)
        self._sensitivity_worker.ready.connect(self._on_sensitivity_ready)
        self._sensitivity_worker.failed.connect(self._on_sensitivity_failed)
        self._sensitivity_worker.completed.connect(self._on_sensitivity_complete)
        self._sensitivity_worker.start()

    def _on_sensitivity_ready(self, source_path, map_path):
        self.sensitivity_maps[source_path] = map_path
        self.export_button.setEnabled(True)
        self._show_sensitivity_dialog(source_path, map_path)

    def _on_sensitivity_failed(self, error):
        QMessageBox.warning(self, tr("Duyarlılık haritası oluşturulamadı"), error[:1200])

    def _on_sensitivity_complete(self):
        self._sensitivity_worker = None
        self.analyze_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.sensitivity_button.setText(tr("HF bölge duyarlılığı"))
        row = self.file_list.currentRow()
        selected = self.paths[row] if 0 <= row < len(self.paths) else None
        self.sensitivity_button.setEnabled(bool(selected and self.scores.get(selected)))
        self.progress_label.setText(tr("Analiz tamamlandı"))

    def _show_sensitivity_dialog(self, source_path, map_path):
        from PIL import Image
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("HF bölge duyarlılığı"))
        dialog.resize(780, 700)
        layout = QVBoxLayout(dialog)
        legend = QHBoxLayout()
        orange_label = QLabel(tr("Turuncu: kapatılınca AI puanı düştü"))
        orange_label.setStyleSheet("color: #B9431D; font-weight: 600;")
        blue_label = QLabel(tr("Mavi: kapatılınca AI puanı yükseldi"))
        blue_label.setStyleSheet("color: #246FB7; font-weight: 600;")
        legend.addWidget(orange_label)
        legend.addStretch(1)
        legend.addWidget(blue_label)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        with Image.open(map_path) as image:
            image.load()
            rgb = image.convert("RGB")
            qimage = QImage(
                rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3,
                QImage.Format.Format_RGB888,
            ).copy()
            pixmap = QPixmap.fromImage(qimage)
            image_label.setPixmap(pixmap.scaled(740, 560, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addLayout(legend)
        layout.addWidget(image_label, 1)
        dialog.exec()

    def _on_failure(self, backend, error):
        self.failures[backend] = error
        self.statusBar().showMessage(f"{MODEL_LABELS.get(backend, backend)}: {error[:180]}", 15000)

    def _on_complete(self):
        self.analyze_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.progress.setValue(100)
        self.progress_label.setText(tr("Analiz tamamlandı"))
        self.statusBar().showMessage(tr("Analiz tamamlandı"), 8000)

    def export_docx(self):
        if not self.scores:
            QMessageBox.information(self, tr("Sonuç yok"), tr("Word raporu oluşturmak için önce görselleri analiz edin."))
            return
        target, _ = QFileDialog.getSaveFileName(self, tr("Word raporunu kaydet"), str(Path.home() / "AI_photo_report.docx"), tr("Word belgesi (*.docx)"))
        if not target:
            return
        if not target.lower().endswith(".docx"):
            target += ".docx"
        try:
            self._write_docx(Path(target))
            QMessageBox.information(self, tr("Dışa aktarma tamamlandı"), f"Word raporu kaydedildi:\n{target}")
        except Exception as exc:
            QMessageBox.critical(self, tr("Dışa aktarma başarısız"), str(exc))

    def _write_docx(self, target: Path):
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt, RGBColor
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        doc = Document()
        section = doc.sections[0]
        section.top_margin = Inches(.55)
        section.bottom_margin = Inches(.55)
        section.left_margin = Inches(.65)
        section.right_margin = Inches(.65)
        style = doc.styles["Normal"]
        style.font.name = "Arial"
        style.font.size = Pt(10)
        style.font.color.rgb = RGBColor(27, 41, 36)
        for index, path_str in enumerate(self.paths):
            if path_str not in self.scores:
                continue
            if index:
                doc.add_page_break()
            if index == 0:
                notice = doc.add_paragraph(
                    tr("AI puanları model çıktılarıdır. Kalibre edilmiş olasılık veya tek başına doğrulama kanıtı olarak yorumlanmamalıdır.")
                )
                notice.paragraph_format.space_after = Pt(6)
                for run in notice.runs:
                    run.italic = True
                    run.font.size = Pt(8)
                    run.font.color.rgb = RGBColor(91, 107, 99)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(6)
            from PIL import Image
            with Image.open(path_str) as image:
                width_px, height_px = image.size
            scale = min(6.3 / width_px, 4.5 / height_px)
            p.add_run().add_picture(
                path_str,
                width=Inches(width_px * scale),
                height=Inches(height_px * scale),
            )
            sensitivity_path = self.sensitivity_maps.get(path_str)
            if sensitivity_path and Path(sensitivity_path).is_file():
                caption = doc.add_paragraph(
                    tr("HF bölge duyarlılığı · turuncu: kapatılınca AI puanı düştü · mavi: kapatılınca AI puanı yükseldi")
                )
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption.paragraph_format.space_after = Pt(2)
                for run in caption.runs:
                    run.italic = True
                    run.font.size = Pt(8)
                    run.font.color.rgb = RGBColor(91, 107, 99)
                from PIL import Image
                with Image.open(sensitivity_path) as image:
                    map_width, map_height = image.size
                map_scale = min(4.2 / map_width, 2.0 / map_height)
                map_paragraph = doc.add_paragraph()
                map_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                map_paragraph.paragraph_format.space_after = Pt(4)
                map_paragraph.add_run().add_picture(
                    sensitivity_path,
                    width=Inches(map_width * map_scale),
                    height=Inches(map_height * map_scale),
                )
            table = doc.add_table(rows=1, cols=3)
            table.style = "Light Shading Accent 1"
            headers = [tr("Algılayıcı"), tr("AI puanı"), tr("Güvenilirlik ağırlığı")]
            for cell, text in zip(table.rows[0].cells, headers):
                cell.text = text
                self._shade(cell, "155E4B")
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            scores = self.scores[path_str]
            for backend in MODEL_ORDER:
                cells = table.add_row().cells
                cells[0].text = MODEL_LABELS[backend]
                value = scores.get(backend)
                cells[1].text = f"{value*100:.1f}%" if value is not None else "—"
                cells[2].text = f"{RELIABILITY[backend]:.3f}" if backend in RELIABILITY else "—"
            weighted = calculate_weighted_ai_score(scores)
            score_table = doc.add_table(rows=2, cols=2)
            score_table.style = "Light Shading Accent 1"
            score_table.cell(0, 0).text = tr("Ağırlıklı AI puanı")
            score_table.cell(0, 1).text = f"{weighted*100:.1f}%" if weighted is not None else "—"
            score_table.cell(1, 0).text = tr("Normalize AI puanı · indeks")
            score_table.cell(1, 1).text = f"{normalize_score_index(weighted)*100:.1f}%" if weighted is not None else "—"
            for cell in score_table.rows[0].cells:
                self._shade(cell, "E7F0EA")
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
            for cell in score_table.rows[1].cells:
                self._shade(cell, "F2F4F1")
        doc.save(target)

    @staticmethod
    def _shade(cell, fill):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        tc_pr = cell._tc.get_or_add_tcPr()
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), fill)
        tc_pr.append(shading)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        row = self.file_list.currentRow()
        if 0 <= row < len(self.paths):
            self.show_selected(row)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--sensitivity-smoke":
        from PIL import Image
        import torch
        from ai_photo_app.aidetector.model import create_detector

        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        source_path = Path(sys.argv[2]).resolve()
        with Image.open(source_path) as source:
            image = source.copy()
        detector = create_detector("hf", device="mps")
        try:
            overlay, changes, baseline = build_sensitivity_overlay(image, detector, source_path)
        finally:
            del detector
            gc.collect()
        out_dir = APP_SUPPORT / "sensitivity_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        path_id = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
        out_path = out_dir / f"{source_path.stem}-{path_id}-frozen-smoke.png"
        overlay.convert("RGB").save(out_path, format="PNG", optimize=True)
        print(json.dumps({
            "image": str(source_path),
            "map": str(out_path),
            "map_size": list(overlay.size),
            "cells": len(changes),
            "hf_baseline_ai_score": baseline,
            "minimum_score_change": min(changes),
            "maximum_score_change": max(changes),
        }, ensure_ascii=False), flush=True)
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--inference-smoke":
        app = QCoreApplication(sys.argv[:1])
        worker = AnalysisWorker([str(Path(sys.argv[2]).resolve())])

        def emit_smoke_scores(values: dict[str, dict[str, float]]) -> None:
            for scores in values.values():
                weighted = calculate_weighted_ai_score(scores)
                scores["raw_weighted_ai_score"] = weighted
                scores["normalized_score_index"] = normalize_score_index(weighted) if weighted is not None else None
            print(json.dumps(values, ensure_ascii=False), flush=True)

        worker.scores_ready.connect(emit_smoke_scores)
        worker.backend_failed.connect(lambda backend, error: print(f"ERROR {backend}: {error}", flush=True))
        worker.completed.connect(app.quit)
        worker.start()
        result = app.exec()
        worker.wait()
        return result
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("AIFoto")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    global ACTIVE_WINDOW, LANGUAGE
    LANGUAGE = str(QSettings("AIFoto", "AI Foto").value("ui_language", "tr"))
    if LANGUAGE not in {"tr", "en"}:
        LANGUAGE = "tr"
    window = MainWindow()
    ACTIVE_WINDOW = window
    window.show()
    return app.exec()


STYLESHEET = """
QMainWindow, QWidget { background: #F2F4F1; color: #1B2924; font-family: 'Helvetica Neue', Arial; font-size: 14px; }
QMenuBar { background: #173C31; color: #FFFFFF; padding: 5px 12px; }
QMenuBar::item:selected, QMenu::item:selected { background: #276B54; }
QMenu { background: #FFFFFF; color: #1B2924; border: 1px solid #D9E0DA; }
QFrame#panel { background: #FFFFFF; border: 1px solid #D9E0DA; border-radius: 12px; }
QLabel#title { font-size: 26px; font-weight: 700; color: #173C31; }
QLabel#muted, QLabel#cardSubtext { color: #5B6B63; font-size: 13px; }
QLabel#sectionLabel, QLabel#cardLabel { color: #5B6B63; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#emptyHint { color: #5B6B63; padding: 10px; }
QLabel#preview { background: #EFF2EF; border: 1px dashed #C7D1C9; border-radius: 10px; color: #5B6B63; }
QLabel#fileName { color: #1B2924; font-weight: 600; }
QFrame#scoreCard { background: #E7F0EA; border: 1px solid #C7DDD0; border-radius: 10px; }
QFrame#scoreCard QLabel { background: transparent; }
QLabel#scoreValue { color: #155E4B; font-size: 34px; font-weight: 700; }
QPushButton { min-height: 42px; padding: 0 16px; border-radius: 8px; font-weight: 650; }
QPushButton#primaryButton { background: #155E4B; color: #FFFFFF; border: 1px solid #155E4B; }
QPushButton#primaryButton:hover { background: #0F4D3D; }
QPushButton#secondaryButton { background: #FFFFFF; color: #155E4B; border: 1px solid #A9BFB1; }
QPushButton#secondaryButton:hover { background: #E7F0EA; }
QPushButton:disabled { background: #D9E0DA; color: #5B6B63; }
QPushButton:focus, QListWidget:focus, QTableWidget:focus { outline: 2px solid #276EF1; }
QListWidget, QTableWidget { background: #FFFFFF; border: 1px solid #D9E0DA; border-radius: 8px; gridline-color: #E7ECE8; alternate-background-color: #F6F8F6; }
QListWidget::item { min-height: 40px; padding: 6px 8px; border-radius: 6px; }
QListWidget::item:selected { background: #E7F0EA; color: #155E4B; }
QHeaderView::section { background: #F2F4F1; color: #42544A; padding: 8px 5px; border: 0; border-bottom: 1px solid #D9E0DA; font-weight: 650; }
QTableWidget::item { padding: 7px 4px; }
QProgressBar { border: 1px solid #D9E0DA; border-radius: 5px; background: #FFFFFF; text-align: center; min-height: 12px; color: #1B2924; }
QProgressBar::chunk { background: #155E4B; border-radius: 4px; }
QStatusBar { background: #F2F4F1; color: #5B6B63; }
"""


if __name__ == "__main__":
    raise SystemExit(main())
