"""Capture the real PySide empty-state UI in both supported languages."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

import main  # noqa: E402


def main_capture() -> None:
    output = ROOT / "screenshots"
    output.mkdir(exist_ok=True)
    app = QApplication(["AI Foto screenshot capture"])
    app.setApplicationName(main.APP_TITLE)
    app.setOrganizationName("AIFoto")
    app.setStyle("Fusion")
    app.setStyleSheet(main.STYLESHEET)

    for language, filename in (("en", "ai-photo-en.png"), ("tr", "ai-foto-tr.png")):
        main.LANGUAGE = language
        window = main.MainWindow()
        window.resize(1240, 820)
        window.show()
        app.processEvents()
        if not window.grab().save(str(output / filename), "PNG"):
            raise RuntimeError(f"Could not capture {language} screenshot")
        window.close()
        app.processEvents()

    app.quit()


if __name__ == "__main__":
    main_capture()
