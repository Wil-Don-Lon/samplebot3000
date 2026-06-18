"""Entry point.

Re-execs itself under the project's .venv interpreter if launched from another
Python (e.g. the conda `base` env), so `python3 main.py` always runs against the
right audio stack (working CoreAudio/PortAudio) and the right library versions
(PySide6, librosa, scikit-learn 1.8 that the models were pickled with).
"""

import os
import sys
from pathlib import Path


def _reexec_in_venv() -> None:
    # One-shot guard: if we've already re-exec'd once, never do it again. This
    # is the hard stop against an infinite exec loop in the (rare) case sys.prefix
    # still doesn't resolve to the venv after exec — e.g. a venv created with
    # --system-site-packages, or conda auto-activation overriding the env.
    if os.environ.get("_SAMPLEBOT_REEXEC") == "1":
        return
    here = Path(__file__).resolve().parent
    venv = here / ".venv"
    venv_py = venv / "bin" / "python"
    if not venv_py.exists():
        return
    # Compare sys.prefix (the active environment dir), NOT sys.executable —
    # .venv/bin/python is usually a symlink to the base interpreter, so the
    # executables resolve to the same binary and can't be told apart. sys.prefix
    # is the one thing that reliably differs between the venv and conda base.
    try:
        in_venv = Path(sys.prefix).resolve() == venv.resolve()
    except OSError:
        in_venv = True  # fail safe: don't risk an exec loop
    if not in_venv:
        # Replace this process with the venv interpreter running the same args.
        os.environ["_SAMPLEBOT_REEXEC"] = "1"
        os.execv(str(venv_py), [str(venv_py), str(here / "main.py"), *sys.argv[1:]])


def main() -> int:
    # Imported here (not at module top) so the re-exec happens before we try to
    # import PySide6 etc., which may not exist in the launching environment.
    from PySide6.QtWidgets import QApplication

    from gui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("audio2inst")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    _reexec_in_venv()
    sys.exit(main())
