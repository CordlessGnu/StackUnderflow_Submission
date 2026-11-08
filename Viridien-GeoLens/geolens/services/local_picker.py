"""Native file/folder dialogs for a local-only GeoLens demo.

Tk runs on a separate process's main thread, not Streamlit's script thread.
The UI always provides editable path fields if Tk/display support is unavailable.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


class PickerError(ValueError):
    pass


_DIALOG = r'''
import json, sys
try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes('-topmost', True)
    except tk.TclError:
        pass
    try:
        kind, initial = sys.argv[1], sys.argv[2]
        if kind == 'directory':
            selected = filedialog.askdirectory(title='GeoLens output directory', initialdir=initial, mustexist=True)
        else:
            types = [('TIFF documents', '*.tif *.tiff')] if kind == 'tiff' else [('JSON annotations', '*.json')]
            selected = filedialog.askopenfilename(title='Select GeoLens input', initialdir=initial, filetypes=types)
        print(json.dumps({'path': selected}))
    finally:
        root.destroy()
except Exception as exc:
    print(json.dumps({'error': str(exc)}))
    sys.exit(1)
'''


def pick_local_path(kind: str, initial: str = '') -> str | None:
    if kind not in ('directory', 'tiff', 'json'):
        raise PickerError('Unsupported picker type.')
    initial_path = Path(initial).expanduser() if initial else Path.home()
    if not initial_path.is_dir():
        initial_path = initial_path.parent
    try:
        result = subprocess.run([sys.executable, '-c', _DIALOG, kind, str(initial_path)],
                                capture_output=True, text=True, timeout=120)
        response = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise PickerError('Native picker unavailable; paste a local path instead.') from exc
    if result.returncode or response.get('error'):
        raise PickerError('Native picker unavailable; paste a local path instead. ' + response.get('error', ''))
    return response.get('path') or None
