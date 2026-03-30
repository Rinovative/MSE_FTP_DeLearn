# MSE_FTP_DeLearn – Setup Guide

Dieses Repository enthält mehrere Deep-Learning-Projekte (PyTorch, Optuna, W&B) und nutzt **uv** für das Environment-Management.

---

## 1. Repository klonen (VS Code)

1. VS Code öffnen
2. `Ctrl + Shift + P`
3. **"Git: Clone"** auswählen
4. Repository-URL einfügen
5. Ordner auswählen
6. **"Open"** klicken

---

## 2. uv installieren (falls noch nicht installiert)

### Windows (PowerShell im VS Code Terminal)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Danach Terminal neu starten und prüfen:

```powershell
uv --version
```

---

## 3. Projekt öffnen

Falls noch nicht offen:

```text
File → Open Folder → MSE_FTP_DeLearn
```

---

## 4. Abhängigkeiten installieren

Im VS Code Terminal:

```powershell
uv sync
```

Das erstellt automatisch:

* `.venv` (virtuelle Umgebung)
* alle benötigten Pakete (inkl. PyTorch, Ruff, Optuna, W&B, etc.)

---

## 5. Interpreter prüfen

VS Code erkennt die `.venv` meist automatisch.

Falls nicht:

1. `Ctrl + Shift + P`
2. **"Python: Select Interpreter"**
3. `.venv\Scripts\python.exe` auswählen

---

## 6. Installation testen

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Erwartet:

```text
True
```

---

## 7. Datensatz einfügen

Die Daten sind **nicht im Repository enthalten**.

Die Struktur ist bereits im Repo vorbereitet.
Füge die Daten entsprechend ein:

```text
data/
├── 01_icosimal/
│   └── data_uniform_224_224_sets/
│       ├── train/
│       └── validate/
├── 02_project/
├── 03_project/
└── ...
```

---

## 8. Projekt verwenden

Nach dem Setup kann alles direkt in VS Code ausgeführt werden:

* Python-Skripte direkt starten
* Jupyter-Notebooks öffnen und Zellen ausführen

Beispiel:

```powershell
python projects/01_cnn_icosimal/src/train.py
```

---

## 9. Neue Pakete hinzufügen

Neue Abhängigkeiten werden über **uv** hinzugefügt.

```powershell
uv add <package>
```

Beispiel:

```powershell
uv add seaborn
```

---

### Wichtig

* **Keine Pakete direkt mit `uv pip install` hinzufügen**
* Ausnahme: **PyTorch CUDA Spezialfall**

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

👉 Grund:

* CUDA-Versionen liegen nicht im Standard-PyPI
* deshalb ist dieser Sonderfall notwendig

Die Konfiguration ist zusätzlich in der `pyproject.toml` hinterlegt → reproduzierbar

---

## 10. Ruff (Linting & Formatting)

Ruff läuft automatisch in VS Code (Format on Save).

Falls Regeln zu strikt sind, können sie in der `pyproject.toml` angepasst werden:

```toml
[tool.ruff.lint]
ignore = ["RULE_CODE"]
```

Beispiel:

```toml
ignore = ["T201"]  # erlaubt print()
```

👉 Neue Regeln einfach dort hinzufügen oder entfernen.

---

## Hinweise

* `data/` enthält nur die Ordnerstruktur (keine Daten im Git)
* GPU wird automatisch verwendet (falls verfügbar)

---

## Troubleshooting

### GPU funktioniert nicht

```powershell
nvidia-smi
```

Falls Problem:

* andere CUDA-Version testen (`cu126`)
* Umstieg auf Docker

---

### Environment neu aufsetzen

```powershell
uv sync --reinstall
```
