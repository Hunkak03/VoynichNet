<<<<<<< HEAD
# Voychinet

Interactive terminal application for **corpus import**, **character-level language model training** (small Transformer), **text generation**, **perplexity scoring**, and **cryptanalytic-style corpus stats** (entropy, IC, Kasiski-style repeats).

## Requirements

- **Python 3.10+** (3.11 recommended)
- **PyTorch** (see `requirements.txt`)
- Optional: `requests`, `beautifulsoup4`, `lxml` for web/Wikipedia import; optional readers for PDF/DOCX/EPUB (commented in `requirements.txt`)

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

### Terminal / live clock

The live status clock uses **ANSI cursor control**. On Windows, use **Windows Terminal** (or ConHost with VT enabled). See `docs/TERMINAL_REQUIREMENTS.txt`.

### Non-interactive / CI

```bash
python main.py --version
python main.py --smoke              # import + init engine, exit 0
python main.py --print-settings     # effective settings JSON
python main.py --list-checkpoints   # checkpoint index (plain text)
```

## Where data goes

| Path | Purpose |
|------|---------|
| `logs/audit.log` | Application log (INFO+); WARNING+ also mirrored to **stderr** |
| `data/` | Imported corpora location (default), `generation_history.json` |
| `outputs/` | Exported generations, corpus exports, training stats |
| `checkpoints/` | Model + tokenizer checkpoints + `registry.json` |
| `voychinet_settings.json` | Persisted UI/settings (next to `config.py`) |

## Tests & CI

```bash
pip install pytest
pytest
```

GitHub Actions workflow (`.github/workflows/ci.yml`) runs compileall, `main.py --smoke`, and pytest.

## Security note on checkpoints

Tokenizer checkpoints use **pickle**. Only load checkpoints you trust. See `docs/CHECKPOINT_SECURITY.txt`.

## License

Add your preferred license file at the repository root if you distribute this project publicly.
=======
# VoynichNet
>>>>>>> fa6ced60246f9600a4274738b18d3e93a53d6bf0
