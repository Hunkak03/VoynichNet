"""
config.py — Themes, Internationalisation, Persistent Settings
"""
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

# Persist next to this module so settings survive different working directories.
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_CONFIG_DIR, "voychinet_settings.json")

# ─── Themes ──────────────────────────────────────────────────────────────────
# Each theme is a dict of ANSI codes keyed by role.
THEMES: Dict[str, Dict[str, str]] = {
    "cyber": {
        "name":    "Cyber (default)",
        "I":  "\033[94m",   # info        — blue
        "S":  "\033[92m",   # success     — green
        "W":  "\033[93m",   # warn        — yellow
        "F":  "\033[91m",   # error       — red
        "M":  "\033[95m",   # magic       — magenta
        "H":  "\033[96m",   # highlight   — cyan
        "B":  "\033[1m",    # bold
        "D":  "\033[2m",    # dim
        "R":  "\033[0m",    # reset
        "border": "\033[94m",
        "logo":   "\033[96m",
        "prompt": "\033[94m",
        "key":    "\033[96m",
        "val":    "\033[92m",
        "clock":  "\033[93m",
    },
    "matrix": {
        "name":    "Matrix",
        "I":  "\033[32m",
        "S":  "\033[92m",
        "W":  "\033[33m",
        "F":  "\033[31m",
        "M":  "\033[32m",
        "H":  "\033[92m",
        "B":  "\033[1m",
        "D":  "\033[2m",
        "R":  "\033[0m",
        "border": "\033[32m",
        "logo":   "\033[92m",
        "prompt": "\033[32m",
        "key":    "\033[92m",
        "val":    "\033[32m",
        "clock":  "\033[92m",
    },
    "crimson": {
        "name":    "Crimson",
        "I":  "\033[91m",
        "S":  "\033[93m",
        "W":  "\033[33m",
        "F":  "\033[31m",
        "M":  "\033[95m",
        "H":  "\033[91m",
        "B":  "\033[1m",
        "D":  "\033[2m",
        "R":  "\033[0m",
        "border": "\033[31m",
        "logo":   "\033[91m",
        "prompt": "\033[91m",
        "key":    "\033[91m",
        "val":    "\033[93m",
        "clock":  "\033[93m",
    },
    "arctic": {
        "name":    "Arctic",
        "I":  "\033[97m",
        "S":  "\033[96m",
        "W":  "\033[93m",
        "F":  "\033[91m",
        "M":  "\033[95m",
        "H":  "\033[97m",
        "B":  "\033[1m",
        "D":  "\033[2m",
        "R":  "\033[0m",
        "border": "\033[97m",
        "logo":   "\033[97m",
        "prompt": "\033[97m",
        "key":    "\033[97m",
        "val":    "\033[96m",
        "clock":  "\033[96m",
    },
    "gold": {
        "name":    "Gold",
        "I":  "\033[33m",
        "S":  "\033[93m",
        "W":  "\033[91m",
        "F":  "\033[31m",
        "M":  "\033[95m",
        "H":  "\033[93m",
        "B":  "\033[1m",
        "D":  "\033[2m",
        "R":  "\033[0m",
        "border": "\033[33m",
        "logo":   "\033[93m",
        "prompt": "\033[33m",
        "key":    "\033[93m",
        "val":    "\033[33m",
        "clock":  "\033[93m",
    },
    "void": {
        "name":    "Void (monochrome)",
        "I":  "\033[37m",
        "S":  "\033[97m",
        "W":  "\033[37m",
        "F":  "\033[37m",
        "M":  "\033[97m",
        "H":  "\033[97m",
        "B":  "\033[1m",
        "D":  "\033[2m",
        "R":  "\033[0m",
        "border": "\033[37m",
        "logo":   "\033[97m",
        "prompt": "\033[37m",
        "key":    "\033[97m",
        "val":    "\033[37m",
        "clock":  "\033[37m",
    },
}

# ─── Internationalisation ─────────────────────────────────────────────────────
I18N: Dict[str, Dict[str, str]] = {
    "en": {
        "lang_name":        "English",
        "main_title":       "VOYCHINET CIPHER ENGINE",
        "status_model":     "MODEL",
        "status_buffer":    "BUFFER",
        "status_node":      "NODE",
        "menu_import":      "Import source file  (txt/pdf/docx/epub/md/rtf)",
        "menu_web":         "Import from URL / Wikipedia",
        "menu_train":       "Train model on corpus",
        "menu_type_train":  "Train on text you type now",
        "menu_save":        "Save checkpoint",
        "menu_load":        "Load checkpoint",
        "menu_generate":    "Generate / Decrypt text",
        "menu_analyze":     "Corpus analysis  (entropy, IC, Kasiski…)",
        "menu_history":     "Generation history",
        "menu_corpus_mgr":  "Corpus manager",
        "menu_settings":    "Settings  (theme, language, model preset…)",
        "menu_exit":        "Exit",
        "prompt":           "VOYCHINET CMD",
        "invalid_cmd":      "Unknown command.",
        "aborted_empty":    "Buffer is empty. Import a file or type text first.",
        "train_steps":      "Training steps",
        "train_lr":         "Learning rate",
        "train_patience":   "Early-stop patience",
        "enter_continue":   "[ Press Enter to continue ]",
        "seed_prompt":      "Seed text",
        "length_prompt":    "Output length",
        "export_prompt":    "Export to file? [y/N]",
        "saved_ok":         "Saved to",
        "no_model":         "No trained model. Train (3) or load (6) first.",
        "checkpoint_saved": "Checkpoint saved",
        "checkpoint_loaded":"Checkpoint loaded",
        "not_found":        "Not found",
        "file_empty":       "File is empty or unreadable.",
        "corpus_loaded":    "Corpus loaded and cleaned",
        "chars":            "chars",
        "vocab":            "vocab size",
        "url_prompt":       "URL or Wikipedia topic",
        "url_fetching":     "Fetching",
        "url_ok":           "Content extracted",
        "url_fail":         "Failed to fetch",
        "type_prompt":      "Type (or paste) text. End with a line containing only '###'",
        "type_done":        "Text ingested into corpus",
        "type_chars":       "characters added",
        "history_empty":    "No generation history yet.",
        "history_title":    "GENERATION HISTORY",
        "no_corpus":        "No corpus loaded.",
        "corpus_list":      "LOADED CORPORA",
        "corpus_clear":     "Clear all corpora? [y/N]",
        "corpus_cleared":   "All corpora cleared.",
        "perplexity":       "Perplexity",
        "settings_title":   "SETTINGS",
        "theme_changed":    "Theme changed to",
        "lang_changed":     "Language changed to",
        "preset_changed":   "Model preset changed to",
        "settings_saved":   "Settings saved.",
        "generating":       "Generating…",
        "analysis_running": "Running analysis…",
        "top_tokens":       "Top tokens",
        "top_chars":        "Top characters",
        "top_bigrams":      "Top bigrams",
        "top_trigrams":     "Top trigrams",
        "kasiski_title":    "Kasiski Examination",
        "no_repeats":       "No repeated sequences found.",
        "shutdown":         "Shutdown requested. Goodbye.",
        "interrupted":      "Interrupted.",
        "appended":         "Appended to existing corpus",
        "replace_corpus":   "Replace corpus or append? [r/A]",
        "clock_rail":       "LOCAL TIME",
        "clock_badge":      "LIVE",
        "clock_no_vt":     "Live clock needs ANSI/VT (e.g. Windows Terminal). Showing static time only.",
        "menu_settings_seed": "Training RNG seed (reproducible runs)",
        "settings_seed_prompt": "Integer seed, or \"none\" for nondeterministic",
        "train_seeded":     "RNG seeded for reproducibility (training_seed={})",
    },
    "es": {
        "lang_name":        "Español",
        "main_title":       "MOTOR CIFRADO VOYCHINET",
        "status_model":     "MODELO",
        "status_buffer":    "BUFFER",
        "status_node":      "NODO",
        "menu_import":      "Importar archivo  (txt/pdf/docx/epub/md/rtf)",
        "menu_web":         "Importar desde URL / Wikipedia",
        "menu_train":       "Entrenar modelo con corpus",
        "menu_type_train":  "Entrenar con texto escrito ahora",
        "menu_save":        "Guardar checkpoint",
        "menu_load":        "Cargar checkpoint",
        "menu_generate":    "Generar / Descifrar texto",
        "menu_analyze":     "Análisis de corpus  (entropía, IC, Kasiski…)",
        "menu_history":     "Historial de generaciones",
        "menu_corpus_mgr":  "Gestor de corpus",
        "menu_settings":    "Configuración  (tema, idioma, modelo…)",
        "menu_exit":        "Salir",
        "prompt":           "VOYCHINET CMD",
        "invalid_cmd":      "Comando desconocido.",
        "aborted_empty":    "Buffer vacío. Importa un archivo o escribe texto primero.",
        "train_steps":      "Pasos de entrenamiento",
        "train_lr":         "Tasa de aprendizaje",
        "train_patience":   "Paciencia parada anticipada",
        "enter_continue":   "[ Pulsa Enter para continuar ]",
        "seed_prompt":      "Texto semilla",
        "length_prompt":    "Longitud de salida",
        "export_prompt":    "¿Exportar a archivo? [s/N]",
        "saved_ok":         "Guardado en",
        "no_model":         "Sin modelo entrenado. Entrena (3) o carga (6) primero.",
        "checkpoint_saved": "Checkpoint guardado",
        "checkpoint_loaded":"Checkpoint cargado",
        "not_found":        "No encontrado",
        "file_empty":       "El archivo está vacío o no es legible.",
        "corpus_loaded":    "Corpus cargado y limpiado",
        "chars":            "caracteres",
        "vocab":            "tamaño vocab",
        "url_prompt":       "URL o tema de Wikipedia",
        "url_fetching":     "Descargando",
        "url_ok":           "Contenido extraído",
        "url_fail":         "Error al descargar",
        "type_prompt":      "Escribe (o pega) texto. Termina con una línea que solo contenga '###'",
        "type_done":        "Texto incorporado al corpus",
        "type_chars":       "caracteres añadidos",
        "history_empty":    "Sin historial de generaciones.",
        "history_title":    "HISTORIAL DE GENERACIONES",
        "no_corpus":        "Sin corpus cargado.",
        "corpus_list":      "CORPUS CARGADOS",
        "corpus_clear":     "¿Limpiar todos los corpus? [s/N]",
        "corpus_cleared":   "Todos los corpus limpiados.",
        "perplexity":       "Perplejidad",
        "settings_title":   "CONFIGURACIÓN",
        "theme_changed":    "Tema cambiado a",
        "lang_changed":     "Idioma cambiado a",
        "preset_changed":   "Modelo cambiado a",
        "settings_saved":   "Configuración guardada.",
        "generating":       "Generando…",
        "analysis_running": "Ejecutando análisis…",
        "top_tokens":       "Tokens más frecuentes",
        "top_chars":        "Caracteres más frecuentes",
        "top_bigrams":      "Bigramas más frecuentes",
        "top_trigrams":     "Trigramas más frecuentes",
        "kasiski_title":    "Examen de Kasiski",
        "no_repeats":       "No se encontraron secuencias repetidas.",
        "shutdown":         "Apagado solicitado. Hasta luego.",
        "interrupted":      "Interrumpido.",
        "appended":         "Añadido al corpus existente",
        "replace_corpus":   "¿Reemplazar corpus o añadir? [r/A]",
        "clock_rail":       "HORA LOCAL",
        "clock_badge":      "EN VIVO",
        "clock_no_vt":     "El reloj en vivo requiere ANSI/VT (p. ej. Windows Terminal). Solo hora estática.",
        "menu_settings_seed": "Semilla RNG de entrenamiento (runs reproducibles)",
        "settings_seed_prompt": "Entero, o \"none\" para no determinista",
        "train_seeded":     "RNG fijada para reproducibilidad (training_seed={})",
    },
}

# ─── Settings dataclass ──────────────────────────────────────────────────────
@dataclass
class Settings:
    theme:        str   = "cyber"
    language:     str   = "en"
    model_preset: str   = "B"
    seq_len:      int   = 64
    batch_size:   int   = 128
    show_clock:   bool  = True
    auto_save:    bool  = False
    max_gen_hist: int   = 20
    web_timeout:  int   = 10
    corpus_names: list  = field(default_factory=list)
    # None = nondeterministic sampling / init; int = torch + python RNG for reproducible training
    training_seed: Optional[int] = 42

    def save(self):
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "Settings":
        if not os.path.exists(SETTINGS_FILE):
            return cls()
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Only keep known fields
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except Exception:
            return cls()

    @property
    def T(self) -> Dict[str, str]:
        """Return current theme colour codes."""
        return THEMES.get(self.theme, THEMES["cyber"])

    @property
    def L(self) -> Dict[str, str]:
        """Return current language strings."""
        return I18N.get(self.language, I18N["en"])
