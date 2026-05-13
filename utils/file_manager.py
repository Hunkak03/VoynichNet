"""
file_manager.py — CheckpointManager
Production-grade checkpoint management.

Improvements over the original
───────────────────────────────
• JSON registry tracks every saved checkpoint with metadata
• SHA-256 integrity hash stored and verified on load
• Versioned checkpoint directories (name_YYYYMMDD_HHMMSS/)
• Graceful multi-encoding fallback when reading text files
• Export generation results to timestamped .txt reports
• list_checkpoints() / delete_checkpoint() / latest_checkpoint()
• Safe pickle load with weights_only torch.load
"""

import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ─── CheckpointManager ────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages model checkpoints with versioning, metadata, and integrity checks.

    Directory layout
    ────────────────
    checkpoints/
    ├── registry.json                        ← master index
    ├── voychinet_20240101_120000/
    │   ├── model.pth                        ← state_dict
    │   ├── tokenizer.pkl                    ← tokenizer object
    │   └── meta.json                        ← metadata + hash
    └── voychinet_20240102_093015/
        └── ...
    """

    REGISTRY_FILE = "registry.json"
    MODEL_FILE    = "model.pth"
    TOK_FILE      = "tokenizer.pkl"
    META_FILE     = "meta.json"

    def __init__(self, base_dir: str = "checkpoints", project_root: Optional[str] = None):
        self._root = Path(project_root).resolve() if project_root else Path.cwd()
        bd = Path(base_dir)
        self.base_dir = bd if bd.is_absolute() else (self._root / bd)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    # ── Registry I/O ──────────────────────────────────────────────────────────

    def _load_registry(self):
        path = self.base_dir / self.REGISTRY_FILE
        self.registry = {"checkpoints": {}, "latest": None}
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("registry root must be an object")
            cps = data.get("checkpoints")
            if not isinstance(cps, dict):
                cps = {}
            self.registry["checkpoints"] = cps
            self.registry["latest"] = data.get("latest")
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as e:
            print(f"[!] Checkpoint registry unreadable ({e}); using empty index.")

    def _save_registry(self) -> None:
        path = self.base_dir / self.REGISTRY_FILE
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            print(f"[!] Could not save registry: {e}")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # ── Hashing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path, chunk: int = 8192) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    # ── Save ──────────────────────────────────────────────────────────────────

    def guardar_progreso(
        self,
        modelo,
        tokenizer,
        nombre:   str  = "voychinet",
        extra:    Dict = None,
    ) -> str:
        """
        Persist model + tokenizer to a versioned directory.
        Returns the checkpoint ID (used for loading).
        """
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_id = f"{nombre}_{ts}"
        ckpt_dir = self.base_dir / ckpt_id
        ckpt_dir.mkdir(exist_ok=True)

        model_path = ckpt_dir / self.MODEL_FILE
        tok_path   = ckpt_dir / self.TOK_FILE
        model_tmp = ckpt_dir / f"{self.MODEL_FILE}.part"
        tok_tmp = ckpt_dir / f"{self.TOK_FILE}.part"

        torch.save(modelo.state_dict(), model_tmp)
        with open(tok_tmp, "wb") as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(model_tmp), str(model_path))
        os.replace(str(tok_tmp), str(tok_path))

        # Build metadata
        param_count = sum(p.numel() for p in modelo.parameters())
        meta = {
            "id":           ckpt_id,
            "name":         nombre,
            "created_at":   ts,
            "vocab_size":   tokenizer.vocab_size,
            "tok_mode":     getattr(tokenizer, "mode", "char"),
            "param_count":  param_count,
            "model_hash":   self._sha256(model_path),
            "tok_hash":     self._sha256(tok_path),
            "extra":        extra or {},
        }

        meta_path = ckpt_dir / self.META_FILE
        meta_tmp = ckpt_dir / f"{self.META_FILE}.part"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(str(meta_tmp), str(meta_path))

        # Update registry
        self.registry["checkpoints"][ckpt_id] = meta
        self.registry["latest"] = ckpt_id
        self._save_registry()

        return ckpt_id

    # ── Load ──────────────────────────────────────────────────────────────────

    def cargar_progreso(
        self,
        modelo_vacio,
        nombre: str = "voychinet",
        verify_hash: bool = True,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load the latest checkpoint matching `nombre` prefix.
        Returns (model_with_weights, tokenizer) or (None, None) if not found.
        """
        matches = sorted(k for k in self.registry["checkpoints"] if k.startswith(nombre))
        if not matches:
            return None, None

        ckpt_id  = matches[-1]
        ckpt_dir = self.base_dir / ckpt_id
        model_path = ckpt_dir / self.MODEL_FILE
        tok_path   = ckpt_dir / self.TOK_FILE

        if not model_path.exists() or not tok_path.exists():
            print(f"[!] Checkpoint files missing for {ckpt_id}")
            return None, None

        # Optional integrity check
        if verify_hash:
            stored_meta = self.registry["checkpoints"][ckpt_id]
            if self._sha256(model_path) != stored_meta.get("model_hash", ""):
                print(f"[!] WARNING: model.pth hash mismatch for {ckpt_id} — file may be corrupted.")
            if self._sha256(tok_path) != stored_meta.get("tok_hash", ""):
                print(f"[!] WARNING: tokenizer.pkl hash mismatch for {ckpt_id}")

        # Load tokenizer
        try:
            with open(tok_path, "rb") as f:
                tokenizer = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, OSError) as e:
            print(f"[!] Invalid or unreadable tokenizer.pkl for {ckpt_id}: {e}")
            return None, None

        # Load model weights (weights_only requires PyTorch 2.0+)
        try:
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location="cpu")
        except (OSError, RuntimeError, pickle.UnpicklingError) as e:
            print(f"[!] Could not load model weights for {ckpt_id}: {e}")
            return None, None
        try:
            modelo_vacio.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"[!] Model architecture mismatch for {ckpt_id}: {e}")
            return None, None
        modelo_vacio.eval()

        return modelo_vacio, tokenizer

    # ── Registry queries ──────────────────────────────────────────────────────

    def listar_checkpoints(self) -> List[Dict]:
        """Return all checkpoint metadata records, newest first."""
        return sorted(
            self.registry["checkpoints"].values(),
            key=lambda m: str(m.get("created_at", "")),
            reverse=True,
        )

    def latest_checkpoint_id(self, nombre: str = "voychinet") -> Optional[str]:
        matches = sorted(k for k in self.registry["checkpoints"] if k.startswith(nombre))
        return matches[-1] if matches else None

    def delete_checkpoint(self, ckpt_id: str) -> bool:
        """Remove checkpoint directory and registry entry. Returns True on success."""
        ckpt_dir = self.base_dir / ckpt_id
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if ckpt_id in self.registry["checkpoints"]:
            del self.registry["checkpoints"][ckpt_id]
            if self.registry.get("latest") == ckpt_id:
                remaining = sorted(self.registry["checkpoints"].keys())
                self.registry["latest"] = remaining[-1] if remaining else None
            self._save_registry()
            return True
        return False

    def render_checkpoint_list(self, color: bool = True) -> str:
        C = {
            "H": "\033[1m", "G": "\033[92m", "Y": "\033[93m",
            "B": "\033[94m", "R": "\033[0m",
        } if color else {k: "" for k in "HGYBR"}

        ckpts = self.listar_checkpoints()
        if not ckpts:
            return f"  {C['Y']}No checkpoints found.{C['R']}"

        latest_id = self.registry.get("latest", "")
        lines = [f"{C['H']}  {'ID':<35} {'Params':>10}  {'Vocab':>6}  {'Mode':<6}  Created{C['R']}"]
        lines.append("  " + "─" * 75)
        for m in ckpts:
            marker = f"{C['G']}★ {C['R']}" if m.get("id") == latest_id else "  "
            cid = str(m.get("id", "?"))[:35]
            lines.append(
                f"{marker}{C['B']}{cid:<35}{C['R']}"
                f" {m.get('param_count', 0):>10,}"
                f"  {m.get('vocab_size', 0):>6}"
                f"  {m.get('tok_mode', '?'):<6}"
                f"  {m.get('created_at', 'N/A')}"
            )
        return "\n".join(lines)

    def get_checkpoint_details(self, ckpt_id: str) -> str:
        """Get detailed metadata for a specific checkpoint."""
        meta = self.registry["checkpoints"].get(ckpt_id)
        if not meta:
            return f"Checkpoint '{ckpt_id}' not found."
        
        vs = meta.get("vocab_size", "N/A")
        pc = meta.get("param_count", "N/A")
        vs_s = f"{vs:,}" if isinstance(vs, int) else str(vs)
        pc_s = f"{pc:,}" if isinstance(pc, int) else str(pc)
        lines = [
            "=" * 70,
            f"  CHECKPOINT DETAILS: {ckpt_id}",
            "=" * 70,
            f"  Created at       : {meta.get('created_at', 'N/A')}",
            f"  Vocab size       : {vs_s}",
            f"  Total parameters : {pc_s}",
            f"  Tokenizer mode   : {meta.get('tok_mode', 'N/A')}",
            f"  Model hash       : {meta.get('model_hash', 'N/A')[:16]}...",
            f"  Tokenizer hash   : {meta.get('tok_hash', 'N/A')[:16]}...",
        ]
        if meta.get('extra'):
            lines.append("  " + "─" * 68)
            lines.append("  Extra metadata:")
            for k, v in meta['extra'].items():
                lines.append(f"    {k:<20}: {v}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ── File reading ──────────────────────────────────────────────────────────

    def leer_txt_crudo(self, ruta: str) -> Optional[str]:
        """
        Read a text file with multi-encoding fallback.
        Tries: UTF-8 → UTF-8-BOM → Latin-1 → CP-1252
        """
        encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")
        for enc in encodings:
            try:
                with open(ruta, "r", encoding=enc) as f:
                    content = f.read()
                return content
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        print(f"[X] Could not read '{ruta}' with any supported encoding.")
        return None

    # ── Export ────────────────────────────────────────────────────────────────

    def exportar_prediccion(
        self,
        output_text: str,
        seed: str,
        strategy: str = "sampling",
        nombre: str = "output",
    ) -> str:
        """
        Export a generation result to a timestamped report in outputs/.
        Returns the file path.
        """
        out_dir = self._root / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = out_dir / f"{nombre}_{ts}.txt"

        with open(fname, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  VOYCHINET — GENERATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Timestamp  : {datetime.now().isoformat()}\n")
            f.write(f"  Seed       : {repr(seed)}\n")
            f.write(f"  Strategy   : {strategy}\n")
            f.write(f"  Length     : {len(output_text)} characters\n")
            f.write("─" * 60 + "\n\n")
            f.write(output_text)
            f.write("\n\n" + "=" * 60 + "\n")

        return str(fname)

    def exportar_corpus(self, text: str, sources: List[str], nombre: str = "corpus") -> str:
        """Export corpus to timestamped file with source metadata."""
        out_dir = self._root / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = out_dir / f"{nombre}_corpus_{ts}.txt"
        
        with open(fname, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  VOYCHINET — CORPUS EXPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Exported at: {datetime.now().isoformat()}\n")
            f.write(f"  Total chars: {len(text):,}\n")
            f.write(f"  Sources    : {len(sources)}\n")
            f.write("─" * 60 + "\n\n")
            for i, src in enumerate(sources, 1):
                f.write(f"  {i}. {src}\n")
            f.write("\n" + "─" * 60 + "\n\n")
            f.write(text)
            f.write("\n\n" + "=" * 60 + "\n")
        
        return str(fname)

    def compare_checkpoints(self, ckpt_id1: str, ckpt_id2: str) -> str:
        """Compare two checkpoints and return analysis."""
        m1 = self.registry["checkpoints"].get(ckpt_id1)
        m2 = self.registry["checkpoints"].get(ckpt_id2)
        
        if not m1 or not m2:
            return "One or both checkpoints not found."
        
        lines = [
            "=" * 70,
            "  CHECKPOINT COMPARISON",
            "=" * 70,
            "",
            f"  {ckpt_id1:<35} vs  {ckpt_id2:<35}",
            "  " + "─" * 68,
            f"  Created     : {str(m1.get('created_at', 'N/A')):<35}  {str(m2.get('created_at', 'N/A'))}",
            f"  Vocab size  : {m1.get('vocab_size', 0):>12,}          {m2.get('vocab_size', 0):>12,}",
            f"  Parameters  : {m1.get('param_count', 0):>12,}          {m2.get('param_count', 0):>12,}",
            f"  Mode        : {m1.get('tok_mode', '?'):<35}  {m2.get('tok_mode', '?')}",
            "  " + "─" * 68,
        ]
        return "\n".join(lines)

    def export_training_stats(
        self, 
        train_losses: list, 
        val_losses: list,
        val_accs: list,
        epochs: int,
        best_loss: float,
    ) -> str:
        """Export training statistics to a report file."""
        out_dir = self._root / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = out_dir / f"training_stats_{ts}.txt"
        
        with open(fname, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("  VOYCHINET — TRAINING STATISTICS\n")
            f.write("=" * 70 + "\n")
            f.write(f"  Generated at: {datetime.now().isoformat()}\n")
            f.write(f"  Total training steps logged : {epochs}\n")
            f.write(f"  Best loss    : {best_loss:.6f}\n")
            f.write("─" * 70 + "\n\n")
            
            f.write("Step | Train Loss | Val Loss | Val Acc\n")
            f.write("─" * 50 + "\n")
            
            max_len = max(len(train_losses), len(val_losses))
            for i in range(max_len):
                tr_loss = train_losses[i] if i < len(train_losses) else None
                val_loss = val_losses[i] if i < len(val_losses) else None
                val_acc = val_accs[i] if i < len(val_accs) else None
                
                tr_str = f"{tr_loss:.6f}" if tr_loss is not None else "N/A"
                vl_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
                ac_str = f"{val_acc:.6f}" if val_acc is not None else "N/A"
                
                f.write(f"{i+1:>5} | {tr_str:>10} | {vl_str:>8} | {ac_str}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        return str(fname)
