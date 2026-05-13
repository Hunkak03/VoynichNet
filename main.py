"""
main.py — VoychinetEngine v4.0

New in this version
───────────────────
• Live status-bar clock — wall-second sync, blink separators, Windows VT auto-enable
• 6 colour themes  (cyber / matrix / crimson / arctic / gold / void)
• EN / ES i18n     (all UI strings externalised)
• Train on typed/pasted text (inline corpus building)
• Universal document reader  (.txt .md .pdf .docx .epub .rtf .html .csv)
• Web/Wikipedia importer     (Wikipedia REST, topic search, generic scraper)
• Multi-source corpus manager (accumulate text from many origins)
• Generation history          (last N entries, disk-backed across sessions)
• Perplexity scoring          (evaluate model on any text)
• Character frequency heatmap (ASCII art bar chart)
• Settings submenu            (theme, language, model preset, seq_len, …)
• Auto-save option            (save checkpoint after every training run)
• Pytest smoke suite + GitHub Actions CI (compile, --smoke, pytest)
• Reproducible training seed (Settings `training_seed`; `none` = nondeterministic)
• CLI flags: `--version`, `--smoke`, `--print-settings`, `--list-checkpoints`
• Audit log WARNING+ mirrored to stderr; checkpoint pickle risk documented in docs/
• All settings persisted to JSON across sessions
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

GEN_HISTORY_FILE = os.path.join(PROJECT_ROOT, "data", "generation_history.json")
VOYNICH_VERSION = "4.0.1"

# ── Core module imports (flat or original layout) ─────────────────────────────
try:
    from tokenizer       import VoynichTokenizer
    from model           import VoynichTransformer, ModelConfig
    from file_manager    import CheckpointManager
    from config          import Settings, THEMES, I18N
    from document_reader import read_document, supported_extensions, preview
    from web_crawler     import (
        fetch_wikipedia_article,
        search_and_fetch_related_topics,
        scrape_url,
        is_wikipedia_url,
        topic_from_wikipedia_url,
    )
except ModuleNotFoundError:
    try:
        from core.tokenizer      import VoynichTokenizer          # type: ignore
        from core.model          import VoynichTransformer, ModelConfig  # type: ignore
        from utils.file_manager  import CheckpointManager         # type: ignore
        from config              import Settings, THEMES, I18N
        from document_reader     import read_document, supported_extensions, preview
        from web_crawler         import (                         # type: ignore
            fetch_wikipedia_article, search_and_fetch_related_topics,
            scrape_url, is_wikipedia_url, topic_from_wikipedia_url,
        )
    except ModuleNotFoundError as e:
        print(f"\n  CRITICAL MODULE ERROR: {e}\n"
              f"  Place all .py files in the same folder as main.py\n")
        sys.exit(1)

try:
    from ui.live_clock import CLOCK_RAIL_ROW, StatusBarClock
    from ui.terminal import enable_vt_processing
except ImportError as _ui_err:
    print(
        "\n  CRITICAL: the `ui` package is missing or broken.\n"
        f"  ({_ui_err})\n"
        "  Ensure `ui/` (live_clock.py, terminal.py) sits next to main.py.\n"
    )
    raise SystemExit(1) from _ui_err


def _clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-copy state tensors; a shallow ``state_dict().copy()`` still aliases live parameters."""
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


# ─── Generation record ───────────────────────────────────────────────────────

@dataclass
class GenRecord:
    timestamp:  str
    seed:       str
    strategy:   str
    length:     int
    output:     str
    perplexity: Optional[float] = None


# ─── Sampling / decoding helpers ─────────────────────────────────────────────

def _top_k(logits: torch.Tensor, k: int) -> int:
    vals, idx = torch.topk(logits, min(k, logits.size(-1)))
    return idx[torch.multinomial(F.softmax(vals, dim=-1), 1)].item()


def _top_p(logits: torch.Tensor, p: float) -> int:
    sl, si = torch.sort(logits, descending=True)
    probs  = F.softmax(sl, dim=-1)
    cum    = torch.cumsum(probs, dim=-1)
    sl[cum - probs > p] = float("-inf")
    return si[torch.multinomial(F.softmax(sl, dim=-1), 1)].item()


def _beam(model: VoynichTransformer, seed: List[int], length: int,
          bw: int, device: torch.device) -> List[int]:
    beams: List[Tuple[float, List[int]]] = [(0.0, list(seed))]
    mxl = model.cfg.max_seq_len
    for _ in range(length):
        cands: List[Tuple[float, List[int]]] = []
        for lp, ids in beams:
            inp = torch.tensor([ids[-mxl:]]).to(device)
            with torch.no_grad():
                lg, _ = model(inp)
            lps = F.log_softmax(lg.squeeze(), dim=-1)
            tv, ti = torch.topk(lps, bw)
            for v, n in zip(tv.tolist(), ti.tolist()):
                cands.append((lp + v, ids + [n]))
        beams = sorted(cands, key=lambda x: -x[0])[:bw]
    return beams[0][1][len(seed):] if beams else []


# ─── Visual helpers ───────────────────────────────────────────────────────────

def _heatmap(freq: List[Tuple[str, int]], T: Dict, width: int = 40) -> str:
    if not freq:
        return ""
    top  = freq[:20]
    maxv = max(n for _, n in top) or 1
    rows = []
    for ch, cnt in top:
        bl = int(cnt / maxv * width)
        bar = T["val"] + "█" * bl + T["D"] + "░" * (width - bl) + T["R"]
        rows.append(f"  {T['H']}{repr(ch):<6}{T['R']} {bar} {T['D']}{cnt:>6}{T['R']}")
    return "\n".join(rows)


def _loss_curve(tr: List[float], va: List[float], T: Dict,
                w: int = 50, h: int = 8) -> str:
    if not tr:
        return ""
    all_ = tr + (va or [])
    mn, mx = min(all_), max(all_)
    rng = (mx - mn) or 1.0
    rows = []
    for r in range(h):
        ratio = 1.0 - r / max(1, h - 1)
        label = f"{mn + ratio * rng:6.4f}"
        row   = []
        for col in range(w):
            it = min(int(col / w * len(tr)), len(tr) - 1)
            nt = (tr[it] - mn) / rng
            cell = " "
            if abs(nt - ratio) < 1 / h:
                cell = f"{T['I']}■{T['R']}"
            if va:
                iv = min(int(col / w * len(va)), len(va) - 1)
                if abs((va[iv] - mn) / rng - ratio) < 1 / h:
                    cell = f"{T['S']}▲{T['R']}"
            row.append(cell)
        rows.append(f"  {T['D']}{label}│{T['R']}" + "".join(row))
    rows.append(f"  {'':>7}└{'─'*w}")
    return "\n".join(rows)


def _pbar(step: int, total: int, loss: float, val: Optional[float], w: int = 28) -> str:
    f = int(step / total * w)
    bar = "█" * f + "░" * (w - f)
    vs  = f"  val {val:.4f}" if val is not None else ""
    return f"  [{bar}] {step:>5}/{total}  loss {loss:.4f}{vs}"


# ─── Engine ───────────────────────────────────────────────────────────────────

class VoychinetEngine:

    PRESETS = {
        "S": ("Small  (4L/4H/128D — fast)",   ModelConfig.small),
        "B": ("Base   (6L/8H/256D — default)", ModelConfig.base),
        "L": ("Large  (8L/8H/512D — GPU)",     ModelConfig.large),
    }

    def __init__(self):
        self.cfg         = Settings.load()
        self.fm          = CheckpointManager(project_root=PROJECT_ROOT)
        self.tokenizer   = VoynichTokenizer(mode="char")
        self.model:      Optional[VoynichTransformer] = None
        self.corpus:     str  = ""
        self.is_trained: bool = False
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_history: List[GenRecord] = []
        self.corpus_sources: List[str]   = []

        _logs = os.path.join(PROJECT_ROOT, "logs")
        _data = os.path.join(PROJECT_ROOT, "data")
        _out  = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(_logs, exist_ok=True)
        os.makedirs(_data, exist_ok=True)
        os.makedirs(_out, exist_ok=True)

        self._setup_logging(_logs)
        self._load_gen_history()

        self.clock = StatusBarClock(CLOCK_RAIL_ROW, self._clock_snapshot)
        self.clock.set_supported(enable_vt_processing() and sys.stdout.isatty())
        if self.cfg.show_clock:
            self.clock.start()

    def _setup_logging(self, logs_dir: str) -> None:
        """File audit log + WARNING+ to stderr (operators see issues without opening files)."""
        self.log = logging.getLogger("VOYCHINET")
        self.log.handlers.clear()
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(os.path.join(logs_dir, "audit.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        self.log.addHandler(fh)
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(logging.Formatter("voychinet %(levelname)s: %(message)s"))
        self.log.addHandler(sh)

    def _load_gen_history(self) -> None:
        if not os.path.isfile(GEN_HISTORY_FILE):
            return
        try:
            with open(GEN_HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                return
            cap = max(1, self.cfg.max_gen_hist)
            out: List[GenRecord] = []
            for item in raw[-cap:]:
                if not isinstance(item, dict):
                    continue
                try:
                    out.append(
                        GenRecord(
                            timestamp=str(item.get("timestamp", "")),
                            seed=str(item.get("seed", "")),
                            strategy=str(item.get("strategy", "")),
                            length=int(item.get("length", 0)),
                            output=str(item.get("output", "")),
                            perplexity=item.get("perplexity"),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            self.gen_history = out
        except (json.JSONDecodeError, OSError) as e:
            self.log.warning("Could not load generation history: %s", e)

    def _persist_gen_history(self) -> None:
        cap = max(1, self.cfg.max_gen_hist)
        tail = self.gen_history[-cap:]
        tmp = GEN_HISTORY_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([asdict(r) for r in tail], f, indent=2)
            os.replace(tmp, GEN_HISTORY_FILE)
        except OSError as e:
            self.log.warning("Could not save generation history: %s", e)
            if os.path.isfile(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _clock_snapshot(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        return self.T, self.L

    # ── Shorthand ─────────────────────────────────────────────────────────────
    @property
    def T(self) -> Dict[str, str]:
        return self.cfg.T

    @property
    def L(self) -> Dict[str, str]:
        return self.cfg.L

    # ── UI helpers ────────────────────────────────────────────────────────────
    def _clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    def _header(self):
        self._clear()
        T = self.T
        print(f"{T['logo']}{T['B']}")
        print(r"""
  ██╗   ██╗ ██████╗ ██╗   ██╗ ██████╗██╗  ██╗██╗███╗   ██╗███████╗████████╗
  ██║   ██║██╔═══██╗╚██╗ ██╔╝██╔════╝██║  ██║██║████╗  ██║██╔════╝╚══██╔══╝
  ██║   ██║██║   ██║ ╚████╔╝ ██║     ███████║██║██╔██╗ ██║█████╗     ██║
  ╚██╗ ██╔╝██║   ██║  ╚██╔╝  ██║     ██╔══██║██║██║╚██╗██║██╔══╝     ██║
   ╚████╔╝ ╚██████╔╝   ██║   ╚██████╗██║  ██║██║██║ ╚████║███████╗   ██║
    ╚═══╝   ╚═════╝    ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝""")
        print(T["R"])
        ai_st  = f"{T['S']}READY{T['R']}" if self.is_trained else f"{T['F']}NULL{T['R']}"
        buf_st = f"{T['S']}{len(self.corpus):,}{T['R']}" if self.corpus else f"{T['D']}0{T['R']}"
        dc     = T["S"] if self.device.type == "cuda" else T["W"]
        ts     = datetime.now().strftime("%H:%M:%S")
        if self.cfg.show_clock and self.clock.is_live:
            time_cell = f"{T['S']}{T['B']}{self.L.get('clock_badge', 'LIVE')}{T['R']}"
        else:
            time_cell = f"{T['clock']}{T['B']}{ts}{T['R']}"
        print(
            f"  {T['D']}v4.0{T['R']}  "
            f"{T['key']}{self.L['status_node']}:{T['R']} {dc}{self.device}{T['R']}  │  "
            f"{T['key']}{self.L['status_model']}:{T['R']} {T['I']}{self.cfg.model_preset}{T['R']}  │  "
            f"{T['key']}IA:{T['R']} {ai_st}  │  "
            f"{T['key']}{self.L['status_buffer']}:{T['R']} {buf_st}  │  "
            f"{time_cell}"
        )
        if self.cfg.show_clock and self.clock.supported:
            print(self.clock.placeholder_line(T, self.L))
        print(f"  {T['border']}{'═'*78}{T['R']}\n")
        if self.cfg.show_clock and self.clock.is_live:
            self.clock.refresh_now()

    def _inp(self, prompt: str) -> str:
        self.clock.suppress()
        try:
            return input(prompt)
        finally:
            self.clock.resume()

    def _log(self, level: str, msg: str):
        """Log message to file and optionally display."""
        if level.upper() == "INFO":
            self.log.info(msg)
        elif level.upper() == "WARNING":
            self.log.warning(msg)
        elif level.upper() == "ERROR":
            self.log.error(msg)

    def _pause(self, s: float = 1.4): time.sleep(s)
    def _wait(self):
        self._inp(f"\n  {self.T['D']}[ {self.L['enter_continue']} ]{self.T['R']}")
    def _ok(self,   m: str): print(f"\n  {self.T['S']}[OK] {m}{self.T['R']}")
    def _err(self,  m: str): print(f"\n  {self.T['F']}[!] {m}{self.T['R']}")
    def _warn(self, m: str): print(f"\n  {self.T['W']}[~] {m}{self.T['R']}")
    def _section(self, title: str):
        T = self.T
        print(f"\n  {T['B']}{T['H']}── {title} ──{T['R']}\n")

    # ── Corpus ingest ─────────────────────────────────────────────────────────
    def _ingest(self, raw: str, label: str):
        T = self.T; L = self.L
        cleaned = self.tokenizer.limpiar_texto(raw)
        if not cleaned or len(cleaned) < 20:
            self._err(L["file_empty"]); return

        if self.corpus:
            ans = self._inp(f"  {T['W']}{L['replace_corpus']}{T['R']} ").strip().lower()
            if ans == "r":
                self.corpus         = cleaned
                self.corpus_sources = [label]
                self.tokenizer      = VoynichTokenizer(mode="char")
            else:
                self.corpus        += " " + cleaned
                self.corpus_sources.append(label)
                self._ok(f"{L['appended']}  (+{len(cleaned):,} {L['chars']})")
        else:
            self.corpus         = cleaned
            self.corpus_sources = [label]

        self.tokenizer.ajustar(self.corpus)
        self._ok(
            f"{L['corpus_loaded']}  "
            f"{T['B']}{len(self.corpus):,}{T['R']} {L['chars']}  │  "
            f"{L['vocab']}: {T['B']}{self.tokenizer.vocab_size}{T['R']}"
        )
        self._pause()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 1 — Import document
    # ─────────────────────────────────────────────────────────────────────────
    def service_import(self):
        T = self.T; L = self.L
        print(f"\n  {T['D']}Supported: {supported_extensions()}{T['R']}")
        fname = self._inp(f"  {T['I']}Filename (in data/ or full path) > {T['R']}").strip()

        path = fname
        if not os.path.isabs(fname):
            candidate = os.path.join(PROJECT_ROOT, "data", fname)
            if os.path.exists(candidate):
                path = candidate

        if not os.path.exists(path):
            self._err(f"{L['not_found']}: {path}"); self._pause(); return

        print(f"\n  {T['D']}Reading…{T['R']}")
        text, fmt = read_document(path)
        if text is None:
            self._err(fmt); self._pause(); return

        print(f"  Format: {T['H']}{fmt}{T['R']}  Preview: {T['D']}{preview(text, 120)}{T['R']}")
        self._ingest(text, f"file:{os.path.basename(fname)}")

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 2 — Web / Wikipedia import
    # ─────────────────────────────────────────────────────────────────────────
    def service_web_import(self):
        T = self.T; L = self.L
        print(f"\n  {T['D']}Enter a URL, Wikipedia title, or search phrase.{T['R']}")
        print(f"  {T['D']}E.g.:  medieval latin  |  vulgar latin  |  https://en.wikipedia.org/wiki/Voynich_manuscript{T['R']}")
        query = self._inp(f"\n  {T['I']}{L['url_prompt']} > {T['R']}").strip()
        if not query:
            return

        print(f"\n  {T['I']}{L['url_fetching']}…{T['R']}")

        # Wikipedia URL
        if is_wikipedia_url(query):
            topic, lang = topic_from_wikipedia_url(query)
            text, src   = fetch_wikipedia_article(topic, lang=lang,
                                                  timeout=self.cfg.web_timeout)

        # Generic URL
        elif query.startswith("http://") or query.startswith("https://"):
            follow = self._inp(
                f"  {T['D']}Follow in-domain links for more content? [y/N] > {T['R']}"
            ).strip().lower() == "y"
            text, src = scrape_url(query, timeout=self.cfg.web_timeout,
                                   follow_links=follow, max_links=4)

        # Topic/phrase search
        else:
            lang = self._inp(
                f"  {T['D']}Wikipedia language code [{T['I']}en{T['R']}{T['D']}] > {T['R']}"
            ).strip() or "en"
            extra_raw = self._inp(
                f"  {T['D']}Extra subtopics, comma-separated (optional) > {T['R']}"
            ).strip()
            extras = [e.strip() for e in extra_raw.split(",") if e.strip()] if extra_raw else []

            if extras:
                results = search_and_fetch_related_topics(
                    query, extras, lang=lang,
                    max_articles=min(6, len(extras) + 1),
                    timeout=self.cfg.web_timeout,
                )
                if not results:
                    self._err(f"{L['url_fail']}: {query}"); self._pause(); return
                combined = "\n\n".join(t for _, t, _ in results)
                print(f"  {T['S']}Fetched {len(results)} articles{T['R']}")
                self._ingest(combined, f"wiki-multi:{query}")
                return
            else:
                text, src = fetch_wikipedia_article(query, lang=lang,
                                                    timeout=self.cfg.web_timeout)

        if text is None:
            self._err(f"{L['url_fail']}: {src}"); self._pause(); return

        print(f"  {T['S']}{L['url_ok']}: {len(text):,} {L['chars']}{T['R']}")
        print(f"  {T['D']}{preview(text, 180)}{T['R']}")
        self._ingest(text, f"web:{query[:50]}")

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 3 — Train on typed text
    # ─────────────────────────────────────────────────────────────────────────
    def service_type_train(self):
        T = self.T; L = self.L
        print(f"\n  {T['D']}{L['type_prompt']}{T['R']}\n")
        lines = []
        while True:
            try:
                line = self._inp(f"  {T['D']}>{T['R']} ")
            except EOFError:
                break
            if line.strip() == "###":
                break
            lines.append(line)
        if not lines:
            return
        raw = "\n".join(lines)
        print(f"  {T['I']}{L['type_chars']}: {len(raw):,}{T['R']}")
        self._ingest(raw, "typed:manual")

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 4 — Train model
    # ─────────────────────────────────────────────────────────────────────────
    def service_train(self):
        T = self.T; L = self.L
        if not self.corpus or len(self.corpus) < 50:
            self._warn(L["aborted_empty"]); self._pause(); return

        self._log("INFO", f"Training initiated. Corpus size: {len(self.corpus)} chars")
        print()
        try:
            STEPS    = int(self._inp(f"  {L['train_steps']} [{T['I']}2000{T['R']}]: ").strip() or "2000")
            LR       = float(self._inp(f"  {L['train_lr']}    [{T['I']}3e-4{T['R']}]: ").strip() or "3e-4")
            PATIENCE = int(self._inp(f"  {L['train_patience']} [{T['I']}5{T['R']}]:    ").strip() or "5")
        except ValueError:
            self._err("Invalid number for steps, learning rate, or patience."); self._pause(); return
        if STEPS < 1 or PATIENCE < 1 or not math.isfinite(LR) or LR <= 0:
            self._err("Steps and patience must be >= 1; learning rate must be a positive finite number.")
            self._pause()
            return

        if self.cfg.training_seed is not None:
            s = int(self.cfg.training_seed)
            random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            print(f"  {T['I']}{L['train_seeded'].format(self.cfg.training_seed)}{T['R']}\n")

        SEQ      = self.cfg.seq_len
        BATCH    = self.cfg.batch_size

        cfg_fn = self.PRESETS[self.cfg.model_preset][1]
        cfg    = cfg_fn(self.tokenizer.vocab_size)
        cfg.max_seq_len = SEQ
        self.model = VoynichTransformer(cfg).to(self.device)
        print(f"\n{self.model.parameter_summary()}\n")

        ids = self.tokenizer.codificar(self.corpus)
        if len(ids) < SEQ + 2:
            self._err("Corpus too short. Use a shorter seq_len in Settings."); self._pause(); return

        xs = torch.tensor([ids[i:i+SEQ]     for i in range(len(ids)-SEQ)])
        ys = torch.tensor([ids[i+1:i+SEQ+1] for i in range(len(ids)-SEQ)])
        sp = max(1, int(len(xs) * 0.9))
        x_tr, y_tr = xs[:sp], ys[:sp]
        x_va = xs[sp:] if len(xs) > sp else xs[:min(BATCH, len(xs))]
        y_va = ys[sp:] if len(ys) > sp else ys[:min(BATCH, len(ys))]

        opt  = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-2)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR/20)
        crit = torch.nn.CrossEntropyLoss(ignore_index=0)

        tr_losses: List[float] = []
        va_losses: List[float] = []
        va_accs:   List[float] = []
        best_val  = float("inf")
        best_state_dict = _clone_state_dict(self.model.state_dict())
        pat       = 0
        LOG_E     = max(1, STEPS // 20)
        VAL_E     = max(1, STEPS // 10)

        print(f"  {T['I']}Training on {self.device}…{T['R']}\n")
        self.model.train()
        t0 = time.time()

        for step in range(1, STEPS + 1):
            idx  = torch.randint(0, len(x_tr), (BATCH,))
            xb, yb = x_tr[idx].to(self.device), y_tr[idx].to(self.device)
            opt.zero_grad()
            loss = crit(self.model.forward_sequence(xb).reshape(-1, self.tokenizer.vocab_size),
                        yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step(); sch.step()
            tr_losses.append(loss.item())

            vl: Optional[float] = None
            if step % VAL_E == 0 or step == STEPS:
                self.model.eval()
                with torch.no_grad():
                    vi = torch.randint(0, len(x_va), (min(BATCH, len(x_va)),))
                    lg = self.model.forward_sequence(x_va[vi].to(self.device))
                    yb_va = y_va[vi].to(self.device)
                    vl = crit(lg.reshape(-1, self.tokenizer.vocab_size), yb_va.reshape(-1)).item()
                    
                    # Compute accuracy
                    preds = lg.argmax(-1)
                    acc = (preds == yb_va).float().mean().item()
                va_losses.append(vl)
                va_accs.append(acc)
                if vl < best_val:
                    best_val = vl
                    best_state_dict = _clone_state_dict(self.model.state_dict())
                    pat = 0
                else:
                    pat += 1
                    if pat >= PATIENCE:
                        print(f"\n  {T['W']}[Early stop]{T['R']}"); break
                self.model.train()

            if step % LOG_E == 0 or step == 1:
                print(_pbar(step, STEPS, loss.item(), vl)
                      + f"  lr {sch.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")

        self.model.eval()
        self.model.load_state_dict(best_state_dict)
        self.is_trained = True
        ds = max(1, len(tr_losses) // 40)
        print(f"\n{_loss_curve(tr_losses[::ds], va_losses, T)}")
        
        # Show validation metrics
        if va_accs:
            best_acc = max(va_accs)
            final_acc = va_accs[-1]
            ppl = math.exp(best_val) if best_val > 0 else float('inf')
            print(f"\n  {T['S']}Validation Metrics:{T['R']}")
            print(f"    Final Accuracy    : {T['B']}{final_acc:.4f}{T['R']}")
            print(f"    Best Accuracy     : {T['B']}{best_acc:.4f}{T['R']}")
            print(f"    Best Perplexity   : {T['B']}{ppl:.4f}{T['R']}")
        
        self._ok(f"Training done  best_val={best_val:.4f}  {time.time()-t0:.1f}s")

        if self.cfg.auto_save:
            ck = self.fm.guardar_progreso(self.model, self.tokenizer,
                                          extra={"preset": self.cfg.model_preset})
            self._ok(f"Auto-saved: {ck}")
        
        # Offer to export training statistics
        if self._inp(f"\n  {T['W']}Export training statistics? [y/N] > {T['R']}").strip().lower() in ("y","s"):
            stats_file = self.fm.export_training_stats(
                tr_losses, va_losses, va_accs, STEPS, best_val
            )
            self._ok(f"Statistics exported to: {stats_file}")
        
        self._pause(2)

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 5 — Save / 6 — Load
    # ─────────────────────────────────────────────────────────────────────────
    def service_save(self):
        if self.model is None:
            self._err("No model. Train first."); self._pause(); return
        try:
            ck = self.fm.guardar_progreso(self.model, self.tokenizer,
                                          extra={"preset": self.cfg.model_preset})
            self._ok(f"{self.L['checkpoint_saved']}: {ck}")
        except Exception as e:
            self.log.error(e); self._err(str(e))
        self._pause()

    def service_load(self):
        T = self.T; L = self.L
        print(f"\n{self.fm.render_checkpoint_list()}\n")
        name = self._inp(
            f"  {T['I']}Prefix [{T['B']}voychinet{T['R']}{T['I']}] > {T['R']}"
        ).strip() or "voychinet"

        dummy = VoynichTransformer(ModelConfig.base(max(self.tokenizer.vocab_size, 10)))
        m, tok = self.fm.cargar_progreso(dummy, nombre=name)
        if m is None:
            self._err(f"{L['not_found']}: {name}"); self._pause(); return

        self.tokenizer  = tok
        self.model = m.to(self.device)
        self.is_trained = True
        self._ok(f"{L['checkpoint_loaded']}  vocab={tok.vocab_size}  params={self.model.count_parameters():,}")
        self._pause()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 7 — Generate
    # ─────────────────────────────────────────────────────────────────────────
    def service_generate(self):
        T = self.T; L = self.L
        if not self.is_trained or self.model is None:
            self._err(L["no_model"]); self._pause(); return

        strats = {"1":"Greedy", "2":"Temperature", "3":"Top-k", "4":"Top-p (nucleus)", "5":"Beam search"}
        self._section("GENERATION")
        for k, v in strats.items():
            print(f"   [{T['key']}{k}{T['R']}] {v}")

        strat = self._inp(f"\n  {T['B']}Strategy > {T['R']}").strip()
        if strat not in strats:
            self._err("Invalid."); self._pause(); return

        seed   = self._inp(f"  {T['W']}{L['seed_prompt']} > {T['R']}").lower().strip() or "a"
        try:
            length = int(self._inp(f"  {L['length_prompt']} [{T['I']}200{T['R']}] > ").strip() or "200")
        except ValueError:
            self._err("Invalid length."); self._pause(); return
        if length < 1:
            self._err("Length must be >= 1."); self._pause(); return
        sids   = self.tokenizer.codificar(seed) or [self.tokenizer.UNK_ID]
        rids: List[int] = []
        self.model.eval()
        print(f"\n  {T['I']}{L['generating']}{T['R']}\n")

        with torch.no_grad():
            if strat == "1":
                ids = list(sids)
                for _ in range(length):
                    inp = torch.tensor([ids[-self.model.cfg.max_seq_len:]]).to(self.device)
                    lg, _ = self.model(inp)
                    n = lg.squeeze(-2).argmax(-1).item(); ids.append(n); rids.append(n)

            elif strat == "2":
                temp = float(self._inp(f"  Temperature [{T['I']}0.8{T['R']}] > ").strip() or 0.8)
                ids  = list(sids)
                for _ in range(length):
                    inp = torch.tensor([ids[-self.model.cfg.max_seq_len:]]).to(self.device)
                    lg, _ = self.model(inp)
                    n = torch.multinomial(F.softmax(lg.squeeze(-2) / max(temp,1e-5), dim=-1), 1).item()
                    ids.append(n); rids.append(n)

            elif strat == "3":
                k = int(self._inp(f"  k [{T['I']}40{T['R']}] > ").strip() or 40)
                ids = list(sids)
                for _ in range(length):
                    inp = torch.tensor([ids[-self.model.cfg.max_seq_len:]]).to(self.device)
                    lg, _ = self.model(inp)
                    n = _top_k(lg.squeeze(), k); ids.append(n); rids.append(n)

            elif strat == "4":
                p = float(self._inp(f"  p [{T['I']}0.9{T['R']}] > ").strip() or 0.9)
                ids = list(sids)
                for _ in range(length):
                    inp = torch.tensor([ids[-self.model.cfg.max_seq_len:]]).to(self.device)
                    lg, _ = self.model(inp)
                    n = _top_p(lg.squeeze(), p); ids.append(n); rids.append(n)

            elif strat == "5":
                bw  = int(self._inp(f"  Beam width [{T['I']}5{T['R']}] > ").strip() or 5)
                rids = _beam(self.model, sids, length, bw, self.device)

        output = self.tokenizer.decodificar(rids)
        full   = seed + output
        ppl    = self._perplexity(full)
        ppl_s  = f"{ppl:.2f}" if ppl is not None else "N/A"

        print(f"  {T['border']}{'─'*72}{T['R']}")
        print(f"  {T['M']}{full}{T['R']}")
        print(f"  {T['border']}{'─'*72}{T['R']}")
        print(f"  {T['D']}{L['perplexity']}: {ppl_s}{T['R']}")

        self.gen_history.append(GenRecord(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            seed, strats[strat], length, full, ppl
        ))
        if len(self.gen_history) > self.cfg.max_gen_hist:
            self.gen_history.pop(0)
        self._persist_gen_history()

        if self._inp(f"\n  {L['export_prompt']} ").strip().lower() in ("y","s"):
            p = self.fm.exportar_prediccion(full, seed, strats[strat])
            self._ok(f"{L['saved_ok']} {p}")
        self._wait()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 7b — Batch generation
    # ─────────────────────────────────────────────────────────────────────────
    def service_batch_generate(self):
        T = self.T; L = self.L
        if not self.is_trained or self.model is None:
            self._err(L["no_model"]); self._pause(); return

        self._section("BATCH GENERATION")
        try:
            num_samples = int(self._inp(f"  Number of samples [{T['I']}5{T['R']}] > ").strip() or "5")
        except ValueError:
            self._err("Invalid number of samples."); self._pause(); return
        if num_samples < 1:
            self._err("Samples must be >= 1."); self._pause(); return
        seed = self._inp(f"  {T['W']}{L['seed_prompt']} > {T['R']}").lower().strip() or "a"
        try:
            length = int(self._inp(f"  {L['length_prompt']} [{T['I']}100{T['R']}] > ").strip() or "100")
        except ValueError:
            self._err("Invalid length."); self._pause(); return
        if length < 1:
            self._err("Length must be >= 1."); self._pause(); return
        sids = self.tokenizer.codificar(seed) or [self.tokenizer.UNK_ID]
        results = []
        
        self.model.eval()
        print(f"\n  {T['I']}Generating {num_samples} samples…{T['R']}\n")
        
        with torch.no_grad():
            for sample_num in range(num_samples):
                ids = list(sids)
                rids = []
                for _ in range(length):
                    inp = torch.tensor([ids[-self.model.cfg.max_seq_len:]]).to(self.device)
                    lg, _ = self.model(inp)
                    n = lg.squeeze(-2).argmax(-1).item()
                    ids.append(n); rids.append(n)
                
                output = self.tokenizer.decodificar(rids)
                full = seed + output
                ppl = self._perplexity(full)
                results.append((full, ppl))
                ppl_str = f"{ppl:.2f}" if ppl is not None else "N/A"
                print(f"  {T['D']}Sample {sample_num+1}/{num_samples}: ppl={ppl_str}{T['R']}")
        
        # Display and optionally export results
        self._section("RESULTS")
        for i, (text, ppl) in enumerate(results, 1):
            ppl_s = f"{ppl:.2f}" if ppl is not None else "N/A"
            print(f"  {T['H']}[{i}]{T['R']} ppl={ppl_s}")
            print(f"     {T['D']}{text[:100]}…{T['R']}\n")
        
        if self._inp(f"\n  {L['export_prompt']} ").strip().lower() in ("y","s"):
            for i, (text, ppl) in enumerate(results, 1):
                p = self.fm.exportar_prediccion(text, seed, "batch_greedy", f"batch_sample_{i}")
                self._ok(f"Saved sample {i}: {p}")
        self._wait()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 8 — Corpus analysis
    # ─────────────────────────────────────────────────────────────────────────
    def service_analyze(self):
        T = self.T; L = self.L
        if not self.corpus:
            self._warn(L["no_corpus"]); self._pause(); return
        print(f"\n  {T['I']}{L['analysis_running']}{T['R']}\n")
        stats = self.tokenizer.obtener_estadisticas(self.corpus)
        print(stats.render())

        self._section("CHARACTER FREQUENCY HEATMAP")
        print(_heatmap(stats.top_chars, T))

        self._section(L["top_bigrams"])
        print(_heatmap(self.tokenizer.analizar_ngramas(self.corpus, 2, 20), T))

        self._section(L["top_trigrams"])
        print(_heatmap(self.tokenizer.analizar_ngramas(self.corpus, 3, 15), T))

        self._section(L["kasiski_title"])
        ks = self.tokenizer.calcular_kasiski(self.corpus, min_len=3)
        if ks:
            for seq, pos in ks[:12]:
                gaps = [pos[i+1]-pos[i] for i in range(len(pos)-1)]
                print(f"  {T['H']}{seq!r:<12}{T['R']} ×{len(pos)}  gaps: {gaps[:6]}")
        else:
            print(f"  {T['D']}{L['no_repeats']}{T['R']}")
        self._wait()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 9 — Generation history
    # ─────────────────────────────────────────────────────────────────────────
    def service_history(self):
        T = self.T; L = self.L
        if not self.gen_history:
            self._warn(L["history_empty"]); self._pause(); return
        self._section(L["history_title"])
        for i, rec in enumerate(reversed(self.gen_history), 1):
            ppl = f"{rec.perplexity:.2f}" if rec.perplexity is not None else "—"
            print(f"  {T['D']}{i:>2}.{T['R']} {T['H']}{rec.timestamp}{T['R']}  "
                  f"{T['key']}strategy:{T['R']}{T['val']}{rec.strategy:<22}{T['R']}  "
                  f"{T['key']}ppl:{T['R']}{T['val']}{ppl:<8}{T['R']}  "
                  f"{T['key']}seed:{T['R']}{T['I']}{repr(rec.seed)}{T['R']}")
            print(f"     {T['D']}{rec.output[:130]}{'…' if len(rec.output)>130 else ''}{T['R']}\n")
        n = self._inp("  Export entry number (Enter to skip) > ").strip()
        if n.isdigit():
            lst = list(reversed(self.gen_history))
            idx = int(n) - 1
            if 0 <= idx < len(lst):
                rec = lst[idx]
                p   = self.fm.exportar_prediccion(rec.output, rec.seed, rec.strategy)
                self._ok(f"{L['saved_ok']} {p}")
        self._wait()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 10 — Corpus manager
    # ─────────────────────────────────────────────────────────────────────────
    def service_corpus_manager(self):
        T = self.T; L = self.L
        self._section(L["corpus_list"])
        if not self.corpus_sources:
            print(f"  {T['D']}{L['no_corpus']}{T['R']}")
        else:
            for i, src in enumerate(self.corpus_sources, 1):
                print(f"   {T['D']}{i:>2}.{T['R']} {T['H']}{src}{T['R']}")
        print(f"\n  Total: {T['val']}{len(self.corpus):,} {L['chars']}{T['R']}  "
              f"Vocab: {T['val']}{self.tokenizer.vocab_size}{T['R']}")
        
        # Export / Clear options
        print(f"\n  {T['B']}Options:{T['R']}")
        print(f"   [{T['key']}1{T['R']}] Export corpus to file")
        print(f"   [{T['key']}2{T['R']}] Clear all corpora")
        choice = self._inp(f"\n  {T['W']}Enter choice [1-2] or press Enter to go back > {T['R']}").strip()
        
        if choice == "1" and self.corpus:
            p = self.fm.exportar_corpus(self.corpus, self.corpus_sources)
            self._ok(f"{L['saved_ok']} {p}")
        elif choice == "2" and self._inp(f"\n  {T['W']}{L['corpus_clear']}{T['R']} ").strip().lower() in ("y","s"):
            self.corpus = ""; self.corpus_sources = []
            self.tokenizer = VoynichTokenizer(mode="char")
            self.is_trained = False; self.model = None
            self._ok(L["corpus_cleared"])
        self._pause()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 11 — Perplexity scorer
    # ─────────────────────────────────────────────────────────────────────────
    def service_perplexity(self):
        T = self.T; L = self.L
        if not self.is_trained or self.model is None:
            self._err(L["no_model"]); self._pause(); return
        self._section("PERPLEXITY SCORER")
        print(f"  {T['D']}Lower = text fits trained distribution better{T['R']}\n")
        text = self._inp("  Text to score > ").strip()
        if not text:
            return
        ppl = self._perplexity(text)
        if ppl is not None:
            print(f"\n  {L['perplexity']}: {T['B']}{ppl:.4f}{T['R']}")
            if math.isfinite(ppl):
                bl  = min(60, int(60 / max(ppl, 1e-9)))
                bar = T["S"] + "█" * bl + T["D"] + "░" * (60 - bl) + T["R"]
                print(f"  {bar}")
        self._wait()

    def _perplexity(self, text: str) -> Optional[float]:
        if not self.model or not self.is_trained:
            return None
        ids = self.tokenizer.codificar(text)
        if len(ids) < 2:
            return None
        mxl = self.model.cfg.max_seq_len
        ids = ids[:mxl + 1]
        x   = torch.tensor([ids[:-1]]).to(self.device)
        y   = torch.tensor(ids[1:]).to(self.device)
        with torch.no_grad():
            lg = self.model.forward_sequence(x).squeeze(0)
            loss = F.cross_entropy(lg, y, ignore_index=0)
        val = math.exp(loss.item())
        return val if math.isfinite(val) else None

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 12 — Settings
    # ─────────────────────────────────────────────────────────────────────────
    def service_settings(self):
        T = self.T; L = self.L
        while True:
            self._header()
            self._section(L["settings_title"])
            opts = {
                "1": f"Theme          [{T['H']}{self.cfg.theme}{T['R']}]",
                "2": f"Language       [{T['H']}{self.cfg.language}{T['R']}]",
                "3": f"Model preset   [{T['H']}{self.cfg.model_preset}{T['R']}]",
                "4": f"Sequence len   [{T['H']}{self.cfg.seq_len}{T['R']}]",
                "5": f"Batch size     [{T['H']}{self.cfg.batch_size}{T['R']}]",
                "6": f"Live clock     [{T['H']}{'ON' if self.cfg.show_clock else 'OFF'}{T['R']}]",
                "7": f"Auto-save      [{T['H']}{'ON' if self.cfg.auto_save else 'OFF'}{T['R']}]",
                "8": f"Max history    [{T['H']}{self.cfg.max_gen_hist}{T['R']}]",
                "9": f"Web timeout    [{T['H']}{self.cfg.web_timeout}s{T['R']}]",
                "10": f"{L['menu_settings_seed']} [{T['H']}{self.cfg.training_seed!r}{T['R']}]",
                "0": "← Back",
            }
            for k, v in opts.items():
                print(f"   [{T['key']}{k}{T['R']}] {v}")

            cmd = self._inp(f"\n  {T['B']}Settings > {T['R']}").strip()
            if cmd == "0":
                break

            elif cmd == "1":
                print(f"\n  Themes:  ", end="")
                for tid, td in THEMES.items():
                    m = "►" if tid == self.cfg.theme else " "
                    print(f"{m}[{T['key']}{tid}{T['R']}]{td['logo']}{td['name']}{td['R']}  ", end="")
                print()
                ch = self._inp("  Theme name > ").strip().lower()
                if ch in THEMES:
                    self.cfg.theme = ch
                    self._ok(f"{L['theme_changed']}: {ch}")
                else:
                    self._err("Unknown theme.")

            elif cmd == "2":
                print(f"\n  Languages: ", end="")
                for lid, ld in I18N.items():
                    m = "►" if lid == self.cfg.language else " "
                    print(f"{m}[{T['key']}{lid}{T['R']}] {ld['lang_name']}  ", end="")
                print()
                ch = self._inp("  Language code > ").strip().lower()
                if ch in I18N:
                    self.cfg.language = ch
                    self._ok(f"{L['lang_changed']}: {ch}")
                else:
                    self._err("Unknown code.")

            elif cmd == "3":
                print()
                for pid, (pd, _) in self.PRESETS.items():
                    m = "►" if pid == self.cfg.model_preset else " "
                    print(f"   {m}[{T['key']}{pid}{T['R']}] {pd}")
                ch = self._inp("  Preset > ").strip().upper()
                if ch in self.PRESETS:
                    self.cfg.model_preset = ch
                    self._ok(f"{L['preset_changed']}: {ch}")
                else:
                    self._err("Unknown preset.")

            elif cmd == "4":
                v = self._inp(f"  Seq len > ").strip()
                if v.isdigit() and int(v) >= 16:
                    self.cfg.seq_len = int(v)

            elif cmd == "5":
                v = self._inp("  Batch size > ").strip()
                if v.isdigit() and int(v) >= 8:
                    self.cfg.batch_size = int(v)

            elif cmd == "6":
                self.cfg.show_clock = not self.cfg.show_clock
                if self.cfg.show_clock:
                    self.clock.set_supported(enable_vt_processing() and sys.stdout.isatty())
                    self.clock.start()
                    if not self.clock.supported:
                        self._warn(L.get("clock_no_vt", I18N["en"]["clock_no_vt"]))
                else:
                    self.clock.stop()

            elif cmd == "7":
                self.cfg.auto_save = not self.cfg.auto_save

            elif cmd == "8":
                v = self._inp("  Max history > ").strip()
                if v.isdigit():
                    self.cfg.max_gen_hist = int(v)

            elif cmd == "9":
                v = self._inp("  Timeout (s) > ").strip()
                if v.isdigit():
                    self.cfg.web_timeout = int(v)

            elif cmd == "10":
                v = self._inp(f"  {L['settings_seed_prompt']} [{T['I']}42{T['R']}] > ").strip().lower()
                if v in ("none", "random", "off", "no"):
                    self.cfg.training_seed = None
                elif v.lstrip("-").isdigit():
                    self.cfg.training_seed = int(v)

            self.cfg.save()
            self._persist_gen_history()
            self._ok(L["settings_saved"])
            self._pause(0.6)

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 13 — Model info
    # ─────────────────────────────────────────────────────────────────────────
    def service_model_info(self):
        if self.model:
            print(f"\n{self.model.parameter_summary()}")
        else:
            self._warn("No model loaded.")
        self._wait()

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE 14 — Checkpoint comparison
    # ─────────────────────────────────────────────────────────────────────────
    def service_checkpoint_comparison(self):
        T = self.T
        print(f"\n{self.fm.render_checkpoint_list()}\n")
        
        choice = self._inp(f"  {T['I']}(C)ompare checkpoints or view (D)etails? [c/d] > {T['R']}").strip().lower()
        
        if choice == "c":
            ckpt1 = self._inp(f"  {T['I']}First checkpoint ID > {T['R']}").strip()
            ckpt2 = self._inp(f"  {T['I']}Second checkpoint ID > {T['R']}").strip()
            if ckpt1 and ckpt2:
                comparison = self.fm.compare_checkpoints(ckpt1, ckpt2)
                print(f"\n{comparison}\n")
        elif choice == "d":
            ckpt_id = self._inp(f"  {T['I']}Checkpoint ID > {T['R']}").strip()
            if ckpt_id:
                details = self.fm.get_checkpoint_details(ckpt_id)
                print(f"\n{details}\n")
        self._wait()

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        entries = [
            ("menu_import",     "📥 ", self.service_import),
            ("menu_web",        "🌐 ", self.service_web_import),
            ("menu_type_train", "⌨️  ", self.service_type_train),
            ("menu_train",      "🧬 ", self.service_train),
            ("menu_save",       "💾 ", self.service_save),
            ("menu_load",       "📂 ", self.service_load),
            ("menu_generate",   "🔮 ", self.service_generate),
            (None,              "🔱 ", self.service_batch_generate),
            ("menu_analyze",    "🔬 ", self.service_analyze),
            ("menu_history",    "🕘 ", self.service_history),
            ("menu_corpus_mgr", "🗂️  ", self.service_corpus_manager),
            (None,              "🧮 ", self.service_perplexity),
            ("menu_settings",   "⚙️  ", self.service_settings),
            (None,              "📊 ", self.service_model_info),
            (None,              "🔍 ", self.service_checkpoint_comparison),
            ("menu_exit",       "❌ ", lambda: sys.exit(0)),
        ]
        extra = {
            8: "Batch generation",
            12: "Perplexity scorer",
            13: "Model architecture info",
            14: "Compare checkpoints",
        }

        while True:
            self._header()
            T = self.T; L = self.L
            for i, (key, icon, _) in enumerate(entries, 1):
                label = L.get(key, extra.get(i, "")) if key else extra.get(i, "")
                print(f"   [{T['key']}{i:>2}{T['R']}] {icon} {label}")

            self.clock.suppress()
            cmd = input(f"\n  {T['B']}{T['prompt']}{L['prompt']} >> {T['R']}").strip()
            self.clock.resume()

            try:
                menu_n = int(cmd)
            except ValueError:
                menu_n = -1
            if 1 <= menu_n <= len(entries):
                idx = menu_n - 1
                try:
                    entries[idx][2]()
                except KeyboardInterrupt:
                    self._warn(L["interrupted"]); self._pause(0.5)
                continue

            self._err(L["invalid_cmd"]); self._pause(0.8)


def _cli_main() -> Optional[int]:
    """Non-interactive entry points (CI, scripting). Return exit code or None to launch TUI."""
    parser = argparse.ArgumentParser(
        prog="voychinet",
        description="Voychinet — corpus, training, and generation CLI.",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    parser.add_argument("--smoke", action="store_true", help="Import stack, init engine, exit 0 (CI).")
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="Print checkpoint registry (plain text) and exit.",
    )
    parser.add_argument(
        "--print-settings",
        action="store_true",
        help="Print effective settings as JSON and exit.",
    )
    args = parser.parse_args()
    if args.version:
        print(VOYNICH_VERSION)
        return 0
    if args.smoke:
        enable_vt_processing()
        VoychinetEngine()
        return 0
    if args.list_checkpoints:
        fm = CheckpointManager(project_root=PROJECT_ROOT)
        print(fm.render_checkpoint_list(color=False))
        return 0
    if args.print_settings:
        print(json.dumps(asdict(Settings.load()), indent=2))
        return 0
    return None


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _code = _cli_main()
    if _code is not None:
        raise SystemExit(_code)
    try:
        enable_vt_processing()
    except Exception:
        pass
    try:
        VoychinetEngine().run()
    except KeyboardInterrupt:
        print(f"\n  \033[2m{I18N['en']['shutdown']}\033[0m\n")
        sys.exit(0)