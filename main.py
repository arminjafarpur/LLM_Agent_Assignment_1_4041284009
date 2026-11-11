#!/usr/bin/env python3
"""Local-LLM reasoning agent (Ollama). Single-file submission.

Usage:  python main.py --input input.json --output output.json

Always writes a valid output file with the original id and a string answer,
even if the model, the server or any step fails (a format-valid fallback is used).
Sections: answers (format detection/extraction) | model (Ollama calls) | prompts | solver (agent) | CLI.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional



# ======================================================================
# ANSWERS
# ======================================================================
# ---------------------------------------------------------------- normalisation
_TRANS = {}
for _i, _c in enumerate("۰۱۲۳۴۵۶۷۸۹"):          # Persian digits
    _TRANS[ord(_c)] = str(_i)
for _i, _c in enumerate("٠١٢٣٤٥٦٧٨٩"):          # Arabic-Indic digits
    _TRANS[ord(_c)] = str(_i)
_TRANS.update({
    ord("٫"): ".", ord("٬"): ",", ord("،"): ",", ord("؛"): ";", ord("؟"): "?",
    ord("−"): "-", ord("–"): "-", ord("—"): "-", ord("ي"): "ی", ord("ك"): "ک",
})
_INVISIBLE = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_PERSIAN = re.compile(r"[\u0600-\u06FF]")


def normalize_text(s) -> str:
    """Persian/Arabic digits -> English digits, remove invisible marks, tidy spaces."""
    s = str(s).translate(_TRANS)
    s = _INVISIBLE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


def natural_key(s: str):
    """Sort key: numbers by value, text case-insensitively (used for tie-breaks)."""
    out = []
    for tok in re.split(r"(\d+)", s.lower()):
        if tok.isdigit():
            out.append((0, int(tok), ""))
        elif tok:
            out.append((1, 0, tok))
    return out


def same_answer(a: str, b: str) -> bool:
    return re.sub(r"\s+", "", str(a).lower()) == re.sub(r"\s+", "", str(b).lower())


# ---------------------------------------------------------------- format spec
@dataclass
class FormatSpec:
    kind: str = "text"            # integer|number|letter|choice|template|digits|text
    choices: list = field(default_factory=list)
    template: str = ""
    digits: int = 0
    source: str = "default"       # heuristic | model | default

    def describe(self) -> str:
        k = self.kind
        if k == "integer":
            return "a single integer (digits only, no units, no words)"
        if k == "number":
            return "a single number (digits only, no units, no words)"
        if k == "letter":
            return "a single capital Latin letter"
        if k == "choice":
            return "exactly one of these options, copied exactly: " + " | ".join(self.choices)
        if k == "template":
            return ("exactly this template, replacing each <...> placeholder with the real value: "
                    + self.template)
        if k == "digits":
            return f"a single {self.digits}-digit code (digits only)"
        return "only the final answer, without any explanation"

    @staticmethod
    def from_dict(d: dict) -> "FormatSpec":
        return FormatSpec(
            kind=str(d.get("kind", "text")),
            choices=[normalize_text(c) for c in d.get("choices", [])],
            template=normalize_text(d.get("template", "")),
            digits=int(d.get("digits", 0) or 0),
            source=d.get("source", "dataset"),
        )


_TPL = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*<\s*([A-Za-z]+)\s*>")
_WORDNUM = {"دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7}


def _detect_template(text: str):
    ms = list(_TPL.finditer(text))
    if not ms:
        return None
    out = text[ms[0].start():ms[0].end()]
    for prev, m in zip(ms, ms[1:]):
        sep = text[prev.end():m.start()]
        if "\n" in sep or len(sep) > 6:
            break
        out += sep + text[m.start():m.end()]
    return out


def _detect_inline_choice(text: str):
    m = re.search(r"یکی از (?:اعداد|حروف|گزینه‌های|عبارت‌های|عبارات)?\s*([^\n]{1,60}?)\s*را\s*برگردان", text)
    if not m:
        return None
    toks = [t for t in re.split(r"[,\s]+", m.group(1)) if t and t not in ("یا", "و")]
    if len(toks) >= 2 and all(re.fullmatch(r"[A-Za-z0-9]{1,3}", t) for t in toks):
        return toks
    return None


def _detect_choice_lines(text: str):
    lines = text.split("\n")

    def cand(ln):
        t = re.sub(r"^[-•*]\s*", "", ln).strip()
        return t if t and not _PERSIAN.search(t) and len(t) <= 80 else None

    i, block = len(lines), []
    while i > 0 and cand(lines[i - 1]) is not None:
        block.insert(0, cand(lines[i - 1]))
        i -= 1
    if len(block) >= 2 and i > 0 and re.search(
            r"برگردان|یکی|گزینه|عبارت|بنویسید", "\n".join(lines[max(0, i - 2):i])):
        return block
    return None


def detect_format(problem: str):
    """Heuristic format detection (no model call). Returns FormatSpec or None."""
    text = normalize_text(problem)
    tpl = _detect_template(text)
    if tpl:
        return FormatSpec(kind="template", template=tpl, source="heuristic")
    ch = _detect_inline_choice(text)
    if ch:
        return FormatSpec(kind="choice", choices=ch, source="heuristic")
    ch = _detect_choice_lines(text)
    if ch:
        return FormatSpec(kind="choice", choices=ch, source="heuristic")
    sents = [p for p in re.split(r"(?<=[.?!])\s+|\n", text)
             if re.search(r"برگردان|بنویسید|مشخص کنید", p)]
    for s in sents:
        if "عدد صحیح" in s:
            return FormatSpec(kind="integer", source="heuristic")
        m = re.search(r"کد\s+(\S+)\s+رقمی", s)
        if m:
            w = m.group(1)
            n = _WORDNUM.get(w) or (int(w) if w.isdigit() else 0)
            if n:
                return FormatSpec(kind="digits", digits=n, source="heuristic")
        if "حرف" in s and "عدد" not in s:
            return FormatSpec(kind="letter", source="heuristic")
        if "عدد" in s:
            return FormatSpec(kind="number", source="heuristic")
    return None


def spec_from_model_json(obj: dict) -> FormatSpec:
    kind = str(obj.get("type", "text")).strip().lower()
    if kind not in ("integer", "number", "letter", "choice", "template", "digits", "text"):
        kind = "text"
    spec = FormatSpec(kind=kind, source="model")
    spec.choices = [normalize_text(c) for c in (obj.get("choices") or []) if str(c).strip()]
    spec.template = normalize_text(obj.get("template") or "")
    try:
        spec.digits = int(obj.get("digits") or 0)
    except (TypeError, ValueError):
        spec.digits = 0
    if kind == "choice" and len(spec.choices) < 2:
        spec.kind = "text"
    if kind == "template" and not _TPL.search(spec.template) and "<" not in spec.template:
        spec.kind = "text"
    if kind == "digits" and spec.digits <= 0:
        spec.kind = "text"
    return spec


# ---------------------------------------------------------------- extraction
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(t: str) -> str:
    t = t or ""
    if "</think>" in t.lower() and "<think>" not in t.lower():
        t = re.split(r"</think>", t, flags=re.I)[-1]
    t = _THINK.sub("", t)
    if "<think>" in t.lower():          # opened but never closed (cut off)
        t = re.split(r"<think>", t, flags=re.I)[0]
    return t.strip()


def _clean(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    s = s.replace("**", "").replace("__", "").replace("`", "").replace("$", "")
    s = s.strip().strip("\"'“”«»").strip()
    s = re.sub(r"^(final answer|answer|پاسخ|جواب)\s*[:：]\s*", "", s, flags=re.I)
    s = s.strip().rstrip(".").strip()
    return s


def _candidates(raw: str, final_only: bool = False):
    text = strip_think(raw)
    out = []
    ms = re.findall(r"final\s*answer\s*[:：]\s*(.+)", text, flags=re.I)
    if ms:
        out.append(ms[-1])
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and not final_only:      # a cut-off reply has a half-written last line: never trust it
        out.append(lines[-1])
    return out


_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(t: str):
    return [m.group(0).replace(",", "") for m in _NUM.finditer(t)]


def _after_equals(t):
    """'320*1.2*0.75 = 288' -> '288' (only when exactly one number follows the last '=')."""
    if "=" in t:
        tail = t.rsplit("=", 1)[1]
        if len(_numbers(tail)) == 1:
            return tail
    return t


def _parse_integer(t, relaxed):
    if len(_numbers(t)) > 1:
        t = _after_equals(t)
    nums = _numbers(t)
    if not nums or (len(nums) > 1 and not relaxed):
        return None
    v = nums[-1]
    if "." in v:
        f = float(v)
        if f != int(f):
            return None
        v = str(int(f))
    return str(int(v))


def _parse_number(t, relaxed):
    if len(_numbers(t)) > 1:
        t = _after_equals(t)
    nums = _numbers(t)
    if not nums or (len(nums) > 1 and not relaxed):
        return None
    v = nums[-1]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def _parse_letter(t, relaxed):
    t = t.strip()
    if re.fullmatch(r"[A-Za-z]", t):
        return t.upper()
    toks = re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", t)
    if not toks:
        return None
    if len({x.upper() for x in toks}) == 1:
        return toks[0].upper()
    return toks[-1].upper() if relaxed else None


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", "", s.lower()).rstrip(".")


def _parse_choice(t, spec, relaxed):
    key = _norm_key(t)
    for ch in spec.choices:
        if key == _norm_key(ch):
            return ch
    low = re.sub(r"\s+", " ", t.lower())
    found = []
    for ch in spec.choices:
        words = re.sub(r"\s+", " ", ch.lower()).split(" ")
        pat = r"(?<![A-Za-z0-9])" + r"\s*".join(re.escape(w) for w in words) + r"(?![A-Za-z0-9])"
        mm = list(re.finditer(pat, low))
        if mm:
            found.append((ch, mm[-1].start()))
    if len(found) == 1:
        return found[0][0]
    if found:
        keep = [f for f in found
                if not any(f[0] != g[0] and _norm_key(f[0]) in _norm_key(g[0]) for g in found)]
        if len(keep) == 1:
            return keep[0][0]
        if relaxed:
            return max(found, key=lambda f: f[1])[0]
    return None


def _parse_digits(t, n, relaxed):
    runs = re.findall(r"\d+", t)
    if len(runs) == 1 and len(runs[0]) == n:
        return runs[0]
    joined = "".join(runs)
    if re.fullmatch(r"[\d\s,\-]+", t) and len(joined) == n:
        return joined
    if relaxed:
        good = [r for r in runs if len(r) == n]
        if good:
            return good[-1]
    return None


def _template_regex(template: str):
    parts = re.split(r"<\s*([A-Za-z]+)\s*>", template)
    rx = ""
    for i, p in enumerate(parts):
        if i % 2 == 0:
            for ch in p:
                if ch.isspace():
                    rx += r"\s*"
                elif ch in "=,:;":
                    rx += r"\s*" + re.escape(ch) + r"\s*"
                else:
                    rx += re.escape(ch)
        else:
            k = p.lower()
            if k in ("integer", "int"):
                rx += r"(-?\d+)(?:\.0+)?"
            elif k in ("number", "float", "decimal", "real"):
                rx += r"(-?\d+(?:\.\d+)?)"
            elif k in ("letter", "char"):
                rx += r"([A-Za-z])"
            else:
                rx += r"([^\s,;]+)"
    return re.compile(rx, re.I), parts


def _parse_template(t, spec):
    try:
        rx, parts = _template_regex(spec.template)
    except re.error:
        return None
    ms = list(rx.finditer(t))
    if not ms:
        return None
    m = ms[-1]
    out, g = "", 1
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out += p
        else:
            out += m.group(g)
            g += 1
    return out


def _parse(t: str, spec: FormatSpec, relaxed: bool):
    if not t:
        return None
    k = spec.kind
    if k == "integer":
        return _parse_integer(t, relaxed)
    if k == "number":
        return _parse_number(t, relaxed)
    if k == "letter":
        return _parse_letter(t, relaxed)
    if k == "choice":
        return _parse_choice(t, spec, relaxed)
    if k == "digits":
        return _parse_digits(t, spec.digits, relaxed)
    if k == "template":
        return _parse_template(t, spec)
    line = t.split("\n")[0].strip()
    return line or None


def extract_answer(raw: str, spec: FormatSpec, relaxed: bool = False, final_only: bool = False):
    """Return the canonical final answer found in `raw`, or None if invalid."""
    if not raw:
        return None
    for c in _candidates(raw, final_only):
        a = _parse(_clean(c), spec, False)
        if a:
            return a
    if relaxed and not final_only:
        tail = strip_think(raw)[-400:]
        a = _parse(_clean(tail), spec, True)
        if a:
            return a
    return None


def last_line_cleaned(raw: str) -> str:
    """Best-effort cleaned last line (used only when nothing valid was found)."""
    c = _candidates(raw)
    return _clean(c[-1]) if c else ""


def fallback_answer(spec: FormatSpec) -> str:
    """A syntactically valid guess, used only when the model gave nothing usable."""
    k = spec.kind
    if k == "choice" and spec.choices:
        return spec.choices[0]
    if k == "letter":
        return "A"
    if k == "digits" and spec.digits:
        return "1" * spec.digits
    if k == "template" and spec.template:
        def sub(m):
            kk = m.group(1).lower()
            return "0" if kk in ("integer", "int", "number", "float", "decimal", "real") else "A"
        return re.sub(r"<\s*([A-Za-z]+)\s*>", sub, spec.template)
    return "0"


# ======================================================================
# MODEL
# ======================================================================
log = logging.getLogger("agent")

PREFERRED_MODEL = "qwen3.8:27b"   # evaluation model announced by the instructor
SECOND_MODEL = "qwen3:8b"         # model used for development on a 6 GB GPU
FINAL_LINE = re.compile(r"FINAL ANSWER:[ \t*]*\S[^\n]*\n", re.I)   # a finished answer line
RETRY_DELAYS = [3.0, 8.0, 15.0]   # waits before retrying a transient server error (HTTP 5xx / connection)


def _is_looping(text: str) -> bool:
    """True when the model is stuck repeating itself: the last 150 characters already
    appeared at least 3 times in the reply (typical 'Try: ... still too long. Try: ...')."""
    if len(text) < 600:
        return False
    return text.count(text[-150:]) >= 3


def _answer_line_done(text: str) -> bool:
    """True when the visible part of the reply (outside any <think> block) already holds a
    complete 'FINAL ANSWER: ...' line. We do not strip the trailing newline on purpose."""
    low = text.lower()
    if "</think>" in low:
        text = text[low.rfind("</think>") + len("</think>"):]
    elif "<think>" in low:
        return False                       # still inside the hidden thinking block
    return bool(FINAL_LINE.search(text))


# ---------------------------------------------------------------- model choice
def list_installed(client) -> list:
    resp = client.list()
    try:
        models = resp["models"]
    except Exception:
        models = getattr(resp, "models", [])
    names = []
    for m in models or []:
        if isinstance(m, dict):
            n = m.get("model") or m.get("name")
        else:
            n = getattr(m, "model", None) or getattr(m, "name", None)
        if n:
            names.append(n)
    return names


def choose_model(cli_model: Optional[str], installed: list) -> str:
    """1) argument / env var  2) qwen3.8:27b if installed  3) qwen3:8b  4) any qwen."""
    explicit = cli_model or os.environ.get("AGENT_MODEL")
    if explicit:
        return explicit
    low = {n.lower(): n for n in installed}
    if PREFERRED_MODEL in low:
        return low[PREFERRED_MODEL]
    if SECOND_MODEL in low:
        return low[SECOND_MODEL]
    qwens = sorted(n for n in installed if "qwen" in n.lower())
    if qwens:
        qwen3 = [n for n in qwens if "qwen3" in n.lower()]
        return (qwen3 or qwens)[0]
    return PREFERRED_MODEL   # nothing found: calls will fail and fallbacks are used


def model_size_b(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    model: str
    big: bool
    think: str = "off"          # off | on | auto
    num_ctx: int = 8192
    num_predict: int = 2048
    max_calls: int = 6          # model calls per problem (format + solves)
    call_timeout: float = 150.0  # max SILENCE (no new token) before a call is abandoned
    call_wall: float = 240.0     # max TOTAL seconds of one call; partial text is kept
    time_limit: float = 300.0   # soft limit per problem, seconds
    stream: bool = True         # read the reply token by token (see LLM.chat)
    host: Optional[str] = None
    hints: bool = False         # optional technique hints in the prompt (grid search, enumeration...)


def build_settings(model: str, think=None, max_calls=None, time_limit=None, host=None, hints=None) -> Settings:
    big = model_size_b(model) >= 20
    st = Settings(
        model=model,
        big=big,
        max_calls=5 if big else 6,
        call_timeout=300.0 if big else 150.0,
        call_wall=400.0 if big else 240.0,
        time_limit=450.0 if big else 300.0,
        host=host or os.environ.get("OLLAMA_HOST"),
    )
    st.hints = bool(hints) or os.environ.get("AGENT_HINTS", "").lower() in ("1", "on", "true")
    st.think = (think or os.environ.get("AGENT_THINK") or "off").lower()
    if st.think not in ("off", "on", "auto"):
        st.think = "off"
    if st.think == "on":
        st.num_predict = 6144
    env_calls = os.environ.get("AGENT_MAX_CALLS")
    env_time = os.environ.get("AGENT_TIME_LIMIT")
    if max_calls or env_calls:
        st.max_calls = int(max_calls or env_calls)
    if time_limit or env_time:
        st.time_limit = float(time_limit or env_time)
    return st


# ---------------------------------------------------------------- Ollama wrapper
class LLM:
    """Counts calls, handles thinking mode, retries transient server errors."""

    def __init__(self, st: Settings, client=None):
        self.st = st
        self.calls = 0          # logical calls (one per chat(), retries are not counted)
        self.requests = 0       # raw HTTP requests, including retries
        self.last_error = ""
        self.last_truncated = False   # True if the last reply was cut (timeout / loop), so it may be unfinished
        self._send_think = True
        self._soft_switch = False
        if client is None:
            import ollama
            client = ollama.Client(host=st.host, timeout=st.call_timeout)
        self.client = client

    def _consume(self, resp, t_start: float):
        """Read a streamed reply. Returns (text, error). Stops early when the model has
        written a complete 'FINAL ANSWER: ...' line, or when the call exceeds call_wall.
        Whatever text arrived before an error or the wall limit is kept (partial salvage)."""
        parts: list = []
        err = None
        checked = 0
        try:
            for chunk in resp:
                try:
                    piece = chunk["message"]["content"] or ""
                except Exception:       # noqa: BLE001
                    piece = getattr(getattr(chunk, "message", None), "content", "") or ""
                parts.append(piece)
                if "\n" in piece and _answer_line_done("".join(parts)):
                    break                          # answer is complete: stop generating, save time
                if len(text_all := "".join(parts)) - checked >= 200:
                    checked = len(text_all)
                    if _is_looping(text_all):
                        err = RuntimeError("repetition loop")
                        break
                if time.time() - t_start > self.st.call_wall:
                    err = TimeoutError(f"call exceeded {self.st.call_wall:.0f}s")
                    break
        except Exception as e:                     # noqa: BLE001  (e.g. ReadTimeout between tokens)
            err = e
        finally:
            close = getattr(resp, "close", None)
            if close:
                try:
                    close()                        # closing the stream makes Ollama stop generating
                except Exception:                  # noqa: BLE001
                    pass
        return "".join(parts), err

    def chat(self, prompt: str, system: Optional[str] = None, temperature: float = 0.6,
             num_predict: Optional[int] = None, json_mode: bool = False) -> Optional[str]:
        self.calls += 1
        self.last_truncated = False
        options = {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": self.st.num_ctx,
            "num_predict": num_predict or self.st.num_predict,
        }
        errors = 0
        for _ in range(8):                      # hard cap on loop iterations
            user = prompt
            if self._soft_switch and self.st.think == "off":
                user = prompt + " /no_think"
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            kwargs = dict(model=self.st.model, messages=messages, options=options)
            if json_mode:
                kwargs["format"] = "json"
            if self._send_think and self.st.think in ("off", "on"):
                kwargs["think"] = self.st.think == "on"
            if self.st.stream:
                kwargs["stream"] = True
            self.requests += 1
            t_start = time.time()
            try:
                resp = self.client.chat(**kwargs)
                if hasattr(resp, "__next__"):           # a stream of chunks
                    text, err = self._consume(resp, t_start)
                    if err is not None:
                        self.last_error = f"{type(err).__name__}: {str(err)[:200]} (partial {len(text)} chars)"
                        log.warning("stream ended early: %s", self.last_error)
                        self.last_truncated = True      # cut text: only an explicit FINAL ANSWER line may be trusted
                        if not text.strip():
                            raise err                   # nothing received: normal error handling
                    return strip_think(text)
                content = resp["message"]["content"]    # non-stream reply (fake client / old behaviour)
                return strip_think(content or "")
            except TypeError as e:            # old python library without `think`
                if "think" in str(e) and self._send_think:
                    self._send_think = False
                    self._soft_switch = True
                    continue
                self.last_error = f"TypeError: {e}"
                log.warning("chat TypeError: %s", e)
                return None
            except Exception as e:            # noqa: BLE001
                status = getattr(e, "status_code", None)
                self.last_error = f"{type(e).__name__}: {str(e)[:200]} (status {status})"
                if "think" in str(e).lower() and self._send_think and status == 400:
                    self._send_think = False  # model does not support the flag
                    continue
                transient = (isinstance(status, int) and status >= 500) or isinstance(e, ConnectionError)
                log.warning("chat error: %s", self.last_error)
                if transient and errors < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[errors])
                    errors += 1
                    continue
                return None                   # timeouts and other errors: do not retry (slow model)
        return None


# ======================================================================
# PROMPTS
# ======================================================================
P_SYSTEM = (
    "You are a meticulous researcher and an exact solver of mathematical and logical problems. "
    "You read Persian problems carefully, reason step by step, and verify every constraint "
    "before you answer."
)

P_RULES = """Rules:
- The problem is in Persian; all digits have already been converted to English digits.
- Use only the facts that matter. Ignore irrelevant details (ages, colours, jobs, names that never affect the result).
- First list the given facts and constraints in a logical order, then reason.
- Before answering, check your candidate against EVERY constraint of the problem.
- If several answers satisfy all constraints, choose the smallest one in natural order (smallest number, alphabetically first letter or option, earliest day or position).
- In truth-teller / liar puzzles, a contradiction inside one case only eliminates that case; test each case.
- If the statements cannot all hold, take the most consistent interpretation. Never leave the answer empty.
- Finish with exactly one last line: FINAL ANSWER: <answer>
  where <answer> is {fmt}. Write nothing after that line."""

P_EX_STEP = """Example (an invented problem, only to show the style):
Problem: A hiker walks 3 km per hour for 2 hours, then 4 km per hour for 3 hours. His backpack weighs 9 kg. How many kilometres does he walk in total? Answer with an integer only.
Reasoning:
1. Facts: 3 km/h for 2 h; 4 km/h for 3 h. (The backpack weight is irrelevant.)
2. First part: 3 x 2 = 6 km. Second part: 4 x 3 = 12 km.
3. Total: 6 + 12 = 18 km.
4. Check: both parts used, units are km. OK.
FINAL ANSWER: 18"""

P_EX_LTM = """Example (an invented problem, only to show the style):
Problem: A shop buys 12 crates with 15 apples in each crate. 20 percent of all apples are rotten and thrown away. How many good apples remain? Answer with an integer only.
Sub-questions:
1. How many apples are there in total? 12 x 15 = 180.
2. How many are rotten? 20 percent of 180 = 36.
3. How many good apples remain? 180 - 36 = 144.
Check: 144 + 36 = 180. OK.
FINAL ANSWER: 144"""

P_STYLE_HEAD = {
    "stepwise": "Solve the problem below step by step.",
    "ltm": ("Solve the problem below with least-to-most reasoning: first write a short numbered "
            "list of sub-questions (easiest first), then answer them in order, using the earlier answers."),
    "analogical": ("First recall one related and simpler problem (2-3 lines) together with the method "
                   "that solves it. Then use that method to solve the problem below step by step."),
}

P_STYLE_TAIL = {
    "stepwise": "Now solve the problem. Keep the reasoning compact and finish with the FINAL ANSWER line.",
    "ltm": "Now solve the problem with sub-questions. Keep it compact and finish with the FINAL ANSWER line.",
    "analogical": "Now write the related problem briefly, solve the real problem, and finish with the FINAL ANSWER line.",
}

P_HINTS = """
Techniques (use the one that fits the problem):
- Paths on a grid: write the list of blocked cells and the bounds first. Then expand level by level: step 0 = the start; step 1 = all legal neighbours; step 2 = their legal unvisited neighbours; and so on until the target appears. That step number is the answer. Before answering, re-check every cell of the path against the blocked list.
- Orderings, assignments, digit codes, key-and-box puzzles: list the candidates and cross out every one that breaks a constraint; the survivors are the answer set.
- Truth-teller / liar puzzles: test each assumption (X truthful, X liar) and keep only the consistent one.
- Scheduling: compute the earliest finish of each activity from its prerequisites.
- Percent and rates: one operation per line, then re-compute once."""

P_STYLES = ["stepwise", "ltm", "analogical"]


def P_solver_prompt(problem: str, fmt_desc: str, style: str, rewritten: str = "", hints: bool = False) -> str:
    parts = [P_STYLE_HEAD[style], P_RULES.format(fmt=fmt_desc) + (P_HINTS if hints else "")]
    if style == "stepwise":
        parts.append(P_EX_STEP)
    elif style == "ltm":
        parts.append(P_EX_LTM)
    parts.append("PROBLEM:\n" + problem)
    if rewritten:
        parts.append("CLEANED SUMMARY (may miss details; the original problem above is authoritative):\n"
                     + rewritten)
    parts.append(P_STYLE_TAIL[style])
    return "\n\n".join(parts)


P_REWRITE = """Rewrite the Persian problem below as a compact English summary with exactly these sections:
GIVEN: every fact and number that can matter.
CONSTRAINTS: the conditions, ordered logically (conditions that fix things first come first).
QUESTION: what must be found.
Drop only details that are clearly irrelevant. Keep all names, labels and numbers exactly as written. Do NOT solve the problem.

PROBLEM:
{problem}"""

P_FORMAT_SPEC = """Read the problem and describe ONLY the required format of the final answer.
Return JSON only, with these keys:
{{"type": "integer|number|letter|choice|template|digits|text", "choices": [], "template": "", "digits": 0}}
- "choice": put every allowed option, copied exactly, in "choices".
- "template": put the exact template with <integer>/<number>/<word> placeholders in "template".
- "digits": the number of digits for a code answer.

PROBLEM:
{problem}"""

P_BASELINE = "{problem}\n\nReturn only the final answer, in exactly the format requested above. No explanation."


# ======================================================================
# SOLVER
# ======================================================================
log = logging.getLogger("agent")

TEMPS = [0.6, 0.7, 0.8, 0.7, 0.8, 0.9, 0.8, 0.9]


@dataclass
class Result:
    answer: str
    calls: int
    elapsed: float
    valid: bool
    source: str = ""                       # vote | salvage | fallback | baseline
    votes: dict = field(default_factory=dict)
    spec_kind: str = ""
    trace: list = field(default_factory=list)   # short per-solve debug info


def _model_format(text: str, llm: LLM) -> FormatSpec | None:
    raw = llm.chat(P_FORMAT_SPEC.format(problem=text), temperature=0.0,
                   num_predict=300, json_mode=True)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return spec_from_model_json(json.loads(m.group(0))) if m else None
    except (ValueError, TypeError):
        return None


def _decided(votes: Counter) -> bool:
    top = votes.most_common()
    if not top:
        return False
    first = top[0][1]
    second = top[1][1] if len(top) > 1 else 0
    return first >= 2 and first > second


def solve_agent(problem: str, llm: LLM, st: Settings, rewrite: bool = False) -> Result:
    t0 = time.time()
    c0 = llm.calls
    deadline = t0 + st.time_limit
    text = normalize_text(problem)

    spec = detect_format(text)
    if spec is None:
        spec = _model_format(text, llm) or FormatSpec(kind="text", source="default")

    rewritten = ""
    if rewrite and time.time() < deadline - 30:
        r = llm.chat(P_REWRITE.format(problem=text), system=P_SYSTEM,
                     temperature=0.2, num_predict=800)
        rewritten = (r or "")[:1500]

    votes: Counter = Counter()
    tails: list = []
    trace: list = []
    fails = 0
    n = 0
    while llm.calls - c0 < st.max_calls and time.time() < deadline - 5:
        style = P_STYLES[n % len(P_STYLES)]
        prompt = P_solver_prompt(text, spec.describe(), style, rewritten, st.hints)
        raw = llm.chat(prompt, system=P_SYSTEM, temperature=TEMPS[n % len(TEMPS)])
        n += 1
        if raw is None:
            trace.append(dict(style=style, extracted=None, tail="<no response> " + llm.last_error))
            fails += 1
            if fails >= 3:
                break
            continue
        fails = 0
        cut = llm.last_truncated          # reply was cut by timeout / repetition loop
        if not cut:
            tails.append(raw[-500:])      # keep only a short tail, never the full reasoning
        body = (raw.rsplit("\n", 1)[0] if "\n" in raw else "") if cut else raw   # drop the half-written last line
        ans = extract_answer(body, spec, final_only=cut)
        trace.append(dict(style=style, extracted=ans, chars=len(raw), tail=raw[-600:]))
        log.info("solve %d (%s) -> %r", n, style, ans)
        if ans:
            votes[ans] += 1
            if _decided(votes):
                break

    if votes:
        best = max(votes.values())
        winners = sorted((a for a, c in votes.items() if c == best), key=natural_key)
        answer, source = winners[0], "vote"
    else:
        answer, source = None, ""
        for tail in reversed(tails):
            answer = extract_answer(tail, spec, relaxed=True)
            if answer:
                source = "salvage"
                break
        if not answer:
            answer, source = fallback_answer(spec), "fallback"

    return Result(answer=answer, calls=llm.calls - c0, elapsed=time.time() - t0, valid=True,
                  source=source, votes=dict(votes), spec_kind=spec.kind, trace=trace)


def solve_baseline(problem: str, llm: LLM, st: Settings) -> Result:
    """The simplest possible solver: one direct call, no reasoning scaffold."""
    t0 = time.time()
    c0 = llm.calls
    text = normalize_text(problem)
    spec = detect_format(text) or FormatSpec(kind="text", source="default")
    raw = llm.chat(P_BASELINE.format(problem=text), temperature=0.0)
    ans = extract_answer(raw or "", spec)
    valid = ans is not None
    if ans is None:
        ans = last_line_cleaned(raw or "")      # kept as-is so invalid formats are visible in eval
    return Result(answer=ans, calls=llm.calls - c0, elapsed=time.time() - t0,
                  valid=valid, source="baseline", spec_kind=spec.kind)


# ======================================================================
# CLI
# ======================================================================
log = logging.getLogger("agent")


def write_output(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _decode(raw: bytes) -> str:
    """Decode a file robustly: UTF-8 (with/without BOM), UTF-16 (Notepad 'Unicode'), then Windows-1256."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1256"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_input(path):
    with open(path, "rb") as f:
        text = _decode(f.read()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # tolerate markdown code fences (```json ... ```) pasted around the JSON
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        data = json.loads("\n".join(lines))
    if isinstance(data, list):
        return data, True
    return [data], False


def heuristic_fallback(problem):
    spec = detect_format(normalize_text(problem)) or FormatSpec(kind="text")
    return fallback_answer(spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local-LLM reasoning agent (Ollama)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=None, help="override model name (else env AGENT_MODEL / auto)")
    ap.add_argument("--think", default=None, choices=["off", "on", "auto"])
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=None, help="soft seconds per problem")
    ap.add_argument("--host", default=None)
    ap.add_argument("--hints", action="store_true", help="add technique hints to the prompt")
    ap.add_argument("--rewrite", action="store_true", help="enable the optional problem-cleaning step")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if (args.verbose or os.environ.get("AGENT_LOG")) else logging.WARNING,
                        stream=sys.stderr, format="%(message)s")

    try:
        items, is_list = load_input(args.input)
    except Exception as e:  # noqa: BLE001
        log.error("cannot read input: %s", e)
        write_output(args.output, {"id": "", "answer": "0"})
        return 1

    results = []
    for it in items:
        it = it if isinstance(it, dict) else {}
        results.append({"id": it.get("id", ""), "answer": heuristic_fallback(it.get("problem", ""))})
    write_output(args.output, results if is_list else results[0])   # safety net first

    llm = st = None
    try:
        import ollama
        client = ollama.Client(host=args.host or os.environ.get("OLLAMA_HOST"))
        try:
            installed = list_installed(client)
        except Exception as e:  # noqa: BLE001
            log.warning("ollama list failed: %s", e)
            installed = []
        st = build_settings(choose_model(args.model, installed), think=args.think,
                            max_calls=args.max_calls, time_limit=args.time_limit, host=args.host,
                            hints=args.hints)
        llm = LLM(st)
        log.info("model=%s big=%s think=%s max_calls=%s", st.model, st.big, st.think, st.max_calls)
    except Exception as e:  # noqa: BLE001
        log.error("model setup failed, using fallbacks: %s", e)

    if llm is not None:
        for i, it in enumerate(items):
            try:
                res = solve_agent(str((it or {}).get("problem", "")), llm, st, rewrite=args.rewrite)
                results[i]["answer"] = str(res.answer)
            except Exception as e:  # noqa: BLE001
                log.error("problem %s failed: %s", results[i]["id"], e)
            write_output(args.output, results if is_list else results[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
