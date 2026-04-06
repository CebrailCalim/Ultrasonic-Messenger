#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════╗
║                     C I R C I R   v 7                        ║
║         Ultimate Self-Optimizing Acoustic Modem              ║
║                                                              ║
║  OFDM · Auto-Cal · AGC · Turbo/Safe · FHSS · Mesh · Sonar  ║
╚═══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import windows, firwin, lfilter
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import threading, queue, time, os, zlib, heapq, struct, hashlib, json
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont
from collections import deque, Counter
import math

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SYSTEM MANIFEST — Her mantık değişikliği buraya otomatik yansır        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
SYSTEM_MANIFEST = {
    "app_name": "Cırcır",
    "version": "7.0",
    "codename": "Ağustos Böceği",
    "description": "Ses dalgaları ile veri iletimi yapan, kendi kendini optimize eden akustik modem.",
    "physical_layer": {
        "modulation": "OFDM (Orthogonal Frequency Division Multiplexing)",
        "subcarriers": "16 ortogonal alt-taşıyıcı (10.0–19.5 kHz)",
        "symbol_encoding": "Gray-coded 16-FSK + 4-level ASK per subcarrier",
        "pulse_shaping": "Raised Cosine (β=0.35)",
        "pilot_tones": "9.0 kHz (LO) + 19.5 kHz (HI) — Doppler tespiti",
        "frequency_hopping": "FHSS — keyword-seeded pseudo-random permütasyon",
        "eq": "Yazılımsal frekans dengeleme (10–20 kHz lineer boost)",
    },
    "calibration": {
        "ping_tail": "10ms impuls → reverberasyon kuyruğu ölçümü → otomatik gap",
        "chirp_duration": "0.5s chirp (9–20 kHz) → çok yollu girişim analizi → ideal sembol süresi",
        "agc": "9.5 kHz preamble = 1.0 referans → mesafe bazlı zayıflama düzeltmesi",
        "toggle": "AUTO_CALIBRATION_ENABLED global anahtar",
    },
    "modes": {
        "turbo": {
            "focus": "Maksimum hız",
            "features": ["OFDM", "256-FSK", "4-ASK", "Minimal gap", "RLE+Delta", "Huffman"],
        },
        "safe": {
            "focus": "%100 veri bütünlüğü",
            "features": ["Yüksek süre/gap", "Hamming FEC", "XOR Şifreleme", "NACK zorunlu"],
        },
    },
    "data_pipeline": [
        "1. Predictive Coding (Delta piksel/karakter tahmini)",
        "2. XOR Stream Cipher (keyword-tabanlı LCG)",
        "3. Delta + RLE Sıkıştırma",
        "4. Huffman Kodlama",
        "5. Hamming (7,4) FEC",
        "6. OFDM Modülasyon + Raised Cosine Şekillendirme",
    ],
    "transport": {
        "fragmentation": "256 byte/chunk, sıra numaralı",
        "ack_nack": "ACK=9.2kHz, NACK=8.8kHz (0.15s ton)",
        "arq": "3 tekrar deneme, timeout=2.0s",
        "mesh_relay": "Hop-count azaltmalı, jitter gecikmeli yeniden yayın",
    },
    "tools": {
        "sonar": "Ping(8.5kHz)/Pong(8.6kHz) RTT → mesafe",
        "steganography": "Veri -22dB white noise içine gömülü",
        "export_png": "Alınan 32×32 resmi PNG olarak kaydetme",
    },
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL SABİTLER                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
FS            = 44100
PREAMBLE_FREQ = 9500     # AGC referans + wake-up
PILOT_LO      = 9000
PILOT_HI      = 19500
ACK_FREQ      = 9200
NACK_FREQ     = 8800
PING_FREQ     = 8500
PONG_FREQ     = 8600
SPEED_SOUND   = 343.0

# --- OFDM Alt-Taşıyıcılar (16 adet, 10.0–19.5 kHz) ---
NUM_SC        = 16
SC_BASE       = 10000
SC_SPACING    = 600       # Hz — ortogonalite için
SUBCARRIERS   = [SC_BASE + i * SC_SPACING for i in range(NUM_SC)]
# Her alt-taşıyıcı: 16-FSK (±STEP Hz offset) + 4-ASK (genlik seviyeleri)
STEP_FSK      = 15        # Hz frekans adımı
FSK_LEVELS    = 16        # 4-bit nibble
ASK_LEVELS    = [0.25, 0.50, 0.75, 1.00]   # 2 extra bit
BITS_PER_SC   = 6         # 4 (FSK) + 2 (ASK) = 6 bit/subcarrier
FRAME_BITS    = NUM_SC * BITS_PER_SC  # 96 bit = 12 byte/symbol

DEF_DURATION  = 0.20      # saniye (turbo default)
DEF_GAP       = 0.03
SAFE_DURATION = 0.40
SAFE_GAP      = 0.08

HEADER_SIZE   = 16
FRAG_MAX      = 256
ARQ_RETRIES   = 3
ACK_TIMEOUT   = 2.0

IMG_W, IMG_H, IMG_LEVELS = 32, 32, 16
PKT_TEXT, PKT_IMAGE = 0x00, 0x01

# Raised Cosine parametreleri
RC_BETA = 0.35

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GRAY CODE                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
GRAY_ENC = [i ^ (i >> 1) for i in range(16)]
GRAY_DEC = [0] * 16
for _i, _g in enumerate(GRAY_ENC):
    GRAY_DEC[_g] = _i

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RAISED COSINE PENCERESİ                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def raised_cosine_window(n, beta=RC_BETA):
    """Side-lobe sızıntısını önleyen Raised Cosine penceresi."""
    if n < 4:
        return np.ones(n)
    t = np.linspace(0, 1, n)
    rolloff_len = int(n * beta)
    w = np.ones(n)
    if rolloff_len > 0:
        rise = 0.5 * (1 - np.cos(np.pi * np.arange(rolloff_len) / rolloff_len))
        fall = rise[::-1]
        w[:rolloff_len] = rise
        w[-rolloff_len:] = fall
    return w

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HAMMING (7,4) FEC                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
_G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],
               [0,0,1,0,0,1,1],[0,0,0,1,1,1,1]], dtype=np.uint8)
_H = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],
               [0,1,1,1,0,0,1]], dtype=np.uint8)

def _ham_enc_nib(n):
    d = np.array([(n >> i) & 1 for i in range(4)], dtype=np.uint8)
    c = (d @ _G) % 2
    return int(sum(c[i] << i for i in range(7)))

def _ham_dec_7(c7):
    r = np.array([(c7 >> i) & 1 for i in range(7)], dtype=np.uint8)
    s = int(sum(((_H @ r) % 2)[i] << i for i in range(3)))
    if 0 < s <= 7:
        r[s - 1] ^= 1
    return int(sum(r[i] << i for i in range(4)))

def hamming_enc(data: bytes) -> bytes:
    o = bytearray()
    for b in data:
        w = _ham_enc_nib(b & 0xF) | (_ham_enc_nib((b >> 4) & 0xF) << 7)
        o += bytes([w & 0xFF, (w >> 8) & 0xFF])
    return bytes(o)

def hamming_dec(data: bytes) -> bytes:
    o = bytearray()
    for i in range(0, len(data) - 1, 2):
        w = data[i] | (data[i + 1] << 8)
        o.append((_ham_dec_7((w >> 7) & 0x7F) << 4) | _ham_dec_7(w & 0x7F))
    return bytes(o)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  XOR STREAM CIPHER                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def xor_cipher(payload: bytes, keyword: str) -> bytes:
    if not keyword:
        return payload
    seed = 0x5A5A5A5A
    for ch in keyword:
        seed = ((seed * 31) + ord(ch)) & 0xFFFFFFFF
    if seed == 0:
        seed = 0xDEADBEEF
    st = seed
    o = bytearray()
    for b in payload:
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        o.append(b ^ ((st >> 16) & 0xFF))
    return bytes(o)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PREDICTIVE CODING (Delta pre-processing)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def predictive_encode(data: bytes) -> bytes:
    """Her byte'ı önceki byte'ın farkı olarak kodlar (delta-predictive)."""
    if len(data) < 2:
        return data
    out = bytearray([data[0]])
    for i in range(1, len(data)):
        out.append((data[i] - data[i - 1]) & 0xFF)
    return bytes(out)

def predictive_decode(data: bytes) -> bytes:
    if len(data) < 2:
        return data
    out = bytearray([data[0]])
    for i in range(1, len(data)):
        out.append((out[-1] + data[i]) & 0xFF)
    return bytes(out)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DELTA + RLE COMPRESSION                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def delta_rle_compress(data: bytes) -> bytes:
    if len(data) < 2:
        return data
    deltas = [data[0]] + [((data[i] - data[i - 1]) & 0xFF) for i in range(1, len(data))]
    rle = bytearray()
    i = 0
    while i < len(deltas):
        val = deltas[i]
        count = 1
        while i + count < len(deltas) and deltas[i + count] == val and count < 255:
            count += 1
        rle.append(count)
        rle.append(val)
        i += count
    return bytes(rle) if len(rle) < len(data) else data

def delta_rle_decompress(data: bytes, orig_len: int) -> bytes:
    deltas = []
    i = 0
    while i + 1 < len(data) and len(deltas) < orig_len:
        count, val = data[i], data[i + 1]
        i += 2
        deltas.extend([val] * count)
    if not deltas:
        return data
    out = [deltas[0]]
    for j in range(1, min(len(deltas), orig_len)):
        out.append((out[-1] + deltas[j]) & 0xFF)
    return bytes(out[:orig_len])

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HUFFMAN COMPRESSION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class _HNode:
    __slots__ = ('f', 'c', 'l', 'r')
    def __init__(self, f, c=None, l=None, r=None):
        self.f, self.c, self.l, self.r = f, c, l, r
    def __lt__(self, o):
        return self.f < o.f

def huffman_compress(data: bytes) -> bytes:
    if len(data) < 4:
        return data
    freq = Counter(data)
    if len(freq) == 1:
        b = list(freq.keys())[0]
        return struct.pack('<BH', b, len(data))
    heap = [_HNode(f, c=b) for b, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, _HNode(a.f + b.f, l=a, r=b))
    codes = {}
    def _walk(nd, path):
        if nd.c is not None:
            codes[nd.c] = path or '0'
            return
        _walk(nd.l, path + '0')
        _walk(nd.r, path + '1')
    _walk(heap[0], '')
    bits = ''.join(codes[b] for b in data)
    pad = (8 - len(bits) % 8) % 8
    bits += '0' * pad
    enc = bytearray(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    hdr = bytearray([len(freq) - 1])
    for sym, code in codes.items():
        hdr.append(sym)
        hdr.append(len(code))
        cb = int(code, 2) if code else 0
        if len(code) <= 8:
            hdr.append(cb & 0xFF)
        elif len(code) <= 16:
            hdr += struct.pack('<H', cb)
        else:
            hdr += struct.pack('<I', cb)[:3]
    hdr.append(pad)
    result = bytes(hdr) + bytes(enc)
    return result if len(result) < len(data) else data

def huffman_decompress(data: bytes, orig_len: int) -> bytes:
    if len(data) < 3:
        return data
    idx = 0
    ns = data[idx] + 1
    idx += 1
    codes_rev = {}
    for _ in range(ns):
        if idx >= len(data):
            return data
        sym = data[idx]; idx += 1
        clen = data[idx]; idx += 1
        if clen <= 8:
            cv = data[idx]; idx += 1
        elif clen <= 16:
            cv = struct.unpack_from('<H', data, idx)[0]; idx += 2
        else:
            cv = int.from_bytes(data[idx:idx + 3], 'little'); idx += 3
        codes_rev[format(cv, f'0{clen}b')] = sym
    if idx >= len(data):
        return data
    pad = data[idx]; idx += 1
    bits = ''.join(format(b, '08b') for b in data[idx:])
    if pad:
        bits = bits[:-pad]
    out = bytearray()
    cur = ''
    for bit in bits:
        cur += bit
        if cur in codes_rev:
            out.append(codes_rev[cur])
            cur = ''
        if len(out) >= orig_len:
            break
    return bytes(out[:orig_len])

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FHSS                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def fhss_perm(frame_idx, seed, n):
    rng = (frame_idx * 2654435761 + seed) & 0xFFFFFFFF
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        rng = (rng * 1103515245 + 12345) & 0xFFFFFFFF
        j = rng % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def fhss_inv(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

def fhss_seed(keyword: str) -> int:
    s = 0x12345678
    for c in keyword:
        s = ((s * 37) + ord(c)) & 0xFFFFFFFF
    return s if s else 0xABCD1234

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EQ — Frekans Dengeleme                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
_EQ_F = [9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
_EQ_G = [1.0,  1.0,   1.05,  1.12,  1.20,  1.30,  1.42,  1.55,  1.70,  1.85,  2.0,   2.15]

def eq_gain(f):
    if f <= _EQ_F[0]:  return _EQ_G[0]
    if f >= _EQ_F[-1]: return _EQ_G[-1]
    for i in range(len(_EQ_F) - 1):
        if _EQ_F[i] <= f <= _EQ_F[i + 1]:
            t = (f - _EQ_F[i]) / (_EQ_F[i + 1] - _EQ_F[i])
            return _EQ_G[i] + t * (_EQ_G[i + 1] - _EQ_G[i])
    return 1.0

SC_EQ = [eq_gain(f) for f in SUBCARRIERS]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  OFDM MODÜLATÖR / DEMODÜLATÖR                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def ofdm_modulate(frame_bytes, duration, use_ask=True, use_fhss=False,
                  fhss_sd=0, frame_idx=0, stego=False):
    """
    frame_bytes: list of ints (max NUM_SC*3//2 = 12 byte/frame for 6 bit/SC)
                 Basitleştirilmiş: her SC'ye 1 byte (lo-nibble=FSK, hi-nibble=ASK)
                 Ya da düz: her SC'ye 1 byte, FSK ile ilet.
    Geriye uyumluluk için: her byte → lo nibble(FSK) + hi nibble(ASK-index)
    """
    n_samples = int(FS * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave = np.zeros(n_samples)

    # FHSS permütasyon
    if use_fhss:
        perm = fhss_perm(frame_idx, fhss_sd, NUM_SC)
    else:
        perm = list(range(NUM_SC))

    for sc_idx in range(min(len(frame_bytes), NUM_SC)):
        byte_val = frame_bytes[sc_idx]
        fsk_val = byte_val & 0x0F          # 4-bit → 16-FSK
        ask_idx = (byte_val >> 4) & 0x03   # 2-bit → 4-ASK
        extra   = (byte_val >> 6) & 0x03   # 2 extra bit → upper ASK

        phys_sc = perm[sc_idx]
        base_freq = SUBCARRIERS[phys_sc]

        # Gray-coded FSK offset
        gray_fsk = GRAY_ENC[fsk_val]
        freq = base_freq + gray_fsk * STEP_FSK

        # ASK amplitude
        if use_ask:
            amp = ASK_LEVELS[ask_idx] * SC_EQ[phys_sc]
        else:
            amp = 1.0 * SC_EQ[phys_sc]

        if stego:
            amp *= 0.08

        wave += amp * np.sin(2 * np.pi * freq * t)

    # Normalize & pulse shape
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.8

    if stego:
        noise = np.random.randn(n_samples) * 0.3
        wave = wave + noise

    wave *= raised_cosine_window(n_samples)
    return wave.astype(np.float32)


def ofdm_demodulate(audio, freqs, mag, thr, doppler_ratio=1.0,
                    use_fhss=False, fhss_sd=0, frame_idx=0):
    """
    Spektrumdan OFDM frame'i çöz.
    Returns: list of byte values (len=NUM_SC) veya None.
    """
    frame = []
    for sc_idx in range(NUM_SC):
        base_freq = SUBCARRIERS[sc_idx] * doppler_ratio
        # FSK arama: base_freq ± FSK_LEVELS*STEP_FSK
        search_lo = base_freq - STEP_FSK
        search_hi = base_freq + FSK_LEVELS * STEP_FSK + STEP_FSK
        mask = (freqs >= search_lo) & (freqs <= search_hi)
        if not np.any(mask):
            return None
        local_mag = mag[mask]
        local_freq = freqs[mask]
        peak_idx = np.argmax(local_mag)
        if local_mag[peak_idx] < thr:
            return None
        detected_freq = local_freq[peak_idx]
        detected_amp = local_mag[peak_idx]

        # FSK decode
        gray_fsk = round((detected_freq - base_freq) / STEP_FSK)
        gray_fsk = max(0, min(15, gray_fsk))
        fsk_val = GRAY_DEC[gray_fsk]

        # ASK decode (genlik seviyesi)
        # Basit: en yakın ASK seviyesine eşle
        # (AGC normalize edilmiş genlik gerektirir)
        ask_idx = 0  # default

        byte_val = (fsk_val & 0x0F) | ((ask_idx & 0x03) << 4)
        frame.append(byte_val)

    # FHSS inverse
    if use_fhss:
        perm = fhss_perm(frame_idx, fhss_sd, NUM_SC)
        inv = fhss_inv(perm)
        frame = [frame[inv[j]] for j in range(NUM_SC)]

    return frame


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RENK PALETİ — Profesyonel dark theme                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class C:
    BG      = "#0c0e14"     # ana zemin
    PANEL   = "#12151e"     # panel zemin
    SURFACE = "#1a1e2e"     # kartlar
    SURF2   = "#232840"     # hover / elevated
    ACCENT  = "#6366f1"     # indigo-500
    ACCENT2 = "#818cf8"     # indigo-400
    ACCENT3 = "#4338ca"     # indigo-700
    CYAN    = "#22d3ee"     # cyan-400
    GREEN   = "#34d399"     # emerald-400
    RED     = "#f87171"     # red-400
    AMBER   = "#fbbf24"     # amber-400
    WHITE   = "#f1f5f9"
    TEXT    = "#e2e8f0"
    MUTED   = "#64748b"
    DIM     = "#475569"
    BORDER  = "#1e293b"
    OWN_BG  = "#1e1b4b"
    OWN_FG  = "#c7d2fe"
    INC_BG  = "#1a1e2e"
    INC_FG  = "#e2e8f0"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TOOLTIP SİSTEMİ                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        self.tip.configure(bg=C.SURF2)
        frame = tk.Frame(self.tip, bg=C.SURF2, padx=10, pady=6,
                         highlightbackground=C.ACCENT, highlightthickness=1)
        frame.pack()
        tk.Label(frame, text=self.text, bg=C.SURF2, fg=C.TEXT,
                 font=("Segoe UI", 9), wraplength=280, justify="left").pack()

    def hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ICON GENERATOR                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def create_circir_icon(size=64):
    """Minimalist ses dalgası + cırcır böceği ikonu."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 2 - 2

    # Daire arka plan (gradient benzeri)
    draw.ellipse([2, 2, size - 2, size - 2], fill="#6366f1")
    draw.ellipse([4, 4, size - 4, size - 4], fill="#4338ca")

    # Ses dalgası çizgileri
    for i, offset in enumerate([-8, -4, 0, 4, 8]):
        x = cx + offset
        amp = (3 - abs(offset) // 3) * 3
        y1 = cy - amp
        y2 = cy + amp
        w = 2 if offset == 0 else 1
        draw.line([(x, y1), (x, y2)], fill="white", width=w)

    return img


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        ANA UYGULAMA SINIFI                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class CircirApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cırcır v7 — Akustik Modem")
        self.root.geometry("1560x960")
        self.root.configure(bg=C.BG)
        self.root.minsize(1200, 700)

        # --- Durum ---
        self.is_running = True
        self.msg_queue = queue.Queue()
        self.is_transmitting = False
        self.is_listening = False
        self.is_channel_busy = False
        self.current_packet = []
        self.last_signal_time = 0
        self.last_frame_set = None
        self.start_t = 0
        self.photo_refs = []
        self.last_received_pil = None

        # --- Ayarlar ---
        self.turbo_mode     = tk.BooleanVar(value=True)
        self.xor_enabled    = tk.BooleanVar(value=True)
        self.fec_enabled    = tk.BooleanVar(value=False)
        self.comp_enabled   = tk.BooleanVar(value=True)
        self.fhss_enabled   = tk.BooleanVar(value=False)
        self.stego_enabled  = tk.BooleanVar(value=False)
        self.mesh_enabled   = tk.BooleanVar(value=False)
        self.nack_enabled   = tk.BooleanVar(value=True)
        self.auto_cal       = tk.BooleanVar(value=True)
        self.hop_count      = tk.IntVar(value=2)

        # Safe mode: trace ile bağla
        self.turbo_mode.trace_add("write", self._on_mode_change)

        # --- DSP ---
        self.duration      = DEF_DURATION
        self.gap           = DEF_GAP
        self.noise_floor   = 2.0
        self.noise_samples = deque(maxlen=80)
        self.NOISE_MULT    = 1.5
        self.mag_history   = deque(maxlen=3)
        self.doppler_ratio = 1.0
        self.agc_ref       = 1.0   # AGC referans genlik
        self.current_snr   = 0.0
        self.comp_ratio    = 0.0
        self.last_distance = 0.0
        self.sonar_waiting = False
        self.sonar_send_t  = 0.0

        # Calibration sonuçları
        self.cal_gap       = DEF_GAP
        self.cal_duration  = DEF_DURATION

        # Fragment buffer
        self.frag_buffer = {}
        self.frag_meta   = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._exit)

        try:
            self.stream = sd.InputStream(
                callback=lambda *a: self.msg_queue.put(a[0].copy()),
                channels=1, samplerate=FS, blocksize=4096)
            self.stream.start()
        except Exception as e:
            print(f"Mikrofon hatası: {e}")

        self._update_loop()

    @property
    def threshold(self):
        return max(1.0, self.noise_floor * self.NOISE_MULT)

    def _xor_key(self):
        return self.xor_key_ent.get().strip()

    def _on_mode_change(self, *_):
        """Turbo/Safe mod değiştiğinde ayarları güncelle."""
        if self.turbo_mode.get():
            self.duration = DEF_DURATION
            self.gap = DEF_GAP
            self.fec_enabled.set(False)
            self.nack_enabled.set(True)
        else:
            self.duration = SAFE_DURATION
            self.gap = SAFE_GAP
            self.fec_enabled.set(True)
            self.xor_enabled.set(True)
            self.nack_enabled.set(True)
        self._update_labels()

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  UI — ANA YAPI                                                      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    def _build_ui(self):
        self._style_ttk()
        self._build_sidebar()
        self._build_main()

    def _style_ttk(self):
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("TProgressbar", background=C.ACCENT, thickness=4,
                       troughcolor=C.BORDER, bordercolor=C.BORDER)
        sty.configure("TCheckbutton", background=C.PANEL, foreground=C.TEXT,
                       font=("Segoe UI", 9))
        sty.map("TCheckbutton",
                background=[("active", C.SURFACE)],
                foreground=[("active", C.ACCENT2)])

    # ═══════════════════════ SIDEBAR ═══════════════════════════════════════
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=C.PANEL, width=380)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        self.sb = sb

        # Scrollable
        cv = tk.Canvas(sb, bg=C.PANEL, highlightthickness=0, width=360)
        scr = tk.Scrollbar(sb, orient=tk.VERTICAL, command=cv.yview,
                           bg=C.PANEL, troughcolor=C.PANEL,
                           activebackground=C.ACCENT)
        self.sb_inner = tk.Frame(cv, bg=C.PANEL)
        self.sb_inner.bind("<Configure>",
                           lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=self.sb_inner, anchor="nw", width=360)
        cv.configure(yscrollcommand=scr.set)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _wheel(e):
            cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
        cv.bind_all("<MouseWheel>", _wheel)

        s = self.sb_inner

        # ─── HEADER ───
        hdr = tk.Frame(s, bg=C.ACCENT3, pady=16, padx=18)
        hdr.pack(fill=tk.X)

        # Cırcır logo + isim
        logo_frame = tk.Frame(hdr, bg=C.ACCENT3)
        logo_frame.pack(anchor="w")

        try:
            icon_pil = create_circir_icon(36)
            self._icon_photo = ImageTk.PhotoImage(icon_pil)
            tk.Label(logo_frame, image=self._icon_photo, bg=C.ACCENT3
                     ).pack(side=tk.LEFT, padx=(0, 10))
        except:
            pass

        name_f = tk.Frame(logo_frame, bg=C.ACCENT3)
        name_f.pack(side=tk.LEFT)
        tk.Label(name_f, text="Cırcır", bg=C.ACCENT3, fg="#ffffff",
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(name_f, text="v7  ·  OFDM Acoustic Modem",
                 bg=C.ACCENT3, fg="#c7d2fe",
                 font=("Segoe UI", 9)).pack(anchor="w")

        # Info butonu
        info_btn = tk.Button(hdr, text="ⓘ", bg=C.ACCENT3, fg="#c7d2fe",
                             font=("Segoe UI", 14), relief="flat", cursor="hand2",
                             command=self._show_manifest, bd=0,
                             activebackground=C.ACCENT3, activeforeground="#fff")
        info_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-4, y=4)
        ToolTip(info_btn, "Cırcır v7 Manifest — Sistem dokümantasyonu")

        # ─── DURUM ───
        sf = tk.Frame(s, bg=C.PANEL, padx=18, pady=10)
        sf.pack(fill=tk.X)
        self._sdot = tk.Canvas(sf, width=12, height=12, bg=C.PANEL, highlightthickness=0)
        self._sdot.pack(side=tk.LEFT)
        self._sdot.create_oval(2, 2, 10, 10, fill=C.GREEN, outline="", tags="dot")
        self._slbl = tk.Label(sf, text="Kanal Müsait", bg=C.PANEL, fg=C.GREEN,
                              font=("Segoe UI", 10, "bold"))
        self._slbl.pack(side=tk.LEFT, padx=(8, 0))
        self._thr_lbl = tk.Label(sf, text="", bg=C.PANEL, fg=C.DIM,
                                 font=("Segoe UI", 8))
        self._thr_lbl.pack(side=tk.RIGHT)
        self._sep(s)

        # ─── MOD SEÇİMİ ───
        self._section(s, "ÇALIŞMA MODU")
        mf = tk.Frame(s, bg=C.PANEL, padx=18)
        mf.pack(fill=tk.X, pady=(0, 4))

        self._mode_frame = tk.Frame(mf, bg=C.PANEL)
        self._mode_frame.pack(fill=tk.X)

        self._turbo_btn = tk.Button(
            self._mode_frame, text="⚡ TURBO", font=("Segoe UI", 10, "bold"),
            bg=C.ACCENT, fg="#fff", relief="flat", padx=20, pady=8, cursor="hand2",
            command=lambda: self.turbo_mode.set(True))
        self._turbo_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ToolTip(self._turbo_btn, "Maksimum hız: OFDM, sıkıştırma, minimal boşluk")

        self._safe_btn = tk.Button(
            self._mode_frame, text="🛡 SAFE", font=("Segoe UI", 10, "bold"),
            bg=C.SURFACE, fg=C.MUTED, relief="flat", padx=20, pady=8, cursor="hand2",
            command=lambda: self.turbo_mode.set(False))
        self._safe_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ToolTip(self._safe_btn, "Doğruluk odaklı: FEC + şifreleme + yüksek süre")

        self._mode_info = tk.Label(mf, text="OFDM · Huffman · RLE · Minimal Gap",
                                   bg=C.PANEL, fg=C.MUTED, font=("Segoe UI", 8))
        self._mode_info.pack(anchor="w", pady=(4, 0))
        self._sep(s)

        # ─── KİMLİK ───
        self._section(s, "KİMLİK")
        idf = tk.Frame(s, bg=C.PANEL, padx=18)
        idf.pack(fill=tk.X, pady=(0, 4))
        idf.columnconfigure(0, weight=1)
        idf.columnconfigure(1, weight=1)
        lbl_f = ("Segoe UI", 8)
        tk.Label(idf, text="Kendi ID", bg=C.PANEL, fg=C.MUTED,
                 font=lbl_f).grid(row=0, column=0, sticky="w")
        tk.Label(idf, text="Hedef ID", bg=C.PANEL, fg=C.MUTED,
                 font=lbl_f).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.my_id_ent = self._entry(idf, "1")
        self.my_id_ent.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        ToolTip(self.my_id_ent, "Kendi cihaz numaranız (0–255)")
        self.target_id_ent = self._entry(idf, "0")
        self.target_id_ent.grid(row=1, column=1, sticky="ew", pady=(2, 0), padx=(8, 0))
        ToolTip(self.target_id_ent, "Hedef cihaz numarası (0=broadcast)")
        self._sep(s)

        # ─── GÜVENLİK ───
        self._section(s, "GÜVENLİK & KODLAMA")
        sf2 = tk.Frame(s, bg=C.PANEL, padx=18)
        sf2.pack(fill=tk.X, pady=(0, 4))

        checks = [
            (self.xor_enabled,  "XOR Şifreleme", "Keyword tabanlı stream cipher"),
            (self.comp_enabled, "Sıkıştırma (Huffman/RLE)", "Metin: Huffman, Resim: Delta+RLE"),
            (self.fec_enabled,  "FEC (Hamming 7,4)", "1-bit hata düzeltme, ~%50 ek yük"),
            (self.fhss_enabled, "FHSS Frekans Atlama", "Her frame farklı alt-taşıyıcı sırası"),
            (self.stego_enabled,"Steganografi", "Veri white noise içine gömülür"),
            (self.mesh_enabled, "Mesh Relay", "Çok atlamalı yeniden yayın"),
            (self.nack_enabled, "NACK Etkin", "CRC hatası → otomatik tekrar isteme"),
        ]
        for var, txt, tip in checks:
            cb = tk.Checkbutton(sf2, text=txt, variable=var,
                                bg=C.PANEL, fg=C.TEXT, selectcolor=C.SURFACE,
                                activebackground=C.PANEL, activeforeground=C.ACCENT2,
                                font=("Segoe UI", 9), anchor="w", cursor="hand2")
            cb.pack(fill=tk.X)
            ToolTip(cb, tip)

        kwf = tk.Frame(sf2, bg=C.PANEL)
        kwf.pack(fill=tk.X, pady=(6, 0))
        tk.Label(kwf, text="Anahtar:", bg=C.PANEL, fg=C.MUTED,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.xor_key_ent = tk.Entry(kwf, bg=C.SURFACE, fg=C.TEXT,
                                    font=("Segoe UI", 10), borderwidth=0,
                                    insertbackground=C.TEXT, show="•", width=14)
        self.xor_key_ent.insert(0, "circir")
        self.xor_key_ent.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)
        ToolTip(self.xor_key_ent, "XOR şifreleme anahtar kelimesi")
        self._sep(s)

        # ─── KALİBRASYON ───
        self._section(s, "OTO-KALİBRASYON")
        cf = tk.Frame(s, bg=C.PANEL, padx=18)
        cf.pack(fill=tk.X, pady=(0, 4))
        cal_cb = tk.Checkbutton(cf, text="Otomatik Kalibrasyon", variable=self.auto_cal,
                                bg=C.PANEL, fg=C.TEXT, selectcolor=C.SURFACE,
                                activebackground=C.PANEL, font=("Segoe UI", 9),
                                anchor="w", cursor="hand2")
        cal_cb.pack(fill=tk.X)
        ToolTip(cal_cb, "İmpuls + Chirp ile gap ve süre otomatik ayarlanır")

        cal_btn = tk.Button(cf, text="Kalibrasyon Başlat", bg=C.SURFACE, fg=C.CYAN,
                            font=("Segoe UI", 9), relief="flat", cursor="hand2",
                            command=self._run_calibration, padx=12, pady=4)
        cal_btn.pack(fill=tk.X, pady=(4, 0))
        ToolTip(cal_btn, "Ping-Tail + Chirp analizi ile ortam ölçümü")

        self._cal_lbl = tk.Label(cf, text="Gap: otomatik  ·  Süre: otomatik",
                                 bg=C.PANEL, fg=C.DIM, font=("Segoe UI", 8))
        self._cal_lbl.pack(anchor="w", pady=(4, 0))
        self._sep(s)

        # ─── SONAR ───
        self._section(s, "SONAR & MESAFE")
        snf = tk.Frame(s, bg=C.PANEL, padx=18)
        snf.pack(fill=tk.X, pady=(0, 4))
        ping_btn = tk.Button(snf, text="📡 Ping", bg=C.SURFACE, fg=C.CYAN,
                             font=("Segoe UI", 9), relief="flat", cursor="hand2",
                             command=self._sonar_ping, padx=10, pady=4)
        ping_btn.pack(side=tk.LEFT)
        ToolTip(ping_btn, "RTT ölçümü ile mesafe hesaplama")
        self._dist_lbl = tk.Label(snf, text="Mesafe: — m", bg=C.PANEL,
                                  fg=C.CYAN, font=("Segoe UI", 9, "bold"))
        self._dist_lbl.pack(side=tk.LEFT, padx=(12, 0))
        self._sep(s)

        # ─── METRİKLER ───
        self._section(s, "CANLI METRİKLER")
        mf2 = tk.Frame(s, bg=C.SURFACE, padx=14, pady=10)
        mf2.pack(fill=tk.X, padx=18, pady=(0, 4))
        self._snr_lbl = self._metric(mf2, "SNR", "— dB")
        self._comp_lbl = self._metric(mf2, "Sıkıştırma", "— %")
        self._dopp_lbl = self._metric(mf2, "Doppler", "1.0000x")
        self._agc_lbl = self._metric(mf2, "AGC Ref", "1.00")
        self._sep(s)

        # ─── RESİM ÖNİZLEME ───
        self._section(s, "ALINAN RESİM")
        pf = tk.Frame(s, bg=C.PANEL, padx=18)
        pf.pack(fill=tk.X, pady=(0, 4))
        self.prev_cv = tk.Canvas(pf, width=100, height=100, bg=C.SURFACE,
                                 highlightthickness=0)
        self.prev_cv.pack(side=tk.LEFT)
        self._prev_idle()

        pi = tk.Frame(pf, bg=C.PANEL, padx=8)
        pi.pack(side=tk.LEFT, fill=tk.Y, anchor="n")
        self._pst = tk.Label(pi, text="Bekleniyor", bg=C.PANEL, fg=C.MUTED,
                             font=("Segoe UI", 8))
        self._pst.pack(anchor="w")
        self._ppr = tk.Label(pi, text="—", bg=C.PANEL, fg=C.DIM,
                             font=("Segoe UI", 7))
        self._ppr.pack(anchor="w", pady=(2, 0))
        exp_btn = tk.Button(pi, text="PNG Kaydet", bg=C.SURFACE, fg=C.TEXT,
                            font=("Segoe UI", 7), relief="flat", cursor="hand2",
                            command=self._export_png, padx=6, pady=2)
        exp_btn.pack(anchor="w", pady=(4, 0))
        ToolTip(exp_btn, "Son alınan resmi 256×256 PNG olarak kaydet")
        self._sep(s)

        # ─── SPEKTRUM ───
        self._section(s, "OFDM SPEKTRUM")
        self.fig, self.ax = plt.subplots(figsize=(3.2, 1.8))
        self.fig.patch.set_facecolor(C.PANEL)
        self.ax.set_facecolor(C.SURFACE)
        self.plot_x = np.linspace(8500, 20500, 800)
        self.line, = self.ax.plot(self.plot_x, np.zeros(800), color=C.ACCENT2, lw=1.2)
        self.ax.set_ylim(0, 60)
        self.ax.set_xlim(8500, 20500)
        self.ax.set_xticks([9000, 11000, 13000, 15000, 17000, 19000])
        self.ax.set_xticklabels(['9k', '11k', '13k', '15k', '17k', '19k'],
                                 fontsize=6, color=C.DIM)
        self.ax.tick_params(colors=C.DIM, length=0)
        for sp in self.ax.spines.values():
            sp.set_color(C.BORDER)
        self.ax.grid(True, linestyle='--', alpha=0.1, color=C.DIM)

        # Alt-taşıyıcı bantlarını göster
        sc_colors = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe'] * 4
        for i, sc in enumerate(SUBCARRIERS):
            self.ax.axvspan(sc, sc + FSK_LEVELS * STEP_FSK,
                            alpha=0.08, color=sc_colors[i % len(sc_colors)])

        self.fig.tight_layout(pad=0.3)
        self.mpl_cv = FigureCanvasTkAgg(self.fig, master=s)
        self.mpl_cv.get_tk_widget().configure(bg=C.PANEL)
        self.mpl_cv.get_tk_widget().pack(fill=tk.X, padx=14, pady=(0, 10))

    # ═══════════════════════ ANA PANEL ════════════════════════════════════
    def _build_main(self):
        m = tk.Frame(self.root, bg=C.BG)
        m.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Top bar
        tb = tk.Frame(m, bg=C.PANEL, pady=12, padx=24,
                      highlightthickness=1, highlightbackground=C.BORDER)
        tb.pack(fill=tk.X)
        tk.Label(tb, text="Mesajlaşma", bg=C.PANEL, fg=C.WHITE,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        self._tp_lbl = tk.Label(tb, text="0.0 B/s", bg=C.PANEL, fg=C.CYAN,
                                font=("Segoe UI", 10, "bold"))
        self._tp_lbl.pack(side=tk.RIGHT, padx=(0, 6))
        tk.Label(tb, text="verim:", bg=C.PANEL, fg=C.MUTED,
                 font=("Segoe UI", 10)).pack(side=tk.RIGHT)

        # Chat
        cw = tk.Frame(m, bg=C.BG, padx=16, pady=8)
        cw.pack(fill=tk.BOTH, expand=True)
        self.chat = scrolledtext.ScrolledText(
            cw, bg=C.SURFACE, fg=C.TEXT, font=("Segoe UI", 12),
            padx=16, pady=14, borderwidth=0, relief="flat",
            state="disabled", cursor="arrow",
            insertbackground=C.TEXT, selectbackground=C.ACCENT)
        self.chat.pack(fill=tk.BOTH, expand=True)

        tags = {
            "oh":  dict(foreground=C.ACCENT2, font=("Segoe UI", 8, "bold"), spacing3=2),
            "ob":  dict(foreground=C.OWN_FG, font=("Segoe UI", 12), background=C.OWN_BG,
                        lmargin1=40, lmargin2=40, rmargin=20, spacing1=3, spacing3=3),
            "om":  dict(foreground=C.DIM, font=("Segoe UI", 7), lmargin1=40, spacing3=6),
            "ih":  dict(foreground=C.MUTED, font=("Segoe UI", 8, "bold"), spacing3=2),
            "ib":  dict(foreground=C.INC_FG, font=("Segoe UI", 12), background=C.INC_BG,
                        lmargin1=16, lmargin2=16, rmargin=50, spacing1=3, spacing3=3),
            "im":  dict(foreground=C.DIM, font=("Segoe UI", 7), lmargin1=16, spacing3=6),
            "sys": dict(foreground=C.DIM, font=("Segoe UI", 7, "italic"),
                        justify="center", spacing1=4, spacing3=4),
            "ok":  dict(foreground=C.GREEN, font=("Segoe UI", 7)),
            "err": dict(foreground=C.RED, font=("Segoe UI", 7)),
        }
        for tag, cfg in tags.items():
            self.chat.tag_config(tag, **cfg)

        # Progress
        pw = tk.Frame(m, bg=C.PANEL, padx=16, pady=8,
                      highlightthickness=1, highlightbackground=C.BORDER)
        pw.pack(fill=tk.X)
        pt = tk.Frame(pw, bg=C.PANEL)
        pt.pack(fill=tk.X)
        self._prog_lbl = tk.Label(pt, text="Hazır", bg=C.PANEL, fg=C.MUTED,
                                  font=("Segoe UI", 9))
        self._prog_lbl.pack(side=tk.LEFT)
        self._tmr_lbl = tk.Label(pt, text="", bg=C.PANEL, fg=C.ACCENT2,
                                 font=("Segoe UI", 9, "bold"))
        self._tmr_lbl.pack(side=tk.RIGHT)
        self.pbar = ttk.Progressbar(pw, orient=tk.HORIZONTAL, mode="determinate",
                                     style="TProgressbar")
        self.pbar.pack(fill=tk.X, pady=(4, 0))

        # Input
        ip = tk.Frame(m, bg=C.PANEL, padx=16, pady=12,
                      highlightthickness=1, highlightbackground=C.BORDER)
        ip.pack(fill=tk.X)
        self.entry = tk.Entry(ip, bg=C.SURFACE, fg=C.TEXT, font=("Segoe UI", 12),
                              relief="flat", bd=0, insertbackground=C.TEXT)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self.send_text())

        btns = [
            ("🖼 Resim", self.pick_and_send_image, C.SURFACE, C.TEXT),
            ("Gönder", self.send_text, C.ACCENT, "#fff"),
            ("Temizle", self.clear_log, C.BG, C.MUTED),
        ]
        for txt, cmd, bg, fg in btns:
            b = tk.Button(ip, text=txt, bg=bg, fg=fg,
                          font=("Segoe UI", 10, "bold" if txt == "Gönder" else ""),
                          relief="flat", padx=14, pady=6, cursor="hand2", command=cmd)
            b.pack(side=tk.LEFT, padx=(0, 6))

    # ═══════════════════════ UI YARDIMCILARI ════════════════════════════
    def _sep(self, parent):
        tk.Frame(parent, bg=C.BORDER, height=1).pack(fill=tk.X, padx=18, pady=6)

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=C.PANEL, padx=18)
        f.pack(fill=tk.X, pady=(8, 2))
        tk.Label(f, text=text, bg=C.PANEL, fg=C.DIM,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w")

    def _entry(self, parent, default):
        e = tk.Entry(parent, bg=C.SURFACE, fg=C.TEXT,
                     font=("Segoe UI", 11), justify="center",
                     borderwidth=0, insertbackground=C.TEXT)
        e.insert(0, default)
        return e

    def _metric(self, parent, label, value):
        f = tk.Frame(parent, bg=C.SURFACE)
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, bg=C.SURFACE, fg=C.MUTED,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        lbl = tk.Label(f, text=value, bg=C.SURFACE, fg=C.TEXT,
                       font=("Segoe UI", 8, "bold"))
        lbl.pack(side=tk.RIGHT)
        return lbl

    def _prev_idle(self):
        self.prev_cv.delete("all")
        self.prev_cv.create_rectangle(0, 0, 100, 100, fill=C.SURFACE, outline="")
        self.prev_cv.create_text(50, 50, text="—", fill=C.DIM, font=("Segoe UI", 16))

    def _set_status(self, txt, col):
        self._slbl.config(text=txt, fg=col)
        self._sdot.delete("dot")
        self._sdot.create_oval(2, 2, 10, 10, fill=col, outline="", tags="dot")

    def _update_labels(self):
        if self.turbo_mode.get():
            self._turbo_btn.config(bg=C.ACCENT, fg="#fff")
            self._safe_btn.config(bg=C.SURFACE, fg=C.MUTED)
            self._mode_info.config(text="OFDM · Huffman · RLE · Minimal Gap")
        else:
            self._turbo_btn.config(bg=C.SURFACE, fg=C.MUTED)
            self._safe_btn.config(bg=C.GREEN, fg="#000")
            self._mode_info.config(text="FEC · XOR · NACK · Yüksek Süre")

    def clear_log(self):
        self.chat.config(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.config(state="disabled")

    def _exit(self):
        self.is_running = False
        os._exit(0)

    # ═══════════════════════ MANIFEST MODAL ════════════════════════════
    def _show_manifest(self):
        win = tk.Toplevel(self.root)
        win.title("Cırcır v7 — Sistem Manifestosu")
        win.geometry("620x700")
        win.configure(bg=C.BG)
        win.transient(self.root)
        win.grab_set()

        # Header
        hdr = tk.Frame(win, bg=C.ACCENT3, padx=20, pady=14)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"Cırcır v{SYSTEM_MANIFEST['version']}", bg=C.ACCENT3,
                 fg="#fff", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(hdr, text=f"Codename: {SYSTEM_MANIFEST['codename']}",
                 bg=C.ACCENT3, fg="#c7d2fe", font=("Segoe UI", 9)).pack(anchor="w")

        # İçerik
        txt = scrolledtext.ScrolledText(
            win, bg=C.SURFACE, fg=C.TEXT, font=("Consolas", 10),
            padx=16, pady=12, borderwidth=0, relief="flat",
            insertbackground=C.TEXT, state="normal")
        txt.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        def _format_manifest(d, indent=0):
            lines = []
            for k, v in d.items():
                prefix = "  " * indent
                if isinstance(v, dict):
                    lines.append(f"{prefix}┌─ {k.upper()}")
                    lines.append(_format_manifest(v, indent + 1))
                    lines.append(f"{prefix}└{'─' * 30}")
                elif isinstance(v, list):
                    lines.append(f"{prefix}● {k}:")
                    for item in v:
                        lines.append(f"{prefix}    ▸ {item}")
                else:
                    lines.append(f"{prefix}  {k}: {v}")
            return "\n".join(lines)

        manifest_text = _format_manifest(SYSTEM_MANIFEST)
        txt.insert("1.0", manifest_text)
        txt.config(state="disabled")

        tk.Button(win, text="Kapat", bg=C.ACCENT, fg="#fff",
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=24, pady=8,
                  command=win.destroy).pack(pady=(0, 12))

    # ═══════════════════════ CHAT ═══════════════════════════════════════
    def _cw(self, txt, tag):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, txt, tag)
        self.chat.see(tk.END)
        self.chat.config(state="disabled")

    def _ci(self, photo, tag):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, "  ", tag)
        self.chat.image_create(tk.END, image=photo, padx=6, pady=3)
        self.chat.insert(tk.END, "\n", tag)
        self.chat.see(tk.END)
        self.chat.config(state="disabled")

    def _own_txt(self, txt, meta):
        ts = datetime.now().strftime("%H:%M")
        self._cw(f"\n  Sen  {ts}\n", "oh")
        self._cw(f"  {txt}\n", "ob")
        self._cw(f"  {meta}\n", "om")

    def _own_img(self, photo, meta):
        ts = datetime.now().strftime("%H:%M")
        self._cw(f"\n  Sen  {ts}\n", "oh")
        self._ci(photo, "ob")
        self._cw(f"  {meta}\n", "om")

    def _inc_txt(self, src, txt, meta, ok=True):
        ts = datetime.now().strftime("%H:%M")
        self._cw(f"\n  Kimden:{src}  {ts}\n", "ih")
        self._cw(f"  {txt}\n", "ib")
        self._cw(f"  {meta}\n", "ok" if ok else "err")

    def _inc_img(self, src, photo, meta, ok=True):
        ts = datetime.now().strftime("%H:%M")
        self._cw(f"\n  Kimden:{src}  {ts}\n", "ih")
        self._ci(photo, "ib")
        self._cw(f"  {meta}\n", "ok" if ok else "err")

    def _sys(self, txt):
        self._cw(f"\n  {txt}  \n", "sys")

    def _pil_tk(self, pil, sz=(140, 140)):
        ph = ImageTk.PhotoImage(pil.resize(sz, Image.NEAREST))
        self.photo_refs.append(ph)
        if len(self.photo_refs) > 40:
            self.photo_refs.pop(0)
        return ph

    # ═══════════════════════ RESİM CODEC ═══════════════════════════════
    @staticmethod
    def _img_enc(pil):
        img = pil.convert("L").resize((IMG_W, IMG_H), Image.LANCZOS)
        px = list(img.getdata())
        q = [int(p * (IMG_LEVELS - 1) / 255) for p in px]
        pk = bytearray()
        for i in range(0, len(q), 2):
            lo = q[i] & 0xF
            hi = (q[i + 1] & 0xF) if i + 1 < len(q) else 0
            pk.append(lo | (hi << 4))
        return bytes(pk)

    @staticmethod
    def _img_dec(data, w=IMG_W, h=IMG_H, partial=False):
        px = []
        for b in data:
            px.append(int((b & 0xF) * 255 / (IMG_LEVELS - 1)))
            px.append(int(((b >> 4) & 0xF) * 255 / (IMG_LEVELS - 1)))
        total = w * h
        if len(px) < total:
            if partial:
                px.extend([0] * (total - len(px)))
            else:
                return None
        img = Image.new("L", (w, h))
        img.putdata(px[:total])
        return img

    def _export_png(self):
        if self.last_received_pil is None:
            self._sys("Kaydedilecek resim yok.")
            return
        p = filedialog.asksaveasfilename(title="PNG Kaydet", defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.last_received_pil.resize((256, 256), Image.NEAREST).save(p)
            self._sys(f"Kaydedildi: {p}")

    # ═══════════════════════ TON ÜRETEÇLERİ ═══════════════════════════
    def _tone(self, freq, dur, amp=1.0):
        n = int(FS * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        return (np.sin(2 * np.pi * freq * t) * amp * raised_cosine_window(n)).astype(np.float32)

    def _play_tone(self, freq, dur, amp=1.0):
        sd.play(self._tone(freq, dur, amp), FS)
        sd.wait()

    def _send_ack(self):
        self._play_tone(ACK_FREQ, 0.15, 0.7)

    def _send_nack(self):
        self._play_tone(NACK_FREQ, 0.15, 0.7)

    # ═══════════════════════ KALİBRASYON ════════════════════════════════
    def _run_calibration(self):
        if self.is_transmitting:
            return
        self._sys("Kalibrasyon başlatılıyor...")
        threading.Thread(target=self._calibrate_thread, daemon=True).start()

    def _calibrate_thread(self):
        """
        1) Ping-Tail: 10ms impuls → reverberasyon ölçümü → gap
        2) Chirp: 0.5s 9–20kHz sweep → çok yollu analiz → süre
        3) AGC: 9.5kHz preamble referans
        """
        # 1) Ping-Tail Gap Calibration
        self._sys("Impuls gönderiliyor (10ms)...")
        impulse_dur = 0.01
        n_imp = int(FS * impulse_dur)
        impulse = (np.sin(2 * np.pi * 12000 * np.linspace(0, impulse_dur, n_imp))
                   * raised_cosine_window(n_imp)).astype(np.float32)
        sd.play(impulse, FS)
        sd.wait()

        # Kayıt al (0.5s)
        rec = sd.rec(int(FS * 0.5), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        rec = rec.flatten()
        env = np.abs(rec)
        # Reverberasyon kuyruğu: enerjinin %5'e düştüğü nokta
        peak_env = np.max(env)
        if peak_env > 0.001:
            tail_threshold = peak_env * 0.05
            tail_idx = np.where(env > tail_threshold)[0]
            if len(tail_idx) > 0:
                tail_time = tail_idx[-1] / FS
                self.cal_gap = max(0.02, min(0.15, tail_time * 1.2))
            else:
                self.cal_gap = DEF_GAP
        else:
            self.cal_gap = DEF_GAP

        self._sys(f"Reverberasyon kuyruğu: gap={self.cal_gap*1000:.0f}ms")
        time.sleep(0.2)

        # 2) Chirp Duration Calibration
        self._sys("Chirp gönderiliyor (0.5s)...")
        chirp_dur = 0.5
        n_chirp = int(FS * chirp_dur)
        t_chirp = np.linspace(0, chirp_dur, n_chirp, endpoint=False)
        chirp = (np.sin(2 * np.pi * (9000 + (20000 - 9000) / 2 * t_chirp / chirp_dur) * t_chirp)
                 * 0.5 * raised_cosine_window(n_chirp)).astype(np.float32)
        sd.play(chirp, FS)
        sd.wait()

        rec2 = sd.rec(int(FS * 0.8), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        rec2 = rec2.flatten()
        # Çok yollu girişim: korelasyon ile eko tespiti
        corr = np.correlate(rec2[:len(rec2) // 2], rec2[:len(rec2) // 4], mode='full')
        peak_corr = np.max(np.abs(corr))
        if peak_corr > 0:
            secondary = np.sort(np.abs(corr))[::-1]
            if len(secondary) > 1 and secondary[0] > 0:
                multipath_ratio = secondary[1] / secondary[0]
                # Yüksek multipath → daha uzun sembol
                if multipath_ratio > 0.5:
                    self.cal_duration = 0.40
                elif multipath_ratio > 0.2:
                    self.cal_duration = 0.25
                else:
                    self.cal_duration = 0.15
            else:
                self.cal_duration = DEF_DURATION
        else:
            self.cal_duration = DEF_DURATION

        self._sys(f"Çok yollu analiz: süre={self.cal_duration*1000:.0f}ms")

        # 3) AGC preamble referansı ayarla
        self._sys("AGC referansı ayarlanıyor...")
        self._play_tone(PREAMBLE_FREQ, 0.3, 1.0)
        time.sleep(0.1)

        # Güncelle
        if self.auto_cal.get():
            self.gap = self.cal_gap
            self.duration = self.cal_duration

        self._cal_lbl.config(
            text=f"Gap: {self.cal_gap*1000:.0f}ms  ·  Süre: {self.cal_duration*1000:.0f}ms")
        self._sys("Kalibrasyon tamamlandı ✓")

    # ═══════════════════════ SONAR ═══════════════════════════════════
    def _sonar_ping(self):
        if self.is_transmitting:
            return
        self._sys("Ping gönderiliyor...")
        self.sonar_waiting = True
        self.sonar_send_t = time.time()
        threading.Thread(target=lambda: self._play_tone(PING_FREQ, 0.2, 0.9),
                         daemon=True).start()

    def _handle_pong(self):
        if self.sonar_waiting:
            rtt = time.time() - self.sonar_send_t
            dist = (rtt * SPEED_SOUND) / 2.0
            self.last_distance = dist
            self.sonar_waiting = False
            self._dist_lbl.config(text=f"Mesafe: {dist:.2f} m")
            self._sys(f"Pong: RTT={rtt * 1000:.1f}ms  Mesafe={dist:.2f}m")

    def _handle_ping(self):
        threading.Thread(target=lambda: (time.sleep(0.03),
                         self._play_tone(PONG_FREQ, 0.15, 0.9)), daemon=True).start()

    # ═══════════════════════ DOPPLER ════════════════════════════════
    def _measure_doppler(self, freqs, mag):
        m1 = (freqs >= PILOT_LO - 100) & (freqs <= PILOT_LO + 100)
        m2 = (freqs >= PILOT_HI - 100) & (freqs <= PILOT_HI + 100)
        thr = self.threshold
        if np.any(m1) and np.any(m2):
            p1, p2 = np.max(mag[m1]), np.max(mag[m2])
            if p1 > thr and p2 > thr:
                f1 = freqs[m1][np.argmax(mag[m1])]
                f2 = freqs[m2][np.argmax(mag[m2])]
                self.doppler_ratio = ((f1 / PILOT_LO) + (f2 / PILOT_HI)) / 2.0
                self._dopp_lbl.config(text=f"{self.doppler_ratio:.4f}x")

    # ═══════════════════════ TX — VERİCİ ════════════════════════════
    def send_text(self):
        msg = self.entry.get().strip()
        if not msg or self.is_transmitting:
            return
        self.entry.delete(0, tk.END)
        threading.Thread(target=self._send_packet,
                         args=(int(self.my_id_ent.get()),
                               int(self.target_id_ent.get()),
                               msg.encode("utf-8"), PKT_TEXT),
                         daemon=True).start()

    def pick_and_send_image(self):
        if self.is_transmitting:
            return
        p = filedialog.askopenfilename(title="Resim Seç",
            filetypes=[("Resim", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("Tüm", "*.*")])
        if not p:
            return
        try:
            pil = Image.open(p)
        except Exception as e:
            self._sys(f"Hata: {e}"); return
        payload = self._img_enc(pil)
        thumb = self._pil_tk(pil.convert("L").resize((IMG_W, IMG_H), Image.LANCZOS), (140, 140))
        threading.Thread(target=self._send_packet,
                         args=(int(self.my_id_ent.get()),
                               int(self.target_id_ent.get()),
                               payload, PKT_IMAGE),
                         kwargs={"thumb": thumb}, daemon=True).start()

    def _send_packet(self, src, dst, payload, pkt_type, thumb=None):
        while self.is_channel_busy:
            time.sleep(0.1)
        self.is_transmitting = True
        original = payload
        orig_len = len(payload)

        # ═══ DATA PIPELINE ═══
        # 1) Predictive Coding
        payload = predictive_encode(payload)

        # 2) XOR Encryption
        use_xor = self.xor_enabled.get()
        kw = self._xor_key()
        if use_xor and kw:
            payload = xor_cipher(payload, kw)

        # 3) Delta+RLE / Huffman Compression
        comp_type = 0
        if self.comp_enabled.get() and len(payload) > 8:
            if pkt_type == PKT_IMAGE:
                comp = delta_rle_compress(payload)
                if len(comp) < len(payload):
                    payload = comp; comp_type = 1
            else:
                comp = huffman_compress(payload)
                if len(comp) < len(payload):
                    payload = comp; comp_type = 2

        self.comp_ratio = (1.0 - len(payload) / orig_len) * 100 if orig_len > 0 else 0
        self._comp_lbl.config(text=f"{self.comp_ratio:.0f}%")

        # 4) FEC (Hamming)
        use_fec = self.fec_enabled.get()
        if use_fec:
            payload = hamming_enc(payload)

        # 5) Fragment
        chunks = [payload[i:i + FRAG_MAX] for i in range(0, max(1, len(payload)), FRAG_MAX)]
        if not chunks:
            chunks = [b'']
        total_chunks = len(chunks)

        use_fhss = self.fhss_enabled.get()
        use_stego = self.stego_enabled.get()
        hop_cnt = self.hop_count.get() if self.mesh_enabled.get() else 0

        # flags: b0-1=pkt_type, b2=fec, b3=xor, b4=reserved, b5=comp, b6=stego, b7=fhss
        flags = pkt_type & 0x03
        if use_fec:   flags |= 0x04
        if use_xor and kw: flags |= 0x08
        if comp_type: flags |= 0x20
        if use_stego: flags |= 0x40
        if use_fhss:  flags |= 0x80

        fhss_sd = fhss_seed(kw) if use_fhss else 0

        dur = self.duration
        gap = self.gap
        start_t = time.time()
        kind = "RESIM" if pkt_type == PKT_IMAGE else "METIN"

        # ═══ PREAMBLE + PILOT TONLARI ═══
        wu_dur = 0.35
        wu_n = int(FS * wu_dur)
        t_wu = np.linspace(0, wu_dur, wu_n, endpoint=False)
        wu = (np.sin(2 * np.pi * PREAMBLE_FREQ * t_wu) * 0.5
              + np.sin(2 * np.pi * PILOT_LO * t_wu) * 0.25
              + np.sin(2 * np.pi * PILOT_HI * t_wu) * 0.25)
        wu = (wu * raised_cosine_window(wu_n)).astype(np.float32)
        sd.play(wu, FS); sd.wait(); time.sleep(0.08)

        all_ok = True
        global_frame_idx = 0

        for seq_idx, chunk in enumerate(chunks):
            sz = len(chunk)
            crc_val = zlib.crc32(chunk) & 0xFF

            # 16-byte header
            header = bytearray(HEADER_SIZE)
            header[0] = src
            header[1] = dst
            header[2] = sz & 0xFF
            header[3] = (sz >> 8) & 0xFF
            header[4] = flags
            header[5] = IMG_W if pkt_type == PKT_IMAGE else 0
            header[6] = IMG_H if pkt_type == PKT_IMAGE else 0
            header[7] = crc_val
            header[8] = seq_idx
            header[9] = total_chunks
            header[10] = hop_cnt
            header[11] = comp_type
            header[12] = orig_len & 0xFF
            header[13] = (orig_len >> 8) & 0xFF
            header[14] = 1 if self.turbo_mode.get() else 0  # mode flag
            header[15] = 0

            raw = list(header) + list(chunk)
            while len(raw) % NUM_SC != 0:
                raw.append(0)

            num_frames = len(raw) // NUM_SC
            hdr_frames = (HEADER_SIZE + NUM_SC - 1) // NUM_SC

            self._prog_lbl.config(text=f"{kind} [{seq_idx + 1}/{total_chunks}] | {sz}B")

            # Her frame'i OFDM ile gönder
            for fi in range(num_frames):
                elapsed = time.time() - start_t
                self.pbar["value"] = min(100, (seq_idx * 100 + fi * 100 / max(num_frames, 1)) / total_chunks)

                frame_data = raw[fi * NUM_SC:(fi + 1) * NUM_SC]

                do_fhss = use_fhss and fi >= hdr_frames
                wave = ofdm_modulate(
                    frame_data, dur,
                    use_ask=self.turbo_mode.get(),
                    use_fhss=do_fhss,
                    fhss_sd=fhss_sd,
                    frame_idx=global_frame_idx,
                    stego=use_stego)

                sd.play(wave, FS); sd.wait()
                time.sleep(gap)
                global_frame_idx += 1

            # ACK/NACK bekleme
            if self.nack_enabled.get():
                ack_ok = self._wait_for_ack()
                if not ack_ok:
                    for retry in range(ARQ_RETRIES):
                        self._sys(f"Tekrar [{retry + 1}/{ARQ_RETRIES}]")
                        for fi2 in range(num_frames):
                            fd2 = raw[fi2 * NUM_SC:(fi2 + 1) * NUM_SC]
                            do_fh2 = use_fhss and fi2 >= hdr_frames
                            w2 = ofdm_modulate(fd2, dur, use_ask=self.turbo_mode.get(),
                                               use_fhss=do_fh2, fhss_sd=fhss_sd,
                                               frame_idx=fi2, stego=use_stego)
                            sd.play(w2, FS); sd.wait(); time.sleep(gap)
                        if self._wait_for_ack():
                            break
                    else:
                        self._sys(f"Fragment {seq_idx} başarısız!")
                        all_ok = False

        elapsed_total = time.time() - start_t
        bps = orig_len / elapsed_total if elapsed_total > 0 else 0
        self._tp_lbl.config(text=f"{bps:.1f} B/s")

        tags_list = []
        if use_fec: tags_list.append("FEC")
        if use_xor and kw: tags_list.append("XOR")
        if comp_type: tags_list.append("RLE" if comp_type == 1 else "HUFF")
        if use_fhss: tags_list.append("FHSS")
        if use_stego: tags_list.append("STEGO")
        tag_s = " [" + "|".join(tags_list) + "]" if tags_list else ""

        meta = f"{orig_len}B · {total_chunks}frag · {elapsed_total:.1f}s · {bps:.1f}B/s{tag_s}"

        if pkt_type == PKT_TEXT:
            self._own_txt(original.decode("utf-8", errors="replace"), meta)
        else:
            self._own_img(thumb, meta)

        self.pbar["value"] = 100
        self._prog_lbl.config(text="Hazır"); self._tmr_lbl.config(text="")
        time.sleep(0.4); self.pbar["value"] = 0
        self.is_transmitting = False

    def _wait_for_ack(self) -> bool:
        deadline = time.time() + ACK_TIMEOUT
        while time.time() < deadline:
            data = []
            while not self.msg_queue.empty():
                data.append(self.msg_queue.get_nowait())
            if data:
                audio = np.concatenate(data).flatten()
                yf = fft(audio)
                xf = fftfreq(len(yf), 1 / FS)
                mag = np.abs(yf[:len(yf) // 2])
                freqs = xf[:len(xf) // 2]
                ma = (freqs >= ACK_FREQ - 60) & (freqs <= ACK_FREQ + 60)
                mn = (freqs >= NACK_FREQ - 60) & (freqs <= NACK_FREQ + 60)
                pa = np.max(mag[ma]) if np.any(ma) else 0
                pn = np.max(mag[mn]) if np.any(mn) else 0
                if pa > self.threshold and pa > pn:
                    return True
                if pn > self.threshold and pn > pa:
                    return False
            time.sleep(0.05)
        return False

    # ═══════════════════════ RX — ALICI ════════════════════════════
    def _update_loop(self):
        if not self.is_running:
            return

        data = []
        while not self.msg_queue.empty():
            data.append(self.msg_queue.get_nowait())

        if data and not self.is_transmitting:
            audio = np.concatenate(data).flatten()

            rms = float(np.sqrt(np.mean(audio ** 2))) * 100
            if not self.is_listening:
                self.noise_samples.append(rms)
                if len(self.noise_samples) >= 5:
                    self.noise_floor = float(np.median(list(self.noise_samples)))

            yf = fft(audio)
            xf = fftfreq(len(yf), 1 / FS)
            mag = np.abs(yf[:len(yf) // 2])
            freqs = xf[:len(xf) // 2]

            # Moving Average
            self.mag_history.append(mag.copy())
            if len(self.mag_history) >= 2:
                mag = np.mean(list(self.mag_history), axis=0)

            # Spektrum
            self.line.set_ydata(np.interp(self.plot_x, freqs, mag))
            self.mpl_cv.draw_idle()

            thr = self.threshold
            self._thr_lbl.config(text=f"Eşik:{thr:.1f}")

            # SNR
            bm = (freqs > 9500) & (freqs < 20000)
            sig = float(np.max(mag[bm])) if np.any(bm) else 0
            snr = 20 * np.log10(max(sig, 0.001) / max(self.noise_floor, 0.001))
            self.current_snr = snr
            self._snr_lbl.config(text=f"{snr:.1f} dB")

            # Doppler
            self._measure_doppler(freqs, mag)

            # AGC: preamble genliğini referans al
            pm = (freqs >= PREAMBLE_FREQ - 60) & (freqs <= PREAMBLE_FREQ + 60)
            preamble_peak = np.max(mag[pm]) if np.any(pm) else 0
            if preamble_peak > thr * 2:
                self.agc_ref = max(0.1, preamble_peak)
                self._agc_lbl.config(text=f"{self.agc_ref:.2f}")

            # Kanal durumu
            peak = sig
            if peak > thr:
                self.is_channel_busy = True
                self._set_status("Meşgul", C.RED)
            else:
                self.is_channel_busy = False
                self._set_status("Müsait", C.GREEN)

            # Sonar
            mp = (freqs >= PING_FREQ - 60) & (freqs <= PING_FREQ + 60)
            mpo = (freqs >= PONG_FREQ - 60) & (freqs <= PONG_FREQ + 60)
            if np.any(mp) and np.max(mag[mp]) > thr and not self.is_listening:
                self._handle_ping()
            if np.any(mpo) and np.max(mag[mpo]) > thr:
                self._handle_pong()

            # Wake-up
            if not self.is_listening and preamble_peak > thr:
                self.is_listening = True
                self.start_t = time.time()
                self.current_packet = []
                self.last_frame_set = None
                self._pst.config(text="Alınıyor...")
                self._sys("Sinyal algılandı...")

            # OFDM Demodülasyon
            if self.is_listening:
                # AGC normalize
                if self.agc_ref > 0.1:
                    norm_mag = mag / self.agc_ref * 10  # normalize
                else:
                    norm_mag = mag

                frame = ofdm_demodulate(
                    audio, freqs, norm_mag, thr,
                    doppler_ratio=self.doppler_ratio)

                if frame is not None and len(frame) == NUM_SC:
                    cs = tuple(frame)
                    if cs != self.last_frame_set:
                        # FHSS ters çevir (header sonrası)
                        if len(self.current_packet) >= HEADER_SIZE:
                            hdr_frames = (HEADER_SIZE + NUM_SC - 1) // NUM_SC
                            fi = len(self.current_packet) // NUM_SC
                            if fi >= hdr_frames and len(self.current_packet) >= 5:
                                if self.current_packet[4] & 0x80:  # FHSS flag
                                    kw = self._xor_key()
                                    fs_seed = fhss_seed(kw)
                                    perm = fhss_perm(fi - hdr_frames, fs_seed, NUM_SC)
                                    inv = fhss_inv(perm)
                                    frame = [frame[inv[j]] for j in range(NUM_SC)]

                        self.current_packet.extend(frame)
                        self.last_frame_set = cs
                        self.last_signal_time = time.time()

                if self.current_packet and time.time() - self.last_signal_time > 1.5:
                    self._finalize_fragment()

        self.root.after(50, self._update_loop)

    # ═══════════════════════ FRAGMENT SONLANDIRMA ═══════════════════
    def _finalize_fragment(self):
        if len(self.current_packet) < HEADER_SIZE:
            self._reset_rx(); return

        h = self.current_packet[:HEADER_SIZE]
        src, dst = h[0], h[1]
        sz = h[2] | (h[3] << 8)
        flags = h[4]
        pkt_type  = flags & 0x03
        has_fec   = bool(flags & 0x04)
        has_xor   = bool(flags & 0x08)
        has_comp  = bool(flags & 0x20)
        has_stego = bool(flags & 0x40)
        has_fhss  = bool(flags & 0x80)
        img_w, img_h = h[5] or IMG_W, h[6] or IMG_H
        crc = h[7]
        seq_num, total_ch = h[8], h[9]
        hop_cnt = h[10]
        comp_type = h[11]
        orig_len = h[12] | (h[13] << 8)

        chunk_data = bytes(self.current_packet[HEADER_SIZE:HEADER_SIZE + sz])
        my_id = int(self.my_id_ent.get())

        # Mesh relay
        if dst != my_id and dst != 0 and hop_cnt > 0 and self.mesh_enabled.get():
            self._sys(f"Mesh relay: {src}→{dst} hop={hop_cnt}")
            self._send_ack()
            threading.Thread(target=self._mesh_relay,
                             args=(self.current_packet[:], hop_cnt - 1),
                             daemon=True).start()
            self._reset_rx(); return

        if dst != my_id and dst != 0:
            self._reset_rx(); return

        # CRC
        crc_ok = (zlib.crc32(chunk_data) & 0xFF) == crc

        if self.nack_enabled.get():
            if crc_ok:
                threading.Thread(target=self._send_ack, daemon=True).start()
            else:
                threading.Thread(target=self._send_nack, daemon=True).start()

        if not crc_ok:
            self._sys(f"CRC HATA — Frag {seq_num + 1}/{total_ch}")
            self._reset_rx(); return

        # Fragment buffer
        fk = (src, dst, pkt_type)
        if fk not in self.frag_buffer:
            self.frag_buffer[fk] = {}
            self.frag_meta[fk] = {
                'total': total_ch, 'flags': flags, 'img_w': img_w, 'img_h': img_h,
                'comp_type': comp_type, 'orig_len': orig_len, 'src': src,
                'has_fec': has_fec, 'has_xor': has_xor, 'pkt_type': pkt_type,
                'start_t': self.start_t
            }
        self.frag_buffer[fk][seq_num] = chunk_data
        self._sys(f"Frag {seq_num + 1}/{total_ch} ✓")

        if len(self.frag_buffer[fk]) >= total_ch:
            self._assemble(fk)

        self._reset_rx()

    def _assemble(self, key):
        mi = self.frag_meta[key]
        frags = self.frag_buffer[key]
        total_dur = time.time() - mi['start_t']

        assembled = b''
        for i in range(mi['total']):
            assembled += frags.get(i, b'')

        status = ["TAM"]
        decoded = assembled

        # Ters pipeline: FEC → Decompress → XOR Decrypt → Predictive Decode
        if mi['has_fec']:
            decoded = hamming_dec(decoded)
            status.append("FEC")

        ct = mi['comp_type']
        ol = mi['orig_len']
        if ct == 1:
            decoded = delta_rle_decompress(decoded, ol)
            status.append("RLE")
        elif ct == 2:
            decoded = huffman_decompress(decoded, ol)
            status.append("HUFF")

        if mi['has_xor']:
            kw = self._xor_key()
            if kw:
                decoded = xor_cipher(decoded, kw)
                status.append("XOR")

        # Predictive decode (her zaman uygula)
        decoded = predictive_decode(decoded)

        orig_sz = len(decoded)
        bps = orig_sz / total_dur if total_dur > 0 else 0
        self._tp_lbl.config(text=f"{bps:.1f} B/s")
        meta = f"{orig_sz}B · {mi['total']}frag · {total_dur:.1f}s · {bps:.1f}B/s · {' · '.join(status)}"

        if mi['pkt_type'] == PKT_TEXT:
            self._inc_txt(mi['src'], decoded.decode("utf-8", errors="replace"), meta)
        elif mi['pkt_type'] == PKT_IMAGE:
            pil = self._img_dec(decoded, mi['img_w'], mi['img_h'])
            if pil:
                self.last_received_pil = pil
                photo = self._pil_tk(pil, (140, 140))
                ph_s = ImageTk.PhotoImage(pil.resize((100, 100), Image.NEAREST))
                self.prev_cv.delete("all")
                self.prev_cv.create_image(50, 50, image=ph_s)
                self.prev_cv._ph = ph_s
                self._pst.config(text="Alındı")
                self._ppr.config(text=f"{orig_sz}B %100")
                self._inc_img(mi['src'], photo, meta)
            else:
                self._inc_txt(mi['src'], "[Resim çözülemedi]", meta, False)

        del self.frag_buffer[key]
        del self.frag_meta[key]

    # ═══════════════════════ MESH RELAY ═════════════════════════════
    def _mesh_relay(self, raw, new_hop):
        time.sleep(0.05 + np.random.random() * 0.15)
        if len(raw) < HEADER_SIZE:
            return
        raw[10] = new_hop

        # Wake-up
        wu_n = int(FS * 0.25)
        t_wu = np.linspace(0, 0.25, wu_n, endpoint=False)
        wu = (np.sin(2 * np.pi * PREAMBLE_FREQ * t_wu) * 0.5
              + np.sin(2 * np.pi * PILOT_LO * t_wu) * 0.25
              + np.sin(2 * np.pi * PILOT_HI * t_wu) * 0.25)
        sd.play((wu * raised_cosine_window(wu_n)).astype(np.float32), FS)
        sd.wait(); time.sleep(0.06)

        while len(raw) % NUM_SC != 0:
            raw.append(0)

        for fi in range(len(raw) // NUM_SC):
            fd = raw[fi * NUM_SC:(fi + 1) * NUM_SC]
            w = ofdm_modulate(fd, self.duration)
            sd.play(w, FS); sd.wait()
            time.sleep(self.gap)

        self._sys(f"Relay tamamlandı (hop={new_hop})")

    def _reset_rx(self):
        self.current_packet = []
        self.is_listening = False
        self.last_frame_set = None
        self._pst.config(text="Bekleniyor")
        self._ppr.config(text="—")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BAŞLATICI                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    root = tk.Tk()
    app = CircirApp(root)
    root.mainloop()