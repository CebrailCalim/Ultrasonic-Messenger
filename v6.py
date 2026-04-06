import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import threading
import queue
import time
from datetime import datetime
import os
import zlib
from PIL import Image, ImageTk
from collections import deque

# =============================================================================
# PARAMETRELER  (v50)
# =============================================================================
FS           = 44100
WAKE_UP_FREQ = 9500

# --- 8 Kanal modu ---
CHANNELS_8 = [
    (10000, 10500),
    (11000, 11500),
    (12000, 12500),
    (13000, 13500),
    (14000, 14500),
    (15000, 15500),
    (16000, 16500),
    (17000, 17500),
]

# --- 16 Kanal modu (yari bant genisligi, 2x throughput) ---
CHANNELS_16 = [
    (10000, 10250),
    (10500, 10750),
    (11000, 11250),
    (11500, 11750),
    (12000, 12250),
    (12500, 12750),
    (13000, 13250),
    (13500, 13750),
    (14000, 14250),
    (14500, 14750),
    (15000, 15250),
    (15500, 15750),
    (16000, 16250),
    (16500, 16750),
    (17000, 17250),
    (17500, 17750),
]

STEP_8     = 30
STEP_16    = 15
HEX_BASE   = 16
DET_WIN_8  = 480
DET_WIN_16 = 240

DEF_DURATION = 0.30
DEF_GAP      = 0.05

# --- Resim sabitleri ---
IMG_W      = 32
IMG_H      = 32
IMG_LEVELS = 16

# --- Paket tipleri ---
PKT_TEXT  = 0x00
PKT_IMAGE = 0x01

# =============================================================================
# GRAY CODE TABLOSU (0-15)
# =============================================================================
GRAY_ENCODE = [i ^ (i >> 1) for i in range(16)]
GRAY_DECODE = [0] * 16
for _i, _g in enumerate(GRAY_ENCODE):
    GRAY_DECODE[_g] = _i

# =============================================================================
# HAMMING (7,4) — FEC
# =============================================================================
_G = np.array([
    [1,0,0,0, 1,1,0],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 0,1,1],
    [0,0,0,1, 1,1,1],
], dtype=np.uint8)

_H = np.array([
    [1,1,0,1,1,0,0],
    [1,0,1,1,0,1,0],
    [0,1,1,1,0,0,1],
], dtype=np.uint8)

def hamming_encode_nibble(nibble):
    d = np.array([(nibble >> i) & 1 for i in range(4)], dtype=np.uint8)
    cw = (d @ _G) % 2
    return int(sum(cw[i] << i for i in range(7)))

def hamming_decode_7bit(code7):
    r = np.array([(code7 >> i) & 1 for i in range(7)], dtype=np.uint8)
    syndrome = (_H @ r) % 2
    s = int(sum(syndrome[i] << i for i in range(3)))
    if s != 0 and s - 1 < 7:
        r[s - 1] ^= 1
    return int(sum(r[i] << i for i in range(4)))

def hamming_encode_bytes(data: bytes) -> bytes:
    out = bytearray()
    for b in data:
        lo, hi = b & 0xF, (b >> 4) & 0xF
        cw_lo, cw_hi = hamming_encode_nibble(lo), hamming_encode_nibble(hi)
        word14 = cw_lo | (cw_hi << 7)
        out.append(word14 & 0xFF)
        out.append((word14 >> 8) & 0xFF)
    return bytes(out)

def hamming_decode_bytes(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data) - 1, 2):
        word14 = data[i] | (data[i+1] << 8)
        lo = hamming_decode_7bit(word14 & 0x7F)
        hi = hamming_decode_7bit((word14 >> 7) & 0x7F)
        out.append((hi << 4) | lo)
    return bytes(out)

# =============================================================================
# XOR STREAM CIPHER — Keyword tabanlı
# =============================================================================
def xor_cipher(payload: bytes, keyword: str) -> bytes:
    """Keyword tabanlı XOR stream cipher.
    Keyword SHA-benzeri bir seed'e dönüştürülür, LCG ile key stream üretilir."""
    if not keyword:
        return payload
    # Keyword'den 32-bit seed üret
    seed = 0x5A5A5A5A
    for ch in keyword:
        seed = ((seed * 31) + ord(ch)) & 0xFFFFFFFF
    if seed == 0:
        seed = 0xDEADBEEF
    state = seed
    out = bytearray()
    for b in payload:
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out.append(b ^ ((state >> 16) & 0xFF))
    return bytes(out)

# =============================================================================
# FREQUENCY EQUALIZATION
# =============================================================================
EQ_FREQS = [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000]
EQ_GAINS = [1.0,   1.05,  1.12,  1.20,  1.30,  1.42,  1.55,  1.70,  1.85]

def eq_gain_for_freq(f):
    if f <= EQ_FREQS[0]:  return EQ_GAINS[0]
    if f >= EQ_FREQS[-1]: return EQ_GAINS[-1]
    for i in range(len(EQ_FREQS) - 1):
        if EQ_FREQS[i] <= f <= EQ_FREQS[i+1]:
            t = (f - EQ_FREQS[i]) / (EQ_FREQS[i+1] - EQ_FREQS[i])
            return EQ_GAINS[i] + t * (EQ_GAINS[i+1] - EQ_GAINS[i])
    return 1.0

# Kanal EQ'larını önceden hesapla — her iki mod için
CHANNEL_EQ_8 = []
for _b1, _b2 in CHANNELS_8:
    CHANNEL_EQ_8.append((eq_gain_for_freq(_b1 + DET_WIN_8/2),
                         eq_gain_for_freq(_b2 + DET_WIN_8/2)))

CHANNEL_EQ_16 = []
for _b1, _b2 in CHANNELS_16:
    CHANNEL_EQ_16.append((eq_gain_for_freq(_b1 + DET_WIN_16/2),
                          eq_gain_for_freq(_b2 + DET_WIN_16/2)))


# --- Renk paleti ---
C_SB      = "#0f172a"
C_SB2     = "#1e293b"
C_SB3     = "#334155"
C_ACCENT  = "#3b82f6"
C_ACCENT2 = "#1d4ed8"
C_TL      = "#f1f5f9"
C_TM      = "#94a3b8"
C_MAIN    = "#f8fafc"
C_WHITE   = "#ffffff"
C_SUCCESS = "#22c55e"
C_DANGER  = "#ef4444"
C_WARN    = "#f59e0b"
C_OWN_BG  = "#dbeafe"
C_OWN_FG  = "#1e40af"
C_INC_BG  = "#f1f5f9"
C_INC_FG  = "#1e293b"
C_BORDER  = "#e2e8f0"


class AcousticMessengerV50:
    def __init__(self, root):
        self.root = root
        self.root.title("AcousticMaster v50")
        self.root.geometry("1480x960")
        self.root.configure(bg=C_MAIN)

        self.is_running       = True
        self.duration         = DEF_DURATION
        self.gap              = DEF_GAP
        self.msg_queue        = queue.Queue()
        self.is_transmitting  = False
        self.is_listening     = False
        self.is_channel_busy  = False
        self.current_packet   = []
        self.last_signal_time = 0
        self.last_frame_set   = None
        self.start_t          = 0
        self.arrival_ts       = ""
        self.photo_refs       = []

        # Mod değişkenleri
        self.safe_mode        = tk.BooleanVar(value=False)
        self.xor_enabled      = tk.BooleanVar(value=True)
        self.channel_mode     = tk.IntVar(value=8)   # 8 veya 16

        # Adaptive Threshold
        self.noise_floor      = 2.5
        self.noise_samples    = deque(maxlen=50)
        self.NOISE_MULT       = 1.5

        # Moving Average
        self.mag_history      = deque(maxlen=3)

        # Export
        self.last_received_pil = None

        # Alıcı taraf: gelen paketin kanal modunu otomatik algılar
        self.rx_channel_mode  = 8
        self.rx_frame_size    = 8

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._exit)

        try:
            self.stream = sd.InputStream(
                callback=lambda *a: self.msg_queue.put(a[0].copy()),
                channels=1, samplerate=FS, blocksize=4096
            )
            self.stream.start()
        except Exception as e:
            print(f"Mikrofon hatasi: {e}")

        self._update_loop()

    # =========================================================================
    # Kanal modu yardımcıları
    # =========================================================================
    def _tx_channels(self):
        return CHANNELS_16 if self.channel_mode.get() == 16 else CHANNELS_8

    def _tx_num_ch(self):
        return self.channel_mode.get()

    def _tx_frame_size(self):
        return self.channel_mode.get()

    def _tx_step(self):
        return STEP_16 if self.channel_mode.get() == 16 else STEP_8

    def _tx_det_win(self):
        return DET_WIN_16 if self.channel_mode.get() == 16 else DET_WIN_8

    def _tx_eq(self):
        return CHANNEL_EQ_16 if self.channel_mode.get() == 16 else CHANNEL_EQ_8

    # =========================================================================
    # DYNAMIC THRESHOLD
    # =========================================================================
    @property
    def threshold(self):
        return max(1.5, self.noise_floor * self.NOISE_MULT)

    def _update_noise_floor(self, rms):
        self.noise_samples.append(rms)
        if len(self.noise_samples) >= 5:
            self.noise_floor = float(np.median(list(self.noise_samples)))

    # =========================================================================
    # UI
    # =========================================================================
    def _build_ui(self):
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=C_SB, width=390)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        self.sb = sb

        # Başlık
        hdr = tk.Frame(sb, bg=C_SB2, pady=18, padx=22)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="AcousticMaster",
                 bg=C_SB2, fg=C_TL, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(hdr, text="v50 · 8/16-Ch · Gray · FEC · XOR-Key · EQ",
                 bg=C_SB2, fg=C_TM, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

        # Durum
        sf = tk.Frame(sb, bg=C_SB, padx=22, pady=10)
        sf.pack(fill=tk.X)
        self._status_dot = tk.Canvas(sf, width=10, height=10,
                                     bg=C_SB, highlightthickness=0)
        self._status_dot.pack(side=tk.LEFT)
        self._status_dot.create_oval(2, 2, 9, 9, fill=C_SUCCESS, outline="", tags="dot")
        self._status_lbl = tk.Label(sf, text="Kanal Musait",
                                    bg=C_SB, fg=C_SUCCESS,
                                    font=("Segoe UI", 10, "bold"))
        self._status_lbl.pack(side=tk.LEFT, padx=(8, 0))
        self._thresh_lbl = tk.Label(sf, text="Esik: 2.50",
                                    bg=C_SB, fg=C_TM, font=("Segoe UI", 8))
        self._thresh_lbl.pack(side=tk.RIGHT)

        self._sep()

        # Kimlik
        self._sec("KIMLIK")
        idf = tk.Frame(sb, bg=C_SB, padx=22)
        idf.pack(fill=tk.X, pady=(0, 4))
        idf.columnconfigure(0, weight=1)
        idf.columnconfigure(1, weight=1)
        tk.Label(idf, text="Kendi ID", bg=C_SB,
                 fg=C_TM, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        tk.Label(idf, text="Hedef ID", bg=C_SB,
                 fg=C_TM, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.my_id_ent = self._dark_entry(idf, "1")
        self.my_id_ent.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.target_id_ent = self._dark_entry(idf, "0")
        self.target_id_ent.grid(row=1, column=1, sticky="ew", pady=(4, 0), padx=(8, 0))

        self._sep()

        # Hız
        self._sec("HIZ AYARLARI")
        for attr, lbl_text, lo, hi, res, default in [
            ("dur", "Sembol Suresi", 0.05, 1.0, 0.05, DEF_DURATION),
            ("gap", "Bosluk Suresi", 0.01, 0.50, 0.01, DEF_GAP),
        ]:
            sf2 = tk.Frame(sb, bg=C_SB, padx=22)
            sf2.pack(fill=tk.X, pady=(0, 4))
            row = tk.Frame(sf2, bg=C_SB)
            row.pack(fill=tk.X)
            tk.Label(row, text=lbl_text, bg=C_SB,
                     fg=C_TM, font=("Segoe UI", 9)).pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text=f"{default:.2f} sn",
                               bg=C_SB, fg=C_ACCENT, font=("Segoe UI", 9, "bold"))
            val_lbl.pack(side=tk.RIGHT)
            scale = tk.Scale(
                sf2, from_=lo, to=hi, resolution=res,
                orient=tk.HORIZONTAL, showvalue=False,
                bg=C_SB, fg=C_TL, troughcolor=C_SB2,
                activebackground=C_ACCENT, highlightthickness=0, bd=0,
                sliderrelief="flat",
                command=lambda v, a=attr, l=val_lbl: self._scale_change(a, v, l)
            )
            scale.set(default)
            scale.pack(fill=tk.X)
            setattr(self, f"{attr}_scale", scale)
            setattr(self, f"{attr}_val_lbl", val_lbl)

        rf = tk.Frame(sb, bg=C_SB, padx=22, pady=6)
        rf.pack(fill=tk.X)
        tk.Button(rf, text="Varsayilani Geri Yukle",
                  bg=C_SB2, fg=C_TM, font=("Segoe UI", 9),
                  relief="flat", cursor="hand2",
                  command=self.reset_defaults).pack(fill=tk.X)

        self._sep()

        # ===== KANAL MODU =====
        self._sec("KANAL MODU")
        chf = tk.Frame(sb, bg=C_SB, padx=22)
        chf.pack(fill=tk.X, pady=(0, 4))
        tk.Radiobutton(
            chf, text="8 Kanal  (8 byte/frame)",
            variable=self.channel_mode, value=8,
            bg=C_SB, fg=C_TL, selectcolor=C_SB2,
            activebackground=C_SB, activeforeground=C_TL,
            font=("Segoe UI", 9), anchor="w"
        ).pack(fill=tk.X)
        tk.Radiobutton(
            chf, text="16 Kanal  (16 byte/frame, 2x hiz)",
            variable=self.channel_mode, value=16,
            bg=C_SB, fg=C_TL, selectcolor=C_SB2,
            activebackground=C_SB, activeforeground=C_TL,
            font=("Segoe UI", 9), anchor="w"
        ).pack(fill=tk.X)
        self._ch_mode_lbl = tk.Label(chf, text="TX: 8-Ch  |  RX: Otomatik",
                                     bg=C_SB, fg=C_WARN, font=("Segoe UI", 8))
        self._ch_mode_lbl.pack(anchor="w", pady=(4, 0))
        self.channel_mode.trace_add("write", lambda *_: self._update_all_labels())

        self._sep()

        # ===== GÜVENLIK & FEC =====
        self._sec("GUVENLIK & FEC")
        opt_f = tk.Frame(sb, bg=C_SB, padx=22)
        opt_f.pack(fill=tk.X, pady=(0, 4))

        tk.Checkbutton(
            opt_f, text="Safe Mode (Hamming FEC)",
            variable=self.safe_mode,
            bg=C_SB, fg=C_TL, selectcolor=C_SB2,
            activebackground=C_SB, activeforeground=C_TL,
            font=("Segoe UI", 9), anchor="w"
        ).pack(fill=tk.X)

        tk.Checkbutton(
            opt_f, text="XOR Sifreleme",
            variable=self.xor_enabled,
            bg=C_SB, fg=C_TL, selectcolor=C_SB2,
            activebackground=C_SB, activeforeground=C_TL,
            font=("Segoe UI", 9), anchor="w"
        ).pack(fill=tk.X)

        # XOR Keyword girişi
        kwf = tk.Frame(opt_f, bg=C_SB)
        kwf.pack(fill=tk.X, pady=(4, 0))
        tk.Label(kwf, text="Anahtar Kelime:", bg=C_SB,
                 fg=C_TM, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.xor_key_ent = tk.Entry(kwf, bg=C_SB2, fg=C_TL,
                                    font=("Segoe UI", 10),
                                    borderwidth=0, insertbackground=C_TL,
                                    show="*", width=16)
        self.xor_key_ent.insert(0, "gizli")
        self.xor_key_ent.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        # Anahtar göster/gizle
        self._show_key = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opt_f, text="Anahtari goster",
            variable=self._show_key,
            bg=C_SB, fg=C_TM, selectcolor=C_SB2,
            activebackground=C_SB, activeforeground=C_TM,
            font=("Segoe UI", 8), anchor="w",
            command=self._toggle_key_vis
        ).pack(fill=tk.X, pady=(2, 0))

        self._safe_lbl = tk.Label(opt_f, text="FEC: Kapali  |  XOR: Acik",
                                  bg=C_SB, fg=C_WARN, font=("Segoe UI", 8))
        self._safe_lbl.pack(anchor="w", pady=(4, 0))
        self.safe_mode.trace_add("write", lambda *_: self._update_all_labels())
        self.xor_enabled.trace_add("write", lambda *_: self._update_all_labels())

        self._sep()

        # Resim önizleme
        self._sec("ALINAN RESIM ONIZLEME")
        pf = tk.Frame(sb, bg=C_SB, padx=22)
        pf.pack(fill=tk.X)

        self.preview_canvas = tk.Canvas(pf, width=120, height=120,
                                        bg=C_SB2, highlightthickness=0)
        self.preview_canvas.pack(side=tk.LEFT)
        self._preview_idle()

        pi = tk.Frame(pf, bg=C_SB, padx=10)
        pi.pack(side=tk.LEFT, fill=tk.Y, anchor="n")
        self._prev_status = tk.Label(pi, text="Bekleniyor",
                                     bg=C_SB, fg=C_TM, font=("Segoe UI", 9))
        self._prev_status.pack(anchor="w")
        self._prev_prog = tk.Label(pi, text="— / — byte",
                                   bg=C_SB, fg=C_TM, font=("Segoe UI", 8))
        self._prev_prog.pack(anchor="w", pady=(4, 0))
        self._prev_size = tk.Label(pi, text="",
                                   bg=C_SB, fg=C_TM, font=("Segoe UI", 8))
        self._prev_size.pack(anchor="w", pady=(2, 0))
        self._export_btn = tk.Button(
            pi, text="PNG Kaydet", bg=C_SB3, fg=C_TL,
            font=("Segoe UI", 8), relief="flat", cursor="hand2",
            command=self._export_png
        )
        self._export_btn.pack(anchor="w", pady=(6, 0))

        self._sep()

        # Spektrum
        self._sec("SPEKTRUM ANALIZI")
        self.fig, self.ax = plt.subplots(figsize=(3.6, 2.6))
        self.fig.patch.set_facecolor(C_SB)
        self.ax.set_facecolor(C_SB2)
        self.plot_x = np.linspace(9000, 18500, 700)
        self.line, = self.ax.plot(self.plot_x, np.zeros(700),
                                  color=C_ACCENT, lw=1.5)
        self.ax.set_ylim(0, 60)
        self.ax.set_xlim(9000, 18500)
        self.ax.set_xticks([10000, 12000, 14000, 16000, 18000])
        self.ax.set_xticklabels(['10k', '12k', '14k', '16k', '18k'],
                                 fontsize=7, color=C_TM)
        self.ax.tick_params(colors=C_TM, length=0)
        for sp in self.ax.spines.values():
            sp.set_color(C_SB2)
        self.ax.grid(True, linestyle='--', alpha=0.15, color=C_TM)

        ch_colors_8 = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444',
                       '#8b5cf6', '#06b6d4', '#f97316', '#ec4899']
        for idx, (b1, b2) in enumerate(CHANNELS_8):
            self.ax.axvspan(b1, b1 + DET_WIN_8, alpha=0.10,
                            color=ch_colors_8[idx % len(ch_colors_8)])

        self.fig.tight_layout(pad=0.4)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=sb)
        self.mpl_canvas.get_tk_widget().configure(bg=C_SB)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.X, padx=16, pady=(0, 16))

    def _build_main(self):
        main = tk.Frame(self.root, bg=C_MAIN)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.main = main

        topbar = tk.Frame(main, bg=C_WHITE, pady=12, padx=24,
                          highlightthickness=1, highlightbackground=C_BORDER)
        topbar.pack(fill=tk.X)
        tk.Label(topbar, text="Mesajlasma",
                 bg=C_WHITE, fg="#0f172a",
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        self._throughput_lbl = tk.Label(topbar, text="0.0 byte/sn",
                                        bg=C_WHITE, fg=C_TM,
                                        font=("Segoe UI", 10))
        self._throughput_lbl.pack(side=tk.RIGHT, padx=(0, 8))
        tk.Label(topbar, text="verim:",
                 bg=C_WHITE, fg=C_TM, font=("Segoe UI", 10)).pack(side=tk.RIGHT)

        chat_wrap = tk.Frame(main, bg=C_MAIN, padx=20, pady=12)
        chat_wrap.pack(fill=tk.BOTH, expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_wrap, bg=C_WHITE, font=("Segoe UI", 13),
            padx=18, pady=16, borderwidth=0, relief="flat",
            state="disabled", cursor="arrow"
        )
        self.chat.pack(fill=tk.BOTH, expand=True)

        for tag, cfg in [
            ("own_hdr",  dict(foreground=C_OWN_FG, font=("Segoe UI", 9, "bold"), spacing3=2)),
            ("own_body", dict(foreground=C_OWN_FG, font=("Segoe UI", 13),
                              background=C_OWN_BG, lmargin1=48, lmargin2=48,
                              rmargin=20, spacing1=4, spacing3=4)),
            ("own_meta", dict(foreground="#93c5fd", font=("Segoe UI", 8),
                              lmargin1=48, spacing3=8)),
            ("inc_hdr",  dict(foreground="#475569", font=("Segoe UI", 9, "bold"), spacing3=2)),
            ("inc_body", dict(foreground=C_INC_FG, font=("Segoe UI", 13),
                              background=C_INC_BG, lmargin1=20, lmargin2=20,
                              rmargin=60, spacing1=4, spacing3=4)),
            ("inc_meta", dict(foreground="#94a3b8", font=("Segoe UI", 8),
                              lmargin1=20, spacing3=8)),
            ("sys",      dict(foreground="#cbd5e1", font=("Segoe UI", 8, "italic"),
                              justify="center", spacing1=6, spacing3=6)),
            ("ok",       dict(foreground=C_SUCCESS, font=("Segoe UI", 8))),
            ("err",      dict(foreground=C_DANGER, font=("Segoe UI", 8))),
        ]:
            self.chat.tag_config(tag, **cfg)

        prog_wrap = tk.Frame(main, bg=C_WHITE, padx=20, pady=10,
                             highlightthickness=1, highlightbackground=C_BORDER)
        prog_wrap.pack(fill=tk.X)
        pt = tk.Frame(prog_wrap, bg=C_WHITE)
        pt.pack(fill=tk.X)
        self._prog_lbl = tk.Label(pt, text="Hazir",
                                  bg=C_WHITE, fg=C_TM, font=("Segoe UI", 9))
        self._prog_lbl.pack(side=tk.LEFT)
        self._timer_lbl = tk.Label(pt, text="",
                                   bg=C_WHITE, fg=C_ACCENT,
                                   font=("Segoe UI", 9, "bold"))
        self._timer_lbl.pack(side=tk.RIGHT)

        sty = ttk.Style()
        sty.theme_use('clam')
        sty.configure("Slim.Horizontal.TProgressbar",
                       background=C_ACCENT, thickness=5,
                       troughcolor=C_BORDER, bordercolor=C_BORDER)
        self.progress = ttk.Progressbar(
            prog_wrap, orient=tk.HORIZONTAL,
            mode="determinate", style="Slim.Horizontal.TProgressbar"
        )
        self.progress.pack(fill=tk.X, pady=(6, 0))

        inp = tk.Frame(main, bg=C_WHITE, padx=20, pady=14,
                       highlightthickness=1, highlightbackground=C_BORDER)
        inp.pack(fill=tk.X)

        self.entry = tk.Entry(
            inp, bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 13),
            relief="solid", bd=1, insertbackground="#0f172a"
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_text())

        self._img_btn = tk.Button(
            inp, text="Resim", bg=C_SB2, fg=C_TL,
            font=("Segoe UI", 11), relief="flat",
            padx=16, pady=8, cursor="hand2",
            command=self.pick_and_send_image
        )
        self._img_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._send_btn = tk.Button(
            inp, text="Gonder", bg=C_ACCENT, fg=C_WHITE,
            font=("Segoe UI", 12, "bold"), relief="flat",
            padx=26, pady=8, cursor="hand2",
            command=self.send_text
        )
        self._send_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            inp, text="Temizle", bg=C_MAIN, fg=C_TM,
            font=("Segoe UI", 11), relief="flat",
            padx=12, pady=8, cursor="hand2",
            command=self.clear_log
        ).pack(side=tk.LEFT)

    # =========================================================================
    # YARDIMCILAR
    # =========================================================================
    def _sep(self):
        tk.Frame(self.sb, bg=C_SB2, height=1).pack(fill=tk.X, padx=22, pady=8)

    def _sec(self, text):
        f = tk.Frame(self.sb, bg=C_SB, padx=22)
        f.pack(fill=tk.X, pady=(10, 3))
        tk.Label(f, text=text, bg=C_SB, fg=C_TM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

    def _dark_entry(self, parent, default):
        e = tk.Entry(parent, bg=C_SB2, fg=C_TL,
                     font=("Segoe UI", 12), justify="center",
                     borderwidth=0, insertbackground=C_TL)
        e.insert(0, default)
        return e

    def _preview_idle(self):
        self.preview_canvas.delete("all")
        self.preview_canvas.create_rectangle(0, 0, 120, 120, fill=C_SB2, outline="")
        self.preview_canvas.create_text(60, 60, text="—",
                                        fill=C_TM, font=("Segoe UI", 20))

    def _set_status(self, text, color):
        self._status_lbl.config(text=text, fg=color)
        self._status_dot.delete("dot")
        self._status_dot.create_oval(2, 2, 9, 9, fill=color, outline="", tags="dot")

    def _scale_change(self, attr, val, lbl):
        v = float(val)
        if attr == "dur":
            self.duration = v
        else:
            self.gap = v
        lbl.config(text=f"{v:.2f} sn")

    def _toggle_key_vis(self):
        self.xor_key_ent.config(show="" if self._show_key.get() else "*")

    def _update_all_labels(self):
        fec = "Acik" if self.safe_mode.get() else "Kapali"
        xor = "Acik" if self.xor_enabled.get() else "Kapali"
        self._safe_lbl.config(text=f"FEC: {fec}  |  XOR: {xor}")
        ch = self.channel_mode.get()
        self._ch_mode_lbl.config(text=f"TX: {ch}-Ch  |  RX: Otomatik")

    def reset_defaults(self):
        self.dur_scale.set(DEF_DURATION)
        self.gap_scale.set(DEF_GAP)
        self._scale_change("dur", DEF_DURATION, self.dur_val_lbl)
        self._scale_change("gap", DEF_GAP, self.gap_val_lbl)

    def clear_log(self):
        self.chat.config(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.config(state="disabled")

    def _exit(self):
        self.is_running = False
        os._exit(0)

    def _export_png(self):
        if self.last_received_pil is None:
            self.add_sys("Kaydedilecek resim yok.")
            return
        path = filedialog.asksaveasfilename(
            title="Resmi PNG Olarak Kaydet",
            defaultextension=".png",
            filetypes=[("PNG Dosyasi", "*.png")]
        )
        if path:
            self.last_received_pil.resize((256, 256), Image.NEAREST).save(path)
            self.add_sys(f"Resim kaydedildi: {path}")

    # =========================================================================
    # SOHBET
    # =========================================================================
    def _chat_write(self, text, tag):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, text, tag)
        self.chat.see(tk.END)
        self.chat.config(state="disabled")

    def _chat_image(self, photo, tag):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, "  ", tag)
        self.chat.image_create(tk.END, image=photo, padx=8, pady=4)
        self.chat.insert(tk.END, "\n", tag)
        self.chat.see(tk.END)
        self.chat.config(state="disabled")

    def add_own_text(self, text, meta):
        ts = datetime.now().strftime("%H:%M")
        self._chat_write(f"\n  Sen  {ts}\n", "own_hdr")
        self._chat_write(f"  {text}\n", "own_body")
        self._chat_write(f"  {meta}\n", "own_meta")

    def add_own_image(self, photo, meta):
        ts = datetime.now().strftime("%H:%M")
        self._chat_write(f"\n  Sen  {ts}\n", "own_hdr")
        self._chat_image(photo, "own_body")
        self._chat_write(f"  {meta}\n", "own_meta")

    def add_inc_text(self, src, text, meta, ok=True):
        ts = datetime.now().strftime("%H:%M")
        self._chat_write(f"\n  Kimden:{src}  {ts}\n", "inc_hdr")
        self._chat_write(f"  {text}\n", "inc_body")
        self._chat_write(f"  {meta}\n", "ok" if ok else "err")

    def add_inc_image(self, src, photo, meta, ok=True):
        ts = datetime.now().strftime("%H:%M")
        self._chat_write(f"\n  Kimden:{src}  {ts}\n", "inc_hdr")
        self._chat_image(photo, "inc_body")
        self._chat_write(f"  {meta}\n", "ok" if ok else "err")

    def add_sys(self, text):
        self._chat_write(f"\n  {text}  \n", "sys")

    # =========================================================================
    # RESIM KODLAMA / COZME
    # =========================================================================
    @staticmethod
    def encode_image(pil_img):
        img = pil_img.convert("L").resize((IMG_W, IMG_H), Image.LANCZOS)
        pixels = list(img.getdata())
        quantized = [int(p * (IMG_LEVELS - 1) / 255) for p in pixels]
        packed = bytearray()
        for i in range(0, len(quantized), 2):
            lo = quantized[i] & 0xF
            hi = (quantized[i + 1] & 0xF) if i + 1 < len(quantized) else 0
            packed.append(lo | (hi << 4))
        return bytes(packed)

    @staticmethod
    def decode_image(data, w=IMG_W, h=IMG_H, partial=False):
        pixels = []
        for byte in data:
            lo = byte & 0xF
            hi = (byte >> 4) & 0xF
            pixels.append(int(lo * 255 / (IMG_LEVELS - 1)))
            pixels.append(int(hi * 255 / (IMG_LEVELS - 1)))
        total = w * h
        if len(pixels) < total:
            if partial:
                pixels.extend([0] * (total - len(pixels)))
            else:
                return None
        img = Image.new("L", (w, h))
        img.putdata(pixels[:total])
        return img

    def _pil_to_tk(self, pil_img, size=(160, 160)):
        photo = ImageTk.PhotoImage(pil_img.resize(size, Image.NEAREST))
        self.photo_refs.append(photo)
        if len(self.photo_refs) > 30:
            self.photo_refs.pop(0)
        return photo

    # =========================================================================
    # ILETIM — Metin
    # =========================================================================
    def send_text(self):
        msg = self.entry.get().strip()
        if not msg or self.is_transmitting:
            return
        my_id  = int(self.my_id_ent.get())
        tgt_id = int(self.target_id_ent.get())
        self.entry.delete(0, tk.END)
        threading.Thread(
            target=self._send_packet,
            args=(my_id, tgt_id, msg.encode("utf-8"), PKT_TEXT),
            daemon=True
        ).start()

    # =========================================================================
    # ILETIM — Resim
    # =========================================================================
    def pick_and_send_image(self):
        if self.is_transmitting:
            return
        path = filedialog.askopenfilename(
            title="Resim Sec",
            filetypes=[
                ("Resim Dosyalari", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff"),
                ("Tum Dosyalar", "*.*")
            ]
        )
        if not path:
            return
        try:
            pil_img = Image.open(path)
        except Exception as e:
            self.add_sys(f"Resim acilamadi: {e}")
            return

        my_id  = int(self.my_id_ent.get())
        tgt_id = int(self.target_id_ent.get())
        payload = self.encode_image(pil_img)

        preview_pil = pil_img.convert("L").resize((IMG_W, IMG_H), Image.LANCZOS)
        thumb = self._pil_to_tk(preview_pil, size=(160, 160))

        threading.Thread(
            target=self._send_packet,
            args=(my_id, tgt_id, payload, PKT_IMAGE),
            kwargs={"thumb": thumb},
            daemon=True
        ).start()

    # =========================================================================
    # ILETIM — Ortak gönderim motoru
    # =========================================================================
    def _send_packet(self, src, dst, payload, pkt_type, thumb=None):
        while self.is_channel_busy:
            time.sleep(0.1)

        self.is_transmitting = True

        # Orijinal payload'ı sakla (chat gösterimi için)
        original_payload = payload

        # --- 1) XOR Cipher (orijinal payload üzerinde) ---
        use_xor = self.xor_enabled.get()
        keyword = self.xor_key_ent.get().strip()
        if use_xor and keyword:
            payload = xor_cipher(payload, keyword)

        # --- 2) Hamming FEC (şifreli payload üzerinde) ---
        use_fec = self.safe_mode.get()
        if use_fec:
            payload = hamming_encode_bytes(payload)

        # --- 3) CRC şifreli+FEC'li son payload üzerinden ---
        sz      = len(payload)
        crc_val = zlib.crc32(payload) & 0xFF

        # Header flags:
        #   bit 0-1: pkt_type
        #   bit 2  : FEC
        #   bit 3  : XOR
        #   bit 4  : channel_mode (0=8ch, 1=16ch)
        ch_mode = self.channel_mode.get()
        frame_size = self._tx_frame_size()

        flags = pkt_type & 0x03
        if use_fec:   flags |= 0x04
        if use_xor:   flags |= 0x08
        if ch_mode == 16: flags |= 0x10

        header = [
            src,
            dst,
            sz & 0xFF,
            (sz >> 8) & 0xFF,
            flags,
            IMG_W if (pkt_type == PKT_IMAGE) else 0,
            IMG_H if (pkt_type == PKT_IMAGE) else 0,
            crc_val,
        ]

        # Header'ı frame_size'a pad'le (8ch=8byte header, 16ch=16byte header)
        while len(header) < frame_size:
            header.append(0)

        raw = header + list(payload)
        while len(raw) % frame_size != 0:
            raw.append(0)

        num_frames = len(raw) // frame_size
        total_est  = 0.5 + num_frames * (self.duration + self.gap)
        start_t    = time.time()

        kind = "RESIM" if pkt_type == PKT_IMAGE else "METIN"
        fec_s = " [FEC]" if use_fec else ""
        xor_s = " [XOR]" if use_xor else ""
        ch_s  = f" [{ch_mode}ch]"
        self._prog_lbl.config(
            text=f"Gonderiliyor... {kind}{fec_s}{xor_s}{ch_s}  |  {sz} byte  |  {num_frames} fr"
        )

        channels = self._tx_channels()
        num_ch   = self._tx_num_ch()
        step     = self._tx_step()
        eq_table = self._tx_eq()

        # 1) Wake-up
        wu_n = int(FS * 0.4)
        wu = np.sin(2*np.pi*WAKE_UP_FREQ*np.linspace(0, 0.4, wu_n)) * windows.tukey(wu_n)
        sd.play(wu, FS); sd.wait(); time.sleep(0.1)

        # 2) Frame frame gönder
        for i in range(0, len(raw), frame_size):
            elapsed   = time.time() - start_t
            remaining = max(0, int(total_est - elapsed))
            self._timer_lbl.config(text=f"{remaining}s kaldi")
            self.progress["value"] = min(100, elapsed / total_est * 100)

            chars = raw[i : i + frame_size]
            t = np.linspace(0, self.duration, int(FS * self.duration))
            wave = np.zeros_like(t)
            for ch_idx, cv in enumerate(chars):
                if ch_idx >= num_ch:
                    break
                b1, b2 = channels[ch_idx]
                lo_nibble = cv % HEX_BASE
                hi_nibble = cv // HEX_BASE
                gray_lo = GRAY_ENCODE[lo_nibble]
                gray_hi = GRAY_ENCODE[hi_nibble]
                freq1 = b1 + gray_lo * step
                freq2 = b2 + gray_hi * step
                eq1, eq2 = eq_table[ch_idx]
                wave += eq1 * np.sin(2*np.pi*freq1*t)
                wave += eq2 * np.sin(2*np.pi*freq2*t)

            final = (wave / (num_ch * 2)) * windows.tukey(len(t))
            sd.play(final.astype(np.float32), FS); sd.wait()
            time.sleep(self.gap)

        elapsed_total = time.time() - start_t
        orig_sz = len(original_payload)
        bps = orig_sz / elapsed_total if elapsed_total > 0 else 0
        self._throughput_lbl.config(text=f"{bps:.1f} byte/sn")

        meta = (f"{orig_sz} byte  ·  {num_frames} fr  ·  "
                f"{elapsed_total:.1f}s  ·  {bps:.1f} B/s{fec_s}{xor_s}{ch_s}")

        if pkt_type == PKT_TEXT:
            self.add_own_text(original_payload.decode("utf-8", errors="replace"), meta)
        else:
            self.add_own_image(thumb, meta)

        self.progress["value"] = 100
        self._prog_lbl.config(text="Hazir")
        self._timer_lbl.config(text="")
        time.sleep(0.5)
        self.progress["value"] = 0
        self.is_transmitting = False

    # =========================================================================
    # ALIM DONGUSU
    # =========================================================================
    def _update_loop(self):
        if not self.is_running:
            return

        data = []
        while not self.msg_queue.empty():
            data.append(self.msg_queue.get_nowait())

        if data and not self.is_transmitting:
            audio = np.concatenate(data).flatten()

            rms = float(np.sqrt(np.mean(audio**2))) * 100
            if not self.is_listening:
                self._update_noise_floor(rms)

            yf    = fft(audio)
            xf    = fftfreq(len(yf), 1 / FS)
            mag   = np.abs(yf[:len(yf)//2])
            freqs = xf[:len(xf)//2]

            # Moving Average
            self.mag_history.append(mag.copy())
            if len(self.mag_history) >= 2:
                mag = np.mean(list(self.mag_history), axis=0)

            # Spektrum
            self.line.set_ydata(np.interp(self.plot_x, freqs, mag))
            self.mpl_canvas.draw_idle()

            thr = self.threshold
            self._thresh_lbl.config(text=f"Esik: {thr:.2f}")

            # Kanal meşguliyeti
            bm   = (freqs > 9000) & (freqs < 18500)
            peak = np.max(mag[bm]) if np.any(bm) else 0
            if peak > thr:
                self.is_channel_busy = True
                self._set_status("Kanal Mesgul", C_DANGER)
            else:
                self.is_channel_busy = False
                self._set_status("Kanal Musait", C_SUCCESS)

            # Wake-up
            wm = (freqs > WAKE_UP_FREQ - 50) & (freqs < WAKE_UP_FREQ + 50)
            wp = np.max(mag[wm]) if np.any(wm) else 0
            if not self.is_listening and wp > thr:
                self.is_listening    = True
                self.start_t         = time.time()
                self.arrival_ts      = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.current_packet  = []
                self.last_frame_set  = None
                self.rx_channel_mode = 0   # henüz bilinmiyor
                self.rx_frame_size   = 0
                self._prev_status.config(text="Aliniyor...")
                self.add_sys("Sinyal alindi, paket bekleniyor...")

            # Veri çözme — her iki modda da dene
            if self.is_listening:
                # İlk frame gelene kadar her iki modu da dene
                # İlk frame'de header okunur, flags'ten kanal modu tespit edilir
                frame_8  = self._try_decode_frame(freqs, mag, thr, CHANNELS_8,  STEP_8,  DET_WIN_8)
                frame_16 = self._try_decode_frame(freqs, mag, thr, CHANNELS_16, STEP_16, DET_WIN_16)

                # Mod henüz belirlenmemişse (ilk frame)
                if self.rx_channel_mode == 0:
                    # İlk tam frame'i yakala — hangisi tam gelirse o moddur
                    if frame_16 is not None and len(frame_16) == 16:
                        # Header flags'e bak: bit4=1 -> 16ch
                        if len(frame_16) >= 5 and (frame_16[4] & 0x10):
                            self.rx_channel_mode = 16
                            self.rx_frame_size   = 16
                            self.current_packet.extend(frame_16)
                            self.last_frame_set   = tuple(frame_16)
                            self.last_signal_time = time.time()
                        elif frame_8 is not None and len(frame_8) == 8:
                            self.rx_channel_mode = 8
                            self.rx_frame_size   = 8
                            self.current_packet.extend(frame_8)
                            self.last_frame_set   = tuple(frame_8)
                            self.last_signal_time = time.time()
                    elif frame_8 is not None and len(frame_8) == 8:
                        # 8-ch header flags bit4=0
                        self.rx_channel_mode = 8
                        self.rx_frame_size   = 8
                        self.current_packet.extend(frame_8)
                        self.last_frame_set   = tuple(frame_8)
                        self.last_signal_time = time.time()

                else:
                    # Mod belirlendi, sadece o modu kullan
                    if self.rx_channel_mode == 16:
                        frame = frame_16
                        expected = 16
                    else:
                        frame = frame_8
                        expected = 8

                    if frame is not None and len(frame) == expected:
                        cs = tuple(frame)
                        if cs != self.last_frame_set:
                            self.current_packet.extend(frame)
                            self.last_frame_set   = cs
                            self.last_signal_time = time.time()
                            self._live_preview()

                if (self.current_packet
                        and time.time() - self.last_signal_time > 1.5):
                    self._finalize()

        self.root.after(50, self._update_loop)

    def _try_decode_frame(self, freqs, mag, thr, channels, step, det_win):
        """Verilen kanal seti ile bir frame decode etmeyi dene.
        Tam frame dönerse list, yoksa None."""
        frame_chars = []
        for b1, b2 in channels:
            m1 = (freqs >= b1) & (freqs <= b1 + det_win)
            m2 = (freqs >= b2) & (freqs <= b2 + det_win)
            if np.any(m1) and np.any(m2):
                i1, i2 = np.argmax(mag[m1]), np.argmax(mag[m2])
                if mag[m1][i1] > thr and mag[m2][i2] > thr:
                    gray_lo = round((freqs[m1][i1] - b1) / step)
                    gray_hi = round((freqs[m2][i2] - b2) / step)
                    gray_lo = max(0, min(15, gray_lo))
                    gray_hi = max(0, min(15, gray_hi))
                    units = GRAY_DECODE[gray_lo]
                    tens  = GRAY_DECODE[gray_hi]
                    cc = max(0, min(255, tens * HEX_BASE + units))
                    frame_chars.append(cc)
                else:
                    return None  # bu kanalda sinyal yok
            else:
                return None
        return frame_chars

    # =========================================================================
    # CANLI RESIM ONIZLEME
    # =========================================================================
    def _live_preview(self):
        fs = self.rx_frame_size or 8
        if len(self.current_packet) < fs:
            return
        pkt_flags = self.current_packet[4]
        if (pkt_flags & 0x03) != PKT_IMAGE:
            return

        img_w = self.current_packet[5] or IMG_W
        img_h = self.current_packet[6] or IMG_H
        sz    = self.current_packet[2] | (self.current_packet[3] << 8)
        img_data = bytes(self.current_packet[fs:])
        if not img_data:
            return

        try:
            # Canlı önizleme: şifreli olabilir, yine de göster (gürültülü görünür)
            pil = self.decode_image(img_data, img_w, img_h, partial=True)
            if pil:
                ph = ImageTk.PhotoImage(pil.resize((120, 120), Image.NEAREST))
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(60, 60, image=ph)
                self.preview_canvas._ph = ph
                recv = len(img_data)
                pct  = int(recv / sz * 100) if sz else 0
                self._prev_prog.config(text=f"{recv} / {sz} byte  ({pct}%)")
                self._prev_size.config(text=f"{img_w}x{img_h}  4-bit gri")
        except Exception:
            pass

    # =========================================================================
    # PAKET SONLANDIRMA
    # =========================================================================
    def _finalize(self):
        total_dur = time.time() - self.start_t - 1.5
        fs = self.rx_frame_size or 8

        if len(self.current_packet) >= fs:
            src      = self.current_packet[0]
            dst      = self.current_packet[1]
            sz       = self.current_packet[2] | (self.current_packet[3] << 8)
            flags    = self.current_packet[4]
            pkt_type = flags & 0x03
            has_fec  = bool(flags & 0x04)
            has_xor  = bool(flags & 0x08)
            is_16ch  = bool(flags & 0x10)
            img_w    = self.current_packet[5] or IMG_W
            img_h    = self.current_packet[6] or IMG_H
            crc      = self.current_packet[7]

            raw_bytes = bytes(self.current_packet[fs : fs + sz])

            my_id = int(self.my_id_ent.get())
            if dst == my_id or dst == 0:
                # CRC kontrolü (şifreli+FEC'li hali üzerinde)
                ok     = (zlib.crc32(raw_bytes) & 0xFF) == crc
                status = "TAM" if ok else "CRC HATA"

                ch_s = f" [{16 if is_16ch else 8}ch]"

                # --- Hamming decode ---
                decoded = raw_bytes
                if has_fec:
                    decoded = hamming_decode_bytes(decoded)
                    status += " [FEC]"

                # --- XOR decipher (kullanıcının keyword'ü ile) ---
                if has_xor:
                    keyword = self.xor_key_ent.get().strip()
                    if keyword:
                        decoded = xor_cipher(decoded, keyword)
                        status += " [XOR]"
                    else:
                        status += " [XOR-ANAHTAR YOK!]"

                orig_sz = len(decoded)
                bps = orig_sz / total_dur if total_dur > 0 else 0
                self._throughput_lbl.config(text=f"{bps:.1f} byte/sn")
                meta = (f"{orig_sz} byte  ·  {total_dur:.1f}s  ·  "
                        f"{bps:.1f} B/s  ·  {status}{ch_s}")

                if pkt_type == PKT_TEXT:
                    text = decoded.decode("utf-8", errors="replace")
                    self.add_inc_text(src, text, meta, ok)

                elif pkt_type == PKT_IMAGE:
                    pil = self.decode_image(decoded, img_w, img_h)
                    if pil:
                        self.last_received_pil = pil
                        photo = self._pil_to_tk(pil, size=(160, 160))
                        ph_s = ImageTk.PhotoImage(pil.resize((120, 120), Image.NEAREST))
                        self.preview_canvas.delete("all")
                        self.preview_canvas.create_image(60, 60, image=ph_s)
                        self.preview_canvas._ph = ph_s
                        self._prev_status.config(text="Son alinan resim")
                        self._prev_prog.config(text=f"{orig_sz} byte  %100")
                        self._prev_size.config(text=f"{img_w}x{img_h}  4-bit gri")
                        self.add_inc_image(src, photo, meta, ok)
                    else:
                        self.add_inc_text(src, "[Resim cozemedi]", meta, False)

        # Sıfırla
        self.current_packet   = []
        self.is_listening     = False
        self.last_frame_set   = None
        self.rx_channel_mode  = 0
        self.rx_frame_size    = 0
        self._prev_status.config(text="Bekleniyor")
        self._prev_prog.config(text="— / — byte")
        self._prev_size.config(text="")


# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = AcousticMessengerV50(root)
    root.mainloop()