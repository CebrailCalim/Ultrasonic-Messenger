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

# =============================================================================
# OCTA-CHANNEL PARAMETRELER  (v48-img)
# =============================================================================
FS           = 44100
WAKE_UP_FREQ = 9500

CHANNELS = [
    (10000, 10500),
    (11000, 11500),
    (12000, 12500),
    (13000, 13500),
    (14000, 14500),
    (15000, 15500),
    (16000, 16500),
    (17000, 17500),
]

STEP       = 30
HEX_BASE   = 16
DET_WIN    = 480
NUM_CH     = 8
FRAME_SIZE = 8

THRESHOLD    = 2.5
DEF_DURATION = 0.30
DEF_GAP      = 0.05

# --- Resim sabitleri ---
# 32x32 gri, 4-bit (16 seviye), nibble-packed
# 32*32 = 1024 piksel / 2 = 512 byte
# 512 byte / 8 byte/frame = 64 frame
# 64 * 0.35s = ~22.4s iletim suresi
IMG_W      = 32
IMG_H      = 32
IMG_LEVELS = 16

# --- Paket tipleri (header[4]) ---
PKT_TEXT  = 0x00
PKT_IMAGE = 0x01

# --- Renk paleti ---
C_SB      = "#0f172a"   # sidebar bg (slate-900)
C_SB2     = "#1e293b"   # sidebar surface (slate-800)
C_SB3     = "#334155"   # sidebar hover (slate-700)
C_ACCENT  = "#3b82f6"   # blue-500
C_ACCENT2 = "#1d4ed8"   # blue-700
C_TL      = "#f1f5f9"   # text light
C_TM      = "#94a3b8"   # text muted
C_MAIN    = "#f8fafc"   # main bg
C_WHITE   = "#ffffff"
C_SUCCESS = "#22c55e"
C_DANGER  = "#ef4444"
C_WARN    = "#f59e0b"
C_OWN_BG  = "#dbeafe"
C_OWN_FG  = "#1e40af"
C_INC_BG  = "#f1f5f9"
C_INC_FG  = "#1e293b"
C_BORDER  = "#e2e8f0"


class AcousticMessengerV48:
    def __init__(self, root):
        self.root = root
        self.root.title("AcousticMaster v48")
        self.root.geometry("1480x960")
        self.root.configure(bg=C_MAIN)

        # Durum
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
        self.photo_refs       = []   # PhotoImage referans deposu (GC koruma)

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
    # UI INSASI
    # =========================================================================
    def _build_ui(self):
        self._build_sidebar()
        self._build_main()

    # ---- Sidebar ----
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=C_SB, width=390)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        self.sb = sb

        # Baslik
        hdr = tk.Frame(sb, bg=C_SB2, pady=18, padx=22)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="AcousticMaster",
                 bg=C_SB2, fg=C_TL, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(hdr, text="v48  ·  Octa-Channel  ·  Base-16  ·  Resim",
                 bg=C_SB2, fg=C_TM, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

        # Durum satiri
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

        # Hiz
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

        # Resim onizleme
        self._sec("ALINAN RESIM ONIZLEME")
        pf = tk.Frame(sb, bg=C_SB, padx=22)
        pf.pack(fill=tk.X)

        self.preview_canvas = tk.Canvas(
            pf, width=120, height=120,
            bg=C_SB2, highlightthickness=0
        )
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

        ch_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444',
                     '#8b5cf6', '#06b6d4', '#f97316', '#ec4899']
        for idx, (b1, b2) in enumerate(CHANNELS):
            self.ax.axvspan(b1, b1 + DET_WIN, alpha=0.18, color=ch_colors[idx])
            self.ax.axvspan(b2, b2 + DET_WIN, alpha=0.18, color=ch_colors[idx])

        self.fig.tight_layout(pad=0.4)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=sb)
        self.mpl_canvas.get_tk_widget().configure(bg=C_SB)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.X, padx=16, pady=(0, 16))

    # ---- Ana panel ----
    def _build_main(self):
        main = tk.Frame(self.root, bg=C_MAIN)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.main = main

        # Ust bar
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

        # Sohbet alani
        chat_wrap = tk.Frame(main, bg=C_MAIN, padx=20, pady=12)
        chat_wrap.pack(fill=tk.BOTH, expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_wrap, bg=C_WHITE, font=("Segoe UI", 13),
            padx=18, pady=16, borderwidth=0, relief="flat",
            state="disabled", cursor="arrow"
        )
        self.chat.pack(fill=tk.BOTH, expand=True)

        # Etiket stilleri
        self.chat.tag_config("own_hdr",
            foreground=C_OWN_FG, font=("Segoe UI", 9, "bold"),
            spacing3=2)
        self.chat.tag_config("own_body",
            foreground=C_OWN_FG, font=("Segoe UI", 13),
            background=C_OWN_BG, lmargin1=48, lmargin2=48,
            rmargin=20, spacing1=4, spacing3=4)
        self.chat.tag_config("own_meta",
            foreground="#93c5fd", font=("Segoe UI", 8),
            lmargin1=48, spacing3=8)
        self.chat.tag_config("inc_hdr",
            foreground="#475569", font=("Segoe UI", 9, "bold"),
            spacing3=2)
        self.chat.tag_config("inc_body",
            foreground=C_INC_FG, font=("Segoe UI", 13),
            background=C_INC_BG, lmargin1=20, lmargin2=20,
            rmargin=60, spacing1=4, spacing3=4)
        self.chat.tag_config("inc_meta",
            foreground="#94a3b8", font=("Segoe UI", 8),
            lmargin1=20, spacing3=8)
        self.chat.tag_config("sys",
            foreground="#cbd5e1", font=("Segoe UI", 8, "italic"),
            justify="center", spacing1=6, spacing3=6)
        self.chat.tag_config("ok",
            foreground=C_SUCCESS, font=("Segoe UI", 8))
        self.chat.tag_config("err",
            foreground=C_DANGER, font=("Segoe UI", 8))

        # Progress bar + zamanlayici
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

        # Giris alani
        inp = tk.Frame(main, bg=C_WHITE, padx=20, pady=14,
                       highlightthickness=1, highlightbackground=C_BORDER)
        inp.pack(fill=tk.X)

        self.entry = tk.Entry(
            inp, bg="#f8fafc", fg="#0f172a",
            font=("Segoe UI", 13),
            relief="solid", bd=1,
            insertbackground="#0f172a"
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_text())

        self._img_btn = tk.Button(
            inp, text="Resim",
            bg=C_SB2, fg=C_TL,
            font=("Segoe UI", 11), relief="flat",
            padx=16, pady=8, cursor="hand2",
            command=self.pick_and_send_image
        )
        self._img_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._send_btn = tk.Button(
            inp, text="Gonder",
            bg=C_ACCENT, fg=C_WHITE,
            font=("Segoe UI", 12, "bold"), relief="flat",
            padx=26, pady=8, cursor="hand2",
            command=self.send_text
        )
        self._send_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            inp, text="Temizle",
            bg=C_MAIN, fg=C_TM,
            font=("Segoe UI", 11), relief="flat",
            padx=12, pady=8, cursor="hand2",
            command=self.clear_log
        ).pack(side=tk.LEFT)

    # =========================================================================
    # YARDIMCILAR — UI
    # =========================================================================
    def _sep(self):
        tk.Frame(self.sb, bg=C_SB2, height=1).pack(fill=tk.X, padx=22, pady=8)

    def _sec(self, text):
        f = tk.Frame(self.sb, bg=C_SB, padx=22) # pady'i buradan kaldırdık
        f.pack(fill=tk.X, pady=(10, 3))         # pack içine taşıdık
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

    # =========================================================================
    # SOHBET MESAJLARI
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
        """PIL Image -> nibble-packed bytes.
        32x32 gri, 16 seviye (4-bit) -> 512 byte."""
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
        """Nibble-packed bytes -> PIL Image."""
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
        """PIL -> PhotoImage; referansi sakla."""
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

        # Gonderilecek resmin gri halini onizleme olarak goster
        preview_pil = pil_img.convert("L").resize((IMG_W, IMG_H), Image.LANCZOS)
        thumb = self._pil_to_tk(preview_pil, size=(160, 160))

        threading.Thread(
            target=self._send_packet,
            args=(my_id, tgt_id, payload, PKT_IMAGE),
            kwargs={"thumb": thumb},
            daemon=True
        ).start()

    # =========================================================================
    # ILETIM — Ortak gonderim motoru
    # =========================================================================
    def _send_packet(self, src, dst, payload, pkt_type, thumb=None):
        # CSMA
        while self.is_channel_busy:
            time.sleep(0.1)

        self.is_transmitting = True
        sz      = len(payload)
        crc_val = zlib.crc32(payload) & 0xFF

        # 8-byte baslik (= 1 tam cerceve)
        # [src, dst, sz_lo, sz_hi, pkt_type, img_w, img_h, crc8]
        header = [
            src,
            dst,
            sz & 0xFF,
            (sz >> 8) & 0xFF,
            pkt_type,
            IMG_W if pkt_type == PKT_IMAGE else 0,
            IMG_H if pkt_type == PKT_IMAGE else 0,
            crc_val,
        ]
        raw = header + list(payload)
        while len(raw) % FRAME_SIZE != 0:
            raw.append(0)

        num_frames = len(raw) // FRAME_SIZE
        total_est  = 0.5 + num_frames * (self.duration + self.gap)
        start_t    = time.time()

        kind = "RESIM" if pkt_type == PKT_IMAGE else "METIN"
        self._prog_lbl.config(
            text=f"Gonderiliyor... {kind}  |  {sz} byte  |  {num_frames} frame"
        )

        # 1) Wake-up tonu
        wu_n  = int(FS * 0.4)
        wu    = np.sin(2*np.pi*WAKE_UP_FREQ*np.linspace(0, 0.4, wu_n)) * windows.tukey(wu_n)
        sd.play(wu, FS); sd.wait(); time.sleep(0.1)

        # 2) Cerceve cerceve gonder
        for i in range(0, len(raw), FRAME_SIZE):
            elapsed   = time.time() - start_t
            remaining = max(0, int(total_est - elapsed))
            self._timer_lbl.config(text=f"{remaining}s kaldi")
            self.progress["value"] = min(100, elapsed / total_est * 100)

            chars = raw[i : i + FRAME_SIZE]
            t = np.linspace(0, self.duration, int(FS * self.duration))
            wave = np.zeros_like(t)
            for ch_idx, cv in enumerate(chars):
                b1, b2 = CHANNELS[ch_idx]
                wave += np.sin(2*np.pi*(b1 + (cv % HEX_BASE) * STEP)*t)
                wave += np.sin(2*np.pi*(b2 + (cv // HEX_BASE) * STEP)*t)
            final = (wave / (NUM_CH * 2)) * windows.tukey(len(t))
            sd.play(final.astype(np.float32), FS); sd.wait()
            time.sleep(self.gap)

        elapsed_total = time.time() - start_t
        bps = sz / elapsed_total if elapsed_total > 0 else 0
        self._throughput_lbl.config(text=f"{bps:.1f} byte/sn")

        meta = (f"{sz} byte  ·  {num_frames} frame  ·  "
                f"{elapsed_total:.1f}s  ·  {bps:.1f} B/s")

        if pkt_type == PKT_TEXT:
            self.add_own_text(payload.decode("utf-8", errors="replace"), meta)
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
            yf    = fft(audio)
            xf    = fftfreq(len(yf), 1 / FS)
            mag   = np.abs(yf[:len(yf)//2])
            freqs = xf[:len(xf)//2]

            # Spektrum
            self.line.set_ydata(np.interp(self.plot_x, freqs, mag))
            self.mpl_canvas.draw_idle()

            # Kanal mesguliyeti
            bm   = (freqs > 9000) & (freqs < 18500)
            peak = np.max(mag[bm]) if np.any(bm) else 0
            if peak > THRESHOLD:
                self.is_channel_busy = True
                self._set_status("Kanal Mesgul", C_DANGER)
            else:
                self.is_channel_busy = False
                self._set_status("Kanal Musait", C_SUCCESS)

            # Wake-up
            wm = (freqs > WAKE_UP_FREQ - 50) & (freqs < WAKE_UP_FREQ + 50)
            wp = np.max(mag[wm]) if np.any(wm) else 0
            if not self.is_listening and wp > THRESHOLD:
                self.is_listening    = True
                self.start_t         = time.time()
                self.arrival_ts      = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.current_packet  = []
                self.last_frame_set  = None
                self._prev_status.config(text="Aliníyor...")
                self.add_sys("Sinyal alindi, paket bekleniyor...")

            # Veri cozme
            if self.is_listening:
                frame_chars = []
                for b1, b2 in CHANNELS:
                    m1 = (freqs >= b1) & (freqs <= b1 + DET_WIN)
                    m2 = (freqs >= b2) & (freqs <= b2 + DET_WIN)
                    if np.any(m1) and np.any(m2):
                        i1, i2 = np.argmax(mag[m1]), np.argmax(mag[m2])
                        if mag[m1][i1] > THRESHOLD and mag[m2][i2] > THRESHOLD:
                            units = round((freqs[m1][i1] - b1) / STEP)
                            tens  = round((freqs[m2][i2] - b2) / STEP)
                            cc    = max(0, min(255, tens * HEX_BASE + units))
                            frame_chars.append(cc)

                if len(frame_chars) == FRAME_SIZE:
                    cs = tuple(frame_chars)
                    if cs != self.last_frame_set:
                        self.current_packet.extend(frame_chars)
                        self.last_frame_set   = cs
                        self.last_signal_time = time.time()
                        self._live_preview()

                if (self.current_packet
                        and time.time() - self.last_signal_time > 1.5):
                    self._finalize()

        self.root.after(50, self._update_loop)

    # =========================================================================
    # CANLI RESIM ONIZLEME (alim sirasinda)
    # =========================================================================
    def _live_preview(self):
        if len(self.current_packet) < FRAME_SIZE:
            return
        if self.current_packet[4] != PKT_IMAGE:
            return

        img_w = self.current_packet[5] or IMG_W
        img_h = self.current_packet[6] or IMG_H
        sz    = self.current_packet[2] | (self.current_packet[3] << 8)
        img_data = bytes(self.current_packet[FRAME_SIZE:])
        if not img_data:
            return

        try:
            pil = self.decode_image(img_data, img_w, img_h, partial=True)
            if pil:
                ph = ImageTk.PhotoImage(pil.resize((120, 120), Image.NEAREST))
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(60, 60, image=ph)
                self.preview_canvas._ph = ph   # GC koruma
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

        if len(self.current_packet) >= FRAME_SIZE:
            src      = self.current_packet[0]
            dst      = self.current_packet[1]
            sz       = self.current_packet[2] | (self.current_packet[3] << 8)
            pkt_type = self.current_packet[4]
            img_w    = self.current_packet[5] or IMG_W
            img_h    = self.current_packet[6] or IMG_H
            crc      = self.current_packet[7]

            raw_bytes = bytes(self.current_packet[FRAME_SIZE : FRAME_SIZE + sz])

            my_id = int(self.my_id_ent.get())
            if dst == my_id or dst == 0:
                ok     = (zlib.crc32(raw_bytes) & 0xFF) == crc
                status = "TAM" if ok else "CRC HATA"
                bps    = len(raw_bytes) / total_dur if total_dur > 0 else 0
                self._throughput_lbl.config(text=f"{bps:.1f} byte/sn")
                meta = (f"{len(raw_bytes)} byte  ·  {total_dur:.1f}s  ·  "
                        f"{bps:.1f} B/s  ·  {status}")

                if pkt_type == PKT_TEXT:
                    text = raw_bytes.decode("utf-8", errors="replace")
                    self.add_inc_text(src, text, meta, ok)

                elif pkt_type == PKT_IMAGE:
                    pil = self.decode_image(raw_bytes, img_w, img_h)
                    if pil:
                        photo = self._pil_to_tk(pil, size=(160, 160))
                        # Sidebar onizlemeyi kalici yap
                        ph_s = ImageTk.PhotoImage(pil.resize((120, 120), Image.NEAREST))
                        self.preview_canvas.delete("all")
                        self.preview_canvas.create_image(60, 60, image=ph_s)
                        self.preview_canvas._ph = ph_s
                        self._prev_status.config(text="Son alinan resim")
                        self._prev_prog.config(text=f"{len(raw_bytes)} byte  %100")
                        self._prev_size.config(text=f"{img_w}x{img_h}  4-bit gri")
                        self.add_inc_image(src, photo, meta, ok)
                    else:
                        self.add_inc_text(src, "[Resim cozemedi]", meta, False)

        # Sifirla
        self.current_packet  = []
        self.is_listening    = False
        self.last_frame_set  = None
        self._prev_status.config(text="Bekleniyor")
        self._prev_prog.config(text="— / — byte")
        self._prev_size.config(text="")


# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = AcousticMessengerV48(root)
    root.mainloop()