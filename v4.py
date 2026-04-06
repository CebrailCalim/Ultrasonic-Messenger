import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import time
from datetime import datetime
import os
import zlib

# =============================================================================
#  OCTA-CHANNEL PARAMETRELERİ  (v48)
#  Değişiklik özeti:
#    · 4 kanal  → 8 kanal   (frame: 4 byte → 8 byte)
#    · Base-10  → Base-16   (kanal başı kapasite: 100 → 256 değer)
#    · STEP 60  → 30 Hz     (dar band'e sığabilmek için)
#    · DET_WIN 700 → 480 Hz (her sub-band 500 Hz genişliğinde)
#    · blocksize 2048 → 4096 (30 Hz adımda yeterli FFT çözünürlüğü)
#    · Varsayılan sembol süresi 0.50 → 0.30 sn
# =============================================================================

FS           = 44100
WAKE_UP_FREQ = 9500

# Her kanal: (f1_base, f2_base)
# f1 ve f2 arası fark 500 Hz → her biri 0..15 adım × 30 Hz = 0..450 Hz yer kaplar
CHANNELS = [
    (10000, 10500),   # Kanal 1
    (11000, 11500),   # Kanal 2
    (12000, 12500),   # Kanal 3
    (13000, 13500),   # Kanal 4
    (14000, 14500),   # Kanal 5
    (15000, 15500),   # Kanal 6
    (16000, 16500),   # Kanal 7
    (17000, 17500),   # Kanal 8
]

STEP        = 30     # Hz — 16 adım × 30 = 480 Hz < 500 Hz band ✓
HEX_BASE    = 16     # Base-16 kodlama
DET_WIN     = 480    # Algılama penceresi genişliği (Hz)
NUM_CH      = 8      # Kanal sayısı
THRESHOLD   = 2.5    # Gürültü eşiği
DEF_DURATION = 0.30  # Varsayılan sembol süresi (sn)
DEF_GAP      = 0.05  # Semboller arası boşluk (sn)
COOLDOWN     = 1.0   # Wake-up sonrası bekleme (sn)


class OctaFrameSyncedNodeV48:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Messenger v48 — Octa-Channel")
        self.root.geometry("1400x950")
        self.root.configure(bg="#f8f9fa")

        # --- DURUM DEĞİŞKENLERİ ---
        self.is_running       = True
        self.duration         = DEF_DURATION
        self.gap              = DEF_GAP
        self.msg_queue        = queue.Queue()
        self.is_transmitting  = False
        self.is_listening     = False
        self.is_channel_busy  = False
        self.current_packet   = []
        self.last_signal_time = 0
        self.last_frame_set   = None  # Tekrar okumayı önleyen kilit

        self.apply_styles()
        self.setup_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.emergency_exit)
        try:
            self.stream = sd.InputStream(
                callback=lambda *args: self.msg_queue.put(args[0].copy()),
                channels=1,
                samplerate=FS,
                blocksize=4096   # ← 2048 → 4096 (30 Hz adım için zorunlu)
            )
            self.stream.start()
        except Exception as e:
            print(f"Mikrofon hatası: {e}")

        self.update_loop()

    # ------------------------------------------------------------------
    #  STİL / UI
    # ------------------------------------------------------------------

    def apply_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(
            "Elegant.Horizontal.TProgressbar",
            background='#0078d4', thickness=22
        )

    def setup_ui(self):
        # --- SIDEBAR ---
        self.sidebar = tk.Frame(
            self.root, bg="#ffffff", width=400,
            padx=25, pady=25,
            highlightthickness=1, highlightbackground="#e1e1e1"
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar, text="KONTROL PANELİ",
            bg="#ffffff", font=("Segoe UI", 16, "bold")
        ).pack(pady=(0, 20))

        tk.Label(self.sidebar, text="Kendi ID (Local):",
                 bg="#ffffff", font=("Segoe UI", 12)).pack(anchor="w")
        self.my_id_ent = tk.Entry(
            self.sidebar, bg="#f3f3f3",
            font=("Segoe UI", 13), justify='center', borderwidth=0
        )
        self.my_id_ent.insert(0, "1")
        self.my_id_ent.pack(fill=tk.X, pady=(5, 15))

        tk.Label(self.sidebar, text="Hedef ID (Remote):",
                 bg="#ffffff", font=("Segoe UI", 12)).pack(anchor="w")
        self.target_id_ent = tk.Entry(
            self.sidebar, bg="#f3f3f3",
            font=("Segoe UI", 13), justify='center', borderwidth=0
        )
        self.target_id_ent.insert(0, "0")
        self.target_id_ent.pack(fill=tk.X, pady=(5, 15))

        # Versiyon + kanal bilgisi
        tk.Label(
            self.sidebar,
            text="v48 · 8 Kanal · Base-16 · ~24 byte/sn",
            bg="#ffffff", fg="#0078d4", font=("Segoe UI", 10, "bold")
        ).pack(pady=(0, 10))

        tk.Label(self.sidebar, text="HIZ AYARLARI",
                 bg="#ffffff", fg="#0078d4",
                 font=("Segoe UI", 12, "bold")).pack(pady=10)

        self.dur_scale = tk.Scale(
            self.sidebar, from_=0.10, to=1.0,
            label="Sembol Süresi (sn)", resolution=0.05,
            orient=tk.HORIZONTAL, bg="#ffffff",
            command=self.update_params
        )
        self.dur_scale.set(DEF_DURATION)
        self.dur_scale.pack(fill=tk.X)

        self.gap_scale = tk.Scale(
            self.sidebar, from_=0.01, to=0.5,
            label="Boşluk Süresi (sn)", resolution=0.01,
            orient=tk.HORIZONTAL, bg="#ffffff",
            command=self.update_params
        )
        self.gap_scale.set(DEF_GAP)
        self.gap_scale.pack(fill=tk.X, pady=10)

        # Canlı hız göstergesi
        self.speed_lbl = tk.Label(
            self.sidebar, text="", bg="#ffffff",
            fg="#555", font=("Segoe UI", 10)
        )
        self.speed_lbl.pack()
        self._refresh_speed_label()

        tk.Button(
            self.sidebar, text="Varsayılanı Geri Yükle",
            fg="#d9534f", font=("Segoe UI", 10, "bold"),
            relief="flat", command=self.reset_defaults
        ).pack(fill=tk.X, pady=(5, 0))

        # Spektrum grafiği
        tk.Label(
            self.sidebar, text="SPEKTRUM ANALİZİ (kHz)",
            bg="#ffffff", font=("Segoe UI", 10, "bold")
        ).pack(pady=(20, 5))

        self.fig, self.ax = plt.subplots(figsize=(4, 3.5))
        self.fig.patch.set_facecolor('#ffffff')
        self.plot_x = np.linspace(9000, 18500, 700)
        self.line, = self.ax.plot(
            self.plot_x, np.zeros(700), color='#0078d4', lw=2
        )
        self.ax.set_facecolor('#f8f9fa')
        self.ax.set_ylim(0, 60)
        self.ax.set_xlim(9000, 18500)
        self.ax.set_xticks([9500, 10500, 12000, 13500, 15000, 16500, 18000])
        self.ax.set_xticklabels(['9.5k', '10.5', '12k', '13.5', '15k', '16.5', '18k'])
        self.ax.grid(True, linestyle='--', alpha=0.3)

        # Kanal band şeritlerini arka planda göster
        band_colors = [
            '#e8f4fd', '#e1f5ee', '#fef9e7', '#fdf0e8',
            '#f0ecfe', '#eaf3de', '#fef9e7', '#fdf0e8'
        ]
        for idx, (b1, b2) in enumerate(CHANNELS):
            self.ax.axvspan(b1, b1 + DET_WIN + STEP,
                            alpha=0.25, color=band_colors[idx], lw=0)
            self.ax.axvspan(b2, b2 + DET_WIN + STEP,
                            alpha=0.25, color=band_colors[idx], lw=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.sidebar)
        self.canvas.get_tk_widget().pack(fill=tk.X)

        # --- SAĞ PANEL ---
        self.main_panel = tk.Frame(
            self.root, bg="#f8f9fa", padx=25, pady=25
        )
        self.main_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.status_lbl = tk.Label(
            self.main_panel, text="● Kanal Müsait",
            bg="#f8f9fa", fg="#28a745",
            font=("Segoe UI", 12, "bold")
        )
        self.status_lbl.pack(anchor="e")

        self.chat_area = scrolledtext.ScrolledText(
            self.main_panel, bg="#ffffff",
            font=("Segoe UI", 14), padx=20, pady=20, borderwidth=0
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=10)

        # Progress + sayaç
        prog_frame = tk.Frame(self.main_panel, bg="#f8f9fa")
        prog_frame.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(
            prog_frame, orient=tk.HORIZONTAL,
            mode='determinate',
            style="Elegant.Horizontal.TProgressbar"
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.timer_lbl = tk.Label(
            prog_frame, text="Kalan: 0s",
            bg="#f8f9fa", fg="#0078d4",
            font=("Segoe UI", 12, "bold")
        )
        self.timer_lbl.pack(side=tk.RIGHT, padx=15)

        # Giriş kutusu
        entry_container = tk.Frame(
            self.main_panel, bg="#ffffff", padx=15, pady=15,
            highlightthickness=1, highlightbackground="#ced4da"
        )
        entry_container.pack(fill=tk.X, pady=10)

        self.entry = tk.Entry(
            entry_container, bg="#ffffff",
            font=("Segoe UI", 15), borderwidth=0
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.send_action())

        tk.Button(
            entry_container, text="Gönder",
            bg="#0078d4", fg="#fff",
            font=("Segoe UI", 12, "bold"),
            relief="flat", padx=35, pady=10,
            command=self.send_action
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            entry_container, text="Temizle",
            font=("Segoe UI", 12), relief="flat",
            command=self.clear_log
        ).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    #  YARDIMCI METODLAR
    # ------------------------------------------------------------------

    def _refresh_speed_label(self):
        """Mevcut ayarlarla teorik throughput'u hesapla ve göster."""
        bps = NUM_CH / (self.duration + self.gap)
        self.speed_lbl.config(
            text=f"Teorik hız: {bps:.1f} byte/sn  "
                 f"({bps * 8:.0f} bit/sn)"
        )

    def update_params(self, _=None):
        self.duration = self.dur_scale.get()
        self.gap      = self.gap_scale.get()
        self._refresh_speed_label()

    def reset_defaults(self):
        self.dur_scale.set(DEF_DURATION)
        self.gap_scale.set(DEF_GAP)
        self.update_params()

    def clear_log(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete('1.0', tk.END)
        self.chat_area.config(state='disabled')

    def emergency_exit(self):
        self.is_running = False
        os._exit(0)

    def add_text(self, text, is_own=False):
        self.chat_area.config(state='normal')
        tag = "own" if is_own else "incoming"
        self.chat_area.insert(tk.END, text, tag)
        self.chat_area.tag_config(
            "own", foreground="#0078d4",
            font=("Segoe UI", 14, "bold")
        )
        self.chat_area.tag_config(
            "incoming", foreground="#2c3e50"
        )
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')

    def send_action(self):
        msg = self.entry.get()
        if msg:
            my_id  = int(self.my_id_ent.get())
            tgt_id = int(self.target_id_ent.get())
            self.entry.delete(0, tk.END)
            threading.Thread(
                target=self.transmit,
                args=(my_id, tgt_id, msg),
                daemon=True
            ).start()

    # ------------------------------------------------------------------
    #  GÖNDERİCİ  (Octa-Channel Base-16)
    # ------------------------------------------------------------------

    def transmit(self, src, dst, text):
        # CSMA — kanal boşalana kadar bekle
        while self.is_channel_busy:
            time.sleep(0.1)

        self.is_transmitting = True

        payload = text.encode('utf-8')
        crc8    = zlib.crc32(payload) & 0xFF
        raw     = [src, dst, len(payload), crc8] + list(payload)

        # 8'in katına tamamla (Octa-Channel frame boyutu)
        while len(raw) % NUM_CH != 0:
            raw.append(0)

        n_frames   = len(raw) // NUM_CH
        total_est  = 0.5 + n_frames * (self.duration + self.gap)
        start_t    = time.time()
        ts         = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 1. Wake-up tonu
        wu_len  = int(FS * 0.4)
        wu_wave = (
            np.sin(2 * np.pi * WAKE_UP_FREQ *
                   np.linspace(0, 0.4, wu_len))
            * windows.tukey(wu_len)
        )
        sd.play(wu_wave, FS)
        sd.wait()
        time.sleep(0.1)

        # 2. Octa-Channel veri gönderimi
        for i in range(0, len(raw), NUM_CH):
            elapsed = time.time() - start_t
            remaining = max(0, int(total_est - elapsed))
            self.timer_lbl.config(text=f"Kalan: {remaining}s")
            self.progress['value'] = (elapsed / total_est) * 100

            chars = raw[i:i + NUM_CH]   # 8 byte
            t     = np.linspace(0, self.duration, int(FS * self.duration))
            wave  = np.zeros_like(t)

            for ch_idx, char_val in enumerate(chars):
                b1, b2 = CHANNELS[ch_idx]
                # Base-16: units = char % 16, tens = char // 16
                f1 = b1 + (char_val % HEX_BASE)  * STEP
                f2 = b2 + (char_val // HEX_BASE) * STEP
                wave += np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

            # 8 kanal × 2 sinüs = 16 bileşen — normalize et
            final = (wave / (NUM_CH * 2)) * windows.tukey(len(t))
            sd.play(final.astype(np.float32), FS)
            sd.wait()
            time.sleep(self.gap)

        elapsed = time.time() - start_t
        self.progress['value'] = 100
        self.add_text(
            f"\n[{ts}] ÇIKIŞ → Hedef:{dst} | "
            f"{len(payload)} byte | {n_frames} frame | "
            f"{elapsed:.2f} sn | "
            f"{len(payload)/elapsed:.1f} byte/sn\n"
            f"SİZ: {text}\n",
            is_own=True
        )
        time.sleep(0.5)
        self.progress['value'] = 0
        self.is_transmitting = False

    # ------------------------------------------------------------------
    #  ALICI DÖNGÜSÜ
    # ------------------------------------------------------------------

    def update_loop(self):
        if not self.is_running:
            return

        data = []
        while not self.msg_queue.empty():
            data.append(self.msg_queue.get_nowait())

        if data and not self.is_transmitting:
            audio = np.concatenate(data).flatten()

            # FFT (4096 noktalı pencere)
            yf    = fft(audio)
            xf    = fftfreq(len(yf), 1 / FS)
            mag   = np.abs(yf[:len(yf) // 2])
            freqs = xf[:len(xf) // 2]

            # Spektrum çizimi
            interp = np.interp(self.plot_x, freqs, mag)
            self.line.set_ydata(interp)
            self.canvas.draw_idle()

            # Kanal meşguliyet tespiti
            band_mag = mag[(freqs > 9000) & (freqs < 18500)]
            peak     = np.max(band_mag) if len(band_mag) else 0

            if peak > THRESHOLD:
                self.is_channel_busy = True
                self.status_lbl.config(text="● Kanal Meşgul", fg="#d9534f")
            else:
                self.is_channel_busy = False
                self.status_lbl.config(text="● Kanal Müsait", fg="#28a745")

            # Wake-up tespiti
            wu_mask = (freqs > WAKE_UP_FREQ - 50) & (freqs < WAKE_UP_FREQ + 50)
            if (not self.is_listening and
                    np.any(wu_mask) and
                    np.max(mag[wu_mask]) > THRESHOLD):
                self.is_listening   = True
                self.start_t        = time.time()
                self.arrival_ts     = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Veri alma
            if self.is_listening:
                frame_chars = self._decode_frame(mag, freqs)

                if len(frame_chars) == NUM_CH:
                    current_set = tuple(frame_chars)
                    if current_set != self.last_frame_set:
                        self.current_packet.extend(frame_chars)
                        self.last_frame_set   = current_set
                        self.last_signal_time = time.time()

                # Sessizlik süresi aşıldıysa paketi kapat
                if (time.time() - self.last_signal_time > 1.5
                        and self.current_packet):
                    self.finalize()

        self.root.after(50, self.update_loop)

    def _decode_frame(self, mag, freqs):
        """
        8 kanalın tamamını tara, her kanaldan 1 byte çöz.
        Tüm kanallar eşik üzerindeyse tam frame (8 eleman) döner,
        aksi hâlde boş liste döner.
        """
        frame_chars = []
        for b1, b2 in CHANNELS:
            m1 = (freqs >= b1) & (freqs <= b1 + DET_WIN)
            m2 = (freqs >= b2) & (freqs <= b2 + DET_WIN)

            if not (np.any(m1) and np.any(m2)):
                break

            i1 = np.argmax(mag[m1])
            i2 = np.argmax(mag[m2])

            if mag[m1][i1] <= THRESHOLD or mag[m2][i2] <= THRESHOLD:
                break

            # Base-16 decode: f1 → units, f2 → tens
            units = round((freqs[m1][i1] - b1) / STEP)
            tens  = round((freqs[m2][i2] - b2) / STEP)

            # Sınır kontrolü (gürültüden kaynaklı taşmayı önle)
            units = max(0, min(units, HEX_BASE - 1))
            tens  = max(0, min(tens,  HEX_BASE - 1))

            char_code = tens * HEX_BASE + units
            frame_chars.append(char_code)

        return frame_chars if len(frame_chars) == NUM_CH else []

    # ------------------------------------------------------------------
    #  PAKET FİNALİZASYONU
    # ------------------------------------------------------------------

    def finalize(self):
        total_dur = time.time() - self.start_t - 1.5

        if len(self.current_packet) >= NUM_CH:
            src, dst, sz, crc = self.current_packet[:4]

            # Başlık dışı veri byte'larını al, null padding'i temizle
            raw_data = bytes(
                c for c in self.current_packet[4:]
                if c != 0
            )[:sz]

            try:
                msg = raw_data.decode('utf-8', errors='replace')
            except Exception:
                msg = repr(raw_data)

            if dst == int(self.my_id_ent.get()) or dst == 0:
                calc_crc = zlib.crc32(raw_data) & 0xFF
                status   = "✓ TAM" if calc_crc == crc else "✗ HATA"
                bps      = sz / total_dur if total_dur > 0 else 0

                self.add_text(
                    f"\n[{self.arrival_ts}] GİRİŞ ← Kimden:{src} | "
                    f"{sz} byte | {total_dur:.2f} sn | "
                    f"{bps:.1f} byte/sn | {status}\n"
                    f"GELEN: {msg}\n"
                )

        # Sıfırla
        self.current_packet   = []
        self.is_listening     = False
        self.last_frame_set   = None


# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = OctaFrameSyncedNodeV48(root)
    root.mainloop()