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
import os, zlib, heapq, struct
from PIL import Image, ImageTk
from collections import deque, Counter

# =============================================================================
# SABITLER (v51)
# =============================================================================
FS            = 44100
WAKE_UP_FREQ  = 9500
PILOT_LO      = 9000      # Doppler pilot 1
PILOT_HI      = 19500     # Doppler pilot 2
ACK_FREQ      = 9200
NACK_FREQ     = 8800
PING_FREQ     = 8500
PONG_FREQ     = 8600
SPEED_SOUND   = 343.0     # m/s

# --- 8 Kanal ---
CHANNELS_8 = [
    (10000,10500),(11000,11500),(12000,12500),(13000,13500),
    (14000,14500),(15000,15500),(16000,16500),(17000,17500),
]
STEP_8, DET_WIN_8 = 30, 480

# --- 16 Kanal  (10kHz–19kHz, 560Hz aralık, 280Hz sub-gap) ---
CHANNELS_16 = [(10000+i*560, 10000+i*560+280) for i in range(16)]
STEP_16, DET_WIN_16 = 15, 240

HEX_BASE     = 16
DEF_DURATION = 0.30
DEF_GAP      = 0.05
HEADER_SIZE  = 16          # byte — her iki modda da sabit
FRAG_MAX     = 256         # byte — fragment sınırı
ARQ_RETRIES  = 3
ACK_TIMEOUT  = 2.0         # saniye

IMG_W, IMG_H, IMG_LEVELS = 32, 32, 16
PKT_TEXT, PKT_IMAGE = 0x00, 0x01

# =============================================================================
# GRAY CODE
# =============================================================================
GRAY_ENC = [i ^ (i >> 1) for i in range(16)]
GRAY_DEC = [0]*16
for _i, _g in enumerate(GRAY_ENC):
    GRAY_DEC[_g] = _i

# =============================================================================
# HAMMING (7,4)
# =============================================================================
_G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],
               [0,0,1,0,0,1,1],[0,0,0,1,1,1,1]], dtype=np.uint8)
_H = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],
               [0,1,1,1,0,0,1]], dtype=np.uint8)

def _ham_enc_nib(n):
    d = np.array([(n>>i)&1 for i in range(4)], dtype=np.uint8)
    c = (d@_G)%2; return int(sum(c[i]<<i for i in range(7)))

def _ham_dec_7(c7):
    r = np.array([(c7>>i)&1 for i in range(7)], dtype=np.uint8)
    s = int(sum(((_H@r)%2)[i]<<i for i in range(3)))
    if 0 < s <= 7: r[s-1] ^= 1
    return int(sum(r[i]<<i for i in range(4)))

def hamming_enc(data: bytes) -> bytes:
    o = bytearray()
    for b in data:
        w = _ham_enc_nib(b&0xF) | (_ham_enc_nib((b>>4)&0xF)<<7)
        o += bytes([w&0xFF, (w>>8)&0xFF])
    return bytes(o)

def hamming_dec(data: bytes) -> bytes:
    o = bytearray()
    for i in range(0, len(data)-1, 2):
        w = data[i]|(data[i+1]<<8)
        o.append((_ham_dec_7((w>>7)&0x7F)<<4)|_ham_dec_7(w&0x7F))
    return bytes(o)

# =============================================================================
# XOR STREAM CIPHER — Keyword
# =============================================================================
def xor_cipher(payload: bytes, keyword: str) -> bytes:
    if not keyword: return payload
    seed = 0x5A5A5A5A
    for ch in keyword: seed = ((seed*31)+ord(ch))&0xFFFFFFFF
    if seed == 0: seed = 0xDEADBEEF
    st = seed; o = bytearray()
    for b in payload:
        st = (st*1103515245+12345)&0xFFFFFFFF
        o.append(b^((st>>16)&0xFF))
    return bytes(o)

# =============================================================================
# HUFFMAN COMPRESSION (Metin)
# =============================================================================
class _HNode:
    __slots__ = ('f','c','l','r')
    def __init__(self, f, c=None, l=None, r=None):
        self.f, self.c, self.l, self.r = f, c, l, r
    def __lt__(self, o): return self.f < o.f

def huffman_compress(data: bytes) -> bytes:
    if len(data) < 4: return data
    freq = Counter(data)
    if len(freq) == 1:
        b = list(freq.keys())[0]
        return struct.pack('<BH', b, len(data))  # tek karakter özel durum
    heap = [_HNode(f, c=b) for b, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, _HNode(a.f+b.f, l=a, r=b))
    root = heap[0]
    codes = {}
    def _walk(nd, path):
        if nd.c is not None: codes[nd.c] = path or '0'; return
        _walk(nd.l, path+'0'); _walk(nd.r, path+'1')
    _walk(root, '')
    bits = ''.join(codes[b] for b in data)
    pad = (8 - len(bits) % 8) % 8
    bits += '0' * pad
    enc = bytearray()
    for i in range(0, len(bits), 8):
        enc.append(int(bits[i:i+8], 2))
    # Header: num_symbols(1) + [(byte, codelen, code_bits)...] + pad(1) + data
    hdr = bytearray()
    hdr.append(len(freq) - 1)  # 0-based count (max 255)
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
    if len(data) < 3: return data
    idx = 0
    ns = data[idx] + 1; idx += 1
    if ns == 1 and idx + 2 <= len(data):
        # Olası tek-karakter formatı kontrol
        pass
    codes_rev = {}
    for _ in range(ns):
        if idx >= len(data): return data
        sym = data[idx]; idx += 1
        clen = data[idx]; idx += 1
        if clen <= 8:
            cv = data[idx]; idx += 1
        elif clen <= 16:
            cv = struct.unpack_from('<H', data, idx)[0]; idx += 2
        else:
            cv = int.from_bytes(data[idx:idx+3], 'little'); idx += 3
        code_str = format(cv, f'0{clen}b')
        codes_rev[code_str] = sym
    if idx >= len(data): return data
    pad = data[idx]; idx += 1
    bits = ''.join(format(b, '08b') for b in data[idx:])
    if pad: bits = bits[:-pad]
    out = bytearray()
    cur = ''
    for bit in bits:
        cur += bit
        if cur in codes_rev:
            out.append(codes_rev[cur]); cur = ''
        if len(out) >= orig_len: break
    return bytes(out[:orig_len])

# =============================================================================
# DELTA + RLE COMPRESSION (Resim)
# =============================================================================
def delta_rle_compress(data: bytes) -> bytes:
    if len(data) < 2: return data
    deltas = [data[0]] + [((data[i]-data[i-1])&0xFF) for i in range(1, len(data))]
    # RLE: (count, value) — count 1-255
    rle = bytearray()
    i = 0
    while i < len(deltas):
        val = deltas[i]; count = 1
        while i+count < len(deltas) and deltas[i+count] == val and count < 255:
            count += 1
        rle.append(count); rle.append(val)
        i += count
    return bytes(rle) if len(rle) < len(data) else data

def delta_rle_decompress(data: bytes, orig_len: int) -> bytes:
    deltas = []
    i = 0
    while i+1 < len(data) and len(deltas) < orig_len:
        count, val = data[i], data[i+1]; i += 2
        deltas.extend([val]*count)
    # Undo delta
    if not deltas: return data
    out = [deltas[0]]
    for j in range(1, min(len(deltas), orig_len)):
        out.append((out[-1]+deltas[j])&0xFF)
    return bytes(out[:orig_len])

# =============================================================================
# FHSS — Frequency Hopping
# =============================================================================
def fhss_permutation(frame_idx, seed, num_ch):
    rng = (frame_idx*2654435761+seed)&0xFFFFFFFF
    perm = list(range(num_ch))
    for i in range(num_ch-1, 0, -1):
        rng = (rng*1103515245+12345)&0xFFFFFFFF
        j = rng%(i+1); perm[i], perm[j] = perm[j], perm[i]
    return perm

def fhss_inv(perm):
    inv = [0]*len(perm)
    for i, p in enumerate(perm): inv[p] = i
    return inv

def fhss_seed_from_key(keyword: str) -> int:
    s = 0x12345678
    for c in keyword: s = ((s*37)+ord(c))&0xFFFFFFFF
    return s if s else 0xABCD1234

# =============================================================================
# EQ
# =============================================================================
_EQ_F = [10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]
_EQ_G = [1.0,  1.05, 1.12, 1.20, 1.30, 1.42, 1.55, 1.70, 1.85, 2.0,  2.15]
def _eq(f):
    if f<=_EQ_F[0]: return _EQ_G[0]
    if f>=_EQ_F[-1]: return _EQ_G[-1]
    for i in range(len(_EQ_F)-1):
        if _EQ_F[i]<=f<=_EQ_F[i+1]:
            t=(f-_EQ_F[i])/(_EQ_F[i+1]-_EQ_F[i]); return _EQ_G[i]+t*(_EQ_G[i+1]-_EQ_G[i])
    return 1.0

EQ8  = [(_eq(b1+DET_WIN_8/2), _eq(b2+DET_WIN_8/2)) for b1,b2 in CHANNELS_8]
EQ16 = [(_eq(b1+DET_WIN_16/2), _eq(b2+DET_WIN_16/2)) for b1,b2 in CHANNELS_16]

# =============================================================================
# RENKLER
# =============================================================================
C_SB='#0f172a'; C_SB2='#1e293b'; C_SB3='#334155'
C_ACC='#3b82f6'; C_ACC2='#1d4ed8'
C_TL='#f1f5f9'; C_TM='#94a3b8'; C_MAIN='#f8fafc'; C_WHITE='#ffffff'
C_OK='#22c55e'; C_ERR='#ef4444'; C_WARN='#f59e0b'
C_OBG='#dbeafe'; C_OFG='#1e40af'; C_IBG='#f1f5f9'; C_IFG='#1e293b'
C_BRD='#e2e8f0'


# #############################################################################
# ANA SINIF
# #############################################################################
class AcousticMasterV51:
    def __init__(self, root):
        self.root = root
        self.root.title("AcousticMaster v51")
        self.root.geometry("1520x960")
        self.root.configure(bg=C_MAIN)

        # --- Durum ---
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
        self.photo_refs       = []
        self.last_received_pil= None

        # --- Kontrol degiskenleri ---
        self.safe_mode     = tk.BooleanVar(value=False)
        self.xor_enabled   = tk.BooleanVar(value=True)
        self.channel_mode  = tk.IntVar(value=16)
        self.fhss_enabled  = tk.BooleanVar(value=False)
        self.stego_enabled = tk.BooleanVar(value=False)
        self.comp_enabled  = tk.BooleanVar(value=True)
        self.mesh_enabled  = tk.BooleanVar(value=False)
        self.hop_count_var = tk.IntVar(value=2)

        # --- DSP durumlari ---
        self.noise_floor   = 2.5
        self.noise_samples = deque(maxlen=60)
        self.NOISE_MULT    = 1.5
        self.mag_history   = deque(maxlen=3)
        self.doppler_ratio = 1.0      # frekans düzeltme çarpanı
        self.current_snr   = 0.0
        self.last_comp_ratio = 0.0
        self.last_distance = 0.0
        self.sonar_waiting = False
        self.sonar_send_t  = 0

        # --- Fragment birleştirme tamponu ---
        self.frag_buffer   = {}       # seq_num -> decoded_bytes
        self.frag_meta     = {}       # bilgi: total_chunks vb.

        # --- RX kanal algılama ---
        self.rx_ch_mode    = 0
        self.rx_frame_size = 0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._exit)

        try:
            self.stream = sd.InputStream(
                callback=lambda *a: self.msg_queue.put(a[0].copy()),
                channels=1, samplerate=FS, blocksize=4096)
            self.stream.start()
        except Exception as e:
            print(f"Mikrofon hatasi: {e}")

        self._update_loop()

    # ======================= YARDIMCI PROPERTY'LER ===========================
    @property
    def threshold(self):
        return max(1.5, self.noise_floor * self.NOISE_MULT)

    def _nf_update(self, rms):
        self.noise_samples.append(rms)
        if len(self.noise_samples) >= 5:
            self.noise_floor = float(np.median(list(self.noise_samples)))

    def _tx_ch(self):
        return CHANNELS_16 if self.channel_mode.get()==16 else CHANNELS_8
    def _tx_n(self):  return self.channel_mode.get()
    def _tx_fs(self): return self.channel_mode.get()  # frame_size = num_ch
    def _tx_step(self): return STEP_16 if self.channel_mode.get()==16 else STEP_8
    def _tx_dw(self):   return DET_WIN_16 if self.channel_mode.get()==16 else DET_WIN_8
    def _tx_eq(self):   return EQ16 if self.channel_mode.get()==16 else EQ8
    def _xor_key(self): return self.xor_key_ent.get().strip()

    # ======================= UI =============================================
    def _build_ui(self):
        self._build_sidebar()
        self._build_main()

    # ---------- sidebar ----------
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=C_SB, width=400)
        sb.pack(side=tk.LEFT, fill=tk.Y); sb.pack_propagate(False)
        self.sb = sb

        # canvas + scrollbar ile scrollable sidebar
        sf_canvas = tk.Canvas(sb, bg=C_SB, highlightthickness=0, width=380)
        sf_scroll = tk.Scrollbar(sb, orient=tk.VERTICAL, command=sf_canvas.yview)
        self.sb_inner = tk.Frame(sf_canvas, bg=C_SB)

        self.sb_inner.bind("<Configure>",
            lambda e: sf_canvas.configure(scrollregion=sf_canvas.bbox("all")))
        sf_canvas.create_window((0, 0), window=self.sb_inner, anchor="nw", width=380)
        sf_canvas.configure(yscrollcommand=sf_scroll.set)

        sf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        sf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse-wheel scroll
        def _on_wheel(e):
            sf_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        sf_canvas.bind_all("<MouseWheel>", _on_wheel)

        s = self.sb_inner  # kısayol

        # --- Başlık ---
        hdr = tk.Frame(s, bg=C_SB2, pady=14, padx=18); hdr.pack(fill=tk.X)
        tk.Label(hdr, text="AcousticMaster", bg=C_SB2, fg=C_TL,
                 font=("Segoe UI",16,"bold")).pack(anchor="w")
        tk.Label(hdr, text="v51 · FHSS · Doppler · ARQ · Mesh · Stego · Sonar",
                 bg=C_SB2, fg=C_TM, font=("Segoe UI",8)).pack(anchor="w",pady=(2,0))

        # --- Durum ---
        sf = tk.Frame(s, bg=C_SB, padx=18, pady=8); sf.pack(fill=tk.X)
        self._sdot = tk.Canvas(sf, width=10, height=10, bg=C_SB, highlightthickness=0)
        self._sdot.pack(side=tk.LEFT)
        self._sdot.create_oval(2,2,9,9, fill=C_OK, outline="", tags="dot")
        self._slbl = tk.Label(sf, text="Kanal Musait", bg=C_SB, fg=C_OK,
                              font=("Segoe UI",9,"bold"))
        self._slbl.pack(side=tk.LEFT, padx=(6,0))
        self._thr_lbl = tk.Label(sf, text="Esik:2.50", bg=C_SB, fg=C_TM,
                                 font=("Segoe UI",7)); self._thr_lbl.pack(side=tk.RIGHT)
        self._sep(s)

        # --- Kimlik ---
        self._sec(s,"KIMLIK")
        idf = tk.Frame(s, bg=C_SB, padx=18); idf.pack(fill=tk.X, pady=(0,4))
        idf.columnconfigure(0, weight=1); idf.columnconfigure(1, weight=1)
        tk.Label(idf, text="Kendi ID", bg=C_SB, fg=C_TM, font=("Segoe UI",8)
                 ).grid(row=0, column=0, sticky="w")
        tk.Label(idf, text="Hedef ID", bg=C_SB, fg=C_TM, font=("Segoe UI",8)
                 ).grid(row=0, column=1, sticky="w", padx=(8,0))
        self.my_id_ent = self._dent(idf,"1")
        self.my_id_ent.grid(row=1, column=0, sticky="ew", pady=(2,0))
        self.target_id_ent = self._dent(idf,"0")
        self.target_id_ent.grid(row=1, column=1, sticky="ew", pady=(2,0), padx=(8,0))
        self._sep(s)

        # --- Hız ---
        self._sec(s,"HIZ AYARLARI")
        for attr, lbl, lo, hi, res, df in [
            ("dur","Sembol",0.05,1.0,0.05,DEF_DURATION),
            ("gap","Bosluk",0.01,0.50,0.01,DEF_GAP)]:
            sf2=tk.Frame(s,bg=C_SB,padx=18); sf2.pack(fill=tk.X,pady=(0,2))
            rr=tk.Frame(sf2,bg=C_SB); rr.pack(fill=tk.X)
            tk.Label(rr,text=lbl,bg=C_SB,fg=C_TM,font=("Segoe UI",8)).pack(side=tk.LEFT)
            vl=tk.Label(rr,text=f"{df:.2f}s",bg=C_SB,fg=C_ACC,font=("Segoe UI",8,"bold"))
            vl.pack(side=tk.RIGHT)
            sc=tk.Scale(sf2,from_=lo,to=hi,resolution=res,orient=tk.HORIZONTAL,
                        showvalue=False,bg=C_SB,fg=C_TL,troughcolor=C_SB2,
                        activebackground=C_ACC,highlightthickness=0,bd=0,sliderrelief="flat",
                        command=lambda v,a=attr,l=vl:self._sc(a,v,l))
            sc.set(df); sc.pack(fill=tk.X)
            setattr(self,f"{attr}_scale",sc); setattr(self,f"{attr}_vl",vl)
        self._sep(s)

        # --- Kanal Modu ---
        self._sec(s,"KANAL MODU")
        chf=tk.Frame(s,bg=C_SB,padx=18); chf.pack(fill=tk.X,pady=(0,2))
        for v, t in [(8,"8-Ch (8 byte/fr)"),(16,"16-Ch (16 byte/fr)")]:
            tk.Radiobutton(chf,text=t,variable=self.channel_mode,value=v,
                bg=C_SB,fg=C_TL,selectcolor=C_SB2,activebackground=C_SB,
                activeforeground=C_TL,font=("Segoe UI",8),anchor="w").pack(fill=tk.X)
        self._sep(s)

        # --- Güvenlik & FEC ---
        self._sec(s,"GUVENLIK & FEC")
        of=tk.Frame(s,bg=C_SB,padx=18); of.pack(fill=tk.X,pady=(0,2))
        for var, txt in [(self.safe_mode,"Safe Mode (Hamming 7,4)"),
                         (self.xor_enabled,"XOR Sifreleme"),
                         (self.comp_enabled,"Sikistirma (Huffman/RLE)")]:
            tk.Checkbutton(of,text=txt,variable=var,bg=C_SB,fg=C_TL,selectcolor=C_SB2,
                activebackground=C_SB,activeforeground=C_TL,font=("Segoe UI",8),
                anchor="w").pack(fill=tk.X)
        kwf=tk.Frame(of,bg=C_SB); kwf.pack(fill=tk.X,pady=(3,0))
        tk.Label(kwf,text="Anahtar:",bg=C_SB,fg=C_TM,font=("Segoe UI",7)).pack(side=tk.LEFT)
        self.xor_key_ent=tk.Entry(kwf,bg=C_SB2,fg=C_TL,font=("Segoe UI",9),
            borderwidth=0,insertbackground=C_TL,show="*",width=14)
        self.xor_key_ent.insert(0,"gizli"); self.xor_key_ent.pack(side=tk.LEFT,padx=(4,0),fill=tk.X,expand=True)
        self._sep(s)

        # --- Gelişmiş Özellikler ---
        self._sec(s,"GELISMIS OZELLIKLER")
        af=tk.Frame(s,bg=C_SB,padx=18); af.pack(fill=tk.X,pady=(0,2))
        for var, txt in [(self.fhss_enabled,"FHSS (Frekans Atlama)"),
                         (self.stego_enabled,"Steganografi Modu"),
                         (self.mesh_enabled,"Mesh Relay")]:
            tk.Checkbutton(af,text=txt,variable=var,bg=C_SB,fg=C_TL,selectcolor=C_SB2,
                activebackground=C_SB,activeforeground=C_TL,font=("Segoe UI",8),
                anchor="w").pack(fill=tk.X)
        hf=tk.Frame(af,bg=C_SB); hf.pack(fill=tk.X,pady=(2,0))
        tk.Label(hf,text="Hop Count:",bg=C_SB,fg=C_TM,font=("Segoe UI",7)).pack(side=tk.LEFT)
        tk.Spinbox(hf,from_=0,to=10,textvariable=self.hop_count_var,width=3,
            bg=C_SB2,fg=C_TL,font=("Segoe UI",9),buttonbackground=C_SB3,
            borderwidth=0).pack(side=tk.LEFT,padx=(4,0))
        self._sep(s)

        # --- Sonar ---
        self._sec(s,"SONAR & MESAFE")
        snf=tk.Frame(s,bg=C_SB,padx=18); snf.pack(fill=tk.X,pady=(0,2))
        tk.Button(snf,text="Ping Gonder",bg=C_SB3,fg=C_TL,font=("Segoe UI",8),
                  relief="flat",cursor="hand2",command=self._sonar_ping
                  ).pack(side=tk.LEFT)
        self._dist_lbl=tk.Label(snf,text="Mesafe: — m",bg=C_SB,fg=C_ACC,
                                font=("Segoe UI",8,"bold"))
        self._dist_lbl.pack(side=tk.LEFT,padx=(10,0))
        self._sep(s)

        # --- Metriler ---
        self._sec(s,"METRIKLER")
        mf=tk.Frame(s,bg=C_SB,padx=18); mf.pack(fill=tk.X,pady=(0,2))
        self._snr_lbl=tk.Label(mf,text="SNR: — dB",bg=C_SB,fg=C_TL,font=("Segoe UI",8))
        self._snr_lbl.pack(anchor="w")
        self._comp_lbl=tk.Label(mf,text="Sikistirma: — %",bg=C_SB,fg=C_TL,font=("Segoe UI",8))
        self._comp_lbl.pack(anchor="w")
        self._dopp_lbl=tk.Label(mf,text="Doppler: 1.0000x",bg=C_SB,fg=C_TL,font=("Segoe UI",8))
        self._dopp_lbl.pack(anchor="w")
        self._sep(s)

        # --- Resim Önizleme ---
        self._sec(s,"ALINAN RESIM")
        pf=tk.Frame(s,bg=C_SB,padx=18); pf.pack(fill=tk.X,pady=(0,2))
        self.prev_cv=tk.Canvas(pf,width=100,height=100,bg=C_SB2,highlightthickness=0)
        self.prev_cv.pack(side=tk.LEFT); self._prev_idle()
        pi=tk.Frame(pf,bg=C_SB,padx=8); pi.pack(side=tk.LEFT,fill=tk.Y,anchor="n")
        self._pst=tk.Label(pi,text="Bekleniyor",bg=C_SB,fg=C_TM,font=("Segoe UI",8))
        self._pst.pack(anchor="w")
        self._ppr=tk.Label(pi,text="—",bg=C_SB,fg=C_TM,font=("Segoe UI",7))
        self._ppr.pack(anchor="w",pady=(2,0))
        tk.Button(pi,text="PNG Kaydet",bg=C_SB3,fg=C_TL,font=("Segoe UI",7),
                  relief="flat",cursor="hand2",command=self._export_png
                  ).pack(anchor="w",pady=(4,0))
        self._sep(s)

        # --- Spektrum ---
        self._sec(s,"SPEKTRUM")
        self.fig, self.ax = plt.subplots(figsize=(3.4,2.0))
        self.fig.patch.set_facecolor(C_SB); self.ax.set_facecolor(C_SB2)
        self.plot_x = np.linspace(8500,20500,800)
        self.line, = self.ax.plot(self.plot_x, np.zeros(800), color=C_ACC, lw=1.2)
        self.ax.set_ylim(0,60); self.ax.set_xlim(8500,20500)
        self.ax.set_xticks([9000,11000,13000,15000,17000,19000])
        self.ax.set_xticklabels(['9k','11k','13k','15k','17k','19k'],fontsize=6,color=C_TM)
        self.ax.tick_params(colors=C_TM,length=0)
        for sp in self.ax.spines.values(): sp.set_color(C_SB2)
        self.ax.grid(True,linestyle='--',alpha=0.12,color=C_TM)
        self.fig.tight_layout(pad=0.3)
        self.mpl_cv = FigureCanvasTkAgg(self.fig, master=s)
        self.mpl_cv.get_tk_widget().configure(bg=C_SB)
        self.mpl_cv.get_tk_widget().pack(fill=tk.X, padx=14, pady=(0,10))

    # ---------- main panel ----------
    def _build_main(self):
        m = tk.Frame(self.root, bg=C_MAIN); m.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # top bar
        tb=tk.Frame(m,bg=C_WHITE,pady=10,padx=20,highlightthickness=1,highlightbackground=C_BRD)
        tb.pack(fill=tk.X)
        tk.Label(tb,text="Mesajlasma",bg=C_WHITE,fg="#0f172a",
                 font=("Segoe UI",13,"bold")).pack(side=tk.LEFT)
        self._tp_lbl=tk.Label(tb,text="0.0 B/s",bg=C_WHITE,fg=C_TM,font=("Segoe UI",9))
        self._tp_lbl.pack(side=tk.RIGHT,padx=(0,6))
        tk.Label(tb,text="verim:",bg=C_WHITE,fg=C_TM,font=("Segoe UI",9)).pack(side=tk.RIGHT)

        # chat
        cw=tk.Frame(m,bg=C_MAIN,padx=16,pady=8); cw.pack(fill=tk.BOTH,expand=True)
        self.chat=scrolledtext.ScrolledText(cw,bg=C_WHITE,font=("Segoe UI",12),
            padx=14,pady=12,borderwidth=0,relief="flat",state="disabled",cursor="arrow")
        self.chat.pack(fill=tk.BOTH,expand=True)
        for tag, cfg in [
            ("oh",  dict(foreground=C_OFG,font=("Segoe UI",8,"bold"),spacing3=2)),
            ("ob",  dict(foreground=C_OFG,font=("Segoe UI",12),background=C_OBG,
                         lmargin1=40,lmargin2=40,rmargin=20,spacing1=3,spacing3=3)),
            ("om",  dict(foreground="#93c5fd",font=("Segoe UI",7),lmargin1=40,spacing3=6)),
            ("ih",  dict(foreground="#475569",font=("Segoe UI",8,"bold"),spacing3=2)),
            ("ib",  dict(foreground=C_IFG,font=("Segoe UI",12),background=C_IBG,
                         lmargin1=16,lmargin2=16,rmargin=50,spacing1=3,spacing3=3)),
            ("im",  dict(foreground="#94a3b8",font=("Segoe UI",7),lmargin1=16,spacing3=6)),
            ("sys", dict(foreground="#cbd5e1",font=("Segoe UI",7,"italic"),justify="center",
                         spacing1=4,spacing3=4)),
            ("ok",  dict(foreground=C_OK,font=("Segoe UI",7))),
            ("err", dict(foreground=C_ERR,font=("Segoe UI",7))),
        ]: self.chat.tag_config(tag, **cfg)

        # progress
        pw=tk.Frame(m,bg=C_WHITE,padx=16,pady=8,highlightthickness=1,highlightbackground=C_BRD)
        pw.pack(fill=tk.X)
        pt=tk.Frame(pw,bg=C_WHITE); pt.pack(fill=tk.X)
        self._prog_lbl=tk.Label(pt,text="Hazir",bg=C_WHITE,fg=C_TM,font=("Segoe UI",8))
        self._prog_lbl.pack(side=tk.LEFT)
        self._tmr_lbl=tk.Label(pt,text="",bg=C_WHITE,fg=C_ACC,font=("Segoe UI",8,"bold"))
        self._tmr_lbl.pack(side=tk.RIGHT)
        sty=ttk.Style(); sty.theme_use('clam')
        sty.configure("S.Horizontal.TProgressbar",background=C_ACC,thickness=4,
                       troughcolor=C_BRD,bordercolor=C_BRD)
        self.pbar=ttk.Progressbar(pw,orient=tk.HORIZONTAL,mode="determinate",
                                   style="S.Horizontal.TProgressbar")
        self.pbar.pack(fill=tk.X,pady=(4,0))

        # input
        ip=tk.Frame(m,bg=C_WHITE,padx=16,pady=10,highlightthickness=1,highlightbackground=C_BRD)
        ip.pack(fill=tk.X)
        self.entry=tk.Entry(ip,bg="#f8fafc",fg="#0f172a",font=("Segoe UI",12),
                            relief="solid",bd=1,insertbackground="#0f172a")
        self.entry.pack(side=tk.LEFT,fill=tk.X,expand=True,ipady=6,padx=(0,8))
        self.entry.bind("<Return>", lambda e: self.send_text())
        for txt, cmd, bg in [("Resim",self.pick_and_send_image,C_SB2),
                              ("Gonder",self.send_text,C_ACC),
                              ("Temizle",self.clear_log,C_MAIN)]:
            tk.Button(ip,text=txt,bg=bg,fg=C_TL if bg!=C_MAIN else C_TM,
                font=("Segoe UI",10,"bold" if txt=="Gonder" else ""),relief="flat",
                padx=14,pady=6,cursor="hand2",command=cmd).pack(side=tk.LEFT,padx=(0,6))

    # ======================= UI Yardımcıları =================================
    def _sep(self, parent):
        tk.Frame(parent, bg=C_SB2, height=1).pack(fill=tk.X, padx=18, pady=6)
    def _sec(self, parent, text):
        f=tk.Frame(parent,bg=C_SB,padx=18); f.pack(fill=tk.X,pady=(8,2))
        tk.Label(f,text=text,bg=C_SB,fg=C_TM,font=("Segoe UI",7,"bold")).pack(anchor="w")
    def _dent(self, par, d):
        e=tk.Entry(par,bg=C_SB2,fg=C_TL,font=("Segoe UI",11),justify="center",
                   borderwidth=0,insertbackground=C_TL)
        e.insert(0,d); return e
    def _prev_idle(self):
        self.prev_cv.delete("all")
        self.prev_cv.create_rect = self.prev_cv.create_rectangle(0,0,100,100,fill=C_SB2,outline="")
        self.prev_cv.create_text(50,50,text="—",fill=C_TM,font=("Segoe UI",16))
    def _set_status(self, txt, col):
        self._slbl.config(text=txt,fg=col)
        self._sdot.delete("dot"); self._sdot.create_oval(2,2,9,9,fill=col,outline="",tags="dot")
    def _sc(self, attr, val, lbl):
        v=float(val)
        if attr=="dur": self.duration=v
        else: self.gap=v
        lbl.config(text=f"{v:.2f}s")
    def clear_log(self):
        self.chat.config(state="normal"); self.chat.delete("1.0",tk.END)
        self.chat.config(state="disabled")
    def _exit(self):
        self.is_running=False; os._exit(0)

    def _export_png(self):
        if self.last_received_pil is None:
            self._sys("Kaydedilecek resim yok."); return
        p=filedialog.asksaveasfilename(title="PNG Kaydet",defaultextension=".png",
            filetypes=[("PNG","*.png")])
        if p:
            self.last_received_pil.resize((256,256),Image.NEAREST).save(p)
            self._sys(f"Kaydedildi: {p}")

    # ======================= CHAT ============================================
    def _cw(self, txt, tag):
        self.chat.config(state="normal"); self.chat.insert(tk.END, txt, tag)
        self.chat.see(tk.END); self.chat.config(state="disabled")
    def _ci(self, photo, tag):
        self.chat.config(state="normal"); self.chat.insert(tk.END,"  ",tag)
        self.chat.image_create(tk.END,image=photo,padx=6,pady=3)
        self.chat.insert(tk.END,"\n",tag); self.chat.see(tk.END)
        self.chat.config(state="disabled")
    def _own_txt(self, txt, meta):
        ts=datetime.now().strftime("%H:%M")
        self._cw(f"\n  Sen  {ts}\n","oh"); self._cw(f"  {txt}\n","ob")
        self._cw(f"  {meta}\n","om")
    def _own_img(self, photo, meta):
        ts=datetime.now().strftime("%H:%M")
        self._cw(f"\n  Sen  {ts}\n","oh"); self._ci(photo,"ob")
        self._cw(f"  {meta}\n","om")
    def _inc_txt(self, src, txt, meta, ok=True):
        ts=datetime.now().strftime("%H:%M")
        self._cw(f"\n  Kimden:{src}  {ts}\n","ih"); self._cw(f"  {txt}\n","ib")
        self._cw(f"  {meta}\n","ok" if ok else "err")
    def _inc_img(self, src, photo, meta, ok=True):
        ts=datetime.now().strftime("%H:%M")
        self._cw(f"\n  Kimden:{src}  {ts}\n","ih"); self._ci(photo,"ib")
        self._cw(f"  {meta}\n","ok" if ok else "err")
    def _sys(self, txt):
        self._cw(f"\n  {txt}  \n","sys")

    def _pil_tk(self, pil, sz=(140,140)):
        ph=ImageTk.PhotoImage(pil.resize(sz,Image.NEAREST))
        self.photo_refs.append(ph)
        if len(self.photo_refs)>40: self.photo_refs.pop(0)
        return ph

    # ======================= RESIM KODLAMA ===================================
    @staticmethod
    def _img_enc(pil):
        img=pil.convert("L").resize((IMG_W,IMG_H),Image.LANCZOS)
        px=list(img.getdata())
        q=[int(p*(IMG_LEVELS-1)/255) for p in px]
        pk=bytearray()
        for i in range(0,len(q),2):
            lo=q[i]&0xF; hi=(q[i+1]&0xF) if i+1<len(q) else 0
            pk.append(lo|(hi<<4))
        return bytes(pk)

    @staticmethod
    def _img_dec(data, w=IMG_W, h=IMG_H, partial=False):
        px=[]
        for b in data:
            px.append(int((b&0xF)*255/(IMG_LEVELS-1)))
            px.append(int(((b>>4)&0xF)*255/(IMG_LEVELS-1)))
        total=w*h
        if len(px)<total:
            if partial: px.extend([0]*(total-len(px)))
            else: return None
        img=Image.new("L",(w,h)); img.putdata(px[:total]); return img

    # ======================= TONE ÜRETEÇLERI =================================
    def _make_tone(self, freq, dur, amplitude=1.0):
        n=int(FS*dur); t=np.linspace(0,dur,n)
        return (np.sin(2*np.pi*freq*t)*amplitude*windows.tukey(n)).astype(np.float32)

    def _play_tone(self, freq, dur, amplitude=1.0):
        sd.play(self._make_tone(freq,dur,amplitude), FS); sd.wait()

    def _send_ack(self):
        self._play_tone(ACK_FREQ, 0.15, 0.8)

    def _send_nack(self):
        self._play_tone(NACK_FREQ, 0.15, 0.8)

    # ======================= SONAR ===========================================
    def _sonar_ping(self):
        if self.is_transmitting: return
        self._sys("Ping gonderiliyor...")
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
            self._sys(f"Pong alindi — RTT: {rtt*1000:.1f}ms — Mesafe: {dist:.2f}m")

    def _handle_ping(self):
        """Ping alındığında otomatik pong gönder."""
        threading.Thread(target=lambda: (time.sleep(0.05),
                         self._play_tone(PONG_FREQ, 0.15, 0.9)), daemon=True).start()

    # ======================= DOPPLER =========================================
    def _measure_doppler(self, freqs, mag):
        """Pilot tonlardan Doppler oranı hesapla."""
        # Pilot LO
        m1 = (freqs >= PILOT_LO-100) & (freqs <= PILOT_LO+100)
        # Pilot HI
        m2 = (freqs >= PILOT_HI-100) & (freqs <= PILOT_HI+100)
        if not (np.any(m1) and np.any(m2)): return
        p1 = np.max(mag[m1]); p2 = np.max(mag[m2])
        thr = self.threshold
        if p1 > thr and p2 > thr:
            f1 = freqs[m1][np.argmax(mag[m1])]
            f2 = freqs[m2][np.argmax(mag[m2])]
            r1 = f1 / PILOT_LO
            r2 = f2 / PILOT_HI
            self.doppler_ratio = (r1 + r2) / 2.0
            self._dopp_lbl.config(text=f"Doppler: {self.doppler_ratio:.4f}x")

    def _doppler_correct(self, freq):
        """Nominal frekansı Doppler düzeltmeli frekansa çevir (alıcı taraf)."""
        if abs(self.doppler_ratio - 1.0) < 0.0001: return freq
        return freq * self.doppler_ratio

    # ======================= TX — GONDER =====================================
    def send_text(self):
        msg=self.entry.get().strip()
        if not msg or self.is_transmitting: return
        self.entry.delete(0,tk.END)
        threading.Thread(target=self._send_packet,
            args=(int(self.my_id_ent.get()), int(self.target_id_ent.get()),
                  msg.encode("utf-8"), PKT_TEXT), daemon=True).start()

    def pick_and_send_image(self):
        if self.is_transmitting: return
        p=filedialog.askopenfilename(title="Resim Sec",
            filetypes=[("Resim","*.png *.jpg *.jpeg *.bmp *.gif *.webp"),("Tum","*.*")])
        if not p: return
        try: pil=Image.open(p)
        except Exception as e: self._sys(f"Hata: {e}"); return
        payload=self._img_enc(pil)
        thumb=self._pil_tk(pil.convert("L").resize((IMG_W,IMG_H),Image.LANCZOS),(140,140))
        threading.Thread(target=self._send_packet,
            args=(int(self.my_id_ent.get()), int(self.target_id_ent.get()),
                  payload, PKT_IMAGE),
            kwargs={"thumb":thumb}, daemon=True).start()

    def _send_packet(self, src, dst, payload, pkt_type, thumb=None):
        while self.is_channel_busy: time.sleep(0.1)
        self.is_transmitting = True
        original = payload
        orig_len = len(payload)

        # === 1) COMPRESSION ===
        comp_type = 0  # 0=none, 1=delta+rle, 2=huffman
        if self.comp_enabled.get() and len(payload) > 8:
            if pkt_type == PKT_IMAGE:
                comp = delta_rle_compress(payload)
                if len(comp) < len(payload):
                    payload = comp; comp_type = 1
            else:
                comp = huffman_compress(payload)
                if len(comp) < len(payload):
                    payload = comp; comp_type = 2
        comp_ratio = (1.0 - len(payload)/orig_len)*100 if orig_len > 0 else 0
        self.last_comp_ratio = comp_ratio
        self._comp_lbl.config(text=f"Sikistirma: {comp_ratio:.0f}%")

        # === 2) XOR ===
        use_xor = self.xor_enabled.get()
        kw = self._xor_key()
        if use_xor and kw:
            payload = xor_cipher(payload, kw)

        # === 3) HAMMING FEC ===
        use_fec = self.safe_mode.get()
        if use_fec:
            payload = hamming_enc(payload)

        # === FRAGMENT ===
        chunks = [payload[i:i+FRAG_MAX] for i in range(0, len(payload), FRAG_MAX)]
        if not chunks: chunks = [b'']
        total_chunks = len(chunks)

        ch_mode = self.channel_mode.get()
        frame_size = ch_mode
        channels = self._tx_ch()
        num_ch = self._tx_n()
        step = self._tx_step()
        eq_table = self._tx_eq()
        use_fhss = self.fhss_enabled.get()
        use_stego = self.stego_enabled.get()
        hop_cnt = self.hop_count_var.get() if self.mesh_enabled.get() else 0

        # flags: bit0-1=pkt_type, bit2=fec, bit3=xor, bit4=16ch, bit5=compressed,
        #        bit6=stego, bit7=fhss
        flags = pkt_type & 0x03
        if use_fec: flags |= 0x04
        if use_xor and kw: flags |= 0x08
        if ch_mode == 16: flags |= 0x10
        if comp_type > 0: flags |= 0x20
        if use_stego: flags |= 0x40
        if use_fhss: flags |= 0x80

        fhss_seed = fhss_seed_from_key(kw) if use_fhss else 0

        start_t = time.time()
        kind = "RESIM" if pkt_type==PKT_IMAGE else "METIN"

        # === WAKE-UP + PILOT TONLARI ===
        wu_dur = 0.4; wu_n = int(FS*wu_dur)
        t_wu = np.linspace(0, wu_dur, wu_n)
        wu = (np.sin(2*np.pi*WAKE_UP_FREQ*t_wu) * 0.5
              + np.sin(2*np.pi*PILOT_LO*t_wu) * 0.25
              + np.sin(2*np.pi*PILOT_HI*t_wu) * 0.25)
        wu = (wu * windows.tukey(wu_n)).astype(np.float32)
        sd.play(wu, FS); sd.wait(); time.sleep(0.1)

        all_ok = True
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
            header[5] = IMG_W if pkt_type==PKT_IMAGE else 0
            header[6] = IMG_H if pkt_type==PKT_IMAGE else 0
            header[7] = crc_val
            header[8] = seq_idx
            header[9] = total_chunks
            header[10] = hop_cnt
            header[11] = comp_type
            header[12] = orig_len & 0xFF
            header[13] = (orig_len >> 8) & 0xFF
            header[14] = 0  # reserved
            header[15] = 0  # reserved

            raw = list(header) + list(chunk)
            while len(raw) % frame_size != 0:
                raw.append(0)

            num_frames = len(raw) // frame_size
            # Header frame sayısı (FHSS'te hop edilmez)
            hdr_frames = HEADER_SIZE // frame_size
            if HEADER_SIZE % frame_size != 0:
                hdr_frames += 1

            self._prog_lbl.config(
                text=f"{kind} [{seq_idx+1}/{total_chunks}] | {sz}B | {num_frames}fr")

            # Her frame'i gönder
            for fi, raw_i in enumerate(range(0, len(raw), frame_size)):
                elapsed = time.time()-start_t
                self.pbar["value"] = min(100, (seq_idx*100 + fi*100/max(num_frames,1))/total_chunks)

                chars = raw[raw_i:raw_i+frame_size]
                t_s = np.linspace(0, self.duration, int(FS*self.duration))
                wave = np.zeros_like(t_s)

                # FHSS: data frame'leri için kanal permütasyonu (header frame'ler sabit)
                if use_fhss and fi >= hdr_frames:
                    perm = fhss_permutation(fi - hdr_frames, fhss_seed, num_ch)
                else:
                    perm = list(range(num_ch))

                for ch_idx in range(min(len(chars), num_ch)):
                    cv = chars[ch_idx]
                    phys_ch = perm[ch_idx]  # fiziksel kanal
                    b1, b2 = channels[phys_ch]
                    g_lo = GRAY_ENC[cv % HEX_BASE]
                    g_hi = GRAY_ENC[cv // HEX_BASE]
                    f1 = b1 + g_lo * step
                    f2 = b2 + g_hi * step
                    eq1, eq2 = eq_table[phys_ch]
                    amp = 0.08 if use_stego else 1.0
                    wave += eq1 * amp * np.sin(2*np.pi*f1*t_s)
                    wave += eq2 * amp * np.sin(2*np.pi*f2*t_s)

                final = wave / (num_ch*2)
                if use_stego:
                    # Cover audio: white noise
                    noise = np.random.randn(len(t_s)) * 0.3
                    final = final + noise
                final = (final * windows.tukey(len(t_s))).astype(np.float32)
                sd.play(final, FS); sd.wait()
                time.sleep(self.gap)

            # === ACK/NACK BEKLEME (her fragment sonrası) ===
            if total_chunks > 1 or True:  # her zaman ACK bekle
                ack_ok = self._wait_for_ack()
                if not ack_ok:
                    # Retry
                    retried = False
                    for retry in range(ARQ_RETRIES):
                        self._sys(f"NACK/Timeout — Tekrar deneme {retry+1}/{ARQ_RETRIES}")
                        # Re-send this chunk
                        for fi2, raw_i2 in enumerate(range(0, len(raw), frame_size)):
                            chars2 = raw[raw_i2:raw_i2+frame_size]
                            t_s2 = np.linspace(0,self.duration,int(FS*self.duration))
                            wave2 = np.zeros_like(t_s2)
                            if use_fhss and fi2 >= hdr_frames:
                                perm2 = fhss_permutation(fi2-hdr_frames, fhss_seed, num_ch)
                            else:
                                perm2 = list(range(num_ch))
                            for ci2 in range(min(len(chars2), num_ch)):
                                cv2 = chars2[ci2]; pc2 = perm2[ci2]
                                bb1,bb2 = channels[pc2]
                                wave2 += eq_table[pc2][0]*np.sin(2*np.pi*(bb1+GRAY_ENC[cv2%16]*step)*t_s2)
                                wave2 += eq_table[pc2][1]*np.sin(2*np.pi*(bb2+GRAY_ENC[cv2//16]*step)*t_s2)
                            f2 = (wave2/(num_ch*2)*windows.tukey(len(t_s2))).astype(np.float32)
                            sd.play(f2,FS); sd.wait(); time.sleep(self.gap)
                        if self._wait_for_ack():
                            retried = True; break
                    if not retried:
                        self._sys(f"Fragment {seq_idx} gonderilemedi!"); all_ok = False

        elapsed_total = time.time() - start_t
        bps = orig_len / elapsed_total if elapsed_total > 0 else 0
        self._tp_lbl.config(text=f"{bps:.1f} B/s")

        tags = []
        if use_fec: tags.append("FEC")
        if use_xor and kw: tags.append("XOR")
        if comp_type: tags.append("COMP" if comp_type==1 else "HUFF")
        if use_fhss: tags.append("FHSS")
        if use_stego: tags.append("STEGO")
        tag_s = " [" + "|".join(tags) + "]" if tags else ""

        meta = (f"{orig_len}B · {total_chunks}frag · "
                f"{elapsed_total:.1f}s · {bps:.1f}B/s{tag_s}")

        if pkt_type == PKT_TEXT:
            self._own_txt(original.decode("utf-8",errors="replace"), meta)
        else:
            self._own_img(thumb, meta)

        self.pbar["value"] = 100
        self._prog_lbl.config(text="Hazir"); self._tmr_lbl.config(text="")
        time.sleep(0.4); self.pbar["value"] = 0
        self.is_transmitting = False

    def _wait_for_ack(self) -> bool:
        """ACK veya NACK tonu dinle. True=ACK, False=NACK/timeout."""
        deadline = time.time() + ACK_TIMEOUT
        while time.time() < deadline:
            data = []
            while not self.msg_queue.empty():
                data.append(self.msg_queue.get_nowait())
            if data:
                audio = np.concatenate(data).flatten()
                yf = fft(audio); xf = fftfreq(len(yf),1/FS)
                mag = np.abs(yf[:len(yf)//2]); freqs = xf[:len(xf)//2]
                # ACK check
                ma = (freqs>=ACK_FREQ-60)&(freqs<=ACK_FREQ+60)
                mn = (freqs>=NACK_FREQ-60)&(freqs<=NACK_FREQ+60)
                pa = np.max(mag[ma]) if np.any(ma) else 0
                pn = np.max(mag[mn]) if np.any(mn) else 0
                if pa > self.threshold and pa > pn:
                    return True
                if pn > self.threshold and pn > pa:
                    return False
            time.sleep(0.05)
        return False  # timeout

    # ======================= RX — ALIM DÖNGÜSÜ ==============================
    def _update_loop(self):
        if not self.is_running: return

        data = []
        while not self.msg_queue.empty():
            data.append(self.msg_queue.get_nowait())

        if data and not self.is_transmitting:
            audio = np.concatenate(data).flatten()
            rms = float(np.sqrt(np.mean(audio**2)))*100
            if not self.is_listening:
                self._nf_update(rms)

            yf = fft(audio); xf = fftfreq(len(yf),1/FS)
            mag = np.abs(yf[:len(yf)//2]); freqs = xf[:len(xf)//2]

            # Moving Average
            self.mag_history.append(mag.copy())
            if len(self.mag_history)>=2:
                mag = np.mean(list(self.mag_history), axis=0)

            # Spektrum güncelle
            self.line.set_ydata(np.interp(self.plot_x, freqs, mag))
            self.mpl_cv.draw_idle()

            thr = self.threshold
            self._thr_lbl.config(text=f"Esik:{thr:.1f}")

            # SNR hesapla
            bm = (freqs>9500)&(freqs<20000)
            sig_power = float(np.max(mag[bm])) if np.any(bm) else 0
            if self.noise_floor > 0.01:
                snr_db = 20*np.log10(max(sig_power,0.001)/max(self.noise_floor,0.001))
            else:
                snr_db = 0
            self.current_snr = snr_db
            self._snr_lbl.config(text=f"SNR: {snr_db:.1f} dB")

            # Doppler ölçümü
            self._measure_doppler(freqs, mag)

            # Kanal meşguliyet
            peak = sig_power
            if peak > thr:
                self.is_channel_busy = True; self._set_status("Mesgul", C_ERR)
            else:
                self.is_channel_busy = False; self._set_status("Musait", C_OK)

            # --- SONAR: Ping/Pong algılama ---
            mp = (freqs>=PING_FREQ-60)&(freqs<=PING_FREQ+60)
            mpo = (freqs>=PONG_FREQ-60)&(freqs<=PONG_FREQ+60)
            if np.any(mp) and np.max(mag[mp])>thr and not self.is_listening:
                self._handle_ping()
            if np.any(mpo) and np.max(mag[mpo])>thr:
                self._handle_pong()

            # --- Wake-up algılama ---
            wm = (freqs>WAKE_UP_FREQ-80)&(freqs<WAKE_UP_FREQ+80)
            wp = np.max(mag[wm]) if np.any(wm) else 0
            if not self.is_listening and wp > thr:
                self.is_listening = True
                self.start_t = time.time()
                self.current_packet = []
                self.last_frame_set = None
                self.rx_ch_mode = 0; self.rx_frame_size = 0
                self._pst.config(text="Aliniyor...")
                self._sys("Sinyal alindi...")

            # --- Veri çözme ---
            if self.is_listening:
                frame_8  = self._try_frame(freqs, mag, thr, CHANNELS_8,  STEP_8,  DET_WIN_8,  8)
                frame_16 = self._try_frame(freqs, mag, thr, CHANNELS_16, STEP_16, DET_WIN_16, 16)

                if self.rx_ch_mode == 0:
                    # İlk frame — mod algıla
                    if frame_16 is not None and len(frame_16)==16:
                        if frame_16[4] & 0x10:
                            self.rx_ch_mode=16; self.rx_frame_size=16
                            self.current_packet.extend(frame_16)
                            self.last_frame_set=tuple(frame_16)
                            self.last_signal_time=time.time()
                        elif frame_8 is not None and len(frame_8)==8:
                            self.rx_ch_mode=8; self.rx_frame_size=8
                            self.current_packet.extend(frame_8)
                            self.last_frame_set=tuple(frame_8)
                            self.last_signal_time=time.time()
                    elif frame_8 is not None and len(frame_8)==8:
                        self.rx_ch_mode=8; self.rx_frame_size=8
                        self.current_packet.extend(frame_8)
                        self.last_frame_set=tuple(frame_8)
                        self.last_signal_time=time.time()
                else:
                    # Mod biliniyor — FHSS ters permütasyon
                    if self.rx_ch_mode == 16:
                        frame = frame_16; exp=16
                    else:
                        frame = frame_8; exp=8

                    if frame is not None and len(frame)==exp:
                        cs = tuple(frame)
                        if cs != self.last_frame_set:
                            # FHSS ters çevir (header zaten çözüldü, data frame'lerde)
                            hdr_frames = HEADER_SIZE // self.rx_frame_size
                            if HEADER_SIZE % self.rx_frame_size: hdr_frames += 1
                            frame_idx = len(self.current_packet) // self.rx_frame_size

                            if frame_idx >= hdr_frames:
                                # Header'dan FHSS flag kontrolü
                                if len(self.current_packet) >= 5 and (self.current_packet[4] & 0x80):
                                    kw = self._xor_key()
                                    fseed = fhss_seed_from_key(kw)
                                    perm = fhss_permutation(frame_idx - hdr_frames, fseed, exp)
                                    inv = fhss_inv(perm)
                                    frame = [frame[inv[j]] for j in range(exp)]

                            self.current_packet.extend(frame)
                            self.last_frame_set = tuple(frame)
                            self.last_signal_time = time.time()

                if (self.current_packet and time.time()-self.last_signal_time > 1.5):
                    self._finalize_fragment()

        self.root.after(50, self._update_loop)

    def _try_frame(self, freqs, mag, thr, channels, step, det_win, num_ch):
        """Bir frame decode etmeyi dene."""
        chars = []
        for b1, b2 in channels:
            # Doppler düzeltme
            db1 = self._doppler_correct(b1)
            db2 = self._doppler_correct(b2)
            m1 = (freqs>=db1)&(freqs<=db1+det_win)
            m2 = (freqs>=db2)&(freqs<=db2+det_win)
            if not (np.any(m1) and np.any(m2)): return None
            i1, i2 = np.argmax(mag[m1]), np.argmax(mag[m2])
            if mag[m1][i1] <= thr or mag[m2][i2] <= thr: return None
            glo = max(0,min(15,round((freqs[m1][i1]-db1)/step)))
            ghi = max(0,min(15,round((freqs[m2][i2]-db2)/step)))
            chars.append(max(0,min(255, GRAY_DEC[ghi]*16+GRAY_DEC[glo])))
        return chars if len(chars)==num_ch else None

    # ======================= FRAGMENT SONLANDIRMA ============================
    def _finalize_fragment(self):
        """Bir fragment alındığında: CRC kontrol, ACK/NACK gönder, birleştir."""
        total_dur = time.time() - self.start_t - 1.5
        fs = self.rx_frame_size or 8

        if len(self.current_packet) < HEADER_SIZE:
            self._reset_rx(); return

        # Header parse
        h = self.current_packet[:HEADER_SIZE]
        src       = h[0]
        dst       = h[1]
        sz        = h[2] | (h[3]<<8)
        flags     = h[4]
        pkt_type  = flags & 0x03
        has_fec   = bool(flags & 0x04)
        has_xor   = bool(flags & 0x08)
        is_16ch   = bool(flags & 0x10)
        has_comp  = bool(flags & 0x20)
        has_stego = bool(flags & 0x40)
        has_fhss  = bool(flags & 0x80)
        img_w     = h[5] or IMG_W
        img_h     = h[6] or IMG_H
        crc       = h[7]
        seq_num   = h[8]
        total_ch  = h[9]
        hop_cnt   = h[10]
        comp_type = h[11]
        orig_len  = h[12] | (h[13]<<8)

        chunk_data = bytes(self.current_packet[HEADER_SIZE : HEADER_SIZE + sz])
        my_id = int(self.my_id_ent.get())

        # --- MESH RELAY ---
        if dst != my_id and dst != 0 and hop_cnt > 0 and self.mesh_enabled.get():
            self._sys(f"Mesh relay: src={src} dst={dst} hop={hop_cnt}")
            self._send_ack()
            # Relay: hop_cnt-1 ile yeniden gönder (jitter)
            threading.Thread(target=self._mesh_relay,
                args=(self.current_packet[:], hop_cnt-1), daemon=True).start()
            self._reset_rx(); return

        if dst != my_id and dst != 0:
            self._reset_rx(); return

        # CRC kontrol
        crc_ok = (zlib.crc32(chunk_data) & 0xFF) == crc

        # ACK veya NACK gönder
        if crc_ok:
            threading.Thread(target=self._send_ack, daemon=True).start()
        else:
            threading.Thread(target=self._send_nack, daemon=True).start()

        if not crc_ok:
            self._sys(f"CRC HATA — Fragment {seq_num+1}/{total_ch}")
            self._reset_rx(); return

        # Fragment'ı tampona ekle
        frag_key = (src, dst, pkt_type)
        if frag_key not in self.frag_buffer:
            self.frag_buffer[frag_key] = {}
            self.frag_meta[frag_key] = {
                'total': total_ch, 'flags': flags, 'img_w': img_w, 'img_h': img_h,
                'comp_type': comp_type, 'orig_len': orig_len, 'src': src,
                'has_fec': has_fec, 'has_xor': has_xor, 'pkt_type': pkt_type,
                'start_t': self.start_t
            }
        self.frag_buffer[frag_key][seq_num] = chunk_data

        self._sys(f"Fragment {seq_num+1}/{total_ch} alindi ✓")

        # Tüm fragmentler tamamlandı mı?
        if len(self.frag_buffer[frag_key]) >= total_ch:
            self._assemble_packet(frag_key)

        self._reset_rx()

    def _assemble_packet(self, key):
        """Tüm fragmentleri birleştir ve mesajı göster."""
        meta_info = self.frag_meta[key]
        frags = self.frag_buffer[key]
        total = meta_info['total']
        total_dur = time.time() - meta_info['start_t']

        # Sıralı birleştir
        assembled = b''
        for i in range(total):
            assembled += frags.get(i, b'')

        status_parts = ["TAM"]

        # === Hamming decode ===
        decoded = assembled
        if meta_info['has_fec']:
            decoded = hamming_dec(decoded)
            status_parts.append("FEC")

        # === XOR decrypt ===
        if meta_info['has_xor']:
            kw = self._xor_key()
            if kw:
                decoded = xor_cipher(decoded, kw)
                status_parts.append("XOR")
            else:
                status_parts.append("XOR-ANAHTAR YOK")

        # === Decompress ===
        ct = meta_info['comp_type']
        ol = meta_info['orig_len']
        if ct == 1:
            decoded = delta_rle_decompress(decoded, ol)
            status_parts.append("RLE")
        elif ct == 2:
            decoded = huffman_decompress(decoded, ol)
            status_parts.append("HUFF")

        orig_sz = len(decoded)
        bps = orig_sz / total_dur if total_dur > 0 else 0
        self._tp_lbl.config(text=f"{bps:.1f} B/s")
        status_str = " · ".join(status_parts)
        meta = f"{orig_sz}B · {total}frag · {total_dur:.1f}s · {bps:.1f}B/s · {status_str}"

        src = meta_info['src']
        pkt_type = meta_info['pkt_type']

        if pkt_type == PKT_TEXT:
            txt = decoded.decode("utf-8", errors="replace")
            self._inc_txt(src, txt, meta, True)
        elif pkt_type == PKT_IMAGE:
            pil = self._img_dec(decoded, meta_info['img_w'], meta_info['img_h'])
            if pil:
                self.last_received_pil = pil
                photo = self._pil_tk(pil, (140,140))
                ph_s = ImageTk.PhotoImage(pil.resize((100,100), Image.NEAREST))
                self.prev_cv.delete("all")
                self.prev_cv.create_image(50,50,image=ph_s)
                self.prev_cv._ph = ph_s
                self._pst.config(text="Alindi")
                self._ppr.config(text=f"{orig_sz}B %100")
                self._inc_img(src, photo, meta, True)
            else:
                self._inc_txt(src, "[Resim cozulemedi]", meta, False)

        # Tampon temizle
        del self.frag_buffer[key]
        del self.frag_meta[key]

    # ======================= MESH RELAY ======================================
    def _mesh_relay(self, raw_packet, new_hop):
        """Paketi azaltılmış hop ile yeniden yayınla."""
        time.sleep(0.05 + np.random.random()*0.15)  # jitter
        if len(raw_packet) < HEADER_SIZE: return
        # Hop count güncelle
        raw_packet[10] = new_hop
        # CRC değişmez (payload aynı)

        fs = 16 if (raw_packet[4] & 0x10) else 8
        channels = CHANNELS_16 if fs==16 else CHANNELS_8
        step = STEP_16 if fs==16 else STEP_8
        eq_table = EQ16 if fs==16 else EQ8
        num_ch = fs

        # Wake-up
        wu_n = int(FS*0.3)
        t_wu = np.linspace(0,0.3,wu_n)
        wu = (np.sin(2*np.pi*WAKE_UP_FREQ*t_wu)*0.5
              + np.sin(2*np.pi*PILOT_LO*t_wu)*0.25
              + np.sin(2*np.pi*PILOT_HI*t_wu)*0.25)
        sd.play((wu*windows.tukey(wu_n)).astype(np.float32), FS); sd.wait()
        time.sleep(0.08)

        while len(raw_packet) % fs != 0:
            raw_packet.append(0)

        for i in range(0, len(raw_packet), fs):
            chars = raw_packet[i:i+fs]
            t_s = np.linspace(0,self.duration,int(FS*self.duration))
            wave = np.zeros_like(t_s)
            for ci in range(min(len(chars),num_ch)):
                cv = chars[ci]
                b1,b2 = channels[ci]
                wave += eq_table[ci][0]*np.sin(2*np.pi*(b1+GRAY_ENC[cv%16]*step)*t_s)
                wave += eq_table[ci][1]*np.sin(2*np.pi*(b2+GRAY_ENC[cv//16]*step)*t_s)
            final = (wave/(num_ch*2)*windows.tukey(len(t_s))).astype(np.float32)
            sd.play(final,FS); sd.wait()
            time.sleep(self.gap)
        self._sys(f"Mesh relay tamamlandi (hop={new_hop})")

    # ======================= RESET ===========================================
    def _reset_rx(self):
        self.current_packet = []
        self.is_listening = False
        self.last_frame_set = None
        self.rx_ch_mode = 0; self.rx_frame_size = 0
        self._pst.config(text="Bekleniyor")
        self._ppr.config(text="—")


# #############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = AcousticMasterV51(root)
    root.mainloop()