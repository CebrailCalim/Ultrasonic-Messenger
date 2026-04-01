Akustik Haberleşme Protokolü
Bu yazılım, standart hoparlör ve mikrofon donanımlarını kullanarak bilgisayarlar arasında ses dalgaları yoluyla dijital veri iletimi sağlayan bir haberleşme protokolüdür. Veriyi yüksek frekanslı ses sinyallerine modüle ederek fiziksel ortamda kablosuz bir veri bağı oluşturur.

Temel Özellikler
Sistem, 10kHz ile 18kHz frekans aralığında çalışan Quad-Channel MFSK modülasyon mimarisine sahiptir. Bu yapı sayesinde aynı anda dört karakter paralel olarak iletilerek veri transfer hızı maksimize edilmiştir.

Veri güvenliği ve iletim kararlılığı için şu teknolojiler kullanılmaktadır:

Hata Kontrolü: zlib tabanlı CRC32 algoritması ile veri bütünlüğü denetlenir.

Ağ Yönetimi: CSMA (Taşıyıcı Algılama) özelliği ile kanal meşguliyeti kontrol edilerek çakışmalar önlenir.

Senkronizasyon: Frame-Sync kilidi ile mükerrer veri okumaları engellenir ve paket doğruluğu sağlanır.

Analiz ve Kullanıcı Arayüzü
Uygulama, kurumsal standartlarda tasarlanmış, yüksek okunabilirlik sunan bir arayüze sahiptir. Kullanıcılar, iletim sürecini milisaniye hassasiyetindeki zaman damgaları ve byte bazlı boyut bilgileriyle takip edebilirler. Arayüzde yer alan spektrum analizörü, 9kHz-18kHz arasındaki sinyal hareketliliğini kHz ölçeğinde anlık olarak raporlar. Ayrıca, iletim hızı kullanıcı tarafından sliderlar aracılığıyla dinamik olarak optimize edilebilir.

Teknik Parametreler
Örnekleme Hızı: 44100 Hz

Frekans Bandı: 9.5 kHz - 17.5 kHz

Modülasyon: Paralel Quad-Channel MFSK

Hata Denetimi: 8-bit Truncated CRC32

Geliştirme Dili: Python (NumPy, SciPy, Matplotlib, SoundDevice)