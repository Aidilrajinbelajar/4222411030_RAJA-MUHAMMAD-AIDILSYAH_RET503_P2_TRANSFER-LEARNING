# Dokumen Desain Awal Pipeline Persepsi (maks. 2 halaman)

**Kelompok** : [isi nama kelompok]
**Proyek RET501** : [isi judul proyek]
**Versi** : 1 (draf awal)

| Komponen | Isi |
|---|---|
| Misi proyek | Robot IoT bergerak berbasis visi ditugaskan memantau kepatuhan K3 di ruang solder. Modul persepsi berfungsi mengenali pemakaian kacamata pelindung dan masker oleh pekerja pada citra kamera, sehingga ketidakpatuhan dapat diketahui dan dilaporkan [sesuaikan aksi robot, mis. memberi peringatan atau mengirim notifikasi]. |
| Kelas objek | Draf kelas proyek: kacamata pelindung dan masker, masing-masing dengan status dipakai atau tidak dipakai [kelas final ditetapkan bersama kelompok]. Dataset praktikum P2 (lima kelas komponen: `arduino_uno`, `esp32`, `kartu_rfid`, `rfid_rc522`, `kosong`) digunakan sebagai uji coba pipeline pelatihan, bukan sebagai kelas proyek. Contoh citra dataset praktikum ditunjukkan pada Gambar 1.1. |
| Kamera & dudukan | Belum ditentukan. Pemilihan dilakukan setelah jarak kerja robot ke pekerja dan tinggi pemasangan diketahui, dengan kriteria: resolusi cukup agar kacamata dan masker terlihat jelas pada jarak kerja, serta antarmuka kompatibel dengan unit komputasi. Parameter kalibrasi dari Pertemuan 2 dipakai untuk koreksi distorsi setelah kamera ditetapkan. |
| Unit komputasi | Belum ditentukan. Perangkat dipilih berdasarkan hasil pengukuran latensi pada kandidat perangkat terhadap anggaran inferensi 35 ms. Pengujian awal dilakukan pada CPU komputer praktikum (PyTorch 2.14.1, CPU). |
| Target kinerja | Akurasi validasi minimal 90% dengan prioritas recall pada kelas tidak memakai APD (usulan), kecepatan pipeline 15 FPS dengan anggaran 67 ms per frame, dan inferensi model maksimal 35 ms. [Sesuaikan jika proyek memiliki target lain.] |
| Kandidat model | (1) **ResNet-18**: inferensi 20,57 ms pada CPU (pipeline 47,1 FPS); pada dataset praktikum mencapai akurasi validasi 99,7% (*feature extraction*) dan 100,0% (fine-tuning parsial). Dipilih sebagai model utama karena memenuhi anggaran 35 ms dan terbukti pada dataset praktikum. (2) **MobileNetV3-Small**: inferensi 6,19 ms pada CPU, disiapkan sebagai cadangan apabila unit komputasi robot lebih lemah; akurasinya belum dievaluasi. Akurasi untuk kacamata dan masker harus diuji ulang pada data proyek karena tugasnya berbeda dari klasifikasi komponen. |
| Strategi TL | Dimulai dengan *feature extraction* (hanya lapisan `fc` yang dilatih), kemudian dilanjutkan fine-tuning parsial pada `layer4` apabila akurasi belum memadai; fine-tuning parsial diduga diperlukan karena kacamata berukuran kecil dan transparan. Pelatihan dari nol tidak dipilih karena hanya mencapai 53,7% pada dataset praktikum. |
| Rencana data | Dataset praktikum: 604 citra pada dua kondisi cahaya (`lab_terang`, `lab_redup`). Data proyek akan dikumpulkan di ruang solder, dengan izin pekerja, mencakup kondisi memakai dan tidak memakai kacamata serta masker, dengan variasi jarak, sudut, pekerja, latar, dan pencahayaan. Target awal minimal 50 citra per kelas. Pembagian data dilakukan per sesi pengambilan untuk mencegah *data leakage*. |
| Risiko | Lihat tabel Risiko dan Mitigasi di bawah. |

![Gambar 1.1 Contoh citra dataset praktikum (lima kelas komponen)](gambar/gambar_4_2.png)

*Gambar 1.1 Contoh citra dataset praktikum (lima kelas komponen)*

## Risiko dan Mitigasi

| No | Risiko | Mitigasi |
|---|---|---|
| 1 | Kacamata pelindung yang transparan dan memantulkan cahaya sulit dikenali, sehingga pelanggaran terlewat | Menambah data dengan variasi sudut dan pencahayaan, memantau recall kelas tidak memakai APD, dan mempertimbangkan fine-tuning parsial |
| 2 | Citra wajah pekerja menimbulkan masalah privasi | Pengambilan data dengan izin, citra tidak dipublikasikan, dan data disimpan terbatas pada kelompok |
| 3 | Akurasi turun pada pekerja, posisi, atau kondisi ruang (asap, pencahayaan, kamera bergerak) yang belum pernah dilihat model | Menambah variasi data dan menguji pada sesi serta lokasi berbeda sebelum integrasi |
| 4 | FPS tidak tercapai karena latensi di perangkat robot lebih tinggi daripada hasil di komputer uji | Mengukur ulang latensi pada perangkat target dan menyiapkan MobileNetV3-Small sebagai cadangan |
| 5 | Model menebak kelas dari tingkat kecerahan (*shortcut learning*) atau terjadi *data leakage* | Data dibagi per sesi dengan kondisi validasi yang sama untuk semua kelas |
