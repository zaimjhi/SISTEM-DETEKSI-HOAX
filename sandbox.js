// sandbox.js

let model;
// Fungsi untuk memuat model
async function loadModel() {
  console.log("Sandbox: Memuat model...");
  try {
    // Dapatkan URL lengkap ke model sesuai nama folder Anda
    const modelURL = chrome.runtime.getURL('folder_output_web/model.json');
    console.log("Memuat model dari:", modelURL);
    model = await tf.loadLayersModel(modelURL);
    console.log("Sandbox: Model berhasil dimuat!");
  } catch (e) {
    console.error("Sandbox: Gagal memuat model", e);
  }
}
loadModel();

// Dengarkan pesan dari 'offscreen.js'
window.addEventListener('message', async (event) => {
  console.log("Sandbox: Menerima teks ->", event.data);

  // NANTINYA LOGIKA PREDIKSI SEBENARNYA ADA DI SINI
  // Untuk sekarang, kita balas dengan data pura-pura
  const isHoax = Math.random() < 0.5; // Hasil acak
  const resultText = isHoax ? "Hasil: Terindikasi Hoax!" : "Hasil: Terlihat Fakta.";
  
  // Kirim hasil kembali ke 'offscreen.js'
  // Pastikan event.source dan event.origin ada sebelum mengirim
  if (event.source && event.origin) {
      event.source.postMessage({ result: resultText }, event.origin);
  }
});