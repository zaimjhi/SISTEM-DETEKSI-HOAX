document.addEventListener('DOMContentLoaded', () => {
  const checkButton = document.getElementById('checkButton');
  const inputText = document.getElementById('inputText');
  const resultText = document.getElementById('resultText');
  const loader = document.getElementById('loader');

  checkButton.addEventListener('click', async () => {
    const text = inputText.value;
    if (!text) {
      // Sembunyikan hasil dan loader jika tidak ada teks
      resultText.style.display = 'none';
      loader.style.display = 'none';
      alert("Silakan masukkan teks dulu.");
      return;
    }

    // Tampilkan loader dan sembunyikan hasil sebelumnya
    loader.style.display = 'block';
    resultText.style.display = 'none';
    resultText.className = ''; // Hapus class warna sebelumnya

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
      });

      // ====================================================================
      // BLOK PENANGANAN ERROR YANG TELAH DIPERBARUI
      // ====================================================================
      if (!response.ok) {
        // Coba baca body error untuk mendapatkan pesan spesifik dari server
        const errorData = await response.json();
        // Gunakan pesan dari server jika ada, jika tidak, gunakan pesan default
        throw new Error(errorData.message || `Server error: ${response.statusText}`);
      }
      // ====================================================================
      // AKHIR DARI BLOK MODIFIKASI
      // ====================================================================

      const data = await response.json();
      
      // Logika tampilan hasil yang telah diperbaiki
      if (data.is_hoax) {
        resultText.innerText = `Hasil: Terindikasi fakta (Score: ${data.prediction_score.toFixed(2)})`;
        resultText.className = 'result-hoax';
      } else {
        resultText.innerText = `Hasil: Terindikasi hoax (Score: ${(1 - data.prediction_score).toFixed(2)})`;
        resultText.className = 'result-fakta';
      }

    } catch (error) {
      console.error("Gagal menghubungi server atau input tidak valid:", error);
      // 'error.message' sekarang akan berisi pesan dinamis dari server
      resultText.innerText = `Error: ${error.message}`;
      resultText.className = 'result-hoax'; // Tampilkan error dengan warna merah
    } finally {
      // Apapun hasilnya, sembunyikan loader dan tampilkan area hasil
      loader.style.display = 'none';
      resultText.style.display = 'block';
    }
  });
});
// document.addEventListener('DOMContentLoaded', () => {
//   const checkButton = document.getElementById('checkButton');
//   const inputText = document.getElementById('inputText');
//   const resultText = document.getElementById('resultText');
//   const loader = document.getElementById('loader');

//   checkButton.addEventListener('click', async () => {
//     const text = inputText.value;
//     if (!text) {
//       // Sembunyikan hasil dan loader jika tidak ada teks
//       resultText.style.display = 'none';
//       loader.style.display = 'none';
//       alert("Silakan masukkan teks dulu.");
//       return;
//     }

//     // Tampilkan loader dan sembunyikan hasil sebelumnya
//     loader.style.display = 'block';
//     resultText.style.display = 'none';
//     resultText.className = ''; // Hapus class warna sebelumnya

//     try {
//       const response = await fetch('http://127.0.0.1:5000/predict', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ text: text }),
//       });

//       if (!response.ok) {
//         throw new Error(`Server error: ${response.statusText}`);
//       }

//       const data = await response.json();
      
//       // Tentukan teks dan class berdasarkan hasil
//       if (data.is_hoax) {
//         resultText.innerText = `Hasil: Terindikasi fakta (Score: ${data.prediction_score.toFixed(2)})`;
//         resultText.className = 'result-hoax';
//       } else {
//         resultText.innerText = `Hasil: Terlihat hoax (Score: ${(1 - data.prediction_score).toFixed(2)})`;
//         resultText.className = 'result-fakta';
//       }

//     } catch (error) {
//       console.error("Gagal menghubungi server:", error);
//       resultText.innerText = "Error: Gagal menghubungi server.";
//       resultText.className = 'result-hoax'; // Tampilkan error dengan warna merah
//     } finally {
//       // Apapun hasilnya, sembunyikan loader dan tampilkan area hasil
//       loader.style.display = 'none';
//       resultText.style.display = 'block';
//     }
//   });
// });