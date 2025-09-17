// offscreen.js

// Dapatkan referensi ke iframe sandbox
const iframe = document.getElementById('sandbox-iframe');

// Dengarkan pesan dari popup.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target === 'offscreen' && message.type === 'predict') {
    // Teruskan pesan ke sandbox.js melalui iframe
    iframe.contentWindow.postMessage(message.data);

    // Siapkan listener untuk menerima balasan dari sandbox
    window.onmessage = (event) => {
      // Kirim balasan kembali ke popup.js
      sendResponse(event.data);
    };
  }
  return true; // Menandakan balasan akan dikirim secara asynchronous
});