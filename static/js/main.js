document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // ป้องกันไม่ให้ฟอร์มส่งข้อมูลแบบปกติ

    const fileInput = document.getElementById('file-input');
    const predictionText = document.getElementById('prediction-text');
    const formData = new FormData();

    formData.append('file', fileInput.files[0]);
    predictionText.textContent = 'กำลังประมวลผล...';

    try {
        const response = await fetch('/', { // ส่ง request ไปที่ route '/' แบบ POST
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            predictionText.textContent = 'เกิดข้อผิดพลาด: ' + result.error;
        } else {
            predictionText.textContent = result.prediction;
        }
    } catch (error) {
        predictionText.textContent = 'เกิดข้อผิดพลาดในการเชื่อมต่อกับเซิร์ฟเวอร์ค่ะ';
    }
});