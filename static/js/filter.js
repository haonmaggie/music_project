document.getElementById('recommendationForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const userId = document.getElementById('userId').value;
    const outputFileName = document.getElementById('outputFileName').value;

    const songPlaycountFile = document.getElementById('songPlaycountFile').files[0];
    const userPlaycountFile = document.getElementById('userPlaycountFile').files[0];
    const completeDatasetFile = document.getElementById('completeDatasetFile').files[0];
    const musicDataFrameFile = document.getElementById('musicDataFrame').files[0];

    const formData = new FormData();
    formData.append('output_file', outputFileName);
    formData.append('user_id', userId);
    formData.append('song_playcount_df', songPlaycountFile);
    formData.append('user_playcount_df', userPlaycountFile);
    formData.append('complete_dataset', completeDatasetFile);
    formData.append('muisc_df', musicDataFrameFile);

    try {
        const response = await fetch('/process_and_recommend', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById('responseMessage').innerText = result.message || 'Recommendations processed successfully!';

        if (result.downloadLink) {
            // 假设后端返回了下载链接
            alert(`数据excel已生成保存: ${result.downloadLink}`);
            goToIndexImmediately()
        }
    } catch (error) {
        document.getElementById('responseMessage').innerText = `Error: ${error.message}`;
    }
});
function goToIndexImmediately() {
        window.location.href = "/"; // 或者使用具体的首页URL
    }