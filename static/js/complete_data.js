document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    const mp3File = document.getElementById('mp3File').files[0];
    const playRecordsFile = document.getElementById('playRecordsFile').files[0];
    const complete_data = document.getElementById('comleteDataset').value;

    const formData = new FormData();
    formData.append('mp3_csv_file', mp3File);
    formData.append('play_records_csv_file', playRecordsFile);
    formData.append('output_csv_path',complete_data)
    fetch('/generate_complete_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('responseMessage').innerText = data.message;
        if (data.status === 'success') {
            alert('Dataset generation successful!');
        } else {
            alert('An error occurred: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An unexpected error occurred.');
    });
});