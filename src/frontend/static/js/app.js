document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadTab = document.getElementById('upload-tab');
    const recordTab = document.getElementById('record-tab');
    const uploadContainer = document.getElementById('upload-container');
    const recordContainer = document.getElementById('record-container');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browse-btn');
    const fileInfo = document.getElementById('file-info');
    const recordBtn = document.getElementById('record-btn');
    const timer = document.getElementById('timer');
    const recordingStatus = document.getElementById('recording-status');
    const classifyBtn = document.getElementById('classify-btn');
    const resultContainer = document.getElementById('result-container');
    const resultClass = document.getElementById('result-class');
    const resultConfidenceValue = document.getElementById('result-confidence-value');
    const confidenceLevel = document.getElementById('confidence-level');
    // const otherProbabilities = document.getElementById('other-probabilities');
    const newClassificationBtn = document.getElementById('new-classification-btn');
    const loadingOverlay = document.getElementById('loading-overlay');

    // State
    let audioFile = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let recordingTimer = null;
    let recordingDuration = 0;

    // Tab switching
    uploadTab.addEventListener('click', () => {
        uploadTab.classList.add('active');
        recordTab.classList.remove('active');
        uploadContainer.classList.add('active');
        recordContainer.classList.remove('active');
    });

    recordTab.addEventListener('click', () => {
        recordTab.classList.add('active');
        uploadTab.classList.remove('active');
        recordContainer.classList.add('active');
        uploadContainer.classList.remove('active');
    });

    // File upload handling
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(e) {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    }

    function handleFile(file) {
        if (!file.type.includes('audio')) {
            fileInfo.textContent = 'Erro: Por favor, selecione um arquivo de áudio.';
            fileInfo.style.color = 'red';
            audioFile = null;
            classifyBtn.disabled = true;
            return;
        }

        audioFile = file;
        fileInfo.textContent = `Arquivo selecionado: ${file.name}`;
        fileInfo.style.color = '#333';
        classifyBtn.disabled = false;
    }

    // Audio recording
    recordBtn.addEventListener('click', toggleRecording);

    async function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.addEventListener('dataavailable', (e) => {
                audioChunks.push(e.data);
            });

            mediaRecorder.addEventListener('stop', () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
                recordingStatus.textContent = 'Gravação concluída!';
                classifyBtn.disabled = false;
            });

            recordingDuration = 0;
            recordingTimer = setInterval(updateTimer, 1000);

            mediaRecorder.start();
            isRecording = true;
            recordBtn.classList.add('recording');
            recordBtn.querySelector('.record-text').textContent = 'Parar Gravação';
            recordingStatus.textContent = 'Gravando...';

            // Auto-stop after 30 seconds
            setTimeout(() => {
                if (isRecording) {
                    stopRecording();
                }
            }, 30000);

        } catch (err) {
            console.error('Erro ao iniciar gravação:', err);
            recordingStatus.textContent = 'Erro ao acessar microfone. Verifique as permissões.';
        }
    }

    function stopRecording() {
        if (!mediaRecorder) return;

        mediaRecorder.stop();
        isRecording = false;
        clearInterval(recordingTimer);

        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Iniciar Gravação';

        const tracks = mediaRecorder.stream.getTracks();
        tracks.forEach(track => track.stop());
    }

    function updateTimer() {
        recordingDuration++;
        const minutes = Math.floor(recordingDuration / 60).toString().padStart(2, '0');
        const seconds = (recordingDuration % 60).toString().padStart(2, '0');
        timer.textContent = `${minutes}:${seconds}`;
    }

    // Sound classification
    classifyBtn.addEventListener('click', classifySound);
    newClassificationBtn.addEventListener('click', resetClassification);

    async function classifySound() {
        if (!audioFile) return;

        loadingOverlay.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', audioFile);

        try {
            const response = await fetch('http://localhost:8001/api/classify', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Erro: ${response.status}`);
            }

            const result = await response.json();
            displayResults(result);
        } catch (err) {
            console.error('Erro ao classificar som:', err);
            alert('Erro ao processar o áudio. Por favor, tente novamente.');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    function displayResults(result) {
        // Hide input controls and show results
        document.querySelector('.audio-input-container').classList.add('hidden');
        resultContainer.classList.remove('hidden');

        // Display main result
        resultClass.textContent = formatClassName(result.class);
        resultConfidenceValue.textContent = `${(result.confidence).toFixed(3)}%`;
        confidenceLevel.style.width = `${result.confidence * 100}%`;
    }

    function formatClassName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    function resetClassification() {
        // Reset UI
        resultContainer.classList.add('hidden');
        document.querySelector('.audio-input-container').classList.remove('hidden');

        // Reset state
        audioFile = null;
        fileInfo.textContent = '';
        classifyBtn.disabled = true;

        // Reset tabs
        uploadTab.click();
    }
});