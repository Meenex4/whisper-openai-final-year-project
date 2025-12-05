/* Audio file preview */
const recorder = document.getElementById('recorder');
const player = document.getElementById('player');
const fileNameDisplay = document.getElementById('file-name');
const audioPlayerWrapper = document.querySelector('.audio-player-wrapper');
const uploadBox = document.querySelector('.upload-box');

recorder.addEventListener('change', function (e) {
    const file = e.target.files[0];
    
    if (file) {
        // Update file name display
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        
        // Create and set audio source
        const url = URL.createObjectURL(file);
        player.src = url;
        
        // Show audio player
        audioPlayerWrapper.classList.add('active');
        
        // Update upload box appearance
        uploadBox.style.borderColor = '#10b981';
        uploadBox.style.background = '#f0fdf4';
    }
});

/* Form submission */
const conversionForm = document.getElementById('conversion-form');
const submitButton = document.getElementById('submit');
const spinner = document.getElementById('spinner');
const submitIcon = document.getElementById('submit-icon');
const submitText = document.getElementById('submit-text');

conversionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    
    // Disable button and show loading state
    submitButton.disabled = true;
    spinner.classList.remove('d-none');
    submitIcon.style.display = 'none';
    submitText.textContent = 'Processing...';

    try {
        const formData = new FormData(conversionForm);
        
        const response = await fetch('/download/', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Conversion failed');
        }

        const blob = await response.blob();
        const downloadUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `${formData.get('filename')}.${formData.get('file_type')}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Success state
        submitText.textContent = 'Success!';
        setTimeout(() => {
            submitText.textContent = 'Convert Audio';
            submitIcon.style.display = 'inline';
        }, 2000);
        
    } catch (error) {
        console.error('Error:', error);
        submitText.textContent = 'Error - Try Again';
        setTimeout(() => {
            submitText.textContent = 'Convert Audio';
            submitIcon.style.display = 'inline';
        }, 3000);
    } finally {
        // Re-enable button
        submitButton.disabled = false;
        spinner.classList.add('d-none');
    }
});

/* Drag and drop functionality */
const uploadLabel = document.querySelector('.upload-label');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    uploadBox.style.borderColor = '#6366f1';
    uploadBox.style.background = '#f0f4ff';
}

function unhighlight(e) {
    uploadBox.style.borderColor = '#e2e8f0';
    uploadBox.style.background = '#f8fafc';
}

uploadLabel.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        recorder.files = files;
        const event = new Event('change');
        recorder.dispatchEvent(event);
    }
}