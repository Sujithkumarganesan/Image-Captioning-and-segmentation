/**
 * Main JavaScript file for Image Captioning & Segmentation App
 */

// Global app configuration
const AppConfig = {
    apiBase: '/api/v1',
    maxFileSize: 16 * 1024 * 1024, // 16MB
    allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'],
    uploadTimeout: 60000, // 60 seconds
};

// Utility functions
const Utils = {
    /**
     * Format file size in human readable format
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Validate file type and size
     */
    validateFile: function(file) {
        const errors = [];
        
        // Check file type
        if (!AppConfig.allowedTypes.includes(file.type)) {
            errors.push('Invalid file type. Please select an image file.');
        }
        
        // Check file size
        if (file.size > AppConfig.maxFileSize) {
            errors.push(`File too large. Maximum size is ${Utils.formatFileSize(AppConfig.maxFileSize)}.`);
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors
        };
    },

    /**
     * Show toast notification
     */
    showToast: function(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast after it's hidden
        toast.addEventListener('hidden.bs.toast', function() {
            toastContainer.removeChild(toast);
        });
    },

    /**
     * Create toast container if it doesn't exist
     */
    createToastContainer: function() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    },

    /**
     * Debounce function
     */
    debounce: function(func, wait, immediate) {
        let timeout;
        return function executedFunction() {
            const context = this;
            const args = arguments;
            const later = function() {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    }
};

// API service
const ApiService = {
    /**
     * Make API request
     */
    request: async function(endpoint, options = {}) {
        const url = `${AppConfig.apiBase}${endpoint}`;
        
        try {
            const response = await fetch(url, {
                timeout: AppConfig.uploadTimeout,
                ...options
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'API request failed');
            }
            
            return data;
        } catch (error) {
            console.error('API request error:', error);
            throw error;
        }
    },

    /**
     * Check API health
     */
    checkHealth: async function() {
        return this.request('/health');
    },

    /**
     * Generate caption for image
     */
    generateCaption: async function(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        return this.request('/caption', {
            method: 'POST',
            body: formData
        });
    },

    /**
     * Perform image segmentation
     */
    segmentImage: async function(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        return this.request('/segment', {
            method: 'POST',
            body: formData
        });
    },

    /**
     * Perform combined analysis
     */
    analyzeImage: async function(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        return this.request('/analyze', {
            method: 'POST',
            body: formData
        });
    }
};

// Model status manager
const ModelStatus = {
    status: {
        captioning: false,
        segmentation: false,
        integrated: false
    },

    /**
     * Update model status display
     */
    updateDisplay: function() {
        Object.keys(this.status).forEach(model => {
            const statusElements = document.querySelectorAll(`[data-model="${model}"]`);
            statusElements.forEach(element => {
                const badge = element.querySelector('.badge');
                if (badge) {
                    if (this.status[model]) {
                        badge.className = 'badge bg-success';
                        badge.textContent = 'Ready';
                    } else {
                        badge.className = 'badge bg-warning';
                        badge.textContent = 'Loading...';
                    }
                }
            });
        });
    },

    /**
     * Check and update status
     */
    checkStatus: async function() {
        try {
            const response = await ApiService.checkHealth();
            if (response.models) {
                this.status = response.models;
                this.updateDisplay();
            }
        } catch (error) {
            console.error('Failed to check model status:', error);
        }
    },

    /**
     * Start periodic status checking
     */
    startChecking: function(interval = 5000) {
        this.checkStatus();
        setInterval(() => this.checkStatus(), interval);
    }
};

// Image upload handler
const ImageUpload = {
    /**
     * Initialize upload functionality
     */
    init: function() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        if (!uploadArea || !fileInput) return;
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input change
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Click to upload
        uploadArea.addEventListener('click', (e) => {
            if (e.target === uploadArea || e.target.closest('.upload-content')) {
                fileInput.click();
            }
        });
    },

    handleDragOver: function(e) {
        e.preventDefault();
        e.currentTarget.classList.add('drag-over');
    },

    handleDragLeave: function(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
    },

    handleDrop: function(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    },

    handleFileSelect: function(e) {
        if (e.target.files.length > 0) {
            this.processFile(e.target.files[0]);
        }
    },

    processFile: function(file) {
        const validation = Utils.validateFile(file);
        
        if (!validation.isValid) {
            validation.errors.forEach(error => {
                Utils.showToast(error, 'danger');
            });
            return;
        }
        
        this.showPreview(file);
    },

    showPreview: function(file) {
        const uploadArea = document.getElementById('upload-area');
        const uploadContent = uploadArea.querySelector('.upload-content');
        const uploadPreview = document.getElementById('upload-preview');
        const previewImage = document.getElementById('preview-image');
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        const submitBtn = document.getElementById('submit-btn');
        
        // Update file info
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = Utils.formatFileSize(file.size);
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            if (previewImage) previewImage.src = e.target.result;
            if (uploadContent) uploadContent.classList.add('d-none');
            if (uploadPreview) uploadPreview.classList.remove('d-none');
            if (submitBtn) submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
};

// Progress handler
const ProgressHandler = {
    show: function(message = 'Processing...') {
        const progressSection = document.getElementById('progress-section');
        const progressMessage = document.querySelector('#progress-section h5');
        
        if (progressSection) {
            progressSection.classList.remove('d-none');
            if (progressMessage) progressMessage.textContent = message;
        }
        
        this.startAnimation();
    },

    hide: function() {
        const progressSection = document.getElementById('progress-section');
        if (progressSection) {
            progressSection.classList.add('d-none');
        }
        
        this.stopAnimation();
    },

    updateProgress: function(percent) {
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            progressBar.style.width = percent + '%';
        }
    },

    startAnimation: function() {
        let progress = 0;
        this.animationInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 90) progress = 90;
            this.updateProgress(progress);
        }, 300);
    },

    stopAnimation: function() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.updateProgress(100);
        }
    }
};

// Smooth scrolling for navigation links
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Image zoom functionality
function initImageZoom() {
    document.querySelectorAll('.zoom-on-hover').forEach(img => {
        img.addEventListener('click', function() {
            showImageModal(this.src, this.alt);
        });
    });
}

function showImageModal(src, alt) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${alt}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body text-center">
                    <img src="${src}" class="img-fluid" alt="${alt}">
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
    
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
    });
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    ImageUpload.init();
    initSmoothScrolling();
    initImageZoom();
    
    // Start model status checking if on homepage
    if (document.querySelector('.model-status')) {
        ModelStatus.startChecking();
    }
    
    // Add fade-in animation to elements
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.feature-card, .step-number, .card').forEach(el => {
        observer.observe(el);
    });
    
    // Show loading overlay if models are not loaded
    const loadingOverlay = document.querySelector('.loading-overlay');
    if (loadingOverlay) {
        // Check if models are loaded every 5 seconds
        const checkModels = setInterval(async () => {
            try {
                const response = await ApiService.checkHealth();
                if (response.models && Object.values(response.models).every(status => status)) {
                    loadingOverlay.style.display = 'none';
                    clearInterval(checkModels);
                    Utils.showToast('All models loaded successfully!', 'success');
                }
            } catch (error) {
                console.error('Error checking model status:', error);
            }
        }, 5000);
    }
});

// Export utilities for global use
window.AppUtils = Utils;
window.AppApi = ApiService;
window.AppProgress = ProgressHandler;
