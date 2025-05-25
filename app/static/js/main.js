/**
 * Main JavaScript file for Emotion Detection App
 *
 * This file contains common functionality used across the application,
 * including utility functions, API calls, and UI enhancements.
 */

// Global app object
window.EmotionApp = {
    config: {
        apiBaseUrl: "",
        supportedFormats: [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/gif",
            "image/bmp",
        ],
        maxFileSize: 16 * 1024 * 1024, // 16MB
    },

    // Available models cache
    availableModels: [],
    currentModel: null,

    // Initialize app
    init: function () {
        console.log("Emotion Detection App initialized");
        this.loadAvailableModels();
        this.initializeCommonEventListeners();
        this.addAnimations();
    },

    /**
     * Load available models from API
     */
    loadAvailableModels: function () {
        fetch("/api/models")
            .then((response) => response.json())
            .then((data) => {
                this.availableModels = data.models;
                this.currentModel = data.current;
                console.log("Available models:", this.availableModels);
            })
            .catch((error) => {
                console.error("Failed to load available models:", error);
            });
    },

    /**
     * Switch to a different model
     * @param {string} modelName - Name of the model to switch to
     * @returns {Promise} Promise that resolves when model is switched
     */
    switchModel: function (modelName) {
        const loadingElement = this.showLoading("Switching model...");

        return fetch("/api/switch_model", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ model: modelName }),
        })
            .then((response) => response.json())
            .then((data) => {
                this.hideLoading(loadingElement);

                if (data.success) {
                    this.currentModel = data.current_model;
                    this.showNotification(
                        "Model switched successfully!",
                        "success"
                    );
                    return data;
                } else {
                    throw new Error(data.error || "Failed to switch model");
                }
            })
            .catch((error) => {
                this.hideLoading(loadingElement);
                this.showNotification(
                    `Failed to switch model: ${error.message}`,
                    "error"
                );
                throw error;
            });
    },

    /**
     * Validate uploaded file
     * @param {File} file - File to validate
     * @returns {Object} Validation result with isValid and error message
     */
    validateFile: function (file) {
        if (!file) {
            return { isValid: false, error: "No file selected" };
        }

        if (file.size > this.config.maxFileSize) {
            return {
                isValid: false,
                error: "File size must be less than 16MB",
            };
        }

        if (!this.config.supportedFormats.includes(file.type)) {
            return {
                isValid: false,
                error: "Please select a valid image file (JPG, PNG, GIF, BMP)",
            };
        }

        return { isValid: true };
    },

    /**
     * Show loading spinner with optional message
     * @param {string} message - Loading message
     * @returns {HTMLElement} Loading element
     */
    showLoading: function (message = "Loading...") {
        const loadingDiv = document.createElement("div");
        loadingDiv.className =
            "position-fixed top-50 start-50 translate-middle";
        loadingDiv.style.zIndex = "9999";
        loadingDiv.innerHTML = `
            <div class="bg-white p-4 rounded shadow text-center">
                <div class="spinner-border text-primary mb-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div>${message}</div>
            </div>
        `;

        document.body.appendChild(loadingDiv);
        return loadingDiv;
    },

    /**
     * Hide loading spinner
     * @param {HTMLElement} loadingElement - Loading element to remove
     */
    hideLoading: function (loadingElement) {
        if (loadingElement && loadingElement.parentNode) {
            loadingElement.parentNode.removeChild(loadingElement);
        }
    },

    /**
     * Show notification message
     * @param {string} message - Message to show
     * @param {string} type - Type of notification (success, error, info, warning)
     */
    showNotification: function (message, type = "info") {
        // Remove existing notifications
        const existingNotifications =
            document.querySelectorAll(".app-notification");
        existingNotifications.forEach((notification) => notification.remove());

        const alertClass = type === "error" ? "alert-danger" : `alert-${type}`;
        const iconClass = this.getNotificationIcon(type);

        const notification = document.createElement("div");
        notification.className = `alert ${alertClass} alert-dismissible fade show app-notification position-fixed`;
        notification.style.cssText =
            "top: 20px; right: 20px; z-index: 9999; max-width: 400px;";
        notification.innerHTML = `
            <i class="${iconClass} me-2"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    },

    /**
     * Get appropriate icon for notification type
     * @param {string} type - Notification type
     * @returns {string} CSS class for icon
     */
    getNotificationIcon: function (type) {
        const icons = {
            success: "fas fa-check-circle",
            error: "fas fa-exclamation-circle",
            warning: "fas fa-exclamation-triangle",
            info: "fas fa-info-circle",
        };
        return icons[type] || icons.info;
    },

    /**
     * Initialize common event listeners
     */
    initializeCommonEventListeners: function () {
        // Handle form submissions with loading states
        document.addEventListener("submit", function (e) {
            const form = e.target;
            if (form.classList.contains("async-form")) {
                e.preventDefault();
                EmotionApp.handleAsyncForm(form);
            }
        });

        // Handle file input changes
        document.addEventListener("change", function (e) {
            if (
                e.target.type === "file" &&
                e.target.accept &&
                e.target.accept.includes("image")
            ) {
                EmotionApp.handleFileSelection(e.target);
            }
        });

        // Add smooth scrolling to anchor links
        document.addEventListener("click", function (e) {
            if (e.target.matches('a[href^="#"]')) {
                e.preventDefault();
                const target = document.querySelector(
                    e.target.getAttribute("href")
                );
                if (target) {
                    target.scrollIntoView({ behavior: "smooth" });
                }
            }
        });
    },

    /**
     * Handle file selection and validation
     * @param {HTMLInputElement} input - File input element
     */
    handleFileSelection: function (input) {
        const file = input.files[0];
        const validation = this.validateFile(file);

        if (!validation.isValid) {
            this.showNotification(validation.error, "error");
            input.value = "";
            return;
        }

        // Show file preview if possible
        this.showFilePreview(file, input);
    },

    /**
     * Show file preview
     * @param {File} file - File to preview
     * @param {HTMLInputElement} input - File input element
     */
    showFilePreview: function (file, input) {
        const previewContainer =
            input.parentElement.querySelector(".file-preview");
        if (!previewContainer) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            previewContainer.innerHTML = `
                <img src="${e.target.result}" class="img-fluid rounded" 
                     style="max-height: 200px;" alt="Preview">
                <div class="mt-2">
                    <small class="text-muted">${
                        file.name
                    } (${EmotionApp.formatFileSize(file.size)})</small>
                </div>
            `;
            previewContainer.style.display = "block";
        };
        reader.readAsDataURL(file);
    },

    /**
     * Format file size for display
     * @param {number} bytes - File size in bytes
     * @returns {string} Formatted file size
     */
    formatFileSize: function (bytes) {
        if (bytes === 0) return "0 Bytes";

        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    },

    /**
     * Handle async form submission
     * @param {HTMLFormElement} form - Form element
     */
    handleAsyncForm: function (form) {
        const formData = new FormData(form);
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;

        // Show loading state
        submitButton.disabled = true;
        submitButton.innerHTML =
            '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';

        fetch(form.action, {
            method: form.method,
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.success) {
                    this.handleFormSuccess(data);
                } else {
                    throw new Error(data.error || "Unknown error occurred");
                }
            })
            .catch((error) => {
                this.showNotification(`Error: ${error.message}`, "error");
            })
            .finally(() => {
                // Restore button state
                submitButton.disabled = false;
                submitButton.textContent = originalText;
            });
    },

    /**
     * Handle successful form submission
     * @param {Object} data - Response data
     */
    handleFormSuccess: function (data) {
        this.showNotification("Operation completed successfully!", "success");

        // Trigger custom event for other components to handle
        document.dispatchEvent(
            new CustomEvent("formSuccess", { detail: data })
        );
    },

    /**
     * Add fade-in animations to elements
     */
    addAnimations: function () {
        // Add intersection observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: "0px 0px -50px 0px",
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("fade-in");
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // Observe cards and other elements
        document
            .querySelectorAll(".card, .alert, .emotion-badge")
            .forEach((el) => {
                observer.observe(el);
            });
    },

    /**
     * Get emotion color based on emotion name
     * @param {string} emotion - Emotion name
     * @returns {string} CSS color class
     */
    getEmotionColor: function (emotion) {
        const colors = {
            anger: "text-danger",
            disgust: "text-warning",
            fear: "text-primary",
            happy: "text-success",
            sad: "text-info",
            surprise: "text-purple",
            neutral: "text-secondary",
            contempt: "text-dark",
        };
        return colors[emotion.toLowerCase()] || "text-muted";
    },

    /**
     * Get emotion icon based on emotion name
     * @param {string} emotion - Emotion name
     * @returns {string} CSS icon class
     */
    getEmotionIcon: function (emotion) {
        const icons = {
            anger: "fas fa-angry",
            disgust: "fas fa-grimace",
            fear: "fas fa-frown",
            happy: "fas fa-smile",
            sad: "fas fa-sad-tear",
            surprise: "fas fa-surprise",
            neutral: "fas fa-meh",
            contempt: "fas fa-grin-squint",
        };
        return icons[emotion.toLowerCase()] || "fas fa-question";
    },

    /**
     * Create emotion visualization chart
     * @param {Object} emotions - Emotion scores
     * @param {HTMLElement} container - Container element
     */
    createEmotionChart: function (emotions, container) {
        // Simple horizontal bar chart
        const maxEmotion = Math.max(...Object.values(emotions));

        container.innerHTML = "";
        Object.entries(emotions).forEach(([emotion, score]) => {
            const percentage = (score * 100).toFixed(1);
            const barWidth = (score / maxEmotion) * 100;

            const emotionDiv = document.createElement("div");
            emotionDiv.className = "emotion-bar mb-2";
            emotionDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="${this.getEmotionColor(emotion)}">
                        <i class="${this.getEmotionIcon(emotion)} me-2"></i>
                        ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                    </span>
                    <span class="text-muted">${percentage}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar bg-primary" style="width: ${barWidth}%"></div>
                </div>
            `;
            container.appendChild(emotionDiv);
        });
    },
};

// Initialize app when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
    window.EmotionApp.init();
});

// Add some global utility functions
window.utils = {
    /**
     * Debounce function calls
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    debounce: function (func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Throttle function calls
     * @param {Function} func - Function to throttle
     * @param {number} limit - Time limit in milliseconds
     * @returns {Function} Throttled function
     */
    throttle: function (func, limit) {
        let inThrottle;
        return function (...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => (inThrottle = false), limit);
            }
        };
    },

    /**
     * Format confidence score for display
     * @param {number} confidence - Confidence score (0-1)
     * @returns {string} Formatted percentage
     */
    formatConfidence: function (confidence) {
        return (confidence * 100).toFixed(1) + "%";
    },
};

// Handle page visibility changes for performance optimization
document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
        console.log("Page hidden - pausing non-essential operations");
    } else {
        console.log("Page visible - resuming operations");
    }
});
