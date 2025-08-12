document.getElementById("upload-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    
    // Get UI elements
    const form = this;
    const resultDiv = document.getElementById("result");
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.textContent;

    try {
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = "Processing...";
        resultDiv.style.display = "none";
        
        // Create and send form data
        const formData = new FormData(form);
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        // Handle HTTP errors
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Handle backend processing errors
        if (!data.success) {
            throw new Error(data.error || "Processing failed");
        }

        // Display results
        resultDiv.style.display = "block";
        document.getElementById("disease").textContent = data.disease;
        document.getElementById("severity").textContent = data.severity;
        
        // Ensure image paths are correct
        const segmentedImg = document.getElementById("segmented-img");
        segmentedImg.src = `/static/output_images/${data.segmented_image}`;
        segmentedImg.onerror = () => {
            segmentedImg.src = ""; // Clear if image fails to load
            console.error("Failed to load segmented image");
        };
        
        // Set report download link
        const downloadLink = document.getElementById("download-report");
        downloadLink.href = `/static/output_images/${data.pdf_report}`;
        
        // Optional: Auto-scroll to results
        resultDiv.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error("Error:", error);
        
        // Show user-friendly error message
        const errorDisplay = document.getElementById("error-message") || createErrorDisplay();
        errorDisplay.textContent = `Error: ${error.message}`;
        errorDisplay.style.display = "block";
        
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
    }
});

// Helper function to create error display element if it doesn't exist
function createErrorDisplay() {
    const errorDiv = document.createElement("div");
    errorDiv.id = "error-message";
    errorDiv.style.color = "red";
    errorDiv.style.margin = "10px 0";
    document.getElementById("upload-form").appendChild(errorDiv);
    return errorDiv;
}





