const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const submitBtn = form.querySelector('button[type="submit"]');
const errorDisplay = document.getElementById("error-message");
const fileInput = document.getElementById("xray");

//  Image Preview Feature
fileInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        let preview = document.getElementById("preview-img");

        if (!preview) {
            preview = document.createElement("img");
            preview.id = "preview-img";
            preview.style.marginTop = "15px";
            preview.style.maxWidth = "100%";
            preview.style.borderRadius = "12px";
            form.appendChild(preview);
        }

        preview.src = URL.createObjectURL(file);
    }
});

//  Form Submit
form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const originalBtnText = submitBtn.innerHTML;

    try {
        // Loading UI 
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span class="loader"></span> Analyzing...`;
        resultDiv.style.display = "none";
        errorDisplay.style.display = "none";

        const formData = new FormData(form);

        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || "Processing failed");
        }

        //  Show Results with animation
        resultDiv.style.display = "block";
        resultDiv.style.opacity = 0;

        document.getElementById("disease").textContent = data.disease;
        document.getElementById("severity").textContent = data.severity;

        const segmentedImg = document.getElementById("segmented-img");
        segmentedImg.src = `/static/output_images/${data.segmented_image}`;

        segmentedImg.onload = () => {
            segmentedImg.style.opacity = 0;
            setTimeout(() => {
                segmentedImg.style.transition = "0.5s";
                segmentedImg.style.opacity = 1;
            }, 100);
        };

        segmentedImg.onerror = () => {
            segmentedImg.src = "";
            showToast(" Failed to load segmented image");
        };

        //  Download link
        const downloadLink = document.getElementById("download-report");
        downloadLink.href = `/static/output_images/${data.pdf_report}`;

        //  Fade-in animation
        setTimeout(() => {
            resultDiv.style.transition = "0.6s";
            resultDiv.style.opacity = 1;
        }, 100);

        //  Scroll to results
        resultDiv.scrollIntoView({ behavior: "smooth" });

        showToast(" Analysis Completed!");

    } catch (error) {
        console.error("Error:", error);
        showToast(` ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
});

//  Toast Notification System (Modern UI)
function showToast(message) {
    let toast = document.createElement("div");
    toast.className = "toast";
    toast.innerText = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("show");
    }, 100);

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}