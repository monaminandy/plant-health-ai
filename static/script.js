// ==========================================
// PLANT HEALTH AI
// FRONTEND JAVASCRIPT
// ==========================================


// ==========================================
// GET HTML ELEMENTS
// ==========================================

const leafImage = document.getElementById("leafImage");
const uploadArea = document.getElementById("uploadArea");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const removeImage = document.getElementById("removeImage");
const analyzeButton = document.getElementById("analyzeButton");


// Result elements
const resultsSection = document.getElementById("results");
const resultImage = document.getElementById("resultImage");
const resultPlant = document.getElementById("resultPlant");
const resultDisease = document.getElementById("resultDisease");
const resultConfidence = document.getElementById("resultConfidence");
const confidenceProgress = document.getElementById("confidenceProgress");
const healthStatus = document.getElementById("healthStatus");
const recommendationList = document.getElementById("recommendationList");


// ==========================================
// IMAGE UPLOAD
// ==========================================

leafImage.addEventListener("change", function () {

    const file = this.files[0];

    if (file) {
        handleImage(file);
    }

});


// ==========================================
// HANDLE IMAGE
// ==========================================

function handleImage(file) {

    // Check image type
    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image file.");

        leafImage.value = "";

        return;
    }


    // Maximum file size = 10 MB
    const maxSize = 10 * 1024 * 1024;

    if (file.size > maxSize) {

        alert("Image size must be less than 10MB.");

        leafImage.value = "";

        return;
    }


    // Read image
    const reader = new FileReader();

    reader.onload = function (event) {

        imagePreview.src = event.target.result;

        uploadArea.style.display = "none";

        previewContainer.style.display = "block";

    };

    reader.readAsDataURL(file);
}


// ==========================================
// REMOVE IMAGE
// ==========================================

removeImage.addEventListener("click", function () {

    leafImage.value = "";

    imagePreview.src = "";

    previewContainer.style.display = "none";

    uploadArea.style.display = "block";

});


// ==========================================
// DRAG & DROP
// ==========================================

uploadArea.addEventListener("dragover", function (event) {

    event.preventDefault();

    uploadArea.style.borderColor = "#2f9e55";

    uploadArea.style.background = "#eef9f1";

});


uploadArea.addEventListener("dragleave", function () {

    uploadArea.style.borderColor = "#b9d5bf";

    uploadArea.style.background = "#f8fbf8";

});


uploadArea.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadArea.style.borderColor = "#b9d5bf";

    uploadArea.style.background = "#f8fbf8";


    const file = event.dataTransfer.files[0];

    if (file) {

        const dataTransfer = new DataTransfer();

        dataTransfer.items.add(file);

        leafImage.files = dataTransfer.files;

        handleImage(file);

    }

});


// ==========================================
// CLICK UPLOAD AREA
// ==========================================

uploadArea.addEventListener("click", function (event) {

    if (event.target.classList.contains("upload-button")) {
        return;
    }

    leafImage.click();

});


// ==========================================
// ANALYZE PLANT
// ==========================================

analyzeButton.addEventListener("click", function () {

    // Check image
    if (!leafImage.files || leafImage.files.length === 0) {

        alert("Please upload a leaf image first.");

        return;
    }


    const file = leafImage.files[0];


    // ======================================
    // LOADING STATE
    // ======================================

    analyzeButton.disabled = true;

    analyzeButton.innerHTML = `
        <span>🧠</span>
        Analyzing Plant...
    `;


    // ======================================
    // CREATE FORM DATA
    // ======================================

    const formData = new FormData();

    formData.append("image", file);


    // ======================================
    // SEND IMAGE TO FLASK
    // ======================================

    fetch("/predict", {

        method: "POST",

        body: formData

    })

    .then(response => {

        if (!response.ok) {

            throw new Error(
                "Server returned an error: " + response.status
            );

        }

        return response.json();

    })


    // ======================================
    // RECEIVE AI RESULT
    // ======================================

    .then(data => {

        console.log("AI Response:", data);


        // Restore button
        analyzeButton.disabled = false;

        analyzeButton.innerHTML = `
            <span>🔬</span>
            Analyze Plant
        `;


        // Check success
        if (!data.success) {

            alert(
                "❌ Analysis failed\n\n" +
                data.message
            );

            return;
        }


        // ==================================
        // DISPLAY RESULT
        // ==================================

        displayResults(data, file);

    })


    // ======================================
    // ERROR HANDLING
    // ======================================

    .catch(error => {

        console.error("Prediction Error:", error);


        analyzeButton.disabled = false;

        analyzeButton.innerHTML = `
            <span>🔬</span>
            Analyze Plant
        `;


        alert(
            "❌ Unable to connect to the AI server.\n\n" +
            "Make sure Flask is running."
        );

    });

});


// ==========================================
// DISPLAY RESULTS
// ==========================================

function displayResults(data, file) {

    // ======================================
    // SHOW RESULT IMAGE
    // ======================================

    const imageURL = URL.createObjectURL(file);

    resultImage.src = imageURL;


    // ======================================
    // PLANT NAME
    // ======================================

    resultPlant.textContent =
        data.plant || "Unknown Plant";


    // ======================================
    // DISEASE
    // ======================================

    resultDisease.textContent =
        data.disease || "Unknown Condition";


    // ======================================
    // CONFIDENCE
    // ======================================

    const confidence =
        Number(data.confidence) || 0;

    resultConfidence.textContent =
        confidence.toFixed(1) + "%";


    // ======================================
    // CONFIDENCE BAR
    // ======================================

    confidenceProgress.style.width = "0%";


    // Small delay so animation is visible
    setTimeout(function () {

        confidenceProgress.style.width =
            Math.min(confidence, 100) + "%";

    }, 100);


    // ======================================
    // HEALTH STATUS
    // ======================================

    if (
        data.status &&
        data.status.toLowerCase().includes("healthy")
    ) {

        healthStatus.textContent =
            "🟢 Healthy";

        healthStatus.style.background =
            "#e8f7ec";

        healthStatus.style.color =
            "#278348";

    } else {

        healthStatus.textContent =
            "🔴 " + (data.status || "Needs Attention");

        healthStatus.style.background =
            "#fff0f0";

        healthStatus.style.color =
            "#d34b4b";
    }


    // ======================================
    // RECOMMENDATIONS
    // ======================================

    recommendationList.innerHTML = "";


    if (
        data.recommendations &&
        data.recommendations.length > 0
    ) {

        data.recommendations.forEach(
            function (recommendation) {

                const li =
                    document.createElement("li");

                li.textContent =
                    "🌱 " + recommendation;

                recommendationList.appendChild(li);

            }
        );

    } else {

        const li =
            document.createElement("li");

        li.textContent =
            "🌿 No specific recommendations available.";

        recommendationList.appendChild(li);

    }


    // ======================================
    // SHOW RESULTS SECTION
    // ======================================

    resultsSection.classList.add("show");


    // ======================================
    // SCROLL TO RESULTS
    // ======================================

    setTimeout(function () {

        resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }, 200);

}


// ==========================================
// NAVIGATION BUTTON
// ==========================================

const navButton =
    document.querySelector(".nav-button");


if (navButton) {

    navButton.addEventListener(
        "click",
        function () {

            document
                .getElementById("home")
                .scrollIntoView({
                    behavior: "smooth"
                });

        }
    );

}


// ==========================================
// KEYBOARD ACCESSIBILITY
// ==========================================

uploadArea.setAttribute(
    "tabindex",
    "0"
);


uploadArea.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter" ||
            event.key === " "
        ) {

            event.preventDefault();

            leafImage.click();

        }

    }
);


// ==========================================
// IMAGE ERROR HANDLING
// ==========================================

imagePreview.addEventListener(
    "error",
    function () {

        alert("Unable to load this image.");

        leafImage.value = "";

        previewContainer.style.display =
            "none";

        uploadArea.style.display =
            "block";

    }
);