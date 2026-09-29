from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np
import os
import json


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "plant_disease_model.keras"
)


CLASS_NAMES_PATH = os.path.join(
    MODEL_FOLDER,
    "class_names.json"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nTrained model not found!\n"
        f"Expected location: {MODEL_PATH}\n\n"
        "Please train the model first."
    )


if not os.path.exists(CLASS_NAMES_PATH):

    raise FileNotFoundError(
        "\nclass_names.json not found!\n"
        f"Expected location: {CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("LOADING PLANT DISEASE MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("✓ Model loaded successfully")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)


print("\nClasses loaded:")

for index, class_name in enumerate(class_names):

    print(
        f"{index}: {class_name}"
    )


print(
    f"\nTotal classes: {len(class_names)}"
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# DISEASE DISPLAY NAMES
# ============================================================

DISEASE_NAMES = {

    "Tomato___Bacterial_spot":
        "Bacterial Spot",

    "Tomato___Early_blight":
        "Early Blight",

    "Tomato___Late_blight":
        "Late Blight",

    "Tomato___Leaf_Mold":
        "Leaf Mold",

    "Tomato___Septoria_leaf_spot":
        "Septoria Leaf Spot",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Spider Mites",

    "Tomato___Target_Spot":
        "Target Spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",

    "Tomato___Tomato_mosaic_virus":
        "Tomato Mosaic Virus",

    "Tomato___healthy":
        "Healthy"

}


# ============================================================
# CARE RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "Tomato___healthy": [

        "Maintain regular watering based on soil moisture.",

        "Provide adequate sunlight for healthy growth.",

        "Keep the growing area clean and well ventilated.",

        "Continue regular inspection of leaves."
    ],


    "Tomato___Bacterial_spot": [

        "Remove severely affected leaves.",

        "Avoid wetting the leaves while watering.",

        "Improve air circulation around the plant.",

        "Monitor nearby plants for similar symptoms."
    ],


    "Tomato___Early_blight": [

        "Remove severely affected leaves.",

        "Avoid overhead watering.",

        "Improve air circulation around the plant.",

        "Keep fallen infected leaves away from the plant."
    ],


    "Tomato___Late_blight": [

        "Remove severely affected plant material.",

        "Avoid excessive leaf moisture.",

        "Improve ventilation around the plants.",

        "Monitor the plant closely for disease progression."
    ],


    "Tomato___Leaf_Mold": [

        "Improve air circulation.",

        "Reduce excessive humidity around the leaves.",

        "Avoid watering the foliage directly.",

        "Remove severely affected leaves."
    ],


    "Tomato___Septoria_leaf_spot": [

        "Remove infected leaves.",

        "Avoid overhead watering.",

        "Keep adequate spacing between plants.",

        "Remove fallen infected plant material."
    ],


    "Tomato___Spider_mites Two-spotted_spider_mite": [

        "Inspect the underside of leaves.",

        "Keep the plant adequately hydrated.",

        "Remove heavily affected leaves.",

        "Monitor the plant regularly for mite activity."
    ],


    "Tomato___Target_Spot": [

        "Remove severely affected leaves.",

        "Improve air circulation.",

        "Avoid prolonged leaf wetness.",

        "Keep the growing area clean."
    ],


    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [

        "Inspect the plant regularly for whitefly activity.",

        "Remove severely affected plants where appropriate.",

        "Keep weeds around the growing area under control.",

        "Monitor nearby plants for similar symptoms."
    ],


    "Tomato___Tomato_mosaic_virus": [

        "Remove severely infected plant material.",

        "Avoid handling healthy plants after infected plants.",

        "Keep gardening tools clean.",

        "Monitor nearby plants for symptoms."
    ]

}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_disease(image_path):

    # ----------------------------------------
    # Load image
    # ----------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # ----------------------------------------
    # Convert image to array
    # ----------------------------------------

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # ----------------------------------------
    # Add batch dimension
    # ----------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ----------------------------------------
    # IMPORTANT
    #
    # Our trained model already contains
    # MobileNetV2 preprocess_input.
    #
    # Therefore we DON'T call
    # preprocess_input here again.
    # ----------------------------------------

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    # ----------------------------------------
    # SHOW ALL 10 CLASS PROBABILITIES
    # ----------------------------------------

    print("\n========================================")
    print("        MODEL PREDICTIONS")
    print("========================================")

    for i, probability in enumerate(probabilities):

        print(
            f"{class_names[i]}: "
            f"{probability * 100:.2f}%"
        )

    print("========================================")

    # ----------------------------------------
    # Find highest probability
    # ----------------------------------------

    predicted_index = np.argmax(
        probabilities
    )

    confidence = float(
        probabilities[predicted_index] * 100
    )

    predicted_class = class_names[
        predicted_index
    ]

    # ----------------------------------------
    # Show final prediction
    # ----------------------------------------

    print(
        "Predicted:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    print("========================================\n")

    return predicted_class, confidence


# ============================================================
# PREDICT API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ----------------------------------------
    # Check image
    # ----------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "message": "No image uploaded."

        }), 400


    image = request.files["image"]


    # ----------------------------------------
    # Check filename
    # ----------------------------------------

    if image.filename == "":

        return jsonify({

            "success": False,

            "message": "No image selected."

        }), 400


    # ----------------------------------------
    # Secure filename
    # ----------------------------------------

    filename = secure_filename(
        image.filename
    )


    # ----------------------------------------
    # Save image
    # ----------------------------------------

    image_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    image.save(
        image_path
    )


    try:

        # ====================================
        # RUN AI MODEL
        # ====================================

        predicted_class, confidence = (
            predict_disease(image_path)
        )


        # ====================================
        # PLANT
        # ====================================

        plant = "Tomato"


        # ====================================
        # DISEASE NAME
        # ====================================

        disease = DISEASE_NAMES.get(

            predicted_class,

            predicted_class.replace(
                "Tomato___",
                ""
            ).replace(
                "_",
                " "
            )

        )


        # ====================================
        # HEALTH STATUS
        # ====================================

        if predicted_class == "Tomato___healthy":

            status = "Healthy"

        else:

            status = "Needs Attention"


        # ====================================
        # RECOMMENDATIONS
        # ====================================

        recommendations = RECOMMENDATIONS.get(

            predicted_class,

            [
                "Monitor the plant regularly.",
                "Maintain suitable watering.",
                "Provide adequate sunlight.",
                "Consult an agricultural expert if symptoms worsen."
            ]

        )


        # ====================================
        # RESPONSE
        # ====================================

        result = {

            "success": True,

            "plant": plant,

            "disease": disease,

            "confidence": round(
                confidence,
                2
            ),

            "status": status,

            "recommendations":
                recommendations

        }


        print("\n========================================")
        print("AI PREDICTION")
        print("========================================")

        print(
            "Class:",
            predicted_class
        )

        print(
            "Disease:",
            disease
        )

        print(
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )


        return jsonify(result)


    except Exception as error:

        print(
            "\nPrediction error:",
            error
        )


        return jsonify({

            "success": False,

            "message":
                "Unable to analyze the image."

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )