import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\Monami\Downloads\PlantVillage-Dataset\raw\color"
)

MODEL_DIR = Path("model")

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SIZE = 0.20

SEED = 42

# Increased from the previous version
INITIAL_EPOCHS = 8

# Increased from the previous version
FINE_TUNE_EPOCHS = 6


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# START
# ============================================================

print("\n========================================")
print("       PLANT HEALTH AI")
print("       TOMATO DISEASE MODEL")
print("========================================\n")


# ============================================================
# CHECK GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")

    for gpu in gpus:
        print(gpu)

else:

    print("No GPU detected.")
    print("Training will use CPU.")


print()


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )


print("Dataset:")
print(DATASET_DIR)


# ============================================================
# FIND TOMATO CLASSES
# ============================================================

tomato_classes = sorted([

    folder.name

    for folder in DATASET_DIR.iterdir()

    if folder.is_dir()
    and folder.name.startswith("Tomato___")

])


print("\n========================================")
print("TOMATO CLASSES")
print("========================================")


for index, class_name in enumerate(tomato_classes):

    print(
        f"{index}: {class_name}"
    )


print(
    f"\nTotal classes: {len(tomato_classes)}"
)


if len(tomato_classes) != 10:

    raise RuntimeError(
        f"Expected 10 Tomato classes, "
        f"but found {len(tomato_classes)}."
    )


# ============================================================
# CREATE CLASS MAPPING
# ============================================================

class_to_index = {

    class_name: index

    for index, class_name
    in enumerate(tomato_classes)

}


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

print("\n========================================")
print("COLLECTING IMAGES")
print("========================================")


image_paths = []

labels = []


valid_extensions = {

    ".jpg",
    ".jpeg",
    ".png"

}


for class_name in tomato_classes:

    class_folder = (
        DATASET_DIR /
        class_name
    )

    class_index = (
        class_to_index[class_name]
    )


    files = [

        file

        for file in class_folder.rglob("*")

        if file.suffix.lower()
        in valid_extensions

    ]


    print(
        f"{class_name}: {len(files)} images"
    )


    for file in files:

        image_paths.append(
            str(file)
        )

        labels.append(
            class_index
        )


image_paths = np.array(
    image_paths
)

labels = np.array(
    labels
)


print(
    "\nTotal Tomato images:",
    len(image_paths)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(

        image_paths,

        labels,

        test_size=VALIDATION_SIZE,

        random_state=SEED,

        stratify=labels

    )
)


print("\n========================================")
print("DATA SPLIT")
print("========================================")


print(
    f"Training images:   {len(train_paths)}"
)

print(
    f"Validation images: {len(val_paths)}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n========================================")
print("CALCULATING CLASS WEIGHTS")
print("========================================")


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_labels),

    y=train_labels

)


class_weights = {

    int(class_index): float(weight)

    for class_index, weight
    in zip(
        np.unique(train_labels),
        class_weights_array
    )

}


for class_index, weight in class_weights.items():

    print(
        f"{tomato_classes[class_index]}: "
        f"{weight:.3f}"
    )


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(
        path
    )


    image = tf.image.decode_image(

        image,

        channels=3,

        expand_animations=False

    )


    image.set_shape(
        [None, None, 3]
    )


    image = tf.image.resize(

        image,

        IMAGE_SIZE

    )


    image = tf.cast(

        image,

        tf.float32

    )


    return image, label


# ============================================================
# CREATE TF.DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(

    (
        train_paths,
        train_labels
    )

)


val_dataset = tf.data.Dataset.from_tensor_slices(

    (
        val_paths,
        val_labels
    )

)


# ============================================================
# SHUFFLE TRAINING DATA
# ============================================================

train_dataset = train_dataset.shuffle(

    buffer_size=len(train_paths),

    seed=SEED,

    reshuffle_each_iteration=True

)


# ============================================================
# LOAD IMAGES
# ============================================================

train_dataset = train_dataset.map(

    load_image,

    num_parallel_calls=tf.data.AUTOTUNE

)


val_dataset = val_dataset.map(

    load_image,

    num_parallel_calls=tf.data.AUTOTUNE

)


# ============================================================
# BATCH
# ============================================================

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)


# ============================================================
# PREFETCH
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    ),

], name="data_augmentation")


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\n========================================")
print("LOADING MOBILENETV2")
print("========================================")


base_model = MobileNetV2(

    input_shape=IMAGE_SIZE + (3,),

    include_top=False,

    weights="imagenet"

)


# Freeze initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(

    shape=IMAGE_SIZE + (3,)

)


x = data_augmentation(
    inputs
)


x = preprocess_input(
    x
)


x = base_model(

    x,

    training=False

)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.Dropout(
    0.35
)(
    x
)


outputs = layers.Dense(

    len(tomato_classes),

    activation="softmax"

)(
    x
)


model = keras.Model(

    inputs,

    outputs

)


# ============================================================
# COMPILE INITIAL MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(

        learning_rate=0.001

    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n========================================")
print("MODEL SUMMARY")
print("========================================")


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=3,

        restore_best_weights=True

    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=1,

        min_lr=1e-7

    ),

    keras.callbacks.ModelCheckpoint(

        filepath=str(

            MODEL_DIR /
            "best_tomato_model.keras"

        ),

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    )

]


# ============================================================
# INITIAL TRAINING
# ============================================================

print("\n========================================")
print("INITIAL TRAINING")
print("========================================\n")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n========================================")
print("FINE-TUNING MOBILENETV2")
print("========================================\n")


base_model.trainable = True


# Freeze all but last 40 layers

for layer in base_model.layers[:-40]:

    layer.trainable = False


# Keep BatchNormalization layers frozen
# This helps stabilize fine-tuning.

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COMPILE FINE-TUNING MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(

        learning_rate=0.00001

    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# FINE-TUNE
# ============================================================

fine_history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model_path = (

    MODEL_DIR /
    "best_tomato_model.keras"

)


if best_model_path.exists():

    print("\nLoading best model...")

    model = keras.models.load_model(
        str(best_model_path)
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n========================================")
print("FINAL VALIDATION")
print("========================================")


val_loss, val_accuracy = model.evaluate(

    val_dataset,

    verbose=1

)


print(
    f"\nFinal Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# CLASS-BY-CLASS EVALUATION
# ============================================================

print("\n========================================")
print("CLASS-BY-CLASS PERFORMANCE")
print("========================================")


true_labels = []

predicted_labels = []


for images, labels_batch in val_dataset:

    predictions = model.predict(

        images,

        verbose=0

    )


    predicted = np.argmax(

        predictions,

        axis=1

    )


    true_labels.extend(
        labels_batch.numpy()
    )

    predicted_labels.extend(
        predicted
    )


true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)


print(
    classification_report(

        true_labels,

        predicted_labels,

        labels=np.arange(
            len(tomato_classes)
        ),

        target_names=tomato_classes,

        digits=3,

        zero_division=0

    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    true_labels,

    predicted_labels,

    labels=np.arange(
        len(tomato_classes)
    )

)


plt.figure(
    figsize=(14, 12)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Tomato Disease Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    len(tomato_classes)
)


plt.xticks(

    tick_marks,

    tomato_classes,

    rotation=90

)


plt.yticks(

    tick_marks,

    tomato_classes

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.tight_layout()


plt.savefig(

    str(
        MODEL_DIR /
        "confusion_matrix.png"
    ),

    dpi=150,

    bbox_inches="tight"

)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    MODEL_DIR /
    "confusion_matrix.png"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model_path = (

    MODEL_DIR /
    "plant_disease_model.keras"

)


model.save(
    str(model_path)
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")


print(
    model_path
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = (

    MODEL_DIR /
    "class_names.json"

)


with open(

    class_names_path,

    "w"

) as file:

    json.dump(

        tomato_classes,

        file,

        indent=4

    )


print(
    "Class names saved:",
    class_names_path
)


# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

accuracy = (

    history.history["accuracy"]

    +

    fine_history.history["accuracy"]

)


val_accuracy_history = (

    history.history["val_accuracy"]

    +

    fine_history.history["val_accuracy"]

)


loss = (

    history.history["loss"]

    +

    fine_history.history["loss"]

)


val_loss_history = (

    history.history["val_loss"]

    +

    fine_history.history["val_loss"]

)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    accuracy,

    label="Training Accuracy"

)


plt.plot(

    val_accuracy_history,

    label="Validation Accuracy"

)


plt.title(
    "Tomato Disease Model Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid(
    True
)


plt.savefig(

    str(
        MODEL_DIR /
        "accuracy.png"
    ),

    dpi=150,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    loss,

    label="Training Loss"

)


plt.plot(

    val_loss_history,

    label="Validation Loss"

)


plt.title(
    "Tomato Disease Model Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.grid(
    True
)


plt.savefig(

    str(
        MODEL_DIR /
        "loss.png"
    ),

    dpi=150,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("       TRAINING COMPLETE!")
print("========================================")


print("\nGenerated files:")

print(
    "✓ model/plant_disease_model.keras"
)

print(
    "✓ model/best_tomato_model.keras"
)

print(
    "✓ model/class_names.json"
)

print(
    "✓ model/accuracy.png"
)

print(
    "✓ model/loss.png"
)

print(
    "✓ model/confusion_matrix.png"
)


print(
    "\nYour improved Tomato disease model is ready! 🌱"
)