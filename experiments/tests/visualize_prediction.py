import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Pfad zum Projekt einfügen (damit deep_vision & multi_spectral gefunden werden)
sys.path.insert(0, "/tf")

from deep_vision.spectrum_unets import derterministic_spectrum_unetb8
from multi_spectral.dataset.dataloader import load_annotated_msis
from multi_spectral.dataset.patches import msi2patches
from deep_vision.data import to_tf_dataset

# ==== KONFIGURATION ====
PATCH_SIZE = 32
CHECKPOINT_PATH = "../experiments/tests/results-1/models/derterministic_spectrum_unetb8_msiv6.2_cutborders_mwp_local_end_80_8_32"
ANNOTATION_PATH = "/tf/datasets/msiv6_recordings/24_04_15_4meter/annotations"
MSI_PATH = "/tf/datasets/msiv6_recordings/24_04_15_4meter/24_04_15_14_53_44"

# ==== MODELL LADEN ====
model = derterministic_spectrum_unetb8(patch_size=PATCH_SIZE, mwp=True)
model.load_weights(CHECKPOINT_PATH).expect_partial()

# ==== DATEN LADEN ====
msi_dict = load_annotated_msis(recording_path=MSI_PATH, annotation_path=ANNOTATION_PATH, merge_wood_and_paper=True)
patched = msi2patches(msi_dict, patch_size=PATCH_SIZE, stride=PATCH_SIZE, remove_unlabeled=False)
dataset = to_tf_dataset(patched, batch_size=1, shuffle=False)

# ==== EIN BILD VORHERSAGEN & ANZEIGEN ====
for image, label in dataset.take(1):
    prediction = model.predict(image)
    predicted_mask = tf.argmax(prediction[0], axis=-1)
    true_mask = tf.squeeze(label[0])

    # RGB-Kanäle extrahieren (z. B. Kanäle 3, 2, 1 → Grün, Rot, Blau)
    rgb_image = image[0][..., [3, 2, 1]].numpy()
    rgb_image = rgb_image / np.max(rgb_image)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(rgb_image)
    axs[0].set_title("RGB")
    axs[0].axis("off")

    axs[1].imshow(true_mask.numpy(), cmap="jet")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(predicted_mask.numpy(), cmap="jet")
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
