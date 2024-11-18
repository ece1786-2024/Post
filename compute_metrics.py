import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from main import Datasets

GPT_VQA_LABELS = {
    Datasets.SMALL: [
        0, 1, 1, 0, 0, 0, 1, 0
    ],
    Datasets.WEB: [
        0, 1, 1, 0, 1, 0, 1, 1
    ]
}

GPT_VQA_IMAGE_ONLY_LABELS = {
    Datasets.SMALL: [
        0, 1, 0, 0, 0, 0, 0, 1
    ],
    Datasets.WEB: [
        0, 1, 0, 0, 0, 0, 1, 1
    ]
}

GPT_VQA_OUTPUTS = {
    Datasets.SMALL: [
        1, 0, 0, 0, 0, 0, 0, 1
    ],
    Datasets.WEB: [
        0, 1, 0, 0, 0, 0, 1, 1
    ]
}

# Calculate the confusion matrices for each dataset against the full tweet label and the image-only label

gs = gridspec.GridSpec(2,2)
figure = plt.figure(figsize=(10, 10))
figure.suptitle("Confusion matrices of the GPT-VQA model against various datasets and labels")

# SMALL dataset, compared against full tweet labels
ax1 = figure.add_subplot(gs[0,0])
ax1.set_title("SMALL dataset, FULL TWEET labels")
cf_matrix_1 = ConfusionMatrixDisplay(confusion_matrix(GPT_VQA_LABELS[Datasets.SMALL], GPT_VQA_OUTPUTS[Datasets.SMALL]))
cf_matrix_1.plot(ax=ax1)

# SMALL dataset, compared against image-only labels
ax2 = figure.add_subplot(gs[0,1])
ax2.set_title("SMALL dataset, IMAGE-ONLY labels")
cf_matrix_2 = ConfusionMatrixDisplay(confusion_matrix(GPT_VQA_IMAGE_ONLY_LABELS[Datasets.SMALL], GPT_VQA_OUTPUTS[Datasets.SMALL]))
cf_matrix_2.plot(ax=ax2)

# WEB dataset, compared against full tweet labels
ax3 = figure.add_subplot(gs[1,0])
ax3.set_title("WEB dataset, FULL TWEET labels")
cf_matrix_3 = ConfusionMatrixDisplay(confusion_matrix(GPT_VQA_LABELS[Datasets.WEB], GPT_VQA_OUTPUTS[Datasets.WEB]))
cf_matrix_3.plot(ax=ax3)

# WEB dataset, compared against image-only labels
ax4 = figure.add_subplot(gs[1,1])
ax4.set_title("WEB dataset, IMAGE-ONLY labels")
cf_matrix_4 = ConfusionMatrixDisplay(confusion_matrix(GPT_VQA_IMAGE_ONLY_LABELS[Datasets.WEB], GPT_VQA_OUTPUTS[Datasets.WEB]))
cf_matrix_4.plot(ax=ax4)

plt.show()

