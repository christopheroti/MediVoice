{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "aacf82c0-d1e7-4a4d-a6ef-6d1c5bff296c",
   "metadata": {},
   "source": [
    "\"\"\"\n",
    "MediVoice AI - Triage Model Training\n",
    "TF-IDF + Logistic Regression\n",
    "\n",
    "Input:\n",
    "    medivoice_triage_dataset_augmented.csv\n",
    "\n",
    "Output:\n",
    "    models/medivoice_triage_model.joblib\n",
    "\n",
    "Metrics:\n",
    "    Accuracy\n",
    "    Precision\n",
    "    Recall\n",
    "    F1-score\n",
    "    Confusion matrix\n",
    "    Classification report\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3f5a9a00-2adf-43bb-ad4d-225a897c6812",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import joblib\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    precision_recall_fscore_support,\n",
    "    classification_report,\n",
    "    confusion_matrix,\n",
    "    ConfusionMatrixDisplay,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41e012ef-f49f-407a-b400-1a5f91e8e7cd",
   "metadata": {},
   "source": [
    " 1. CONFIGURATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "09ab9558-066e-45c7-b9a2-48bac0b08aa3",
   "metadata": {},
   "outputs": [],
   "source": [
    "DATASET_PATH = \"data/medivoice_triage_dataset_augmented.csv\"\n",
    "MODEL_DIR = \"models\"\n",
    "MODEL_PATH = os.path.join(MODEL_DIR, \"medivoice_triage_model.joblib\")\n",
    "CM_PATH = os.path.join(MODEL_DIR, \"confusion_matrix.png\")\n",
    "\n",
    "RANDOM_STATE = 42"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59ae41b5-85c7-44a0-88f7-32930e40404d",
   "metadata": {},
   "source": [
    "2. LOAD DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "bd87190c-bbe2-4a74-aa7e-08e916cc7b26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "======================================================================\n",
      "MEDIVOICE AI - TRIAGE MODEL TRAINING\n",
      "======================================================================\n"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "\nDataset not found:\n./data/medivoice_triage_dataset_augmented.csv\n\nPlace medivoice_triage_dataset_augmented.csv inside your project's data/ folder.",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mFileNotFoundError\u001b[39m                         Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[12]\u001b[39m\u001b[32m, line 6\u001b[39m\n\u001b[32m      2\u001b[39m print(\u001b[33m\"MEDIVOICE AI - TRIAGE MODEL TRAINING\"\u001b[39m)\n\u001b[32m      3\u001b[39m print(\u001b[33m\"=\"\u001b[39m * \u001b[32m70\u001b[39m)\n\u001b[32m      4\u001b[39m \n\u001b[32m      5\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28;01mnot\u001b[39;00m os.path.exists(DATASET_PATH):\n\u001b[32m----> \u001b[39m\u001b[32m6\u001b[39m     raise FileNotFoundError(\n\u001b[32m      7\u001b[39m         \u001b[33mf\"\\nDataset not found:\\n{DATASET_PATH}\\n\\n\"\u001b[39m\n\u001b[32m      8\u001b[39m         \u001b[33m\"Place medivoice_triage_dataset_augmented.csv inside your \"\u001b[39m\n\u001b[32m      9\u001b[39m         \u001b[33m\"project's data/ folder.\"\u001b[39m\n",
      "\u001b[31mFileNotFoundError\u001b[39m: \nDataset not found:\n./data/medivoice_triage_dataset_augmented.csv\n\nPlace medivoice_triage_dataset_augmented.csv inside your project's data/ folder."
     ]
    }
   ],
   "source": [
    "print(\"=\" * 70)\n",
    "print(\"MEDIVOICE AI - TRIAGE MODEL TRAINING\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "if not os.path.exists(DATASET_PATH):\n",
    "    raise FileNotFoundError(\n",
    "        f\"\\nDataset not found:\\n{DATASET_PATH}\\n\\n\"\n",
    "        \"Place medivoice_triage_dataset_augmented.csv inside your \"\n",
    "        \"project's data/ folder.\"\n",
    "    )\n",
    "\n",
    "df = pd.read_csv(DATASET_PATH)\n",
    "\n",
    "required_columns = [\"text\", \"chief_complaint\", \"triage_level\"]\n",
    "\n",
    "missing = [c for c in required_columns if c not in df.columns]\n",
    "\n",
    "if missing:\n",
    "    raise ValueError(f\"Missing required columns: {missing}\")\n",
    "\n",
    "df = df.dropna(subset=[\"text\", \"chief_complaint\", \"triage_level\"])\n",
    "\n",
    "# Combine the patient's description and chief complaint.\n",
    "df[\"input_text\"] = (\n",
    "    df[\"chief_complaint\"].astype(str)\n",
    "    + \" \"\n",
    "    + df[\"text\"].astype(str)\n",
    ")\n",
    "\n",
    "# Normalize labels.\n",
    "df[\"triage_level\"] = df[\"triage_level\"].str.lower().str.strip()\n",
    "\n",
    "allowed_labels = {\"low\", \"moderate\", \"high\"}\n",
    "\n",
    "invalid_labels = set(df[\"triage_level\"].unique()) - allowed_labels\n",
    "\n",
    "if invalid_labels:\n",
    "    raise ValueError(f\"Unexpected triage labels: {invalid_labels}\")\n",
    "\n",
    "print(f\"\\nTotal samples: {len(df)}\")\n",
    "\n",
    "print(\"\\nClass distribution:\")\n",
    "print(df[\"triage_level\"].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "888186bc-be1c-4066-a2b7-234866679c72",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5229cda-b630-4862-8a2f-e96a527e0e50",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c063b321-4b52-4f41-9dbc-ce3ba2119718",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a32a9551-9e2f-4bb0-91ab-0b5af1217c2e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (MediVoice ML)",
   "language": "python",
   "name": "medivoice-ml"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
