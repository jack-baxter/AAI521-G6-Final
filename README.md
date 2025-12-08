# Solar Panel Defect Detection
# AAI-521 Final Project – Group 6 (Fall 2025)
# Jamshed Nabizada, Jack Baxter, Naima Botros

Hey! This is our final project for Dr. Albuyeh's class. We built a binary classifier to detect defects in solar panels using the ELPV dataset. Two models are compared: a custom CNN (baseline) and a fine-tuned ResNet18.

# Project Structure
adjustedg6project.ipynb      <- Main notebook with everything: data loading, EDA, training, eval, benchmarking
/content/drive/MyDrive/ELPV_SOLAR_DATA/  <- Expected dataset location (mounted in Colab)

# Dataset
- ELPV solar panel images (train/valid/test splits)
- Original YOLO-format labels → converted to binary (0 = clean, 1 = any defect)
- Heavily imbalanced (~98% defective)

# Dependencies
Run the first cell:
!pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn Pillow

# How to Run (Google Colab)
1. Open adjustedg6project.ipynb in Colab
2. Mount your Google Drive (cell already there)
3. Make sure your dataset is at /MyDrive/ELPV_SOLAR_DATA/
4. Run all cells top to bottom

The notebook will:
- Copy data locally for speed
- Build DataFrame + show class distribution
- Perform EDA (sample images, imbalance plot)
- Train both models (10 epochs, batch 32)
- Plot confusion matrices, ROC, Precision-Recall curves
- Benchmark inference speed and model size

# Key Results
ResNet18 (fine-tuned):
- PR-AUC ≈ 0.984
- Model size ≈ 43 MB
- Inference ≈ 2.6 ms/image (GPU, batch=1)

Baseline CNN:
- Similar PR-AUC but 3× larger (~130 MB)
- Slightly faster raw inference

→ ResNet18 wins for real-world deployment.

# Possible Improvements
- Stronger augmentation
- Use original bounding boxes (object detection
- Try EfficientNet / MobileNet for even smaller/faster models
- Stratified sampling or focal loss for the imbalance

— Jack, Jamshed & Naima
