# 🗑️ Garbage Classification using Deep Learning

Proyek klasifikasi gambar sampah menggunakan Convolutional Neural Network (CNN) dengan Transfer Learning untuk submission akhir Dicoding Machine Learning Course.

## 👤 Author Information
- **Nama:** Ilham Sholahuddin  
- **Email:** ilhamsholahuddin161@gmail.com  
- **ID Dicoding:** ilham_sholahuddin16

---

## 📋 Deskripsi Proyek

Project ini mengimplementasikan model deep learning untuk mengklasifikasikan gambar sampah ke dalam 10 kategori berbeda. Model dibangun menggunakan arsitektur Transfer Learning dengan **MobileNetV2** sebagai base model, dilengkapi dengan layer tambahan untuk fine-tuning pada dataset garbage classification.

### 🎯 Tujuan
- Mengklasifikasikan gambar sampah secara otomatis
- Mencapai akurasi minimal 95% pada validation set
- Mengimplementasikan anti-overfitting techniques
- Menyediakan model dalam berbagai format (SavedModel, TFLite, TensorFlow.js)

---

## 📊 Dataset

**Source:** [Garbage Classification v2 - Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

### Kategori Sampah (10 Kelas):
1. Battery
2. Biological
3. Brown-glass
4. Cardboard
5. Clothes
6. Green-glass
7. Metal
8. Paper
9. Plastic
10. Shoes

### Dataset Statistics:
- **Total Images:** ~2500+ images
- **Image Size:** 224x224 pixels (setelah preprocessing)
- **Format:** RGB images
- **Split Ratio:** 
  - Training: 70%
  - Validation: 20%
  - Testing: 10%

---

## 🛠️ Technologies & Libraries

### Core Libraries:
```python
- Python 3.x
- TensorFlow 2.19.0
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV (cv2)
- PIL (Pillow)
- scikit-image
```

### Environment:
- Google Colab
- GPU: T4 (CUDA support)
- Kaggle API for dataset download

---

## 🏗️ Model Architecture

### Transfer Learning: MobileNetV2

Model menggunakan **MobileNetV2** pretrained pada ImageNet sebagai feature extractor, dengan layer tambahan:

```
Input (224, 224, 3)
    ↓
MobileNetV2 (pretrained, frozen)
    ↓
Conv2D (64 filters, 3x3, ReLU)
    ↓
MaxPooling2D
    ↓
Conv2D (128 filters, 3x3, ReLU)
    ↓
MaxPooling2D
    ↓
GlobalAveragePooling2D
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (10, Softmax)
```

### Training Configuration:

#### Initial Training:
- **Optimizer:** Adam (learning rate: 1e-3)
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Size:** 32
- **Epochs:** 10-20

#### Fine-Tuning:
- **Unfreezing:** Last 50 layers of MobileNetV2
- **Optimizer:** Adam (learning rate: 1e-5)
- **Additional Epochs:** 10

---

## 🔄 Data Preprocessing & Augmentation

### Preprocessing Steps:
1. **Resize:** Semua gambar di-resize ke 224x224 pixels
2. **Normalization:** Pixel values dinormalisasi ke range [0, 1]
3. **Data Splitting:** Train/Val/Test split dengan stratified sampling

### Data Augmentation (Training Set):
```python
- Random Rotation: ±20 degrees
- Random Zoom: 0.2
- Random Horizontal Flip
- Random Brightness: 0.2
- Random Contrast: 0.2
- Random Width/Height Shift: 0.1
```

### Anti-Overfitting Techniques:
- ✅ Dropout layers (0.3)
- ✅ L2 Regularization
- ✅ Data Augmentation
- ✅ Early Stopping
- ✅ Proper Train/Val/Test split
- ✅ Batch Normalization

---

## 📈 Model Performance

### Metrics:
- **Training Accuracy:** ~98%
- **Validation Accuracy:** >95%
- **Test Accuracy:** >95%

### Evaluation Metrics:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Training & Validation Loss Curves

---

## 📦 Model Export Formats

Model tersedia dalam 3 format:

### 1. **SavedModel** (TensorFlow)
```python
model.save('garbage_classifier_model')
```
- Format: `.pb` + variables
- Use case: Production deployment, TensorFlow Serving

### 2. **TFLite** (TensorFlow Lite)
```python
converter = tf.lite.TFLiteConverter.from_saved_model('garbage_classifier_model')
tflite_model = converter.convert()
```
- Format: `.tflite`
- Use case: Mobile & IoT devices (Android, iOS, Edge devices)

### 3. **TensorFlow.js**
```bash
tensorflowjs_converter \
    --input_format=keras \
    garbage_classifier_model.h5 \
    tfjs_model/
```
- Format: `.json` + weight shards
- Use case: Web browsers, Node.js applications

---

## 📊 Visualization Examples

### Training History
- Accuracy & Loss curves untuk training dan validation
- Monitoring overfitting/underfitting

### Confusion Matrix
- Heatmap untuk melihat performa per kelas
- Identifikasi kelas yang sering misclassified

### Sample Predictions
- Visualisasi prediksi model pada test set
- Confidence scores untuk setiap prediksi

---

## 🎓 Key Learnings

1. **Transfer Learning:** Memanfaatkan pretrained MobileNetV2 untuk feature extraction
2. **Fine-Tuning:** Unfreezing layer terakhir untuk adaptasi dataset spesifik
3. **Data Augmentation:** Meningkatkan generalisasi model
4. **Anti-Overfitting:** Kombinasi dropout, regularization, dan proper validation
5. **Model Deployment:** Export ke multiple formats untuk berbagai platform

---

## 🔮 Future Improvements

- [ ] Implementasi ensemble methods (voting classifier)
- [ ] Eksplorasi arsitektur lain (EfficientNet, ResNet)
- [ ] Dataset augmentation dengan synthetic data
- [ ] Real-time classification menggunakan webcam
- [ ] Deploy ke cloud (AWS, GCP, Azure)
- [ ] Mobile app development (Android/iOS)
- [ ] Web application dengan TensorFlow.js

---

## 📄 License

This project is created for educational purposes as part of Dicoding Deep Learning Course submission.

---

## 🙏 Acknowledgments

- **Dataset:** Sumn2u - Garbage Classification v2 (Kaggle)
- **Dicoding Indonesia:** Deep Learning Path
- **TensorFlow Team:** Framework & Documentation
- **MobileNetV2 Paper:** Efficient ConvNets for Mobile Vision Applications

---
