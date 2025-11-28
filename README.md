# 🥑 End-to-End Deep Learning Crop Classification System

## 🚀 Live Demo
[Click here to open the Live Demo](https://crop-classifier-q3yf.onrender.com)

This project demonstrates a complete end-to-end Data Science pipeline, from model engineering and training to full-stack web application deployment. It classifies images of agricultural crops using Deep Learning.

---

## 📸 Deployment Visualization
The screenshot below shows a successful classification (e.g., Avocado plant) along with integrated botanical facts.

![Crop Classifier Sample Webapp](https://github.com/krish5143/Crop_classifier/blob/master/crop_classifier_sample_webapp.png?raw=true)

---

## ✨ Key Features
- **140-Class Multi-Classification:** Recognizes 140 distinct crop species.  
- **Transfer Learning:** Utilizes pre-trained ResNet-18 CNN on 35,000+ images.  
- **Full-Stack Deployment:** Containerized and served via Streamlit on Render.  
- **Botanical Insights:** Displays contextual facts about each predicted crop.  

---

## 🛠️ Technical Stack

| Component      | Technology        | Purpose                                         |
|----------------|-----------------|------------------------------------------------|
| Language       | Python 3.9+      | Model development & scripting                  |
| Deep Learning  | PyTorch, TorchVision | CNN model building & training              |
| Model          | ResNet-18        | Transfer Learning architecture                 |
| Web App        | Streamlit        | Frontend interface for image submission & prediction |
| Deployment     | Render           | Hosting & serving the application             |
| Data Handling  | NumPy, Pandas, PIL | Image loading, transformations, and data manipulation |

---

## 📂 Project Architecture
The pipeline leverages Transfer Learning, retraining the final layer of ResNet-18 to output predictions for 140 crop classes.

---

## ⚙️ Model Training & Metrics
- **Dataset Size:** 35,000+ images across 140 classes  
- **Training:** 5 epochs, Adam optimizer, CrossEntropyLoss  
- **Serialization:** Model weights and class mapping saved as `crop_classifier_model.pkl` using `joblib`
