# 🤖🎧 UrbanSonic-CNNs

This project applies deep learning techniques to classify urban soundscape components, encompassing both anthropophony (e.g., traffic, human voices, and socio-communal sounds) and natural sounds (e.g., biophony). Specifically, a convolutional neural network (CNN) is used to classify urban sounds based on their representation as Mel spectrograms generated from audio recordings collected in urban areas.

---

## 🧠 Project

The CNN is trained to recognize diverse urban soundscape components across different datasets, allowing its classification performance to be evaluated. The model is based on the pre-trained EfficientNet CNN architecture and leverages modern audio and image preprocessing techniques.

---

## 🗂️ Class Taxonomy

The sounds are organized according to the urban soundscape taxonomy, structured into five main classes:

| Code  | Class Name              | Description                                           | Example              |
|-------|--------------------------|-------------------------------------------------------|----------------------|
| **MT** | Motorised Transport      | Land, air, or water transport sounds                  | Bus                  |
| **VM** | Voices or Music          | Human voice, laughter, singing, or music             | People talking       |
| **EM** | Electro-mechanical       | Electrical, mechanical, or construction sounds        | Fan                  |
| **S**  | Social-communal          | Sounds relevant to a community                        | Bells                |
| **BI** | Biophony                 | Sounds from wild or domesticated animals              | Birds songs       |

---

## ⚙️ Requirements

This project was developed and tested in the following environment:

- **Operating System:** Windows 10 / 11  
- **Python Version:** 3.9.7  
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU  
- **Main Libraries:**

```bash
Library            Version
------------------ ---------
numpy              1.23.5
pandas             2.2.3
matplotlib         3.9.4
seaborn            0.13.2
scikit-learn       1.6.1
tensorflow         2.10.0
keras (tf.keras)   2.10.0
librosa            0.10.1
soundfile          0.12.1
joblib             1.4.2
IPython            8.12.3
```
**This repository is part of Urbanphony efforts. A soundscape project boosted from the heart of Mantra, Science, Art and Technoogy**
