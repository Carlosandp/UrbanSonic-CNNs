# 🌆 UrbanSonic-CNNs

This project applies machine learning to classify urban sounds according to the **Urbanphony** taxonomy, which groups environmental sounds into representative categories of the city soundscape.

---

## 🧠 Project

A classification model is trained to recognize audio recordings belonging to classes such as motorized transport, human voice, machinery, social events, and animal sounds, among others. The model is built on efficient architectures like EfficientNet and leverages modern audio and image preprocessing techniques.

---

## 🗂️ Class Taxonomy

The sounds are organized according to the **Urbanphony** taxonomy, structured into five main classes:

| Code  | Class Name              | Description                                           | Example              |
|-------|--------------------------|-------------------------------------------------------|----------------------|
| **TM** | Motorised Transport      | Land, air, or water transport sounds                  | Bus                  |
| **VM** | Voices or Music          | Human voice, laughter, singing, or music             | People talking       |
| **EM** | Electro-mechanical       | Electrical, mechanical, or construction sounds        | Fan                  |
| **S**  | Social/Communal          | Sounds relevant to a community                        | Bells                |
| **BI** | Biophony                 | Sounds from wild or domesticated animals              | Birds singing        |

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
