# 🌆 Urbanphonic Sound Classifier

Este proyecto aplica aprendizaje automático para clasificar sonidos urbanos siguiendo la taxonomía de **Urbanphony**, que agrupa los sonidos ambientales en categorías representativas del paisaje sonoro de la ciudad.

---

## 🧠 Proyecto

Se entrena un modelo de clasificación para reconocer grabaciones de audio pertenecientes a clases como transporte motorizado, voz humana, maquinaria, eventos sociales y sonidos de animales, entre otros. El modelo se construye sobre arquitecturas eficientes como EfficientNet y utiliza técnicas modernas de preprocesamiento de audio e imágenes.

---

## 🗂️ Taxonomía de Clases

Los sonidos se organizan según la taxonomía **Urbanphony**, estructurada en cinco clases principales:

| Código | Nombre de la Clase         | Descripción                                          | Ejemplo             |
|--------|----------------------------|------------------------------------------------------|----------------------|
| **TM** | Motorised Transport        | Transporte terrestre, aéreo o acuático               | Autobús              |
| **VM** | Voices or Music            | Voz humana, risa, canto o música                     | Personas hablando    |
| **EM** | Electro-mechanical         | Sonidos eléctricos, mecánicos o de construcción      | Ventilador           |
| **S**  | Social/Communal            | Sonidos relevantes para una comunidad                | Campanas             |
| **BI** | Biophony                   | Sonidos de animales silvestres o domesticados        | Aves cantando        |

---

## ⚙️ Requisitos

Este proyecto fue desarrollado y probado en el siguiente entorno:

- **Sistema operativo:** Windows 10 / 11
- **Python:** 3.9.7
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Librerías principales:**

```bash
Librería           Versión
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
