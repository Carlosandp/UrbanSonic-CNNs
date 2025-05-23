# 🌆 Urbanphonic Sound Classifier

Este proyecto utiliza aprendizaje automático para clasificar sonidos urbanos según la taxonomía de **Urbanphony**, que divide los sonidos del entorno urbano en categorías significativas.

---

## 🧠 Proyecto

Entrenamos un modelo para reconocer y clasificar grabaciones de audio en distintas categorías relevantes para el paisaje sonoro urbano, como transporte, voz humana, maquinaria, sonidos comunitarios y fauna.

---

## 🗂️ Taxonomía de Clases

Los sonidos se organizan según la taxonomía **Urbanphony**, la cual se divide en:

### 🔊 Urbanphonic Classes

| Código | Nombre de la Clase         | Descripción                                          | Ejemplo             |
|--------|----------------------------|------------------------------------------------------|----------------------|
| **TM** | Motorised Transport        | Transporte terrestre, aéreo o acuático               | Autobús              |
| **VM** | Voices or Music            | Voz humana, risa, canto o música                     | Personas hablando    |
| **EM** | Electro-mechanical         | Sonidos eléctricos, mecánicos o de construcción      | Ventilador           |
| **S**  | Social/Communal            | Sonidos relevantes para una comunidad                | Campanas             |
| **BI** | Biophony                   | Sonidos de animales silvestres o domesticados        | Aves cantando        |

---

## 🔧 Requisitos

- Python 3.8+
- TensorFlow / PyTorch
- librosa
- numpy
- matplotlib

Instalación de dependencias:

```bash
pip install -r requirements.txt

