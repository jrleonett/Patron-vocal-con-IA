# Identificación de patrones vocales tras cirugía de voz y laringe para reafirmar el género

![Licencia](https://img.shields.io/badge/Licencia-GNU%20GPL%20v3-blue)
![GitHub](https://img.shields.io/badge/Python-3.8%2B-green)
![GitHub](https://img.shields.io/badge/Estado-Activo-brightgreen)

La cirugía de voz y laringe es una intervención común en la reafirmación de género para personas transgénero y no binarias. Este procedimiento puede cambiar significativamente las características vocales, como la frecuencia fundamental (F0), la calidad vocal, y los formantes. Sin embargo, evaluar la efectividad de estos cambios desde una perspectiva objetiva y cuantificable sigue siendo un desafío. Este estudio explora cómo el cómputo forense y la ciencia forense pueden colaborar en la identificación de patrones vocales tras estas cirugías. 

En este estudio científico para el ambito del cómputo forense, me centraré en el uso de modelos de inteligencia artificial (IA) y matemáticos, además de librerías de Python, para analizar, identificar y verificar patrones vocales relacionados con la cirugía.

![image](https://drive.google.com/uc?export=view&id=1tTdu0kQOmxi_B8mW7e-rv01Fk4Q-d74e)

---

## Marco Teórico.

### Cómputo forense y la ciencia forense.
El cómputo forense o informática forense, emplea herramientas tecnológicas para investigar y analizar datos digitales, mientras que la ciencia forense aplica principios científicos para resolver preguntas legales. En el contexto del análisis vocal, estas disciplinas pueden integrarse para:

- Detectar cambios en las características acústicas.
- Identificar patrones únicos en la señal vocal.
- Proveer evidencia objetiva de las modificaciones vocales.

### Inteligencia artificial en análisis vocal.
Modelos de IA como redes neuronales profundas (DNN), redes neuronales convolucionales (CNN) y modelos basados en atención han demostrado ser efectivos en el análisis de señales vocales. Estos modelos pueden extraer características acústicas y clasificarlas con alta precisión.

---

## Metodología.

### 1. Recolección de datos.
- **Muestra:** Grabaciones de voz pre y post cirugía de individuos que han realizado la cirugía de voz y laringe para reafirmar su género.
- **Parámetros Acústicos a Analizar:**
  - Frecuencia fundamental (F0).
  - Duración de fonemas.
  - Formantes (F1, F2, F3).
  - Rasgos melódicos y prosódicos.

### 2. Modelos Matemáticos.

- **Transformada de Fourier:** Para analizar espectros de frecuencia.
- **Modelos Estadísticos:**
  - Análisis de varianza (ANOVA) para medir diferencias significativas entre muestras pre y post quirúrgicas.
  - Regresión logística para evaluar la probabilidad de identificación correcta de género.

### 3. Implementación con IA.

#### Modelosr Recomendados:

1. **Convolutional Neural Networks (CNN):** Para extracción y clasificación de características acústicas.
2. **Long Short-Term Memory (LSTM):** Para análisis de patrones temporales en la señal vocal.
3. **Transformer-based Models (e.g., Wav2Vec):** Para representación robusta de características vocales.

#### Algoritmo de entrenamiento:
1. **Preprocesamiento:** Normalización y eliminación de ruido.
2. **Extracción de Características:** Uso de Mel-Frequency Cepstral Coefficients (MFCCs).
3. **Entrenamiento y Validación:**
   - División de los datos en conjunto de entrenamiento (80%) y prueba (20%).
   - Métricas de evaluación: exactitud, precisión, recall, F1-score.

---

## Herramientas y librerías en Python.

### Librerías Principales.

1. **Librosa:** Para análisis y preprocesamiento de señales de audio.
   ```python
   import librosa
   import librosa.display
   
y, sr = librosa.load("audio_post_surgery.wav")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   ```

2. **PyTorch/TensorFlow:** Para el diseño y entrenamiento de modelos de IA.
   ```python
   import torch
   from torch import nn
   from torch.utils.data import DataLoader
   ```

3. **SciPy y NumPy:** Para análisis matemático y estadístico.
   ```python
   from scipy.fft import fft
   import numpy as np
   
signal_fft = fft(signal)
   ```

4. **PraatIO:** Para analizar y manipular características acústicas específicas.

### Pipeline en Python

```python
# Preprocesamiento de audio
import librosa
import librosa.display
import numpy as np

# Cargar audio
signal, sr = librosa.load("audio_post_surgery.wav")

# Extracción de MFCCs
mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

# Visualización
import matplotlib.pyplot as plt
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title("MFCCs")
plt.show()

# Entrenamiento de un modelo simple (Ejemplo con PyTorch)
import torch
from torch import nn
from torch.utils.data import DataLoader

class VocalModel(nn.Module):
    def __init__(self):
        super(VocalModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Clasificación binaria
        )

    def forward(self, x):
        return self.fc(x)

# Definir el modelo
model = VocalModel()
# Continuar con el entrenamiento...
```

---

## Resultados esperados.

Se espera que el análisis forense de las señales vocales proporcione una evaluación objetiva de los cambios acústicos. Los modelos de IA deben lograr una alta precisión (>90%) en la identificación de patrones asociados a la cirugía vocal.

---

## Referencias.
1. Smith, J. et al. (2020). "Voice Pattern Recognition Post-Gender Affirmation Surgery". Journal of Forensic Science.
2. Brown, A. et al. (2019). "Machine Learning Techniques for Voice Analysis". IEEE Transactions on Audio, Speech, and Language Processing.
3. Librosa Development Team. (2021). "Librosa: Audio and Music Analysis in Python". https://librosa.org
4. TensorFlow Developers. (2021). "TensorFlow: An Open Source Machine Learning Framework". https://www.tensorflow.org
5. Scikit-learn Developers. (2021). "Scikit-learn: Machine Learning in Python". https://scikit-learn.org

---
# Cómo citar este trabajo
Usa la siguiente entrada BibTeX si utilizas este trabajo en tu investigación:
```bash
@article{joséRLeonett,
  title={Patrones vocales con IA},
  author={José R. Leonett},
  year={2024}
}
```

**Licencia**
- Este proyecto está bajo la licencia GNU General Public License v3.0. Consulta el archivo LICENSE para más detalles.


