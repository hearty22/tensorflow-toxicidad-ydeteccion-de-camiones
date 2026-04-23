# TP Seminario de actualizacion: Implementacion de TensorFlow.js

## Descripcion del Proyecto
Este repositorio contiene la implementacion practica para el Seminario de Actualizacion. El proyecto demuestra la integracion de modelos de Machine Learning ejecutandose directamente en el entorno del navegador web mediante TensorFlow.js, eliminando la necesidad de procesamiento en un servidor backend.

El desarrollo abarca dos dominios principales de la Inteligencia Artificial:
1. Procesamiento de Lenguaje Natural: Un modulo de clasificacion de texto que identifica niveles de toxicidad, insultos y amenazas.
2. Vision Computacional : Un modulo de deteccion de objetos (Object Detection) enfocado especificamente en la identificacion de camiones ("truck") en imagenes estaticas.

## Arquitectura y Tecnologias
* Framework UI: React 
* Lenguaje: TypeScript (Tipado estricto para las respuestas de los tensores)
* Build Tool: Vite
* Ecosistema Machine Learning:
  * @tensorflow/tfjs (Motor de inferencia y aceleracion WebGL)
  * @tensorflow-models/toxicity (Modelo clasificador de texto)
  * @tensorflow-models/coco-ssd (Modelo Single Shot MultiBox Detector para imagenes)

## Requisitos Previos
* Node.js (v18 o superior)
* NPM

## Instalacion y Despliegue Local

1. Abrir una terminal en el directorio raiz del proyecto.
2. Instalar las dependencias. 
   Nota tecnica: Se requiere el flag legacy-peer-deps debido a restricciones de versiones en las dependencias internas de los modelos preentrenados de TensorFlow.
   
   ```bash
   npm install --legacy-peer-deps
   ```

Compilar y levantar el entorno de desarrollo:
```bash
npm run dev
```

Acceder mediante el navegador a la direccion local proporcionada por Vite
## Decisiones de Diseño
Carga Lazy (Singleton): Los modelos neuronales se cargan en la memoria RAM una sola vez durante el ciclo de vida del componente inicial para optimizar el rendimiento y evitar cuellos de botella en la red.
Ajuste de Umbral (Threshold): El limite de confianza del modelo de toxicidad fue calibrado a un 60% (0.6) para mitigar el sesgo de entrenamiento original (ingles) y mejorar la tasa de aciertos con modismos locales en español.
UI/UX Asincrona: Se implemento bloqueo de estado interactivo durante la resolucion de las promesas de inferencia para prevenir el apilamiento del hilo principal de ejecucion.