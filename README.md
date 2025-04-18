# Detector-Yolov5

# 🧠 Gender Recognition (male and female aspects) com YOLOv5

Este projeto foi desenvolvido como uma introdução prática ao universo da **Visão Computacional** utilizando **Redes Neurais Convolucionais (CNNs)** com a arquitetura **YOLOv5**. O objetivo foi realizar o treinamento de uma rede para **detecção e reconhecimento de gênero** a partir de imagens, utilizando **PyTorch**, **Ultralytics YOLOv5** e um **dataset do Roboflow**.

## 📷 Dataset

O dataset utilizado está disponível na plataforma Roboflow:

- 🔗 [Gender Recognition Dataset - Roboflow](https://universe.roboflow.com/rehabcv/gender-recognition-ukake/dataset/4)

Este dataset contém imagens anotadas para identificação de **masculino** e **feminino**, sendo utilizado diretamente no formato compatível com o YOLOv5.

---

## ⚙️ Tecnologias Utilizadas

- 🧠 YOLOv5 - [Ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- 🖥️ GPU NVIDIA RTX 2060
- 🔥 PyTorch
- 📷 OpenCV
- 📦 Roboflow
- 💻 Python com foco em Orientação a Objetos (OO)

---

## 🚀 Treinamento

O treinamento foi realizado utilizando a seguinte configuração:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

---

## 📊 Resultados

Durante o treinamento, o modelo apresentou os resultados ao longo das 350 épocas. A seguir, estão destacados os principais indicadores:

- [📊 Resultados do Modelo](./resultados_metricas.md)


---

## 🧠 Considerações Finais

Este projeto foi uma introdução prática ao universo da **Visão Computacional** com foco em **detecção de objetos usando YOLOv5**. A experiência permitiu:

- Consolidar conhecimentos em **Redes Neurais Convolucionais (CNNs)**;
- Aplicar conceitos de **Orientação a Objetos (OO)** no desenvolvimento de scripts;
- Utilizar frameworks modernos como **PyTorch**, ferramentas como **Roboflow** e, além disso, **OpenCV** para visão computacional.
- Experiência em visão computacional utilizando GPUs com processamento local.

---
