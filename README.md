# Detector-Yolov5

# 🧠 Gender Recognition com YOLOv5

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
- 📦 Roboflow (para obtenção e formatação do dataset)
- 💻 Python com foco em Orientação a Objetos (OO)

---

## 🚀 Treinamento

O treinamento foi realizado utilizando a seguinte configuração:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

# O treinamento foi iniciado com o seguinte comando:
```bash 
python train.py --img 640 --batch 16 --epochs 350 --data ./data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --name gender_yolo_model
```
---

## 📊 Resultados

Durante o treinamento, o modelo apresentou resultados satisfatórios ao longo das 350 épocas. A seguir, estão destacados os principais indicadores:

### Métricas de Desempenho

- **Precisão (Precision)**
- **Revocação (Recall)**
- **mAP@0.5 (Mean Average Precision)**
- **Curvas de Loss (perda) e outros indicadores**

As imagens das métricas geradas estão disponíveis na pasta `runs/train/gender_yolo_model_results/` e podem ser visualizadas abaixo:

<p align="center">
  <img src="/runs/train/exp2/confusion_matrix.png" width="500" alt="Matriz de Confusão">
</p>

<p align="center">
  <img src="/runs/train/exp2/P_curve.png" width="500" alt="Curva de Precisão">
</p>

<p align="center">
  <img src="/runs/train/exp2/R_curve.png" width="500" alt="Curva de Revocação">
</p>

<p align="center">
  <img src="/runs/train/exp2/PR_curve.png" width="500" alt="Curva de Precisão-Revocação">
</p>

<p align="center">
  <img src="/runs/train/exp2/F1_curve.png" width="500" alt="Curva F1">
</p>

<p align="center">
  <img src="/runs/train/exp2/results.png" width="500" alt="Resultados do Treinamento">
</p>

- [📊 Resultados do Modelo](./resultados_metricas.md)


---

## 🧠 Considerações Finais

Este projeto foi uma introdução prática ao universo da **Visão Computacional** com foco em **detecção de objetos usando YOLOv5**. A experiência permitiu:

- Consolidar conhecimentos em **Redes Neurais Convolucionais (CNNs)**;
- Aplicar conceitos de **Orientação a Objetos (OO)** no desenvolvimento de scripts;
- Utilizar frameworks modernos como **PyTorch** e ferramentas como o **Roboflow** para formatação de datasets.

O treinamento realizado com uma **GPU NVIDIA RTX 2060** evidenciou a eficiência do hardware aliado a técnicas avançadas de deep learning, possibilitando uma boa generalização do modelo.

---


