import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Adiciona o diretório do YOLOv5 ao path para importações
sys.path.append(r"D:\Teste Yolo\yolov5")

# Importa as classes e funções necessárias
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

# Caminho para o modelo treinado
MODEL_PATH = r"D:\Teste Yolo\yolov5\runs\train\exp2\weights\best.pt"

# Definição das classes (ajuste conforme seu dataset)
CLASSES = ['female', 'male']  # Definindo os nomes das classes

# Configuração do dispositivo (GPU ou CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Carregar modelo YOLO
model = DetectMultiBackend(MODEL_PATH, device=device)
model.warmup(imgsz=(1, 3, 640, 640))  # Preparar modelo

# Inicializa a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera!")
    exit()

# Abre arquivo para salvar detecções
output_file = open("resultados.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem!")
        break

    # Pré-processamento da imagem
    img = torch.from_numpy(frame).to(device)
    img = img.permute(2, 0, 1).float() / 255.0  # Normaliza
    img = img.unsqueeze(0)  # Adiciona batch dimension

    # Inferência
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.5)[0]  # Limiar de confiança 50%

    # Processa e salva as detecções
    detections = []
    for det in pred:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        cls = int(cls)  # Converte a classe para inteiro
        class_name = CLASSES[cls]  # Nome da classe detectada

        # Cria a string de saída
        detection_str = f"{class_name} ({conf:.2f})"
        detections.append(detection_str)

        # Desenha a caixa de detecção
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, detection_str, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibe no terminal
    if detections:
        print("\n".join(detections))

    # Salva no arquivo TXT
    output_file.write("\n".join(detections) + "\n")

    # Mostrar o vídeo com detecções
    cv2.imshow("YOLOv5 Detection", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fecha o arquivo e libera os recursos
output_file.close()
cap.release()
cv2.destroyAllWindows()
