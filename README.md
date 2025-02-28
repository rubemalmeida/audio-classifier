# Classificador de Sons Não Vocais

Sistema baseado em GPT/Whisper para identificação e transcrição de sons não vocais, como sirenes, quedas de objetos, colisões, motores de veículos, etc.

## Descrição

Este projeto é parte de um trabalho de pesquisa que visa desenvolver um sistema capaz de transformar sons não vocais em descrições textuais. Utilizamos um modelo baseado no Whisper da OpenAI, fine-tuned para identificar diferentes categorias de sons ambientais.

## Estrutura do Projeto

```
audio-classifier/
├── src/
│   ├── ml/            # Módulos de machine learning
│   ├── backend/        # API FastAPI
│   └── frontend/       # Interface web (Flask)
├── data/
│   ├── sounds/         # Dados de treinamento
│   └── trained_model/  # Modelos salvos
└── reports/            # Documentação e relatórios
```

## Funcionalidades

- Interface web para upload ou gravação de áudios
- API REST para processamento e classificação de áudios
- Suporte para arquivos .wav
- Processamento automático para frequência de 16kHz
- Limite de 30 segundos por áudio
- Modelo treinado para identificar diversas categorias de sons não vocais

## Requisitos

- Python 3.8+
- PyTorch
- Whisper
- FastAPI
- Flask
- Outras dependências especificadas em requirements.txt

## Instalação

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o backend: `python -m src.backend.main`
4. Execute o frontend: `python -m src.frontend.app`

## Uso

1. Acesse a interface web em http://localhost:5000
2. Faça upload de um arquivo de áudio ou grave um novo
3. Clique em "Classificar Som"
4. Visualize os resultados da classificação

## Treinamento do Modelo

Para treinar um novo modelo:

```bash
cd src/ml
python train.py
```

