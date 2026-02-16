import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

# ==========================================
# 1. Подготовка данных (Dataset)
# ==========================================

class PestAudioDataset(Dataset):
    def __init__(self, num_samples=200, sample_rate=16000, duration=2, use_spectrogram=False):
        """
        Инициализация датасета.
        """
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.duration = duration
        self.seq_len = sample_rate * duration
        self.use_spectrogram = use_spectrogram
        
        # Генерация синтетических данных (на репозиторий датасет не загружал) (0: нет вредителя, 1: есть вредитель)
        # Вредитель имитируется синусоидой определенной частоты (например, 500 Гц) на фоне шума
        self.labels = np.random.randint(0, 2, num_samples)
        
    def __len__(self):
        return self.num_samples

    def _generate_signal(self, label):
        t = np.linspace(0, self.duration, self.seq_len)
        # Фоновый шум
        noise = np.random.normal(0, 0.1, self.seq_len).astype(np.float32)
        
        if label == 1:
            # Сигнал вредителя: амплитудно-модулированная синусоида (имитация жужжания)
            freq = 500 + np.random.randint(-20, 20) # Небольшой разброс частоты
            signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            # Добавляем случайные паузы в сигнале (вредитель не жужжит непрерывно)
            mask = np.random.choice([0, 1], size=self.seq_len, p=[0.3, 0.7])
            signal = signal * mask
            return noise + signal
        else:
            # Только шум или природные звуки (белый шум)
            return noise

    def __getitem__(self, idx):
        label = self.labels[idx]
        waveform = self._generate_signal(label)
        waveform_tensor = torch.tensor(waveform).unsqueeze(0) # [1, Seq_Len]

        # Опционально: MFCC (мел-частотные кепстральные коэффициенты) для Transformer
        if self.use_spectrogram:
            mfcc_transform = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13,
                melkwargs={'n_fft': 400, 'hop_length': 50, 'n_mels': 64}
            )
            # Преобразуем в [Features, Time] -> [Time, Features] для Transformer
            features = mfcc_transform(waveform_tensor).squeeze(0).transpose(0, 1)
            return features, label
        
        # Для 1D CNN возвращаем сырой сигнал [1, Seq_Len]
        return waveform_tensor, label

# ==========================================
# 2. Архитектуры моделей
# ==========================================

# --- Модель 1: 1D CNN ---
class AudioCNN1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(AudioCNN1D, self).__init__()
        # Вход: [Batch, 1, 32000] (при 16kHz и 2 сек)
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=64, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, 32, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Глобальный пулинг
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Модель 2: Transformer ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=13, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        
        # Проекция входных данных (MFCC) в размерность модели
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Позиционное кодирование (упрощенное, обучаемое)
        # В реальной задаче лучше использовать Sinusoidal или Rotary PE
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model)) # Макс длина 1000
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        x = self.input_projection(x) # [Batch, Time, d_model]
        
        # Добавляем позиционное кодирование (обрезаем до длины последовательности)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer блок
        x = self.transformer_encoder(x)
        
        # Классификация по первому токену или среднему пулингу
        # Здесь используем среднее по временной оси
        x = x.mean(dim=1)
        
        return self.classifier(x)

# ==========================================
# 3. Функции обучения и валидации
# ==========================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(dataloader), acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return running_loss / len(dataloader), acc, f1

# ==========================================
# 4. Основной цикл с Кросс-Валидацией
# ==========================================

def run_experiment(model_type='cnn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Experiment: {model_type.upper()} on {device} ---")
    
    # Создаем общий датасет (не загружаем в память сразу, если он большой)
    # Для CNN используем сырой сигнал (use_spectrogram=False)
    # Для Transformer используем MFCC (use_spectrogram=True)
    use_spec = True if model_type == 'transformer' else False
    full_dataset = PestAudioDataset(num_samples=200, use_spectrogram=use_spec)
    
    # Настройка K-Fold
    k_folds = 5
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Списки для метрик по фолдам
    fold_results = []
    
    # Получаем метки для StratifiedKFold
    labels = full_dataset.labels
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"FOLD {fold+1}/{k_folds}")
        
        # Семплеры и лоадеры
        train_subsampler = Subset(full_dataset, train_ids)
        test_subsampler = Subset(full_dataset, test_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=16, shuffle=False)
        
        # Инициализация модели
        if model_type == 'cnn':
            model = AudioCNN1D().to(device)
        else:
            # Для Transformer нужно знать размерность входа (MFCC features = 13)
            model = TimeSeriesTransformer(input_dim=13).to(device)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Цикл обучения
        epochs = 10
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            # Можно добавить валидацию здесь, но для простоты сделаем в конце эпохи или в конце фолда
        
        # Оценка фолда
        val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion, device)
        print(f"Fold {fold+1} Accuracy: {val_acc:.4f}, F1-Score: {val_f1:.4f}")
        fold_results.append(val_acc)
        
    print(f"\nAverage Accuracy for {model_type.upper()}: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

# ==========================================
# 5. Запуск
# ==========================================

if __name__ == "__main__":
    # Эксперимент 1: 1D CNN на сырых данных
    run_experiment(model_type='cnn')
    
    # Эксперимент 2: Transformer на спектральных признаках
    run_experiment(model_type='transformer')
