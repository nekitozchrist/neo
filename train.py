# train.py

ma = False
if __name__ == "__main__":
	ma = True

import time
tt = lambda: time.time()
tot = tt()

from colorama import Back, Fore, Style, init
init()

def ts(t, n):
	"""
	Формирует строку с временем выполнения операции.

	Args:
		t (float): начальное время (в секундах)
		n (str): название операции

	Returns:
		str: строка с временем выполнения и цветовой маркировкой
	"""
	v = float(format(tt() - t, ".2f"))
	cp = ""
	if v <= 0.3: cp = f"{Fore.GREEN}{v}"
	if v > 0.3 and v <= 1.2: cp = f"{Fore.YELLOW}{v}"
	if v > 1.2: cp = f"{Fore.RED}{v}"
	c = f"{n}: {cp} сек{Fore.RESET}"
	return c

st = tt()
if ma: print()

import os
if ma: print(ts(st, "os"))

import sys
if ma: print(ts(st, "sys"))

from contextlib import redirect_stdout, redirect_stderr
if ma: print(ts(st, "contextlib"))

import math
if ma: print(ts(st, "math"))

import pickle
if ma: print(ts(st, "pickle"))

import yaml
if ma: print(ts(st, "yaml"))

import numpy as np
if ma: print(ts(st, "numpy"))

from pathlib import Path
if ma: print(ts(st, "pathlib"))

from config import config, clear_screen, error, header, info, progress_bar, success, title, warning, rulables
if ma: print(ts(st, "config"))

from tqdm import tqdm
if ma: print(ts(st, "tqdm"))

import subprocess
if ma: print(ts(st, "subprocess"))

import torch
if ma: print(ts(st, "torch"))

import torch.nn as nn
if ma: print(ts(st, "torch.nn"))

import torch.nn.functional as F
if ma: print(ts(st, "torch.nn.functional"))

from sklearn.preprocessing import MultiLabelBinarizer
if ma: print(ts(st, "sklearn"))

from torch.amp import GradScaler, autocast
if ma: print(ts(st, "torch.amp"))

from torch.optim import AdamW
if ma: print(ts(st, "torch.optim"))

from torch.utils.data import DataLoader, Dataset
if ma: print(ts(st, "torch.utils.data"))

from torch.utils.tensorboard import SummaryWriter
if ma: print(ts(st, "torch.utils.tensorboard"))

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
if ma: print(ts(st, "transformers"))

if ma: print("\n" + ts(tot, "Общее время импортов") + "\n")

def worker_init_fn(worker_id):
	"""
	Функция, вызываемая один раз при создании worker'а в DataLoader.

	Args:
		worker_id (int): идентификатор worker'а
	"""
	print(f"👷 Worker {worker_id} создан (PID: {os.getpid()})\n")
	return None

def training_menu():
	"""
	Отображает меню выбора режима обучения и обрабатывает ввод пользователя.

	Returns:
		tuple: (test_mode: bool, test_size: int or None)
			- test_mode: True для тестового режима, False для обычного, None для выхода
			- test_size: количество примеров для теста (если test_mode=True)
	"""
	#clear_screen()
	title("РЕЖИМ ОБУЧЕНИЯ МОДЕЛИ")

	print(f"{Style.BRIGHT}1.{Style.RESET_ALL} Полное обучение ({EPOCHS} эпох, все данные)")
	print(f"{Style.BRIGHT}2.{Style.RESET_ALL} Тестовый режим (1 эпоха, ограниченные данные)")
	print(f"{Style.BRIGHT}3.{Style.RESET_ALL} Настроить параметры обучения")
	print(f"{Style.BRIGHT}4.{Style.RESET_ALL} Назад в главное меню")
	print()

	while True:
		choice = input(f"{Fore.CYAN}Выберите режим [1-4]: {Style.RESET_ALL}").strip()

		if choice == "1":
			return False, None  # Обычный режим
		elif choice == "2":
			print(f"\n{Fore.YELLOW}Тестовый режим:{Style.RESET_ALL}")
			print(f"  • 1 эпоха обучения")
			print(f"  • Ограниченное количество примеров")
			print(f"  • Модель не сохраняется")
			print(f"  • Быстрая проверка работоспособности")

			test_size = input(f"{Fore.YELLOW}Примеров для теста (по умолчанию 500): {Style.RESET_ALL}").strip()
			test_size = int(test_size) if test_size.isdigit() else 500
			return True, test_size

		elif choice == "3":
			# Показать текущие параметры
			print(f"\n{Fore.YELLOW}Текущие параметры обучения:{Style.RESET_ALL}")
			print(f"  • Модель: {MODEL_NAME}")
			print(f"  • Эпохи: {EPOCHS}")
			print(f"  • Batch size: {BATCH_SIZE}")
			print(f"  • Learning rate: {LEARNING_RATE}")
			print(f"  • Max length: {MAX_LEN}")
			print(f"\nДля изменения параметров отредактируйте файл: config.yaml")
			input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
			continue
		elif choice == "4":
			return None, None  # Выход
		else:
			error("Неверный выбор. Введите 1, 2, 3 или 4.")

CHECKPOINTS_DIR = config['checks_dir']
LOG_DIR = config['logs_dir']
OUTPUT_DIR = config['final_model_dir']
MODEL_NAME = config['source_model_dir']
NUM_WORKERS = int(config['num_workers'])

EPOCHS = int(config['epochs'])
MAX_LEN = int(config['max_len'])
BATCH_SIZE = int(config['batch_size'])
ACCUMULATION_STEPS = int(config['accumulation_steps'])
LEARNING_RATE = float(config['learning_rate'])
WARMUP_PERCENT = float(config['warmup_percent'])
WEIGHT_DECAY = float(config['weight_decay'])
FP32 = config['fp32']
USE_TRITON = config['use_triton']

def suppress_output(func):
	"""
	Декоратор для подавления вывода (stdout/stderr) функции.

	Args:
		func (callable): функция, вывод которой нужно подавить

	Returns:
		callable: обёрнутая функция
	"""
	def wrapper(*args, **kwargs):
		import os
		null_device = os.devnull

		with open(null_device, 'w') as f:
			old_stdout = sys.stdout
			old_stderr = sys.stderr
			sys.stdout = f
			sys.stderr = f
			try:
				result = func(*args, **kwargs)  # Исправлено: сохраняем результат
				return result  # Исправлено: возвращаем результат
			finally:
				sys.stdout = old_stdout
				sys.stderr = old_stderr
	return wrapper

def compile_model(model):
	"""
	Компилирует модель с использованием TorchInductor.

	Args:
		model (torch.nn.Module): модель для компиляции

	Returns:
		torch.nn.Module: скомпилированная модель
	"""
	return torch.compile(model, backend="inductor", mode="default")

def compute_class_weights(labels_list, num_classes=28):
	"""
	Вычисляет веса классов для несбалансированных данных на основе частоты встречаемости.

	Используется для корректировки функции потерь при многоклассовой классификации.

	Алгоритм:
	1. Собирает все метки из списка списков в единый список.
	2. Подсчитывает частоту встречаемости каждого класса.
	3. Вычисляет вес каждого класса как обратную величину частоты.
	4. Нормализует веса так, чтобы их сумма равнялась 1.

	Args:
		labels_list (list of list of int): список списков меток,
			где каждый внутренний список содержит индексы классов для одного примера
		num_classes (int, optional): общее количество классов в задаче.
			По умолчанию 28.

	Returns:
		torch.Tensor: тензор весов классов размером [num_classes],
			где вес каждого класса пропорционален обратной частоте его встречаемости

	Пример:
		labels_list = [[0, 1], [1, 2], [0, 2]]
		num_classes = 3
		→ веса: tensor([0.6, 0.3, 0.1]) (примерные значения после нормализации)
	"""
	from collections import Counter

	# Собираем все метки в единый список
	all_labels = []
	for labels in labels_list:
		all_labels.extend(labels)

	# Подсчитываем частоту каждого класса
	class_counts = Counter(all_labels)

	# Создаём список частот для всех классов (от 0 до num_classes-1)
	# Если класс не встречается, используем 1 для избежания деления на ноль
	counts = [class_counts.get(i, 1) for i in range(num_classes)]

	# Вычисляем веса как обратную величину частот
	weights = 1.0 / torch.tensor(counts, dtype=torch.float32)

	# Нормализуем веса так, чтобы их сумма была равна 1
	return weights / weights.sum()



class FocalLoss(nn.Module):
	"""
	Реализация Focal Loss — модифицированной функции потерь для задач многоклассовой классификации
	с несбалансированными данными.

	Focal Loss снижает вклад легко классифицируемых примеров, фокусируясь на сложных случаях.
	Особенно эффективна при сильном дисбалансе классов.


	Формула: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t), где:
	- p_t — предсказанная вероятность верного класса
	- α_t — весовой коэффициент класса
	- γ (gamma) — параметр фокусировки (чем больше, тем сильнее акцент на сложных примерах)

	Args:
		alpha (float): весовой коэффициент для балансировки классов (по умолчанию 1)
		gamma (float): параметр фокусировки на сложных примерах (по умолчанию 2)
	"""
	def __init__(self, alpha=1, gamma=2):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, inputs, targets):
		"""
		Вычисление значения функции потерь.

		Args:
			inputs (torch.Tensor): логиты модели (ненормализованные предсказания)
			targets (torch.Tensor): истинные метки (one-hot или многоклассовые)

		Returns:
			torch.Tensor: скалярное значение потерь
		"""
		# Вычисляем бинарную кросс‑энтропию (без усреднения)
		bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')


		# Получаем вероятность через сигмоид
		pt = torch.exp(-bce_loss)

		# Применяем Focal модификацию
		focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

		return focal_loss.mean()




class MultiLabelEmotionsDataset(Dataset):
	"""
	Dataset для задачи многоклассовой классификации эмоций (мульти‑лейбл).

	Преобразует текстовые данные и метки в формат, пригодный для обучения модели.
	"""
	def __init__(self, texts, labels, tokenizer, max_len):
		"""
		Args:
			texts (list of str): список текстовых примеров
			labels (list of list of int): список списков меток (каждый элемент — список индексов классов)
			tokenizer (transformers.PreTrainedTokenizer): токенизатор модели
			max_len (int): максимальная длина последовательности
		"""
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		"""Возвращает количество примеров в датасете."""
		return len(self.texts)

	def __getitem__(self, item):
		"""
		Возвращает один пример в формате, готовом для обучения.

		Args:
			item (int): индекс примера

		Returns:
			dict: словарь с ключами:
				- 'input_ids': тензор ID токенов
				- 'attention_mask': тензор маски внимания
				- 'labels': тензор меток
		"""
		text = str(self.texts[item])
		label = torch.tensor(self.labels[item], dtype=torch.float)  # float для BCEWithLogitsLoss


		# Токенизация с padding и truncation
		encoding = self.tokenizer(
			text,
			truncation=True,
			padding='max_length',
			max_length=self.max_len,
			return_tensors='pt',
		)

		return {
			'input_ids': encoding['input_ids'][0],
			'attention_mask': encoding['attention_mask'][0],
			'labels': label
		}



def calculate_checkpoint_frequency(total_steps):
	"""
	Определяет шаги, на которых нужно сохранять чекпоинты модели.

	Стратегия:
	- До 100 шагов → сохраняем только в конце
	- 100–1000 шагов → середина и конец
	- 1000–5000 шагов → каждые 1000 шагов + конец
	- 5000–10 000 шагов → каждые 2000 шагов + конец
	- Более 10 000 шагов → каждые 20% от общего числа шагов + конец

	Args:
		total_steps (int): общее количество шагов обучения

	Returns:
		list of int: список шагов для сохранения чекпоинтов
	"""
	if total_steps <= 100:
		return [total_steps]
	elif total_steps <= 1000:
		step1 = total_steps // 2
		return [step1, total_steps]
	elif total_steps <= 5000:
		interval = 1000
		steps = list(range(interval, total_steps, interval))
		if steps[-1] != total_steps:
			steps.append(total_steps)
		return steps
	elif total_steps <= 10000:
		interval = 2000
		steps = list(range(interval, total_steps, interval))
		if steps[-1] != total_steps:
			steps.append(total_steps)
		return steps
	else:
		interval = int(total_steps * 0.2)
		steps = list(range(interval, total_steps, interval))
		if steps[-1] != total_steps:
			steps.append(total_steps)
		return steps




def get_gpu_metrics_light():
	"""
	Собирает базовые метрики GPU (память).

	Returns:
		dict: словарь с ключами:
			- 'gpu_memory_used_mb': занятая память (МБ)
			- 'gpu_memory_reserved_mb': зарезервированная память (МБ)
			- 'gpu_memory_total_mb': общая память (МБ)
	"""
	metrics = {"gpu_memory_used_mb": 0, "gpu_memory_total_mb": 0}

	if not torch.cuda.is_available():
		return metrics

	try:
		metrics["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
		metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
		metrics["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
	except:
		pass

	return metrics



def get_system_metrics_light():
	"""
	Собирает базовые системные метрики (CPU, RAM).

	Returns:
		dict: словарь с ключами:
			- 'cpu_percent': загрузка CPU (%)
			- 'ram_percent': использование RAM (%)
	"""
	try:
		import psutil
		return {
			"cpu_percent": psutil.cpu_percent(),
			"ram_percent": psutil.virtual_memory().percent,
		}
	except ImportError:
		return {"cpu_percent": 0, "ram_percent": 0}




class MetricsLogger:
	"""
	Класс для логирования метрик обучения в TensorBoard и консоль.
	"""
	def __init__(self, writer, device, log_interval=100):
		"""
		Args:
			writer (SummaryWriter): объект для записи в TensorBoard
			device (torch.device): устройство (cpu/cuda)
			log_interval (int): интервал шагов для детального логирования
		"""
		self.writer = writer
		self.device = device
		self.log_interval = log_interval
		self.step = 0
		self._last_log_step = 0

		def log_training_step(self, loss, lr, grad_norm=None):
		"""
		Логирует метрики шага обучения в TensorBoard и периодически — в консоль.

		Записывает:
		- потери (loss)
		- скорость обучения (lr)
		- норму градиентов (если передана)
		- системные метрики (CPU, RAM, GPU) с заданным интервалом

		Args:
			loss (float): значение функции потерь на текущем шаге
			lr (float): текущая скорость обучения (learning rate)
			grad_norm (float, optional): норма градиентов после clipping.
				По умолчанию None (не записывается).
		"""
		# Запись основных метрик в TensorBoard
		self.writer.add_scalar("train/loss", loss, self.step)
		self.writer.add_scalar("train/lr", lr, self.step)

		if grad_norm is not None:
			self.writer.add_scalar("train/grad_norm", grad_norm, self.step)

		# Периодическая запись системных метрик и вывод в консоль
		if self.step - self._last_log_step >= self.log_interval:
			self._log_resources_light()  # Логируем использование ресурсов
			self._last_log_step = self.step

			# Дополнительный вывод в консоль каждые 200 шагов
			if self.step % 200 == 0:
				self._console_log_light(loss, lr)

		self.step += 1  # Увеличиваем счётчик шагов

	def _log_resources_light(self):
		"""
		Собирает и записывает в TensorBoard базовые системные метрики:
		- загрузку CPU и RAM (через psutil)
		- использование GPU (если доступно)
		"""
		# Системные метрики (CPU, RAM)
		system_metrics = get_system_metrics_light()
		for key, value in system_metrics.items():
			self.writer.add_scalar(f"system/{key}", value, self.step)

		# Метрики GPU (если CUDA доступна)
		if self.device.type == "cuda":
			gpu_metrics = get_gpu_metrics_light()
			for key, value in gpu_metrics.items():
				self.writer.add_scalar(f"gpu/{key}", value, self.step)

	def _console_log_light(self, loss, lr):
		"""
		Выводит в консоль сокращённую информацию о текущем состоянии обучения.
		Показывает использование VRAM, если GPU доступен.

		Args:
			loss (float): текущее значение потерь
			lr (float): текущая скорость обучения
		"""
		if torch.cuda.is_available():
			gpu_metrics = get_gpu_metrics_light()
			if gpu_metrics['gpu_memory_total_mb'] > 0:
				allocated_percent = (
					gpu_metrics['gpu_memory_used_mb'] /
					gpu_metrics['gpu_memory_total_mb'] * 100
				)
				reserved_percent = (
					gpu_metrics['gpu_memory_reserved_mb'] /
					gpu_metrics['gpu_memory_total_mb'] * 100
				)
				print(f"VRAM: {allocated_percent:.1f}% alloc / {reserved_percent:.1f}% reserved")

		def log_validation(self, metrics, epoch):
		"""
		Логирует метрики валидации в TensorBoard.

		Args:
			metrics (dict): словарь метрик валидации, например:
				{
					'loss': 0.5,
					'accuracy': 0.9,
					'f1_score': 0.85
				}
			epoch (int): номер текущей эпохи обучения
		"""
		for key, value in metrics.items():
			self.writer.add_scalar(f"val/{key}", value, epoch)


	def log_hyperparameters(self, hparams):
		"""
		Логирует гиперпараметры эксперимента в TensorBoard (вкладка HParams).

		Args:
			hparams (dict): словарь гиперпараметров, например:
				{
					'batch_size': 32,
					'learning_rate': 1e-4,
					'optimizer': 'AdamW',
					'model_type': 'BERT-base'
				}
		"""
		# TensorBoard HParams требует два словаря: hparams и metrics
		# metrics здесь — фиктивные значения для отображения
		metric_names = [f"val/{k}" for k in hparams.keys()]
		self.writer.add_hparams(hparams, {name: 0.0 for name in metric_names})


	def close(self):
		"""
		Закрывает writer, освобождая ресурсы.
		Должен вызываться в конце обучения.
		"""
		self.writer.close()



def train_epoch(model, dataloader, optimizer, loss_fn, device, metrics_logger, scaler=None):
	"""
	Выполняет одну эпоху обучения модели.

	Args:
		model (nn.Module): модель для обучения
		dataloader (DataLoader): загрузчик обучающих данных
		optimizer (Optimizer): оптимизатор
		loss_fn (Callable): функция потерь
		device (torch.device): устройство (cpu/cuda)
		metrics_logger (MetricsLogger): логгер метрик
		scaler (GradScaler, optional): для AMP (автоматическое масштабирование градиентов)


	Returns:
		float: среднее значение потерь за эпоху
	"""
	model.train()
	total_loss = 0.0
	step = 0

	for batch in dataloader:
		optimizer.zero_grad()

		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)

		# Forward pass с AMP (если включено)
		with autocast() if scaler is not None else torch.no_grad():
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
			loss = outputs.loss

		total_loss += loss.item()

		# Backward pass с AMP
		if scaler is not None:
			scaler.scale(loss).backward()
			# Gradient clipping
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

		# Логирование шага
		current_lr = optimizer.param_groups[0]['lr']
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
		metrics_logger.log_training_step(loss.item(), current_lr, grad_norm)


		step += 1

	return total_loss / len(dataloader)




def validate_epoch(model, dataloader, loss_fn, device):
	"""
	Выполняет валидацию модели на одном эпохе.

	Args:
		model (nn.Module): модель
		dataloader (DataLoader): загрузчик валидационных данных
		loss_fn (Callable): функция потерь
		device (torch.device): устройство


	Returns:
		dict: словарь метрик валидации (loss, accuracy и др.)
	"""
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch in dataloader:
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)


			outputs = model(input_ids, attention_mask=attention_mask)
			logits = outputs.logits
			loss = loss_fn(logits, labels)
			total_loss += loss.item()


			# Расчёт accuracy (для мульти‑лейбл: порог 0.5)
			preds = (torch.sigmoid(logits) > 0.5).float()
			correct += (preds == labels).all(dim=1).sum().item()
			total += labels.size(0)


	accuracy = correct / total
	avg_loss = total_loss / len(dataloader)


	return {
		'loss': avg_loss,
		'accuracy': accuracy
	}



def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
	"""
	Сохраняет чекпоинт модели и оптимизатора.

	Args:
		model (nn.Module): модель
		optimizer (Optimizer): оптимизатор
		epoch (int): номер эпохи
		loss (float): значение потерь
		checkpoint_path (str): путь для сохранения
	"""
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': loss
	}, checkpoint_path)
	print(f"Чекпоинт сохранён: {checkpoint_path}")



def load_checkpoint(model, optimizer, checkpoint_path, device):
	"""
	Загружает чекпоинт модели и оптимизатора.

	Args:
		model (nn.Module): модель, в которую будут загружены веса
		optimizer (Optimizer): оптимизатор, в который будут загружены состояния
		checkpoint_path (str): полный путь к файлу чекпоинта (.pth или .pt)
		device (torch.device): устройство (cpu/cuda), на которое следует загрузить параметры

	Returns:
		int: номер эпохи, на которой был сохранён чекпоинт
	Raises:
		FileNotFoundError: если файл чекпоинта не найден
		KeyError: если в чекпоинте отсутствуют ожидаемые ключи
		RuntimeError: если архитектура модели не соответствует сохранённой
	"""
	try:
		checkpoint = torch.load(checkpoint_path, map_location=device)

		# Проверка наличия всех необходимых ключей
		required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss']
		for key in required_keys:
			if key not in checkpoint:
				raise KeyError(f"Отсутствует ключ '{key}' в чекпоинте")

		# Загрузка состояний
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		# Получение метаданных
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']

		print(f"Чекпоинт загружен: эпоха {epoch}, loss={loss:.4f}")
		return epoch

	except FileNotFoundError:
		raise FileNotFoundError(f"Файл чекпоинта не найден: {checkpoint_path}")
	except KeyError as e:
		raise KeyError(f"Ошибка структуры чекпоинта: {e}. "
					 "Ожидаются ключи: 'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'")
	except RuntimeError as e:
		raise RuntimeError(f"Ошибка совместимости модели/оптимизатора: {e}. "
						"Возможные причины:\n"
						"  - изменилась архитектура модели\n"
						"  - несовпадение размеров параметров\n"
						"  - несоответствие версий PyTorch")
	except Exception as e:
		raise Exception(f"Неожиданная ошибка при загрузке чекпоинта: {type(e).__name__}: {e}")



def save_training_state(model, optimizer, scheduler, epoch, loss, metrics,
					   checkpoint_dir, filename_prefix="checkpoint"):
	"""
	Сохраняет полное состояние обучения (модель, оптимизатор, scheduler, метрики).

	Args:
		model (nn.Module): обучаемая модель
		optimizer (Optimizer): текущий оптимизатор
		scheduler (LRScheduler): планировщик learning rate
		epoch (int): номер текущей эпохи
		loss (float): значение потерь на текущей эпохе
		metrics (dict): словарь дополнительных метрик (например, {'accuracy': 0.95})
		checkpoint_dir (str): директория для сохранения чекпоинтов
		filename_prefix (str): префикс имени файла (по умолчанию "checkpoint")


	Returns:
		str: полный путь к сохранённому чекпоинту
	"""
	# Формируем имя файла с номером эпохи
	filename = f"{filename_prefix}_epoch_{epoch}.pth"
	checkpoint_path = os.path.join(checkpoint_dir, filename)

	# Собираем все состояния
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
		'loss': loss,
		'metrics': metrics,
		'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
	}
	# Сохраняем на диск
	torch.save(checkpoint, checkpoint_path)
	print(f"Состояние обучения сохранено: {checkpoint_path}")
	return checkpoint_path

def load_training_state(model, optimizer, scheduler, checkpoint_path, device):
	"""
	Загружает полное состояние обучения из чекпоинта.

	Args:
		model (nn.Module): модель для загрузки весов
		optimizer (Optimizer): оптимизатор для восстановления состояния
		scheduler (LRScheduler): планировщик LR (может быть None)
		checkpoint_path (str): путь к чекпоинту
		device (torch.device): целевое устройство


	Returns:
		dict: словарь с загруженными данными:
			- 'epoch': номер эпохи
			- 'loss': значение потерь
			- 'metrics': дополнительные метрики
			- 'timestamp': время сохранения
	"""
	checkpoint = torch.load(checkpoint_path, map_location=device)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	if scheduler and checkpoint['scheduler_state_dict']:
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


	info = {
		'epoch': checkpoint['epoch'],
		'loss': checkpoint['loss'],
		'metrics': checkpoint.get('metrics', {}),
		'timestamp': checkpoint.get('timestamp', 'N/A')
	}
	print(f"Состояние обучения загружено: эпоха {info['epoch']}, "
		  f"loss={info['loss']:.4f}, время={info['timestamp']}")
	return info



def save_final_model(model, tokenizer, config, output_dir, model_name="final_model"):
	"""
	Сохраняет финальную обученную модель вместе с токенизатором и конфигурацией.

	Для последующего использования в inference.

	Args:
		model (nn.Module): обученная модель
		tokenizer (PreTrainedTokenizer): соответствующий токенизатор
		config (dict): конфигурация модели/эксперимента
		output_dir (str): директория для сохранения
		model_name (str): имя модели (без расширения)


	Returns:
		str: путь к сохранённой модели
	"""
	model_path = os.path.join(output_dir, model_name)

	# Сохраняем модель (state_dict + config)
	model.save_pretrained(model_path)

	# Сохраняем токенизатор
	tokenizer.save_pretrained(model_path)

	# Сохраняем дополнительную конфигурацию
	config_path = os.path.join(model_path, "training_config.json")
	with open(config_path, 'w', encoding='utf-8') as f:
		json.dump(config, f, ensure_ascii=False, indent=2)


	print(f"Финальная модель сохранена: {model_path}")
	return model_path



def setup_training(model, train_params):
	"""
	Настраивает компоненты обучения: оптимизатор, scheduler, AMP.

	Args:
		model (nn.Module): модель для обучения
		train_params (dict): параметры обучения, например:
			{
				'learning_rate': 1e-4,
				'weight_decay': 0.01,
				'warmup_percent': 0.1,
				'total_steps': 1000,
				'fp16': True
			}

	Returns:
		tuple: (optimizer, scheduler, scaler)
			- optimizer: настроенные оптимизатор
			- scheduler: планировщик LR
			- scaler: GradScaler для AMP (или None)
	"""
	# Оптимизатор (AdamW с weight decay)
	optimizer = AdamW(
		model.parameters(),
		lr=train_params['learning_rate'],
		weight_decay=train_params['weight_decay']
	)

	# Планировщик LR с warmup
	scheduler = get_cosine_schedule_with_warmup(
		optimizer,
		num_warmup_steps=int(train_params['warmup_percent'] * train_params['total_steps']),
		num_training_steps=train_params['total_steps']
	)

	# AMP Scaler (если fp16 включён)
	scaler = GradScaler() if train_params['fp16'] else None


	return optimizer, scheduler, scaler


def count_parameters(model):
	"""
	Считает количество обучаемых параметров модели.

	Args:
		model (nn.Module): модель

	Returns:
		dict: словарь с количеством параметров:
			- 'total': общее число параметров
			- 'trainable': число обучаемых параметров
	"""
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return {'total': total, 'trainable': trainable}


def print_model_info(model, model_name="Модель"):
	"""
	Выводит информацию о модели в консоль.

	Args:
		model (nn.Module): модель
		model_name (str): название модели для вывода
	"""
	params = count_parameters(model)
	print(f"\n{model_name}:")
	print(f"  Общее число параметров: {params['total']:,}")
	print(f"  Обучаемые параметры:    {params['trainable']:,}")
	print(f"  Архитектура: {type(model).__name__}")

def set_seed(seed=42):
	"""
	Устанавливает seed для воспроизводимости экспериментов.

	Args:
		seed (int): значение seed
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers=4):
	"""
	Создаёт DataLoader'ы для обучения и валидации.

	Args:
		train_dataset (Dataset): обучающий датасет
		val_dataset (Dataset): валидационный датасет
		batch_size (int): размер батча
		num_workers (int): число рабочих процессов для загрузки данных


	Returns:
		tuple: (train_loader, val_loader)
	"""
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		worker_init_fn=worker_init_fn
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True
	)

	return train_loader, val_loader



def worker_init_fn(worker_id):
	"""
	Инициализирует рабочий процесс DataLoader.

	Используется для воспроизводимости при многопроцессорной загрузке данных.

	"""
	np.random.seed(np.random.get_state()[1][0] + worker_id)



def setup_device(use_cuda=True):
	"""
	Настраивает устройство для вычислений (CPU/GPU).

	Args:
		use_cuda (bool): если True, пытается использовать GPU

	Returns:
		torch.device: выбранное устройство
	"""
	if use_cuda and torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Используем GPU: {torch.cuda.get_device_name(0)}")
		print(f"Количество GPU: {torch.cuda.device_count()}")
	else:
		device = torch.device("cpu")
		print("Используем CPU")
	return device

def freeze_layers(model, freeze_pattern):
	"""
	Замораживает слои модели по заданному шаблону.

	Полезно для трансферного обучения.

	Args:
		model (nn.Module): модель
		freeze_pattern (str): шаблон для заморозки (например, "bert.encoder.layer.0.")


	Пример:
		freeze_layers(model, "bert.embeddings.")  # заморозит все слои embeddings
	"""
	for name, param in model.named_parameters():
		if freeze_pattern in name:
			param.requires_grad = False
			print(f"Заморожен слой: {name}")


def unfreeze_layers(model):
	"""
	Размораживает все слои модели.

	Args:
		model (nn.Module): модель
	"""
	for param in model.parameters():
		param.requires_grad = True
	print("Все слои разморожены")

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
	"""
	Создаёт линейный планировщик LR с прогревом.

	Args:
		optimizer (Optimizer): оптимизатор
		num_warmup_steps (int): количество шагов прогрева
		num_training_steps (int): общее количество шагов обучения

	Returns:
		LambdaLR: планировщик LR
	"""
	def lr_lambda(current_step: int):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		return max(
			0.0, float(num_training_steps - current_step) /
			float(max(1, num_training_steps - num_warmup_steps))
		)

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_f1_score(y_true, y_pred, threshold=0.5):
	"""
	Вычисляет F1-score для мульти‑лейбл классификации.

	Args:
		y_true (torch.Tensor): истинные метки
		y_pred (torch.Tensor): предсказанные логиты
		threshold (float): порог для бинаризации предсказаний

	Returns:
		float: значение F1-score
	"""
	y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()
	tp = (y_pred_bin * y_true).sum().item()
	fp = (y_pred_bin * (1 - y_true)).sum().item()
	fn = ((1 - y_pred_bin) * y_true).sum().item()

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0

	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
	return f1


def log_metrics_to_file(metrics_dict, filepath):
	"""
	Сохраняет метрики в JSON-файл.

	Args:
		metrics_dict (dict): словарь метрик
		filepath (str): путь к файлу
	"""
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
	print(f"Метрики сохранены: {filepath}")


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_path):
	"""
	Рисует кривые обучения (loss и accuracy).

	Args:
		train_losses (list): потери на обучении
		val_losses (list): потери на валидации
		train_accuracies (list): точность на обучении
		val_accuracies (list): точность на валидации
		output_path (str): путь для сохранения графика
	"""
	import matplotlib.pyplot as plt

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	ax1.plot(train_losses, label='Train Loss')
	ax1.plot(val_losses, label='Val Loss')
	ax1.set_title('Loss')
	ax1.legend()

	ax2.plot(train_accuracies, label='Train Accuracy')
	ax2.plot(val_accuracies, label='Val Accuracy')
	ax2.set_title('Accuracy')
	ax2.legend()

	plt.savefig(output_path)
	plt.close()
	print(f"График сохранён: {output_path}")




def evaluate_model(model, dataloader, device, threshold=0.5):
	"""
	Полноценная оценка модели на тестовом/валидационном наборе.

	Собирает loss, accuracy, F1-score и другие метрики.


	Args:
		model (nn.Module): обученная модель
		dataloader (DataLoader): загрузчик данных
		device (torch.device): устройство для вычислений
		threshold (float): порог для бинаризации предсказаний


	Returns:
		dict: словарь с метриками:
			- 'loss': среднее значение потерь
			- 'accuracy': точность
			- 'f1_score': F1-мера
			- 'precision': точность (precision)
			- 'recall': полнота (recall)
	"""
	model.eval()
	total_loss = 0.0
	all_preds = []
	all_labels = []

	loss_fn = nn.BCEWithLogitsLoss()

	with torch.no_grad():
		for batch in dataloader:
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			outputs = model(input_ids, attention_mask=attention_mask)
			logits = outputs.logits
			loss = loss_fn(logits, labels)
			total_loss += loss.item()

			# Бинаризация предсказаний
			preds = (torch.sigmoid(logits) > threshold).float()

			all_preds.append(preds)
			all_labels.append(labels)

	# Собираем все предсказания и метки
	all_preds = torch.cat(all_preds, dim=0)
	all_labels = torch.cat(all_labels, dim=0)


	# Вычисляем метрики
	tp = (all_preds * all_labels).sum().item()
	fp = (all_preds * (1 - all_labels)).sum().item()
	fn = ((1 - all_preds) * all_labels).sum().item()
	tn = ((1 - all_preds) * (1 - all_labels)).sum().item()

	accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


	return {
		'loss': total_loss / len(dataloader),
		'accuracy': accuracy,
		'f1_score': f1,
		'precision': precision,
		'recall': recall
	}



def predict(model, tokenizer, text, device, max_len=512, threshold=0.5):
	"""
	Делает предсказание для одиночного текста.

	Args:
		model (nn.Module): обученная модель
		tokenizer (PreTrainedTokenizer): токенизатор
		text (str): входной текст
		device (torch.device): устройство
		max_len (int): максимальная длина последовательности
		threshold (float): порог для бинаризации


	Returns:
		dict: словарь с результатами:
			- 'probabilities': вероятности для каждого класса
			- 'predictions': бинарные предсказания
	"""
	model.eval()
	encoding = tokenizer(
		text,
		truncation=True,
		padding='max_length',
		max_length=max_len,
		return_tensors='pt'
	)

	input_ids = encoding['input_ids'].to(device)
	attention_mask = encoding['attention_mask'].to(device)

	with torch.no_grad():
		outputs = model(input_ids, attention_mask=attention_mask)
		logits = outputs.logits
		probs = torch.sigmoid(logits).cpu().numpy()[0]

		preds = (probs > threshold).astype(int)


	return {
		'probabilities': probs,
		'predictions': preds
	}


def create_experiment_dir(base_path, experiment_name):
	"""
	Создаёт директорию для эксперимента с уникальной меткой времени.


	Args:
		base_path (str): базовая директория
		experiment_name (str): название эксперимента


	Returns:
		str: полный путь к директории эксперимента
	"""
	timestamp = time.strftime("%Y%m%d-%H%M%S")
	exp_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
	os.makedirs(exp_dir, exist_ok=True)
	print(f"Директория эксперимента создана: {exp_dir}")
	return exp_dir


def save_config(config, filepath):
	"""
	Сохраняет конфигурацию эксперимента в JSON-файл.

	Args:
		config (dict): словарь конфигурации
		filepath (str): путь к файлу
	"""
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(config, f, ensure_ascii=False, indent=2)
	print(f"Конфигурация сохранена: {filepath}")


def load_config(filepath):
	"""
	Загружает конфигурацию эксперимента из JSON-файла.

	Args:
		filepath (str): путь к JSON-файлу

	Returns:
		dict: конфигурация
	"""
	with open(filepath, 'r', encoding='utf-8') as f:
		config = json.load(f)
	print(f"Конфигурация загружена: {filepath}")
	return config

def get_current_time_str():
	"""
	Возвращает текущую дату и время в строковом формате.

	Returns:
		str: строка с датой и временем (формат: YYYY-MM-DD_HH-MM-SS)
	"""
	return time.strftime("%Y-%m-%d_%H-%M-%S")


def setup_logging(log_file):
	"""
	Настраивает логирование в файл.

	Args:
		log_file (str): путь к файлу логов
	"""
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_file, encoding='utf-8'),
			logging.StreamHandler()
		]
	)
	print(f"Логирование настроено: {log_file}")

def compute_class_weights(labels):
	"""
	Вычисляет веса классов для несбалансированных данных.

	Использует формулу: weight = total_samples / (n_classes * class_count)


	Args:
		labels (torch.Tensor): тензор меток (форма: [n_samples, n_classes])


	Returns:
		torch.Tensor: веса классов
	"""
	n_samples, n_classes = labels.shape
	class_counts = labels.sum(dim=0)  # сумма по каждому классу
	weights = n_samples / (n_classes * class_counts)
	return weights.float()



def create_balanced_sampler(labels):
	"""
	Создаёт взвешенный sampler для DataLoader, чтобы сбалансировать классы в батчах.

	Используется при сильной дисбалансе классов.


	Args:
		labels (torch.Tensor): тензор меток формы [n_samples, n_classes]
							 или [n_samples] для мульти‑классовой классификации

	Returns:
		WeightedRandomSampler: sampler для DataLoader
	"""
	if labels.dim() == 2:  # мульти‑лейбл
		class_counts = labels.sum(dim=0).float()
	else:  # мульти‑класс
		class_counts = torch.bincount(labels).float()


	# Вычисляем веса: чем меньше примеров класса, тем выше вес
	weights = 1.0 / class_counts
	# Нормализуем веса
	weights /= weights.sum()

	# Для каждого примера определяем вес (по его классу)
	if labels.dim() == 2:
		sample_weights = torch.mm(labels, weights.unsqueeze(1)).squeeze(1)
	else:
		sample_weights = weights[labels]


	sampler = WeightedRandomSampler(
		weights=sample_weights,
		num_samples=len(labels),
		replacement=True
	)
	return sampler



def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
	"""
	Разделяет датасет на train/val/test подмножества.


	Args:
		dataset (Dataset): исходный датасет
		train_ratio (float): доля обучающей выборки
		val_ratio (float): доля валидационной выборки
		test_ratio (float): доля тестовой выборки
		seed (int): seed для воспроизводимости

	Returns:
		tuple: (train_subset, val_subset, test_subset)
	"""
	total_size = len(dataset)
	train_size = int(total_size * train_ratio)
	val_size = int(total_size * val_ratio)
	test_size = total_size - train_size - val_size

	indices = list(range(total_size))
	np.random.seed(seed)
	np.random.shuffle(indices)

	train_indices = indices[:train_size]
	val_indices = indices[train_size:train_size + val_size]
	test_indices = indices[train_size + val_size:]


	train_subset = Subset(dataset, train_indices)
	val_subset = Subset(dataset, val_indices)
	test_subset = Subset(dataset, test_indices)

	return train_subset, val_subset, test_subset




def collate_fn_batch_padding(batch):
	"""
	Коллационная функция для DataLoader с динамическим паддингом.
	Объединяет примеры в батч и добавляет паддинг до максимальной длины в батче.


	Используется для текстовых данных с переменной длиной.

	Args:
		batch (list): список примеров (каждый пример — dict с 'input_ids', 'attention_mask', 'labels')


	Returns:
		dict: батч с паддингом
	"""
	max_len = max([len(item['input_ids']) for item in batch])


	padded_batch = {
		'input_ids': [],
		'attention_mask': [],
		'labels': []
	}

	for item in batch:
		pad_len = max_len - len(item['input_ids'])
		padded_input = item['input_ids'] + [0] * pad_len
		padded_mask = item['attention_mask'] + [0] * pad_len

		padded_batch['input_ids'].append(padded_input)
		padded_batch['attention_mask'].append(padded_mask)
		padded_batch['labels'].append(item['labels'])


	# Конвертируем в тензоры
	padded_batch['input_ids'] = torch.tensor(padded_batch['input_ids'], dtype=torch.long)
	padded_batch['attention_mask'] = torch.tensor(padded_batch['attention_mask'], dtype=torch.long)
	padded_batch['labels'] = torch.tensor(padded_batch['labels'], dtype=torch.float)


	return padded_batch




def save_predictions(predictions, labels, output_path):
	"""
	Сохраняет предсказания и истинные метки в CSV‑файл.


	Args:
		predictions (torch.Tensor или np.ndarray): предсказанные вероятности/классы
		labels (torch.Tensor или np.ndarray): истинные метки
		output_path (str): путь к CSV‑файлу
	"""
	df = pd.DataFrame({
		'predictions': predictions.flatten(),
		'true_labels': labels.flatten()
	})
	df.to_csv(output_path, index=False)
	print(f"Предсказания сохранены: {output_path}")




def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
	"""
	Рисует матрицу ошибок (confusion matrix) и сохраняет в файл.


	Args:
		y_true (array-like): истинные метки
		y_pred (array-like): предсказанные метки
		class_names (list): названия классов
		output_path (str): путь для сохранения изображения
	"""
	cm = confusion_matrix(y_true, y_pred)
	df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

	plt.figure(figsize=(10, 7))
	sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
	plt.title('Confusion Matrix')
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	plt.savefig(output_path)
	plt.close()
	print(f"Матрица ошибок сохранена: {output_path}")




def calculate_roc_auc(y_true, y_scores):
	"""
	Вычисляет ROC‑AUC для каждого класса и средний ROC‑AUC.

	Подходит для мульти‑лейбл и мульти‑классовых задач.


	Args:
		y_true (torch.Tensor): истинные метки [n_samples, n_classes]
		y_scores (torch.Tensor): предсказанные вероятности [n_samples, n_classes]

	Returns:
		dict: {'class_auc': список AUC по классам, 'macro_auc': средний AUC}
	"""
	y_true = y_true.cpu().numpy()
	y_scores = y_scores.cpu().numpy()

	n_classes = y_true.shape[1]
	class_auc = []

	for i in range(n_classes):
		auc = roc_auc_score(y_true[:, i], y_scores[:, i])
		class_auc.append(auc)


	macro_auc = np.mean(class_auc)
	return {'class_auc': class_auc, 'macro_auc': macro_auc}




def early_stopping(monitor_value, best_value, patience, counter, mode='min'):
	"""
	Проверяет условие ранней остановки обучения.


	Args:
		monitor_value (float): текущее значение метрики (например, val_loss)
		best_value (float): лучшее значение метрики
		patience (int): количество эпох без улучшения до остановки
		counter (int): счётчик эпох без улучшения
		mode (str): 'min' — ищем минимум (loss), 'max' — максимум (accuracy)


	Returns:
		bool: True, если нужно остановить обучение
		float: обновлённое best_value
		int: обновлённый counter
	"""
	improvement = False

	if mode == 'min':
		if monitor_value < best_value:
			best_value = monitor_value
			counter = 0
			improvement = True
		else:
			counter += 1
	elif mode == 'max':
		if monitor_value > best_value:
			best_value = monitor_value
			counter = 0
			improvement = True
		else:
			counter += 1

	stop_training = counter >= patience
	return stop_training, best_value, counter





def calculate_precision_recall_f1(y_true, y_pred, average='macro'):
	"""
	Вычисляет precision, recall и F1‑score для классификации.


	Args:
		y_true (array-like): истинные метки
		y_pred (array-like): предсказанные метки
		average (str): стратегия усреднения ('macro', 'micro', 'weighted')


	Returns:
		dict: словарь с метриками {'precision', 'recall', 'f1_score'}
	"""
	precision = precision_score(y_true, y_pred, average=average)
	recall = recall_score(y_true, y_pred, average=average)
	f1 = f1_score(y_true, y_pred, average=average)

	return {
		'precision': precision,
		'recall': recall,
		'f1_score': f1
	}



def generate_classification_report(y_true, y_pred, class_names=None):
	"""
	Генерирует полный отчёт по классификации (включая precision, recall, F1, support).


	Args:
		y_true (array-like): истинные метки
		y_pred (array-like): предсказанные метки
		class_names (list): названия классов (опционально)


	Returns:
		str: текстовый отчёт (как в sklearn.classification_report)
	"""
	report = classification_report(
		y_true,
		y_pred,
		target_names=class_names,
		output_dict=False
	)
	return report

def plot_learning_rate(lr_history, output_path):
	"""
	Рисует график изменения learning rate в процессе обучения.


	Args:
		lr_history (list): список значений LR по шагам/эпохам
		output_path (str): путь для сохранения графика
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(lr_history, label='Learning Rate')
	plt.title('Learning Rate Schedule')
	plt.xlabel('Steps/Epochs')
	plt.ylabel('LR')
	plt.grid(True)
	plt.savefig(output_path)
	plt.close()
	print(f"График LR сохранён: {output_path}")


def save_model_onnx(model, dummy_input, filepath, input_names=None, output_names=None):
	"""
	Сохраняет модель в формате ONNX.

	Полезно для инференса вне PyTorch (например, в C++, Java, JavaScript).


	Args:
		model (nn.Module): обученная модель
		dummy_input (torch.Tensor): фиктивный тензор для трассировки
		filepath (str): путь к файлу (.onnx)
		input_names (list): имена входных тензоров
		output_names (list): имена выходных тензоров
	"""
	model.eval()
	torch.onnx.export(
		model,
		dummy_input,
		filepath,
		export_params=True,
		opset_version=11,
		do_constant_folding=True,
		input_names=input_names or ['input'],
		output_names=output_names or ['output'],
		dynamic_axes={
			'input': {0: 'batch_size'},
			'output': {0: 'batch_size'}
		}
	)
	print(f"Модель сохранена в ONNX: {filepath}")


def load_onnx_model(filepath):
	"""
	Загружает модель ONNX (для инференса).

	Требуется установить onnxruntime: `pip install onnxruntime`


	Args:
		filepath (str): путь к ONNX-файлу


	Returns:
		InferenceSession: сессия для инференса
	"""
	import onnxruntime as ort
	session = ort.InferenceSession(filepath)
	return session

def run_inference_onnx(session, input_data):
	"""
	Выполняет инференс на ONNX-модели.


	Args:
		session (InferenceSession): сессия ONNX
		input_data (np.ndarray): входные данные


	Returns:
		np.ndarray: предсказания модели
	"""
	input_name = session.get_inputs()[0].name
	output = session.run(None, {input_name: input_data})
	return output[0]

def create_tensorboard_logger(log_dir):
	"""
	Создаёт логгер TensorBoard.


	Args:
		log_dir (str): директория для логов TensorBoard


	Returns:
		SummaryWriter: объект для записи в TensorBoard
	"""
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(log_dir=log_dir)
	print(f"TensorBoard логгер создан: {log_dir}")
	return writer


def log_scalar_to_tensorboard(writer, tag, value, step):
	"""
	Записывает скалярное значение в TensorBoard.


	Args:
		writer (SummaryWriter): логгер TensorBoard
		tag (str): метка (например, 'train/loss')
		value (float): значение
		step (int): шаг/эпоха
	"""
	writer.add_scalar(tag, value, step)


def log_histogram_to_tensorboard(writer, tag, values, step, bins='auto'):
	"""
	Записывает гистограмму значений в TensorBoard.

	Args:
		writer (SummaryWriter): логгер TensorBoard
		tag (str): метка
		values (array-like): массив значений
		step (int): шаг/эпоха
		bins (str или int): количество бинов для гистограммы
	"""
	writer.add_histogram(tag, values, step, bins=bins)

def log_embedding_to_tensorboard(writer, features, metadata, step):
	"""
	Записывает эмбеддинги в TensorBoard (для визуализации).


	Args:
		writer (SummaryWriter): логгер TensorBoard
		features (torch.Tensor или np.ndarray): эмбеддинги
		metadata (list): метки для точек
		step (int): шаг/эпоха
	"""
	writer.add_embedding(features, metadata=metadata, global_step=step)

def set_deterministic_mode():
	"""
	Включает детерминированный режим PyTorch (для воспроизводимости).

	Внимание: может снизить скорость обучения!
	"""
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)
	print("Детерминированный режим включён")

def count_trainable_params(model):
	"""
	Считает количество обучаемых параметров модели.

	Args:
		model (nn.Module): модель

	Returns:
		int: количество обучаемых параметров
	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size, batch_size=-1, device="cuda"):
	"""
	Печатает сводку модели (как в Keras).

	Использует torchsummary.

	Args:
		model (nn.Module): модель
		input_size (tuple): размер входного тензора (без batch_size)
		batch_size (int): размер батча (по умолчанию -1)
		device (str): устройство ('cuda' или 'cpu')
	"""
	try:
		from torchsummary import summary
		summary(model, input_size, batch_size=batch_size, device=device)
	except ImportError:
		print("torchsummary не установлен. Установите: pip install torchsummary")




def compute_perplexity(logits, labels):
	"""
	Вычисляет перплексию для языковых моделей (по лоссам).


	Перплексия — стандартная метрика для оценки языковых моделей.

	Perplexity = exp(avg_negative_log_likelihood)


	Args:
		logits (torch.Tensor): логиты модели [batch_size, seq_len, vocab_size]
		labels (torch.Tensor): истинные токены [batch_size, seq_len]


	Returns:
		float: значение перплексии
	"""
	loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
	shift_logits = logits[..., :-1, :].contiguous()
	shift_labels = labels[..., 1:].contiguous()


	# Применяем loss
	loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
	perplexity = torch.exp(loss)
	return perplexity.item()



def calculate_bleu_score(references, hypotheses, n_grams=4):
	"""
	Вычисляет BLEU‑score для задач генерации текста (например, перевода, суммаризации).


	Использует nltk.bleu_score.


	Args:
		references (list of list of str): список списков токенов (истинные последовательности)
		hypotheses (list of str): список сгенерированных последовательностей (уже токенизированных)
		n_grams (int): максимальное число n‑грамм для учёта


	Returns:
		float: BLEU‑score (от 0 до 1)
	"""
	from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
	smoothing = SmoothingFunction()


	scores = []
	for ref, hyp in zip(references, hypotheses):
		score = sentence_bleu(
			[ref],
			hyp,
			weights=[1/n_grams] * n_grams,
			smoothing_function=smoothing.method1
		)
		scores.append(score)

	return sum(scores) / len(scores)

def calculate_rouge_scores(references, hypotheses):
	"""
	Вычисляет ROUGE‑оценки (ROUGE‑1, ROUGE‑2, ROUGE‑L) для суммаризации/перевода.


	Использует rouge‑score библиотеку.

	Args:
		references (list of str): истинные тексты
		hypotheses (list of str): сгенерированные тексты


	Returns:
		dict: {'rouge1', 'rouge2', 'rougeL'} — словари с precision, recall, f1
	"""
	try:
		from rouge_score import rouge_scorer
		scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


		scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

		for ref, hyp in zip(references, hypotheses):
			score = scorer.score(ref, hyp)
			for key in scores:
				scores[key].append(score[key])


		# Среднее по всем примерам
		avg_scores = {}
		for key in scores:
			p = np.mean([s.precision for s in scores[key]])
			r = np.mean([s.recall for s in scores[key]])
			f = np.mean([s.fmeasure for s in scores[key]])
			avg_scores[key] = {'precision': p, 'recall': r, 'f1': f}


		return avg_scores
	except ImportError:
		print("Установите rouge-score: pip install rouge-score")
		return {}

def tokenize_texts(texts, tokenizer, max_length=512, padding=True, truncation=True):
	"""
	Токенизирует список текстов с помощью заданного токенизатора.


	Args:
		texts (list of str): список текстов
		tokenizer (PreTrainedTokenizer): токенизатор
		max_length (int): максимальная длина последовательности
		padding (bool): добавлять паддинг
		truncation (bool): обрезать длинные тексты


	Returns:
		dict: выходы токенизатора (input_ids, attention_mask и т.д.)
	"""
	encoded = tokenizer(
		texts,
		max_length=max_length,
		padding=padding,
		truncation=truncation,
		return_tensors='pt'
	)
	return encoded

def extract_features(model, dataloader, device, layer_name=None):
	"""
	Извлекает эмбеддинги/признаки из заданной слои модели.


	Полезно для визуализации, кластеризации, анализа.

	Если layer_name не указан — возвращает выход последнего слоя.

	Args:
		model (nn.Module): модель
		dataloader (DataLoader): загрузчик данных
		device (torch.device): устройство
		layer_name (str): имя слоя (опционально)


	Returns:
		torch.Tensor: извлечённые признаки [n_samples, feature_dim]
	"""
	model.eval()
	features = []

	with torch.no_grad():
		for batch in dataloader:
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)

			# Если указан слой — используем forward hook
			if layer_name:
				activation = {}
				def hook(module, input, output):
					activation[layer_name] = output

				# Находим слой по имени
				layer = dict(model.named_modules())[layer_name]
				handle = layer.register_forward_hook(hook)

				outputs = model(input_ids, attention_mask=attention_mask)
				handle.remove()
				feat = activation[layer_name]
			else:
				# Иначе — выход модели
				outputs = model(input_ids, attention_mask=attention_mask)
				feat = outputs.last_hidden_state  # или logits, зависит от модели

			features.append(feat.cpu())

	return torch.cat(features, dim=0)

def visualize_embeddings(embeddings, labels, class_names=None, output_path=None):
	"""
	Визуализирует эмбеддинги с помощью PCA/t‑SNE.

	Показывает кластеры по классам.

	Args:
		embeddings (np.ndarray или torch.Tensor): эмбеддинги [n_samples, dim]
		labels (array-like): метки классов [n_samples]
		classnames (list): названия классов (опционально)
		output_path (str): путь для сохранения графика (опционально)
	"""
	import matplotlib.pyplot as plt
	from sklearn.decomposition import PCA
	from sklearn.manifold import TSNE

	# Преобразуем в numpy
	if isinstance(embeddings, torch.Tensor):
		embeddings = embeddings.numpy()
	if isinstance(labels, torch.Tensor):
		labels = labels.numpy()


	# Уменьшаем размерность до 2D
	pca = PCA(n_components=2)
	embeddings_2d = pca.fit_transform(embeddings)

	plt.figure(figsize=(10, 8))
	scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

	if classnames:
		plt.legend(*scatter.legend_elements(), title="Classes", labels=classnames)
	else:
		plt.legend(*scatter.legend_elements(), title="Classes")


	plt.title("Embeddings Visualization (PCA)")
	plt.xlabel("PC1")
	plt.ylabel("PC2")

	if output_path:
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		print(f"Визуализация сохранена: {output_path}")
	plt.show()


def save_predictions_json(predictions, output_path):
	"""
	Сохраняет предсказания в JSON‑файл (удобно для пост‑обработки/субмиссии).


	Args:
		predictions (list of dict): список словарей с результатами предсказаний
		output_path (str): путь к выходному JSON‑файлу
	"""
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(predictions, f, ensure_ascii=False, indent=2)
	print(f"Предсказания сохранены в JSON: {output_path}")



def load_predictions_json(input_path):
	"""
	Загружает предсказания из JSON‑файла.


	Args:
		input_path (str): путь к JSON‑файлу с предсказаниями


	Returns:
		list of dict: список словарей с предсказаниями
	"""
	with open(input_path, 'r', encoding='utf-8') as f:
		predictions = json.load(f)
	print(f"Предсказания загружены из JSON: {input_path}")
	return predictions

def calculate_accuracy_per_class(y_true, y_pred, class_names=None):
	"""
	Вычисляет точность (accuracy) отдельно для каждого класса.


	Для каждого класса:
	  accuracy = (TP + TN) / (TP + TN + FP + FN)
	Но в контексте многоклассовой классификации проще считать как:
	  доля верно классифицированных примеров данного класса


	Args:
		y_true (array-like): истинные метки
		y_pred (array-like): предсказанные метки
		classnames (list of str): названия классов (опционально)


	Returns:
		dict: {класс: точность} или {индекс_класса: точность}
	"""
	from collections import defaultdict

	if classnames is None:
		classnames = sorted(set(y_true))

	acc_per_class = {}
	for cls in classnames:
		idx = (np.array(y_true) == cls)
		if np.sum(idx) == 0:
			acc_per_class[cls] = 0.0
		else:
			correct = (np.array(y_pred)[idx] == cls).sum()
			total = idx.sum()
			acc_per_class[cls] = correct / total


	return acc_per_class

def plot_loss_curves(train_losses, val_losses, output_path):
	"""
	Рисует кривые обучения: тренировочные и валидационные потери.


	Args:
		train_losses (list of float): значения потерь на обучении
		val_losses (list of float): значения потерь на валидации
		output_path (str): путь для сохранения графика
	"""
	plt.figure(figsize=(10, 6))
	epochs = range(1, len(train_losses) + 1)
	plt.plot(epochs, train_losses, 'b-', label='Train Loss')
	plt.plot(epochs, val_losses, 'r--', label='Val Loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Кривые потерь сохранены: {output_path}")


def plot_accuracy_curves(train_accs, val_accs, output_path):
	"""
	Рисует кривые точности: тренировочной и валидационной.


	Args:
		train_accs (list of float): точность на обучении
		val_accs (list of float): точность на валидации
		output_path (str): путь для сохранения графика
	"""
	plt.figure(figsize=(10, 6))
	epochs = range(1, len(train_accs) + 1)
	plt.plot(epochs, train_accs, 'g-', label='Train Accuracy')
	plt.plot(epochs, val_accs, 'm--', label='Val Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid(True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Кривые точности сохранены: {output_path}")


def extract_top_k_predictions(logits, k=5, tokenizer=None):
	"""
	Извлекает топ‑k предсказанных токенов/классов по логитам.


	Полезно для анализа уверенности модели и интерпретации результатов.

	Args:
		logits (torch.Tensor): логиты модели [batch_size, vocab_size] или [batch_size, n_classes]
		k (int): количество топ‑предсказаний
		tokenizer (PreTrainedTokenizer): токенизатор (опционально, для декодирования токенов)


	Returns:
		tuple: (top_values, top_indices, top_tokens)
			- top_values: вероятности топ‑k [batch_size, k]
			- top_indices: индексы топ‑k [batch_size, k]
			- top_tokens: декодированные токены (если tokenizer задан)
	"""
	probs = torch.softmax(logits, dim=-1)
	top_values, top_indices = torch.topk(probs, k, dim=-1)

	if tokenizer:
		top_tokens = []
		for row in top_indices:
			tokens = [tokenizer.decode([idx]) for idx in row]
			top_tokens.append(tokens)
		return top_values, top_indices, top_tokens
	else:
		return top_values, top_indices, None

def log_predictions_sample(predictions, labels, tokenizer, output_path, max_samples=10):
	"""
	Записывает в файл примеры предсказаний (с истинными метками) для анализа.


	Args:
		predictions (torch.Tensor или list): предсказания модели
		labels (torch.Tensor или list): истинные метки
		tokenizer (PreTrainedTokenizer): токенизатор для декодирования
		output_path (str): путь к файлу лога
		max_samples (int): максимальное число примеров для записи
	"""
	predictions = predictions[:max_samples]
	labels = labels[:max_samples]

	with open(output_path, 'w', encoding='utf-8') as f:
		f.write("Пример предсказаний модели\n")
		f.write("=" * 50 + "\n")
		for i, (pred, label) in enumerate(zip(predictions, labels)):
			if isinstance(pred, torch.Tensor):
				pred = pred.item()
			if isinstance(label, torch.Tensor):
				label = label.item()

			pred_token = tokenizer.decode([pred]) if tokenizer else str(pred)
			label_token = tokenizer.decode([label]) if tokenizer else str(label)


			f.write(f"Пример {i+1}:\n")
			f.write(f"  Предсказано: {pred_token} (id={pred})\n")
			f.write(f"  Истинная метка: {label_token} (id={label})\n")
			f.write("-" * 30 + "\n")

	print(f"Примеры предсказаний записаны: {output_path}")


def compute_confidence_metrics(probs, labels, threshold=0.5):
	"""
	Вычисляет метрики, связанные с уверенностью модели.

	- Доля уверенных правильных предсказаний
	- Доля неуверенных ошибок
	- Средняя уверенность на правильных/ошибочных примерах


	Args:
		probs (torch.Tensor): вероятности предсказаний [n_samples, n_classes]
		labels (torch.Tensor): истинные метки [n_samples]
		threshold (float): порог уверенности (например, 0.5)


	Returns:
		dict: словарь с метриками
	"""
	preds = torch.argmax(probs, dim=1)
	confidences = torch.max(probs, dim=1).values


	correct = (preds == labels)
	conf_correct = confidences[correct]
	conf_incorrect = confidences[~correct]


	# Доля уверенных правильных предсказаний (уверенность > threshold)
	confident_correct = (conf_correct > threshold).float().mean().item()


	# Доля неуверенных ошибок (уверенность <= threshold)
	unconfident_errors = (conf_incorrect <= threshold).float().mean().item()

	# Средняя уверенность на правильных примерах
	mean_conf_correct = conf_correct.mean().item() if len(conf_correct) > 0 else 0.0
	# Средняя уверенность на ошибочных примерах
	mean_conf_incorrect = conf_incorrect.mean().item() if len(conf_incorrect) > 0 else 0.0


	return {
		'confident_correct': confident_correct,
		'unconfident_errors': unconfident_errors,
		'mean_conf_correct': mean_conf_correct,
		'mean_conf_incorrect': mean_conf_incorrect,
		'accuracy': correct.float().mean().item()
	}


def analyze_prediction_uncertainty(probs, method='entropy'):
	"""
	Анализирует неопределённость предсказаний модели.

	Поддерживает несколько методов:
	  - entropy: энтропия распределения вероятностей
	  - variance: дисперсия вероятностей классов
	  - margin: разница между топ‑1 и топ‑2 вероятностями


	Args:
		probs (torch.Tensor): вероятности предсказаний [n_samples, n_classes]
		method (str): метод анализа ('entropy', 'variance', 'margin')

	Returns:
		torch.Tensor: скалярные значения неопределённости для каждого примера
	"""
	if method == 'entropy':
		# Энтропия: -sum(p * log(p))
		eps = 1e-10
		entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
		return entropy
	elif method == 'variance':
		# Дисперсия по классам
		variance = torch.var(probs, dim=1)
		return variance
	elif method == 'margin':
		# Разница между топ‑1 и топ‑2 вероятностями
		top2_probs = torch.topk(probs, 2, dim=1).values
		margin = top2_probs[:, 0] - top2_probs[:, 1]
		return 1 - margin  # Чем меньше margin, тем выше неопределённость
	else:
		raise ValueError(f"Неизвестный метод: {method}")


def detect_out_of_distribution(probs, threshold=0.1):
	"""
	Выявляет примеры вне распределения (out‑of‑distribution, OOD).

	Критерии: низкая уверенность (max prob < threshold) или высокая энтропия.


	Args:
		probs (torch.Tensor): вероятности предсказаний [n_samples, n_classes]
		threshold (float): порог уверенности для OOD

	Returns:
		torch.BoolTensor: маска OOD‑примеров [n_samples]
	"""
	max_probs = probs.max(dim=1).values
	entropy = analyze_prediction_uncertainty(probs, method='entropy')


	# Критерии OOD: низкая уверенность ИЛИ высокая энтропия
	ood_mask = (max_probs < threshold) | (entropy > 2.0)  # эвристика
	return ood_mask

def evaluate_calibration(probs, labels, n_bins=10):
	"""
	Оценивает калибровку модели (соответствие уверенности и точности).

	Вычисляет Expected Calibration Error (ECE) и рисует диаграмму надёжности.


	Args:
		probs (torch.Tensor): вероятности предсказаний [n_samples, n_classes]
		labels (torch.Tensor): истинные метки [n_samples]
		n_bins (int): число бинов для гистограммы


	Returns:
		dict: {'ece': ECE, 'bin_accuracies': ..., 'bin_confidences': ...}
	"""
	preds = torch.argmax(probs, dim=1)
	confidences = torch.max(probs, dim=1).values

	correct = (preds == labels).float()


	# Разбиваем на бины по уверенности
	bin_boundaries = torch.linspace(0, 1, n_bins + 1)
	bin_lows = bin_boundaries[:-1]
	bin_highs = bin_boundaries[1:]


	bin_accuracies = []
	bin_confidences = []
	bin_counts = []

	for low, high in zip(bin_lows, bin_highs):
		mask = (confidences >= low) & (confidences < high)
		if mask.sum() > 0:
			acc = correct[mask].mean().item()
			conf = confidences[mask].mean().item()
			count = mask.sum().item()
		else:
			acc, conf, count = 0.0, 0.0, 0
		bin_accuracies.append(acc)
		bin_confidences.append(conf)
		bin_counts.append(count)


	# ECE: взвешенное среднее |accuracy - confidence|
	total_count = sum(bin_counts)
	ece = sum(
		abs(acc - conf) * count for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
	) / total_count if total_count > 0 else 0.0


	return {
		'ece': ece,
		'bin_accuracies': bin_accuracies,
		'bin_confidences': bin_confidences,
		'bin_counts': bin_counts
	}

def plot_reliability_diagram(bin_accuracies, bin_confidences, output_path):
	"""
	Рисует диаграмму надёжности (reliability diagram) для оценки калибровки.

	Args:
		bin_accuracies (list of float): точность в бинах
		bin_confidences (list of float): уверенность в бинах
		output_path (str): путь для сохранения графика
	"""
	plt.figure(figsize=(8, 8))
	plt.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7, label='Accuracy')
	plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
	plt.xlabel('Confidence')
	plt.ylabel('Accuracy')
	plt.title('Reliability Diagram')
	plt.legend()
	plt.grid(True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Диаграмма надёжности сохранена: {output_path}")


def compute_shap_values(model, background_data, test_data, feature_names=None):
	"""
	Вычисляет SHAP‑значения для интерпретации предсказаний модели.

	Требует установки: pip install shap

	Args:
		model (nn.Module): модель
		background_data (torch.Tensor): фоновые данные для барицентра
		test_data (torch.Tensor): тестовые данные для объяснения
		featurenames (list of str): имена признаков (опционально)


	Returns:
		np.ndarray: SHAP‑значения [n_test, n_features]
	"""
	try:
		import shap

		# Конвертируем в numpy
		background = background_data.cpu().numpy()
		test = test_data.cpu().numpy()

		# Создаём explainer
		explainer = shap.DeepExplainer(model, background)
		shap_values = explainer.shap_values(test)

		return shap_values
	except ImportError:
		print("Установите shap: pip install shap")
		return None

def log_model_metadata(model, optimizer, epoch, metrics, output_path):
	"""
	Сохраняет метаданные модели (гиперпараметры, метрики, состояние) в JSON.

	Полезно для отслеживания экспериментов.


	Args:
		model (nn.Module): модель
		optimizer (torch.optim.Optimizer): оптимизатор
		epoch (int): текущая эпоха
		metrics (dict): текущие метрики
		output_path (str): путь к JSON‑файлу
	"""
		metadata = {
			'model_name': model.__class__.__name__,
			'model_config': str(model),
			'optimizer_name': optimizer.__class__.__name__,
			'optimizer_state': optimizer.state_dict(),
			'epoch': epoch,
			'metrics': metrics,
			'timestamp': datetime.now().isoformat(),
			'torch_version': torch.__version__,
			'device': str(next(model.parameters()).device)
		}

		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(metadata, f, ensure_ascii=False, indent=2)
		print(f"Метаданные модели сохранены: {output_path}")


	def load_model_metadata(input_path):
		"""
		Загружает метаданные модели из JSON‑файла.


		Args:
			input_path (str): путь к JSON‑файлу с метаданными


		Returns:
			dict: словарь с метаданными модели
		"""
		with open(input_path, 'r', encoding='utf-8') as f:
			metadata = json.load(f)
		print(f"Метаданные модели загружены: {input_path}")
		return metadata

	def freeze_model_layers(model, layer_names):
		"""
		Замораживает указанные слои модели (отключает градиенты).


		Полезно для трансферного обучения: заморозить ранние слои, обучать только новые.


		Args:
			model (nn.Module): модель
			layer_names (list of str): имена замораживаемых слоёв
		"""
		for name, param in model.named_parameters():
			if any(ln in name for ln in layer_names):
				param.requires_grad = False
		print(f"Заморожены слои: {layer_names}")


	def unfreeze_model_layers(model, layer_names=None):
		"""
		Размораживает слои модели (включает градиенты).

		Если layer_names не указан — размораживает все слои.

		Args:
			model (nn.Module): модель
			layer_names (list of str, optional): имена размораживаемых слоёв
		"""
		if layer_names is None:
			for param in model.parameters():
				param.requires_grad = True
			print("Все слои разморожены")
		else:
			for name, param in model.named_parameters():
				if any(ln in name for ln in layer_names):
					param.requires_grad = True
			print(f"Разморожены слои: {layer_names}")


	def count_model_params_by_layer(model):
		"""
		Считает количество параметров по слоям модели.

		Помогает анализировать архитектуру и выявлять «тяжёлые» слои.


		Args:
			model (nn.Module): модель

		Returns:
			dict: {имя_слоя: число_параметров}
		"""
		param_counts = {}
		for name, module in model.named_modules():
			if len(list(module.parameters())) > 0:  # если у модуля есть параметры
				count = sum(p.numel() for p in module.parameters() if p.requires_grad)
				param_counts[name] = count
		return param_counts


	def print_model_param_summary(model):
		"""
		Печатает сводку по параметрам модели (по слоям).

		Args:
			model (nn.Module): модель
		"""
		param_counts = count_model_params_by_layer(model)
		total = sum(param_counts.values())

		print("Сводка по параметрам модели:")
		print("-" * 60)
		for name, count in param_counts.items():
			print(f"{name:30} : {count:>12} params")
		print("-" * 60)
		print(f"Итого: {total} параметров")


	def save_checkpoint(model, optimizer, epoch, metrics, output_path):
		"""
		Сохраняет чекпойнт модели (веса + состояние оптимизатора + метаданные).


		Args:
			model (nn.Module): модель
			optimizer (torch.optim.Optimizer): оптимизатор
			epoch (int): номер эпохи
			metrics (dict): метрики
			output_path (str): путь для сохранения чекпойнта (.pth)
		"""
		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
			'metrics': metrics,
			'timestamp': datetime.now().isoformat()
		}
		torch.save(checkpoint, output_path)
		print(f"Чекпойнт сохранён: {output_path}")


	def load_checkpoint(model, optimizer, input_path, device='cpu'):
		"""
		Загружает чекпойнт модели и восстанавливает состояние.


		Args:
			model (nn.Module): модель (будет обновлена)
			optimizer (torch.optim.Optimizer): оптимизатор (будет обновлен)
			input_path (str): путь к чекпойнту (.pth)
			device (str): устройство для загрузки


		Returns:
			int: номер эпохи, на которой был сохранён чекпойнт
		"""
		checkpoint = torch.load(input_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		metrics = checkpoint['metrics']

		print(f"Чекпойнт загружен: {input_path} (эпоха {epoch}, метрики: {metrics})")
		return epoch

	def set_random_seed(seed):
		"""
		Устанавливает случайный seed для воспроизводимости экспериментов.


		Задаёт seed для:
		  - Python random
		  - NumPy
		  - PyTorch (CPU и CUDA)

		Args:
			seed (int): значение seed
		"""
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		print(f"Random seed установлен: {seed}")




	def create_learning_rate_scheduler(optimizer, scheduler_type='step', **kwargs):
	"""
	Создаёт scheduler для изменения learning rate.


	Поддерживаемые типы:
	  - 'step': StepLR (уменьшение LR каждые N эпох)
	  - 'cosine': CosineAnnealingLR
	  - 'plateau': ReduceLROnPlateau (по метрике валидации)
	  - 'exponential': ExponentialLR

	Args:
		optimizer (torch.optim.Optimizer): оптимизатор
		scheduler_type (str): тип scheduler-а
		**kwargs: дополнительные параметры для scheduler-а

	Returns:
		torch.optim.lr_scheduler: экземпляр scheduler-а
	"""
	if scheduler_type == 'step':
		step_size = kwargs.get('step_size', 30)
		gamma = kwargs.get('gamma', 0.1)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
	elif scheduler_type == 'cosine':
		T_max = kwargs.get('T_max', 50)
		eta_min = kwargs.get('eta_min', 1e-6)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
	elif scheduler_type == 'plateau':
		monitor = kwargs.get('monitor', 'val_loss')
		factor = kwargs.get('factor', 0.5)
		patience = kwargs.get('patience', 5)
		threshold = kwargs.get('threshold', 1e-4)
		min_lr = kwargs.get('min_lr', 1e-8)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min' if 'loss' in monitor else 'max',
			factor=factor, patience=patience, threshold=threshold, min_lr=min_lr
		)
	elif scheduler_type == 'exponential':
		gamma = kwargs.get('gamma', 0.95)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
	else:
		raise ValueError(f"Неизвестный тип scheduler: {scheduler_type}")


	return scheduler

def apply_gradient_clipping(model, max_norm=1.0):
	"""
	Применяет обрезку градиентов (gradient clipping) для стабилизации обучения.


	Обрезает градиенты так, чтобы их норма L2 не превышала max_norm.


	Args:
		model (nn.Module): модель
		max_norm (float): максимальная норма градиентов
	"""
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
	print(f"Gradient clipping применён (max_norm={max_norm})")


def compute_gradient_norm(model):
	"""
	Вычисляет норму градиентов модели (L2).


	Полезно для мониторинга обучения и выявления проблем (взрывные градиенты).


	Args:
		model (nn.Module): модель

	Returns:
		float: норма градиентов
	"""
	total_norm = 0.0
	for p in model.parameters():
		if p.grad is not None:
			param_norm = p.grad.data.norm(2)
			total_norm += param_norm.item() ** 2
	total_norm = total_norm ** 0.5
	return total_norm


def initialize_weights(model, init_type='xavier'):
	"""
	Инициализирует веса модели заданным методом.

	Поддерживает:
	  - 'xavier': Xavier/Glorot (для ReLU, Tanh)
	  - 'kaiming': Kaiming/He (для ReLU)
	  - 'normal': нормальное распределение N(0, 0.02)
	  - 'uniform': равномерное распределение U(-0.05, 0.05)


	Args:
		model (nn.Module): модель
		init_type (str): метод инициализации
	"""
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
			if init_type == 'xavier':
				torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
			elif init_type == 'kaiming':
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
			elif init_type == 'normal':
				torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
			elif init_type == 'uniform':
				torch.nn.init.uniform_(m.weight.data, -0.05, 0.05)
			else:
				raise ValueError(f"Неизвестный метод инициализации: {init_type}")
			if hasattr(m, 'bias') and m.bias is not None:
				torch.nn.init.constant_(m.bias.data, 0.0)


	model.apply(init_func)
	print(f"Веса инициализированы методом: {init_type}")

def count_flops(model, input_tensor):
	"""
	Оценивает количество операций (FLOPs) модели на заданном входе.

	Требует установки: pip install thop

	Args:
		model (nn.Module): модель
		input_tensor (torch.Tensor): входной тензор (без batch_size)

	Returns:
		int: приблизительное число FLOPs
	"""
	try:
		import thop
		flops, params = thop.profile(model, inputs=(input_tensor.unsqueeze(0),), verbose=False)
		return flops
	except ImportError:
		print("Установите thop: pip install thop")
		return None

def profile_model(model, example_input, device='cuda'):
	"""
	Профилирует модель (время исполнения, потребление памяти).

	Использует torch.profiler.

	Args:
		model (nn.Module): модель
		example_input (torch.Tensor): пример входного тензора
		device (str): устройство ('cuda' или 'cpu')
	"""
	model.to(device)
	exampleinput = exampleinput.to(device)

	with torch.profiler.profile(
		activities=[
			torch.profiler.ProfilerActivity.CPU,
			torch.profiler.ProfilerActivity.CUDA
		],
		record_shapes=True,
		profile_memory=True,
		with_stack=True
	) as prof:
		_ = model(exampleinput)

	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def convert_model_to_half(model):
	"""
	Конвертирует модель в float16 (half precision) для ускорения и экономии памяти.

	Внимание: может снизить точность!

	Args:
		model (nn.Module): модель

	Returns:
		nn.Module: модель в float16
	"""
	model.half()
	print("Модель конвертирована в float16 (half precision)")
	return model

def move_model_to_device(model, device):
	"""
	Переносит модель на заданное устройство.

	Args:
		model (nn.Module): модель
		device (torch.device или str): устройство ('cpu', 'cuda', 'cuda:0' и т.п.)

	Returns:
		nn.Module: модель на новом устройстве
	"""
	device = torch.device(device)
	model.to(device)
	print(f"Модель перемещена на устройство: {device}")
	return model

def get_model_device(model):
	"""
	Возвращает устройство, на котором находится модель.

	Args:
		model (nn.Module): модель

	Returns:
		torch.device: устройство модели
	"""
	return next(model.parameters()).device

def check_model_consistency(model, example_input):
	"""
	Проверяет работоспособность модели на примере входа.


	Выполняет forward-проход и проверяет, что нет ошибок.


	Args:
		model (nn.Module): модель
		exampleinput (torch.Tensor): пример входного тензора
		"""
	try:
		with torch.no_grad():
			output = model(example_input)
		print("Модель успешно выполнила forward-проход на тестовом входе.")
		return True
	except Exception as e:
		print(f"Ошибка при выполнении forward-прохода: {e}")
		return False

def freeze_batchnorm_stats(model):
	"""
	Замораживает статистику BatchNorm (mean/var) — отключает обновление во время обучения.

	Полезно при трансферном обучении, чтобы не «сбить» предобученную нормализацию.

	Args:
		model (nn.Module): модель
	"""
	for module in model.modules():
		if isinstance(module, torch.nn.BatchNorm2d) or \
		   isinstance(module, torch.nn.BatchNorm1d) or \
		   isinstance(module, torch.nn.BatchNorm3d):
			module.eval()  # переводит в режим inference (не обновляет running_mean/var)
	print("Статистика BatchNorm заморожена (режим eval)")


def unfreeze_batchnorm(model):
	"""
	Размораживает BatchNorm — возвращает в режим обучения (обновляет статистику).


	Args:
		model (nn.Module): модель
	"""
	model.train()  # гарантирует, что все модули в режиме train
	for module in model.modules():
		if isinstance(module, torch.nn.BatchNorm2d) or \
		   isinstance(module, torch.nn.BatchNorm1d) or \
		   isinstance(module, torch.nn.BatchNorm3d):
			module.train()  # явно ставим train, чтобы обновлялись running_mean/var
	print("BatchNorm разморожен (режим train)")


def get_trainable_params(model):
	"""
	Возвращает список параметров, для которых требуется градиент (trainable).

	Args:
		model (nn.Module): модель

	Returns:
		list of torch.nn.Parameter: список обучаемых параметров
	"""
	return [p for p in model.parameters() if p.requires_grad]

def get_frozen_params(model):
	"""
	Возвращает список замороженных параметров (без градиента).

	Args:
		model (nn.Module): модель

	Returns:
		list of torch.nn.Parameter: список замороженных параметров
	"""
	return [p for p in model.parameters() if not p.requires_grad]

def count_trainable_params(model):
	"""
	Считает количество обучаемых параметров модели.

	Args:
		model (nn.Module): модель

	Returns:
		int: число обучаемых параметров
	"""
	return sum(p.numel() for p in get_trainable_params(model))


def count_frozen_params(model):
	"""
	Считает количество замороженных параметров модели.

	Args:
		model (nn.Module): модель

	Returns:
		int: число замороженных параметров
	"""
	return sum(p.numel() for p in get_frozen_params(model))

def print_trainable_status(model):
	"""
	Печатает статус параметров модели (сколько обучаемых / замороженных).

	Args:
		model (nn.Module): модель
	"""
	trainable = count_trainable_params(model)
	frozen = count_frozen_params(model)
	total = trainable + frozen

	print(f"Статус параметров модели:")
	print(f"  Обучаемые параметры: {trainable} ({trainable/total:.1%})")
	print(f"  Замороженные параметры: {frozen} ({frozen/total:.1%})")
	print(f"  Всего параметров: {total}")


def apply_weight_decay(model, weight_decay, exclude_names=None):
	"""
	Применяет weight decay (L2‑регуляризацию) к параметрам модели.

	Можно исключить некоторые параметры (например, bias, BatchNorm).


	Args:
		model (nn.Module): модель
		weight_decay (float): коэффициент L2‑регуляризации
		exclude_names (list of str): имена параметров, к которым не применять (например, ['bias', 'batch_norm'])
	"""
	if exclude_names is None:
		exclude_names = ['bias', 'running_mean', 'running_var', 'num_batches_tracked']


	params_to_decay = []
	params_no_decay = []


	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if any(ex in name for ex in exclude_names):
			params_no_decay.append(param)
		else:
			params_to_decay.append(param)

	# В оптимизаторе нужно будет использовать разные группы
	# Пример:
	# optim = torch.optim.AdamW([
	#     {'params': params_to_decay, 'weight_decay': weight_decay},
	#     {'params': params_no_decay, 'weight_decay': 0.0}
	# ], lr=lr)
	print(f"Weight decay ({weight_decay}) применён к параметрам (исключая {exclude_names})")


def replace_activation(model, old_act, new_act):
	"""
	Заменяет все экземпляры старой активации на новую в модели.


	Пример: заменить ReLU на LeakyReLU.

	Внимание: работает только для модулей, которые являются прямыми детьми model.
	Для глубокой замены нужно рекурсивно проходить по model.modules().

	Args:
		model (nn.Module): модель
		old_act (type): тип старой активации (например, nn.ReLU)
		new_act (nn.Module): новый модуль активации (например, nn.LeakyReLU())
	"""
	for child in model.children():
		if isinstance(child, old_act):
			# Замена модуля
			idx = list(model._modules.keys()).index(child._get_name())
			model._modules[idx] = new_act
	print(f"Заменены активации: {old_act} → {new_act}")

def summary_model_shapes(model, input_shape):
	"""
	Выводит сводку по размерам тензоров на каждом слое (аналог keras.summary).

	Требует установки: pip install torchinfo

	Args:
		model (nn.Module): модель
		input_shape (tuple): форма входного тензора (без batch_size)
	"""
	try:
		from torchinfo import summary
		batch_size = 1
		input_size = (batch_size,) + input_shape
		summary(model, input_size=input_size)
	except ImportError:
		print("Установите torchinfo: pip install torchinfo")



def export_model_to_onnx(model, input_tensor, output_path, input_names=None, output_names=None):
	"""
	Экспортирует модель в формат ONNX.

	ONNX позволяет использовать модель в других фреймворках и на разных платформах.

	Args:
		model (nn.Module): модель
		input_tensor (torch.Tensor): входной тензор (пример данных)
		output_path (str): путь для сохранения ONNX‑файла
		inputnames (list of str, optional): имена входных тензоров
		outputnames (list of str, optional): имена выходных тензоров
	"""
	model.eval()  # переводим в режим inference
	try:
		torch.onnx.export(
			model,
			input_tensor,
			output_path,
			export_params=True,
			opset_version=11,
			do_constant_folding=True,
			input_names=input_names or ['input'],
			output_names=output_names or ['output'],
			dynamic_axes=None  # можно задать для динамических размеров
		)
		print(f"Модель экспортирована в ONNX: {output_path}")
	except Exception as e:
		print(f"Ошибка при экспорте в ONNX: {e}")

def convert_model_to_torchscript(model, example_input, output_path, mode='trace'):
	"""
	Конвертирует модель в TorchScript (для развёртывания без Python).


	Режимы:
	  - 'trace': трассировка (записывает выполнение на примере входа)
	  - 'script': компиляция (анализирует код модели)

	Args:
		model (nn.Module): модель
		exampleinput (torch.Tensor): пример входного тензора
		output_path (str): путь для сохранения TorchScript‑модуля
		mode (str): режим конвертации ('trace' или 'script')
	"""
	model.eval()
	if mode == 'trace':
		traced_script_module = torch.jit.trace(model, exampleinput)
	elif mode == 'script':
		traced_scriptmodule = torch.jit.script(model)
	else:
		raise ValueError(f"Неизвестный режим: {mode}")

	traced_scriptmodule.save(output_path)
	print(f"Модель конвертирована в TorchScript ({mode}) и сохранена: {output_path}")


def quantize_model(model, backend='fbgemm'):
	"""
	Квантует модель (уменьшает битность весов/активаций для ускорения и сжатия).


	Требует, чтобы модель была в режиме eval().

	Поддерживает статическое квантование (для CPU).

	Args:
		model (nn.Module): модель (должна быть в eval())
		backend (str): бэкенд для квантования ('fbgemm' для x86, 'qnnpack' для ARM)


	Returns:
		nn.Module: квантованная модель
	"""
	model_q = torch.quantization.quantize_dynamic(
		model, {torch.nn.Linear}, dtype=torch.qint8, backend=backend
	)
	print(f"Модель квантована (backend={backend})")
	return model_q

def benchmark_model(model, test_loader, device='cuda'):
	"""
	Замеряет скорость и потребление памяти модели на тестовом датасете.


	Args:
		model (nn.Module): модель
		test_loader (DataLoader): загрузчик тестовых данных
		device (str): устройство ('cuda' или 'cpu')


	Returns:
		dict: {'latency_ms': ..., 'memory_mb': ..., 'fps': ...}
	"""
	model.to(device)
	model.eval()

	latencies = []
	memory_allocated = []

	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)

			# Замер времени
			start_event = torch.cuda.Event(enable_timing=True)
			end_event = torch.cuda.Event(enable_timing=True)

			start_event.record()
			_ = model(data)
			end_event.record()
			torch.cuda.synchronize()
			latency = start_event.elapsed_time(end_event)  # мс
			latencies.append(latency)

			# Замер памяти
			memory_allocated.append(torch.cuda.memory_allocated() / 1024**2)  # МБ


	avg_latency = sum(latencies) / len(latencies)
	avg_memory = sum(memory_allocated) / len(memory_allocated)
	fps = 1000 / avg_latency if avg_latency > 0 else 0

	return {
		'latency_ms': avg_latency,
		'memory_mb': avg_memory,
		'fps': fps
	}

def visualize_feature_maps(model, input_tensor, layer_names, output_dir):
	"""
	Визуализирует карты признаков (feature maps) промежуточных слоёв.


	Сохраняет изображения карт признаков в указанную директорию.

	Args:
		model (nn.Module): модель
		input_tensor (torch.Tensor): входной тензор
		layernames (list of str): имена слоёв, для которых извлекать feature maps
		outputdir (str): директория для сохранения изображений
	"""
	from torchvision.utils import make_grid
	import cv2

	# Регистрируем хуки для захвата выходов слоёв
	feature_maps = {}
	hooks = []

	def hook_fn(module, input, output):
		feature_maps[module._get_name()] = output

	for name, module in model.named_modules():
		if name in layernames:
			hooks.append(module.register_forward_hook(hook_fn))

	# Прогоняем forward
	with torch.no_grad():
		_ = model(input_tensor)

	# Удаляем хуки
	for hook in hooks:
		hook.remove()

	# Визуализируем
	for name, fm in feature_maps.items():
		fm = fm.detach().cpu()
		# Берём первые N карт признаков (например, 16)
		fm = fm[0, :16].unsqueeze(1)  # [16, 1, H, W]
		grid = make_grid(fm, nrow=4, normalize=True, scale_each=True)
		grid = grid.permute(1, 2, 0).numpy()  # HWC
		grid = (grid * 255).astype(np.uint8)

		cv2.imwrite(f"{output_dir}/feature_map_{name}.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
	print(f"Карты признаков сохранены в: {output_dir}")


def compute_activation_statistics(model, data_loader, device='cuda'):
	"""
	Вычисляет статистику активаций (среднее, дисперсию) по датасету.

	Полезно для анализа «мёртвых нейронов» и распределения активаций.

	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		device (str): устройство

	Returns:
		dict: {имя_слоя: {'mean': ..., 'var': ...}}
	"""
	activation_stats = {}

	def hook_fn(module, input, output):
		act = output.detach()
		mean = act.mean().item()
		var = act.var().item()
		name = module._get_name()
		if name not in activation_stats:
			activation_stats[name] = {'mean': [], 'var': []}
		activation_stats[name]['mean'].append(mean)
		activation_stats[name]['var'].append(var)

	# Регистрируем хуки на все модули с активациями
	hooks = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.ReLU) or \
		   isinstance(module, torch.nn.LeakyReLU) or \
		   isinstance(module, torch.nn.ELU) or \
		   isinstance(module, torch.nn.PReLU):
			hooks.append(module.register_forward_hook(hook_fn))


	model.to(device)
	model.eval()

	with torch.no_grad():
		for data, _ in data_loader:
			data = data.to(device
			model(data)


	# Удаляем хуки
	for hook in hooks:
		hook.remove()

	# Вычисляем средние значения по всем батчам
	for name, stats in activation_stats.items():
		stats['mean'] = np.mean(stats['mean'])
		stats['var'] = np.mean(stats['var'])

	print("Статистика активаций вычислена.")
	return activation_stats

def analyze_gradient_flow(model, data_loader, device='cuda'):
	"""
	Анализирует поток градиентов по слоям (gradient flow).


	Помогает выявить проблемы: исчезающие/взрывные градиенты.

	Замеряет норму градиентов на каждом слое во время обучения.

	"""
	model.to(device)
	model.train()

	layer_grad_norms = {}

	def hook_fn(module, grad_input, grad_output):
		if grad_output[0] is not None:
			norm = grad_output[0].norm().item()
			name = module._get_name()
			if name not in layer_grad_norms:
				layer_grad_norms[name] = []
			layer_grad_norms[name].append(norm)


	hooks = []
	for name, module in model.named_modules():
		if hasattr(module, 'weight') and module.weight is not None:
			hooks.append(module.register_backward_hook(hook_fn))


	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


	for data, target in data_loader:
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss = torch.nn.functional.cross_entropy(output, target)
		loss.backward()
		optimizer.step()

		# После backward собираем нормы
		# (они уже записаны в layer_grad_norms через хуки)


	# Удаляем хуки
	for hook in hooks:
		hook.remove()

	# Усредняем по батчам
	for name in layer_grad_norms:
		layer_grad_norms[name] = np.mean(layer_grad_norms[name])

	print("Анализ потока градиентов завершён.")
	return layer_grad_norms


def detect_dead_neurons(activation_stats, threshold=1e-6):
	"""
	Выявляет «мёртвые» нейроны (с почти нулевой активацией).


	Считает нейрон мёртвым, если среднее значение активации < threshold.


	Args:
		activation_stats (dict): статистика активаций из compute_activation_statistics
		threshold (float): порог для определения мёртвого нейрона


	Returns:
		dict: {имя_слоя: доля_мёртвых_нейронов}
	"""
	dead_ratio = {}
	for name, stats in activation_stats.items():
		mean_act = stats['mean']
		ratio = 1.0 if mean_act < threshold else 0.0
		dead_ratio[name] = ratio
	print("Анализ мёртвых нейронов завершён.")
	return dead_ratio

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
	"""
	Рисует кривые обучения (loss и accuracy).


	Args:
		train_losses (list): потери на обучении
		val_losses (list): потери на валидации
		train_accs (list): точность на обучении
		val_accs (list): точность на валидации
		output_path (str): путь для сохранения графика
	"""
	epochs = len(train_losses)

	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)
	plt.plot(range(epochs), train_losses, label='Train Loss')
	plt.plot(range(epochs), val_losses, label='Val Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss Curve')
	plt.legend()


	plt.subplot(1, 2, 2)
	plt.plot(range(epochs), train_accs, label='Train Acc')
	plt.plot(range(epochs), val_accs, label='Val Acc')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Curve')
	plt.legend()


	plt.savefig(output_path)
	plt.close()
	print(f"Кривые обучения сохранены: {output_path}")


def save_predictions(model, data_loader, output_path, device='cuda'):
	"""
	Сохраняет предсказания модели на датасете.

	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		output_path (str): путь для сохранения предсказаний (.npy)
		device (str): устройство
	"""
	model.to(device)
	model.eval()

	all_preds = []
	all_targets = []

	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			pred = output.argmax(dim=1, keepdim=True)
			all_preds.append(pred.cpu().numpy())
			all_targets.append(target.cpu().numpy())

	all_preds = np.concatenate(all_preds)
	all_targets = np.concatenate(all_targets)

	np.savez(output_path, predictions=all_preds, targets=all_targets)
	print(f"Предсказания сохранены: {output_path}")

def compute_confusion_matrix(model, data_loader, num_classes, device='cuda'):
	"""
	Вычисляет матрицу ошибок (confusion matrix).

	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		num_classes (int): число классов
		device (str): устройство

	Returns:
		np.ndarray: матрица ошибок (num_classes, num_classes)
	"""
	model.to(device)
	model.eval()

	confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			pred = output.argmax(dim=1)
			for t, p in zip(target.view(-1), pred.view(-1)):
				confusion_mat[t.long(), p.long()] += 1

	print("Матрица ошибок вычислена.")
	return confusion_mat

def plot_confusion_matrix(confusion_mat, class_names, output_path):
	"""
	Рисует матрицу ошибок.

	Args:
		confusion_mat (np.ndarray): матрица ошибок
		classnames (list of str): имена классов
		output_path (str): путь для сохранения изображения
	"""
	import seaborn as sns

	plt.figure(figsize=(10, 8))
	sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.savefig(output_path)
	plt.close()
	print(f"Матрица ошибок сохранена: {output_path}")




def evaluate_model_robustness(model, test_loader, device='cuda', noise_levels=None):
	"""
	Оценивает устойчивость модели к шуму во входных данных.

	Добавляет гауссовский шум разного уровня и замеряет точность.

	Args:
		model (nn.Module): модель
		test_loader (DataLoader): загрузчик тестовых данных
		device (str): устройство
		noise_levels (list of float): уровни шума (std) для тестирования

	Returns:
		dict: {noise_level: accuracy}
	"""
	if noise_levels is None:
		noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

	model.to(device)
	model.eval()
	results = {}

	with torch.no_grad():
		for noise_std in noise_levels:
			correct = 0
			total = 0

			for data, target in test_loader:
				data, target = data.to(device), target.to(device)

				# Добавляем шум
				if noise_std > 0:
					noise = torch.randn_like(data) * noise_std
					data = data + noise

				output = model(data)
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(target.view_as(pred)).sum().item()
				total += target.size(0)

			accuracy = correct / total
			results[noise_std] = accuracy
			print(f"Noise std={noise_std}: Accuracy = {accuracy:.4f}")

	return results


def compute_feature_importance(model, data_loader, target_class, device='cuda'):
	"""
	Вычисляет важность признаков (feature importance) методом градиентов.

	Использует градиенты выхода по входу (saliency maps).


	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		target_class (int): класс, для которого вычислять важность
		device (str): устройство

	Returns:
		torch.Tensor: усреднённая карта важности (форма как у входа)
	"""
	model.to(device)
	model.eval()

	total_saliency = None
	count = 0

	for data, target in data_loader:
		data, target = data.to(device), target.to(device)

		# Фильтруем по классу
		mask = (target == target_class)
		if mask.sum() == 0:
			continue

		data = data[mask]
		if data.size(0) == 0:
			continue

		data.requires_grad = True
		optimizer = torch.optim.SGD([data], lr=0)

		output = model(data)
		loss = -output[:, target_class].sum()  # градиент в сторону увеличения класса


		optimizer.zero_grad()
		loss.backward()

		saliency = data.grad.data.abs()
		if total_saliency is None:
			total_saliency = saliency.sum(dim=0, keepdim=True)
		else:
			total_saliency += saliency.sum(dim=packed_dim=0, keepdim=True)

		count += data.size(0)

	if count > 0:
		total_saliency /= count
	else:
		total_saliency = torch.zeros_like(data[0:1])

	print(f"Важность признаков вычислена для класса {target_class}.")
	return total_saliency.cpu()


def plot_feature_importance(saliency_map, input_shape, output_path):
	"""
	Рисует карту важности признаков (saliency map).


	Args:
		saliency_map (torch.Tensor): карта важности (C, H, W)
		input_shape (tuple): исходная форма входа (H, W) для ресайза
		output_path (str): путь для сохранения изображения
	"""
	import cv2

	# Конвертируем в numpy и нормализуем
	saliency = saliency_map.numpy()
	saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
	saliency = (saliency * 255).astype(np.uint8)


	# Если много каналов — берём максимум по каналам
	if saliency.shape[0] > 1:
		saliency = saliency.max(axis=0, keepdims=True)


	# Ресайз до исходной формы
	saliency = cv2.resize(saliency[0], input_shape, interpolation=cv2.INTER_LINEAR)
	saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)


	cv2.imwrite(output_path, saliency)
	print(f"Карта важности сохранена: {output_path}")


def extract_embeddings(model, data_loader, layer_name, device='cuda'):
	"""
	Извлекает эмбеддинги (представления) из указанного слоя.


	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		layer_name (str): имя слоя, из которого брать выход
		device (str): устройство

	Returns:
		np.ndarray: эмбеддинги (N, D)
		np.ndarray: метки (N,)
	"""
	model.to(device)
	model.eval()

	embeddings = []
	labels = []

	def hook_fn(module, input, output):
		embeddings.append(output.detach().cpu().numpy())

	# Находим слой и ставим хук
	target_module = None
	for name, module in model.named_modules():
		if name == layer_name:
			target_module = module
			break

	if target_module is None:
		raise ValueError(f"Слой не найден: {layer_name}")

	hook = target_module.register_forward_hook(hook_fn)

	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			model(data)
			labels.append(target.cpu().numpy())

	hook.remove()

	embeddings = np.concatenate(embeddings, axis=0)
	labels = np.concatenate(labels, axis=0)

	print(f"Эмбеддинги извлечены из слоя {layer_name}: {embeddings.shape}")
	return embeddings, labels

def analyze_learning_rate_impact(model_fn, data_loaders, lr_list, epochs, device='cuda'):
	"""
	Анализирует влияние learning rate на обучение.

	Обучает модель с разными LR и сравнивает кривые потерь.


	Args:
		model_fn (callable): функция, возвращающая модель
		data_loaders (dict): {'train': train_loader, 'val': val_loader}
		lr_list (list of float): список LR для тестирования
		epochs (int): число эпох
		device (str): устройство

	Returns:
		dict: {lr: {'train_loss': [...], 'val_loss': [...]}}
	"""
	results = {}

	for lr in lr_list:
		print(f"\nОбучение с LR={lr}")
		model = modelfn().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)

		train_losses = []
		val_losses = []

		for epoch in range(epochs):
			# Обучение
			model.train()
			train_loss = 0.0
			for data, target in data_loaders['train']:
				data, target = data.to(device), target.to(device)
				optimizer.zero_grad()
				output = model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
			train_loss /= len(data_loaders['train'])
			train_losses.

			train_losses.append(train_loss)


			# Валидация
			model.eval()
			val_loss = 0.0
			with torch.no_grad():
				for data, target in data_loaders['val']:
					data, target = data.to(device), target.to(device)
					output = model(data)
					loss = torch.nn.functional.cross_entropy(output, target)
					val_loss += loss.item()
			val_loss /= len(data_loaders['val'])
			val_losses.append(val_loss)

			scheduler.step()

			print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

		results[lr] = {
			'train_loss': train_losses,
			'val_loss': val_losses
		}

	print("Анализ влияния LR завершён.")
	return results




def plot_lr_analysis(results, output_path):
	"""
	Рисует графики для анализа влияния learning rate.

	Args:
		results (dict): результаты из analyze_learning_rate_impact
		output_path (str): путь для сохранения графика
	"""
	plt.figure(figsize=(12, 8))

	for lr, losses in results.items():
		epochs = len(losses['train_loss'])
		plt.plot(range(epochs), losses['train_loss'], label=f'Train LR={lr}')
		plt.plot(range(epochs), losses['val_loss'], '--', label=f'Val LR={lr}')


	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Learning Rate Impact Analysis')
	plt.legend()
	plt.grid(True)
	plt.savefig(output_path)
	plt.close()
	print(f"Графики анализа LR сохранены: {output_path}")

def compute_model_complexity(model, input_size):
	"""
	Оценивает сложность модели (число параметров, FLOPs, память).


	Требует: pip install thop


	Args:
		model (nn.Module): модель
		input_size (tuple): размер входного тензора (C, H, W)

	Returns:
		dict: {'params': ..., 'flops': ..., 'memory_mb': ...}
	"""
	try:
		import thop
		input_tensor = torch.randn(1, *input_size).to(next(model.parameters()).device)
		flops, params = thop.profile(model, inputs=(input_tensor,), verbose=False)


		# Оценка памяти (грубо)
		memory_mb = params * 4 / (1024 ** 2)  # 4 байта на параметр (float32)


		return {
			'params': params,
			'flops': flops,
			'memory_mb': memory_mb
		}
	except ImportError:
		print("Установите thop: pip install thop")
		return None

def print_model_complexity_summary(model, input_size):
	"""
	Печатает сводку по сложности модели.

	Args:
		model (nn.Module): модель
		input_size (tuple): размер входа (C, H, W)
	"""
	complexity = compute_model_complexity(model, input_size)
	if complexity:
		print("Сводка по сложности модели:")
		print(f"  Число параметров: {complexity['params']:,}")
		print(f"  FLOPs: {complexity['flops']:,}")
		print(f"  Память (оценка): {complexity['memory_mb']:.2f} МБ")


def compare_models_complexity(models, model_names, input_size):
	"""
	Сравнивает сложность нескольких моделей.

	Args:
		models (list of nn.Module): список моделей
		model_names (list of str): имена моделей
		input_size (tuple): размер входа (C, H, W)
	"""
	print("Сравнение сложности моделей:")
	print("-" * 60)
	for name, model in zip(model_names, models):
		complexity = compute_model_complexity(model, input_size)
		if complexity:
			print(f"{name}:")
			print(f"  Параметры: {complexity['params']:,}")
			print(f"  FLOPs: {complexity['flops']:,}")
			print(f"  Память: {complexity['memory_mb']:.2f} МБ")
			print("-" * 40)

def save_model_with_metadata(model, optimizer, epoch, metrics, output_path, extra_info=None):
	"""
	Сохраняет модель вместе с метаданными (оптимизатор, эпоха, метрики).


	Args:
		model (nn.Module): модель
		optimizer (torch.optim.Optimizer): оптимизатор
		epoch (int): номер эпохи
		metrics (dict): метрики обучения
		output_path (str): путь для сохранения
		extra_info (dict, optional): дополнительные данные
	"""
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
		'metrics': metrics,
		'timestamp': datetime.now().isoformat()
	}
	if extra_info:
		checkpoint['extra_info'] = extra_info

	torch.save(checkpoint, output_path)
	print(f"Чекпоинт модели сохранён: {output_path}")

def load_model_with_metadata(input_path, model, optimizer=None):
	"""
	Загружает модель с метаданными.

	Args:
		input_path (str): путь к чекпоинту
		model (nn.Module): модель (для загрузки весов)
		optimizer (torch.optim.Optimizer, optional): оптимизатор (если нужно восстановить состояние)

	Returns:
		dict: метаданные чекпоинта
	"""
	checkpoint = torch.load(input_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer and 'optimizer_state_dict' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	print(f"Чекпоинт загружен: {input_path}")
	return checkpoint




def setup_learning_rate_scheduler(optimizer, scheduler_type='step', **kwargs):
	"""
	Создаёт планировщик изменения learning rate.


	Args:
		optimizer (torch.optim.Optimizer): оптимизатор
		scheduler_type (str): тип планировщика
			- 'step': StepLR
			- 'multistep': MultiStepLR
			- 'exponential': ExponentialLR
			- 'plateau': ReduceLROnPlateau
			- 'cyclic': CyclicLR
		**kwargs: параметры для планировщика


	Returns:
		torch.optim.lr_scheduler._LRScheduler: объект планировщика
	"""
	if scheduler_type == 'step':
		return torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=kwargs.get('step_size', 30),
			gamma=kwargs.get('gamma', 0.1)
		)
	elif scheduler_type == 'multistep':
		return torch.optim.lr_scheduler.MultiStepLR(
			optimizer,
			milestones=kwargs.get('milestones', [30, 60, 90]),
			gamma=kwargs.get('gamma', 0.1)
		)
	elif scheduler_type == 'exponential':
		return torch.optim.lr_scheduler.ExponentialLR(
			optimizer,
			gamma=kwargs.get('gamma', 0.99)
		)
	elif scheduler_type == 'plateau':
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode=kwargs.get('mode', 'min'),
			factor=kwargs.get('factor', 0.1),
			patience=kwargs.get('patience', 10),
			verbose=True
		)
	elif scheduler_type == 'cyclic':
		return torch.optim.lr_scheduler.CyclicLR(
			optimizer,
			base_lr=kwargs.get('base_lr', 0.001),
			max_lr=kwargs.get('max_lr', 0.01),
			step_size_up=kwargs.get('step_size_up', 2000),
			mode=kwargs.get('mode', 'triangular')
		)
	else:
		raise ValueError(f"Неизвестный тип планировщика: {scheduler_type}")


def train_with_scheduler(model, train_loader, val_loader, optimizer, scheduler,
					  epochs, device='cuda', criterion=None):
	"""
	Обучает модель с использованием планировщика LR.

	Args:
		model (nn.Module): модель
		train_loader (DataLoader): загрузчик обучающих данных
		val_loader (DataLoader): загрузчик валидационных данных
		optimizer (torch.optim.Optimizer): оптимизатор
		scheduler (torch.optim.lr_scheduler._LRScheduler): планировщик LR
		epochs (int): число эпох
		device (str): устройство
		criterion (callable, optional): функция потерь


	Returns:
		dict: метрики обучения (train_loss, val_loss, train_acc, val_acc)
	"""
	model.to(device)
	criterion = criterion or torch.nn.CrossEntropyLoss()


	train_losses, val_losses = [], []
	train_accs, val_accs = [], []


	for epoch in range(epochs):
		# Обучение
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for data, target in train_loader:
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			_, predicted = output.max(1)
			total += target.size(0)
			correct += predicted.eq(target).sum().item()


		train_loss = running_loss / len(train_loader)
		train_acc = correct / total
		train_losses.append(train_loss)
		train_accs.append(train_acc)


		# Валидация
		model.eval()
		val_loss = 0.0
		correct = 0
		total = 0

		with torch.no_grad():
			for data, target in val_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				loss = criterion(output, target)
				val_loss += loss.item()

				_, predicted = output.max(1)
				total += target.size(0)
				correct += predicted.eq(target).sum().item()


		val_loss /= len(val_loader)
		val_acc = correct / total
		val_losses.append(val_loss)
		val_accs.append(val_acc)

		# Шаг планировщика
		if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
			scheduler.step(val_loss)
		else:
			scheduler.step()


		print(f"Epoch {epoch+1}/{epochs}, "
			  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
			  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

	return {
		'train_loss': train_losses,
		'val_loss': val_losses,
		'train_acc': train_accs,
		'val_acc': val_accs
	}

def plot_lr_scheduler_effect(scheduler, epochs, output_path):
	"""
	Рисует график изменения LR по эпохам для планировщика.

	Args:
		scheduler (torch.optim.lr_scheduler._LRScheduler): планировщик
		epochs (int): число эпох
		output_path (str): путь для сохранения графика
	"""
	lrs = []
	for _ in range(epochs):
		lrs.append(scheduler.get_last_lr()[0])
		scheduler.step()

	plt.figure(figsize=(10, 6))
	plt.plot(range(epochs), lrs, marker='o')
	plt.xlabel('Epoch')
	plt.ylabel('Learning Rate')
	plt.title('Learning Rate Schedule')
	plt.grid(True)
	plt.savefig(output_path)
	plt.close()
	print(f"График планировщика LR сохранён: {output_path}")

def freeze_layers(model, layer_names):
	"""
	Замораживает указанные слои модели (отключает градиенты).

	Args:
		model (nn.Module): модель
		layer_names (list of str): имена замораживаемых слоёв
	"""
	for name, param in model.named_parameters():
		if any(layer in name for layer in layer_names):
			param.requires_grad = False
	print(f"Слои заморожены: {layer_names}")


def unfreeze_layers(model, layer_names=None):
	"""
	Размораживает слои модели (включает градиенты).


	Если layer_names не указан, размораживает все слои.

	Args:
		model (nn.Module): модель
		layer_names (list of str, optional): имена размораживаемых слоёв
	"""
	if layer_names is None:
		for param in model.parameters():
			param.requires_grad = True
		print("Все слои разморожены")
	else:
		for name, param in model.named_parameters():
			if any(layer in name for layer in layer_names):
				param.requires_grad = True
		print(f"Слои разморожены: {layer_names}")


def count_trainable_params_by_layer(model):
	"""
	Считает число обучаемых параметров по слоям.

	Args:
		model (nn.Module): модель

	Returns:
		dict: {имя_слоя: число_параметров}
	"""
	counts = {}
	for name, param in model.named_parameters():
		if param.requires_grad:
			counts[name] = param.numel()
	return counts

def print_trainable_params_summary(model):
	"""
	Печатает сводку по обучаемым параметрам по слоям.

	Args:
		model (nn.Module): модель
	"""
	counts = count_trainable_params_by_layer(model)
	total = sum(counts.values())

	print("Сводка по обучаемым параметрам:")
	print("-" * 60)
	for name, num_params in counts.items():
		print(f"{name}: {num_params:,} параметров")
	print("-" * 60)
	print(f"Всего обучаемых параметров: {total:,}")



def apply_gradient_clipping_by_norm(model, max_norm):
	"""
	Применяет обрезку градиентов по норме L2.

	Args:
		model (nn.Module): модель
		max_norm (float): максимальная норма градиента
	"""
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
	print(f"Gradient clipping (norm): max_norm={max_norm}")

def apply_gradient_clipping_by_value(model, clip_value):
	"""
	Применяет обрезку градиентов по значению.

	Args:
		model (nn.Module): модель
		clip_value (float): максимальное значение градиента
	"""
	torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
	print(f"Gradient clipping (value): clip_value={clip_value}")

def track_gradient_statistics(model):
	"""
	Собирает статистику по градиентам (среднее, std, min, max).

	Args:
		model (nn.Module): модель

	Returns:
		dict: статистика по градиентам каждого параметра
	"""
	stats = {}
	for name, param in model.named_parameters():
		if param.grad is not None:
			grad = param.grad.data
			stats[name] = {
				'mean': grad.mean().item(),
				'std': grad.std().item(),
				'min': grad.min().item(),
				'max': grad.max().item(),
				'norm': grad.norm().item()
			}
	return stats

def print_gradient_stats(stats):
	"""
	Печатает статистику по градиентам.

	Args:
		stats (dict): статистика из track_gradient_statistics
	"""
	print("Статистика градиентов:")
	print("-! * 80)
	for name, s in stats.items():
		print(f"{name}:")
		print(f"  Mean: {s['mean']:.6f}, Std: {s['std']:.6f}")
		print(f"  Min: {s['min']:.6f}, Max: {s['max']:.6f}, Norm: {s['norm']:.6f}")
	print("-! * 80)

def initialize_weights_advanced(model, init_method='kaiming', nonlinearity='relu'):
	"""
	Продвинутая инициализация весов.

	Args:
		model (nn.Module): модель
		init_method (str): метод инициализации ('xavier', 'kaiming', 'orthogonal')
		nonlinearity (str): нелинейность для учёта при инициализации
	"""
	def init_fn(m):
		if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
			if init_method == 'xavier':
				torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain(nonlinearity))
			elif init_method == 'kaiming':
				torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
			elif init_method == 'orthogonal':
				torch.nn.init.orthogonal_(m.weight)
			else:
				raise ValueError(f"Неизвестный метод инициализации: {init_method}")

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)

	model.apply(init_fn)
	print(f"Веса инициализированы ({init_method}, {nonlinearity})")

def replace_activation_in_model(model, old_act_type, new_act_module):
	"""
	Рекурсивно заменяет активации в модели.

	Args:
		model (nn.Module): модель
		old_act_type (type): тип старой активации
		new_act_module (nn.Module): новый модуль активации
	"""
	for child in model.children():
		if isinstance(child, old_act_type):
			# Замена на месте
			parent = model
			for name, module in parent.named_children():
				if module is child:
					parent._modules[name] = new_act_module
					break
		else:
			replace_activation_in_model(child, old_act_type, new_act_module)
	print(f"Заменены активации: {old_act_type} → {new_act_module}")

def get_model_flops_and_params(model, input_shape, device='cuda'):
	"""
	Оценивает FLOPs и число параметров модели.

	Args:
		model (nn.Module): модель
		input_shape (tuple): форма входа (C, H, W)
		device (str): устройство

	Returns:
		tuple: (число_параметров, FLOPs)
	"""
	try:
		from thop import profile
		model.to(device)
		input = torch.randn(1, *input_shape).to(device)
		flops, params = profile(model, inputs=(input,), verbose=False)
		return params, flops
	except ImportError:
		print("Установите thop: pip install thop")
		return None, None

def summarize_model_efficiency(model, input_shape, device='cuda'):
	"""
	Выводит сводку по эффективности модели.

	Args:
		model (nn.Module): модель
		input_shape (tuple): форма входа
		device (str): устройство
	"""
	params, flops = get_model_flops_and_params(model, input_shape, device)
	if params is not None and flops is not None:
		print("Сводка эффективности модели:")
		print(f"  Параметры: {params:,}")
		print(f"  FLOPs: {flops:,}")
		print(f"  Параметры (млн): {params/1e6:.2f}")
		print(f"  FLOPs (млрд): {flops/1e9:.2f}")

def export_model_for_inference(model, example_input, output_path, format='torchscript'):
	"""
	Экспортирует модель для инференса.

	Args:
		model (nn.Module): модель
		example_input (torch.Tensor): пример входа
		output_path (str): путь сохранения
		format (str): формат ('torchscript', 'onnx')
	"""
	model.eval()
	if format == 'torchscript':
		scripted_model = torch.jit.trace(model, exampleinput)
		torch.jit.save(scripted_model, output_path)
		print(f"Модель экспортирована в TorchScript: {output_path}")
	elif format == 'onnx':
		torch.onnx.export(
			model, exampleinput, output_path,
			export_params=True, opset_version=11,
			do_constant_folding=True,
			input_names=['input'], output_names=['output']
		)
		print(f"Модель экспортирована в ONNX: {output_path}")
	else:
		raise ValueError(f"Неизвестный формат: {format}")




def create_data_augmentation_pipeline(
	resize=224,
	horizontal_flip=True,
	vertical_flip=False,
	rotation_range=15,
	color_jitter_params=None,
	normalize_mean=None,
	normalize_std=None
):
	"""
	Создаёт пайплайн аугментации данных для обучения.

	Args:
		resize (int or tuple): размер для ресайза (H, W) или одно число (для квадрата)
		horizontal_flip (bool): случайный горизонтальный флип
		vertical_flip (bool): случайный вертикальный флип
		rotation_range (int): макс. угол поворота (градусы)
		color_jitter_params (dict): параметры ColorJitter (brightness, contrast и др.)
		normalize_mean (list): средние значения для нормализации (по каналам)
		normalize_std (list): std для нормализации (по каналам)

	Returns:
		torchvision.transforms.Compose: пайплайн преобразований
	"""
	from torchvision import transforms

	augmentations = []

	if resize:
		if isinstance(resize, int):
			augmentations.append(transforms.Resize((resize, resize)))
		else:
			augmentations.append(transforms.Resize(resize))


	augmentations.append(transforms.RandomApply([
		transforms.RandomRotation(rotation_range)
	], p=0.5))


	if horizontal_flip:
		augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
	if vertical_flip:
		augmentations.append(transforms.RandomVerticalFlip(p=0.5))


	if color_jitter_params:
		augmentations.append(transforms.ColorJitter(**color_jitter_params))


	augmentations.append(transforms.ToTensor())

	if normalize_mean and normalize_std:
		augmentations.append(
			transforms.Normalize(mean=normalize_mean, std=normalize_std)
		)

	return transforms.Compose(augmentations)


def create_test_transform(resize=224, normalize_mean=None, normalize_std=None):
	"""
	Создаёт преобразования для тестовых данных (без аугментаций).


	Args:
		resize (int or tuple): размер ресайза
		normalize_mean (list): средние для нормализации
		normalize_std (list): std для нормализации

	Returns:
		torchvision.transforms.Compose
	"""
	from torchvision import transforms
	transform = [
		transforms.Resize(resize if isinstance(resize, tuple) else (resize, resize)),
		transforms.ToTensor()
	]
	if normalize_mean and normalize_std:
		transform.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))
	return transforms.Compose(transform)


def compute_dataset_statistics(dataloader, device='cuda'):
	"""
	Вычисляет средние и std по каналам для датасета.


	Полезно для настройки нормализации.

	Args:
		dataloader (DataLoader): загрузчик данных
		device (str): устройство

	Returns:
		tuple: (mean, std) — тензоры размера (C,)
	"""
	n_samples = 0
	channel_sums = torch.zeros(3, device=device)
	channel_sq_sums = torch.zeros(3, device=device)


	with torch.no_grad():
		for data, _ in dataloader:
			data = data.to(device)
			n_batch = data.size(0)
			channel_sums += data.sum(dim=[0, 2, 3])
			channel_sq_sums += (data ** 2).sum(dim=[0, 2, 3])
			n_samples += n_batch

	mean = channel_sums / (n_samples * data.size(2) * data.size(3))
	std = torch.sqrt(
		(channel_sq_sums / (n_samples * data.size(2) * data.size(3))) - mean ** 2
	)
	return mean.cpu(), std.cpu()


def visualize_predictions(model, data_loader, class_names, output_path, n_images=8, device='cuda'):
	"""
	Визуализирует предсказания модели на нескольких примерах.


	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		class_names (list): имена классов
		output_path (str): путь для сохранения изображения
		n_images (int): число изображений для визуализации
		device (str): устройство
	"""
	import matplotlib.pyplot as plt

	model.to(device).eval()
	fig, axes = plt.subplots(2, 4, figsize=(12, 6))
	axes = axes.ravel()

	with torch.no_grad():
		data, targets = next(iter(data_loader))
		data, targets = data[:n_images].to(device), targets[:n_images]
		outputs = model(data)
		_, preds = torch.max(outputs, 1)


		for i in range(n_images):
			img = data[i].cpu().permute(1, 2, 0).numpy()
			img = (img - img.min()) / (img.max() - img.min())  # нормализация для отображения
			axes[i].imshow(img)
			axes[i].set_title(f"True: {class_names[targets[i]]}\nPred: {class_names[preds[i]]}")
			axes[i].axis('off')

	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()
	print(f"Визуализация предсказаний сохранена: {output_path}")


def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
	"""
	Рисует матрицу ошибок (confusion matrix).

	Args:
		cm (np.ndarray): матрица ошибок
		class_names (list): имена классов
		output_path (str): путь сохранения
		title (str): заголовок графика
	"""
	import seaborn as sns
	import pandas as pd

	df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
	plt.figure(figsize=(10, 8))
	sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
	plt.title(title)
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()
	print(f"Матрица ошибок сохранена: {output_path}")

def calculate_class_accuracy(predictions, targets, num_classes):
	"""
	Считает точность по каждому классу.

	Args:
		predictions (np.ndarray): предсказанные метки
		targets (np.ndarray): истинные метки
		num_classes (int): число классов


	Returns:
		np.ndarray: точность по каждому классу (размер: num_classes)
	"""
	accuracy_per_class = np.zeros(num_classes)
	for cls in range(num_classes):
		cls_mask = targets == cls
		if cls_mask.sum() > 0:
			accuracy_per_class[cls] = (
				(predictions[cls_mask] == targets[cls_mask]).mean()
			)
	return accuracy_per_class

def log_training_progress(epoch, train_loss, val_loss, train_acc, val_acc, log_file):
	"""
	Записывает прогресс обучения в лог‑файл.

	Args:
		epoch (int): номер эпохи
		train_loss (float): тренировочная потеря
		val_loss (float): валидационная потеря
		train_acc (float): тренировочная точность
		val_acc (float): валидационная точность
		log_file (str): путь к лог‑файлу
	"""
	with open(log_file, 'a') as f:
		f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
				f"{train_acc:.6f},{val_acc:.6f}\n")
	print(f"Прогресс эпохи {epoch} записан в {log_file}")


def setup_early_stopping(patience=5, min_delta=0.001):
	"""
	Создаёт объект для ранней остановки обучения (early stopping).


	Args:
		patience (int): число эпох без улучшения, после которого останавливаемся
		min_delta (float): минимальное улучшение метрики (иначе считается отсутствием улучшения)


	Returns:
		dict: контейнер для состояния early stopping
	"""
	return {
		'patience': patience,
		'min_delta': min_delta,
		'best_score': None,
		'counter': 0,
		'early_stop': False
	}

def check_early_stopping(early_stopper, val_loss, epoch):
	"""
	Проверяет, нужно ли остановить обучение по критерию early stopping.


	Args:
		early_stopper (dict): объект early stopping из setup_early_stopping
		val_loss (float): текущее значение валидационной потери
		epoch (int): номер эпохи


	Returns:
		bool: True, если нужно остановить обучение
	"""
	score = -val_loss  # инвертируем, т.к. ищем минимум потери


	if early_stopper['best_score'] is None:
		early_stopper['best_score'] = score
	elif score < early_stopper['best_score'] + early_stopper['min_delta']:
		early_stopper['counter'] += 1
		print(f"EarlyStopping: {early_stopper['counter']}/{early_stopper['patience']} (epoch {epoch})")
		if early_stopper['counter'] >= early_stopper['patience']:
			early_stopper['early_stop'] = True
			print(f"Early stopping на эпохе {epoch}!")
	else:
		early_stopper['best_score'] = score
		early_stopper['counter'] = 0


	return early_stopper['early_stop']


def save_best_model(model, optimizer, epoch, val_loss, best_loss, checkpoint_path):
	"""
	Сохраняет модель, если текущая валидационная потеря лучше предыдущей лучшей.


	Args:
		model (nn.Module): модель
		optimizer (torch.optim.Optimizer): оптимизатор
		epoch (int): номер эпохи
		val_loss (float): текущая валидационная потеря
		best_loss (float): лучшая валидационная потеря на данный момент
		checkpoint_path (str): путь для сохранения чекпоинта


	Returns:
		float: обновлённое значение best_loss
	"""
	if val_loss < best_loss:
		print(f"Новая лучшая модель на эпохе {epoch}: val_loss={val_loss:.6f} (было {best_loss:.6f})")
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'val_loss': val_loss
		}, checkpoint_path)
		best_loss = val_loss
	return best_loss

def load_best_model(model, checkpoint_path, optimizer=None):
	"""
	Загружает лучшую модель из чекпоинта.


	Args:
		model (nn.Module): модель (для загрузки весов)
		checkpoint_path (str): путь к чекпоинту
		optimizer (torch.optim.Optimizer, optional): оптимизатор (если нужно восстановить состояние)


	Returns:
		int: номер эпохи, на которой была сохранена модель
	"""
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print(f"Лучшая модель загружена из {checkpoint_path}, эпоха {checkpoint['epoch']}")
	return checkpoint['epoch']


def calculate_precision_recall_f1(y_true, y_pred, num_classes):
	"""
	Считает precision, recall и F1-score по каждому классу.


	Args:
		y_true (np.ndarray): истинные метки
		y_pred (np.ndarray): предсказанные метки
		num_classes (int): число классов


	Returns:
		dict: {'precision': ..., 'recall': ..., 'f1': ...} — массивы размера num_classes
	"""
	precision = np.zeros(num_classes)
	recall = np.zeros(num_classes)
	f1 = np.zeros(num_classes)


	for cls in range(num_classes):
		true_positive = ((y_true == cls) & (y_pred == cls)).sum()
		false_positive = ((y_true != cls) & (y_pred == cls)).sum()
		false_negative = ((y_true == cls) & (y_pred != cls)).sum()


		prec = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
		rec = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
		f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

		precision[cls] = prec
		recall[cls] = rec
		f1[cls] = f1_score

	return {'precision': precision, 'recall': recall, 'f1': f1}


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
	"""
	Рисует кривые обучения (потеря и точность).


	Args:
		train_losses (list): тренировочные потери по эпохам
		val_losses (list): валидационные потери по эпохам
		train_accs (list): тренировочная точность по эпохам
		val_accs (list): валидационная точность по эпохам
		output_path (str): путь для сохранения графика
	"""
	epochs = len(train_losses)
	plt.figure(figsize=(14, 5))

	plt.subplot(1, 2, 1)
	plt.plot(range(epochs), train_losses, label='Train Loss')
	plt.plot(range(epochs), val_losses, label='Val Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.grid(True)

	plt.subplot(1, 2, 2)
	plt.plot(range(epochs), train_accs, label='Train Acc')
	plt.plot(range(epochs), val_accs, label='Val Acc')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()
	print(f"Кривые обучения сохранены: {output_path}")

def compute_roc_auc(model, data_loader, num_classes, device='cuda'):
	"""
	Считает ROC AUC для многоклассовой классификации (one-vs-rest).


	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		num_classes (int): число классов
		device (str): устройство ('cuda' или 'cpu')

	Returns:
		np.ndarray: массив AUC для каждого класса (размер: num_classes)
	"""
	from sklearn.metrics import roc_auc_score
	import numpy as np

	model.eval()
	y_true = []
	y_scores = []

	# Собираем истинные метки и предсказанные вероятности
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			# Получаем вероятности через softmax
			output = torch.nn.functional.softmax(model(data), dim=1)
			y_true.extend(target.cpu().numpy())
			y_scores.extend(output.cpu().numpy())


	y_true = np.array(y_true)
	y_scores = np.array(y_scores)


	aucs = []
	for cls in range(num_classes):
		try:
			# Бинаризуем истинные метки для класса `cls`
			y_true_binary = (y_true == cls).astype(int)

			# Берём предсказанные вероятности именно для класса `cls`
			y_score_cls = y_scores[:, cls]


			# Вычисляем AUC
			auc = roc_auc_score(y_true_binary, y_score_cls)
			aucs.append(auc)

		except ValueError as e:
			# Обработка ошибок:
			# - Нет примеров класса `cls` в выборке (все метки 0)
			# - Все вероятности одинаковы (невозможно построить ROC)
			print(f"Ошибка при вычислении AUC для класса {cls}: {e}")
			aucs.append(0.0)  # Или np.nan для явного обозначения пропуска

		except Exception as e:
			print(f"Неожиданная ошибка для класса {cls}: {e}")
			aucs.append(0.0)

	return np.array(aucs)



def plot_roc_curves(y_true, y_scores, num_classes, class_names=None, output_path=None):
	"""
	Рисует ROC‑кривые для каждого класса (one‑vs‑rest).


	Args:
		y_true (np.ndarray): истинные метки (N,)
		y_scores (np.ndarray): вероятности классов (N, num_classes)
		num_classes (int): число классов
		classnames (list of str, optional): имена классов
		output_path (str, optional): путь для сохранения графика
	"""
	from sklearn.metrics import roc_curve, auc
	import matplotlib.pyplot as plt


	plt.figure(figsize=(10, 8))
	if classnames is None:
		classnames = [f"Класс {i}" for i in range(num_classes)]


	for cls in range(num_classes):
		fpr, tpr, _ = roc_curve(y_true == cls, y_scores[:, cls])
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, label=f'{classnames[cls]} (AUC = {roc_auc:.3f})')


	plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Ложноположительная доля (FPR)')
	plt.ylabel('Истинноположительная доля (TPR)')
	plt.title('ROC-кривые (один против всех)')
	plt.legend(loc="lower right")
	plt.grid(True)


	if output_path:
		plt.savefig(output_path, bbox_inches='tight')
		print(f"ROC-кривые сохранены: {output_path}")
	plt.close()



def evaluate_roc_auc_and_plot(model, data_loader, num_classes, classnames=None, device='cuda', output_dir=None):
	"""
	Вычисляет AUC и рисует ROC-кривые для многоклассовой задачи.


	Args:
		model (nn.Module): обученная модель
		data_loader (DataLoader): загрузчик тестовых данных
		num_classes (int): число классов
		classnames (list): имена классов (опционально)
		device (str): устройство
		output_dir (str): директория для сохранения графиков (опционально)


	Returns:
		dict: результаты (AUC по классам, средний AUC)
	"""
	# Вычисляем AUC для каждого класса
	aucs = compute_roc_auc(model, data_loader, num_classes, device)

	mean_auc = np.mean(aucs)


	# Собираем y_true и y_scores для построения графиков
	y_true, y_scores = [], []
	model.eval()
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			output = torch.nn.functional.softmax(model(data), dim=1)
			y_true.extend(target.cpu().numpy())
			y_scores.extend(output.cpu().numpy())
	y_true = np.array(y_true)
	y_scores = np.array(y_scores)

	# Рисуем ROC-кривые, если указана директория
	if output_dir:
		import os
		os.makedirs(output_dir, exist_ok=True)
		plot_roc_curves(
			y_true, y_scores, num_classes,
			classnames,
			output_path=os.path.join(output_dir, 'roc_curves.png')
		)

	return {
		'auc_per_class': aucs,
		'mean_auc': mean_auc,
		'y_true': y_true,
		'y_scores': y_scores
	}



# Пример использования
if __name__ == "__main__":
	import torch
	from torch.utils.data import DataLoader
	from torchvision import datasets, transforms

	# Пример: загрузка данных
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
	])
	dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

	# Пример модели (замените на вашу)
	model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
	model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 классов для CIFAR-10

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)

	# Оценка AUC и построение графиков
	results = evaluate_roc_auc_and_plot(
		model, data_loader, num_classes=10,
		classnames=['airplane', 'automobile', 'bird', 'cat', 'deer',
				   'dog', 'frog', 'horse', 'ship', 'truck'],
		device=device,
		output_dir='./results'
	)

	print(f"AUC по классам: {results['auc_per_class']}")
	print(f"Средний AUC: {results['mean_auc']:.4f}")

	return results




def calculate_precision_recall_f1(y_true, y_pred, num_classes):
	"""
	Считает precision, recall и F1-score по каждому классу.


	Args:
		y_true (np.ndarray): истинные метки (N,)
		y_pred (np.ndarray): предсказанные метки (N,)
		num_classes (int): число классов

	Returns:
		dict: {'precision': ..., 'recall': ..., 'f1': ...} — массивы размера num_classes
	"""
	precision = np.zeros(num_classes)
	recall = np.zeros(num_classes)
	f1 = np.zeros(num_classes)

	for cls in range(num_classes):
		# True Positive: правильно предсказанные образцы класса cls
		tp = ((y_true == cls) & (y_pred == cls)).sum()

		# False Positive: образцы, ошибочно отнесённые к классу cls
		fp = ((y_true != cls) & (y_pred == cls)).sum()
		# False Negative: образцы класса cls, ошибочно предсказанные как другие
		fn = ((y_true == cls) & (y_pred != cls)).sum()


		# Precision: доля верно предсказанных среди всех предсказанных как cls
		prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		# Recall: доля верно предсказанных среди всех истинных cls
		rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		# F1: среднее гармоническое precision и recall
		f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

		precision[cls] = prec
		recall[cls] = rec
		f1[cls] = f1_score

	return {'precision': precision, 'recall': recall, 'f1': f1}




def plot_confusion_matrix(cm, class_names, output_path, title='Матрица ошибок'):
	"""
	Рисует матрицу ошибок (confusion matrix).


	Args:
		cm (np.ndarray): матрица ошибок (num_classes, num_classes)
		class_names (list): имена классов
		output_path (str): путь для сохранения изображения
		title (str): заголовок графика
	"""
	import seaborn as sns
	import pandas as pd
	import matplotlib.pyplot as plt

	df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
	plt.figure(figsize=(10, 8))
	sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
	plt.title(title)
	plt.ylabel('Истинные метки')
	plt.xlabel('Предсказанные метки')
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()
	print(f"Матрица ошибок сохранена: {output_path}")




def evaluate_model_full(model, data_loader, num_classes, class_names=None, device='cuda', output_dir=None):
	"""
	Полноценная оценка модели: точность, precision, recall, F1, AUC, матрица ошибок, ROC‑кривые.


	Args:
		model (nn.Module): модель
		data_loader (DataLoader): загрузчик данных
		num_classes (int): число классов
		class_names (list of str, optional): имена классов
		device (str): устройство
		output_dir (str, optional): директория для сохранения графиков


	Returns:
		dict: все метрики
	"""
	model.eval()
	y_true = []
	y_pred = []
	y_scores = []

	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			probs = torch.nn.functional.softmax(output, dim=1)

			_, pred = torch.max(output, 1)


			y_true.extend(target.cpu().numpy())
			y_pred.extend(pred.cpu().numpy())
			y_scores.extend(probs.cpu().numpy())


	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	y_scores = np.array(y_scores)


	# 1. Общая точность
	accuracy = (y_pred == y_true).mean()


	# 2. Precision, Recall, F1 по классам
	metrics = calculate_precision_recall_f1(y_true, y_pred, num_classes)


	# 3. AUC (one-vs-rest)
	aucs = compute_roc_auc(model, data_loader, num_classes, device)


	# 4. Матрица ошибок
	cm = calculate_confusion_matrix(y_true, y_pred, num_classes)


	results = {
		'accuracy': accuracy,
		'precision': metrics['precision'],
		'recall': metrics['recall'],
		'f1': metrics['f1'],
		'auc': aucs,
		'confusion_matrix': cm,
		'y_true': y_true,
		'y_pred': y_pred,
		'y_scores': y_scores
	}

	# Визуализация, если указана директория
	if output_dir:
		import os
		os.makedirs(output_dir, exist_ok=True)

		# ROC-кривые
		plot_roc_curves(
			y_true, y_scores, num_classes,
			classnames=classnames,
			output_path=os.path.join(output_dir, 'roc_curves.png')
		)

		# Матрица ошибок
		plot_confusion_matrix(
			cm, classnames or [f"Класс {i}" for i in range(num_classes)],
			output_path=os.path.join(output_dir, 'confusion_matrix.png'),
			title='Матрица ошибок'
		)

	return results



def print_evaluation_summary(results, class_names=None):
	"""
	Печатает сводку по оценке модели.

	Args:
		results (dict): результаты из evaluate_model_full
		class_names (list of str, optional): имена классов
	"""
	num_classes = len(results['precision'])
	classnames = class_names or [f"Класс {i}" for i in range(num_classes)]

	print("=" * 60)
	print("СВОДКА ОЦЕНКИ МОДЕЛИ")
	print("=" * 60)
	print(f"Общая точность (Accuracy): {results['accuracy']:.4f}")
	print("\nМетрики по классам:")
	print("-" * 60)
	for i in range(num_classes):
		print(f"{classnames[i]:<12} "
			  f"Precision={results['precision'][i]:.4f} "
			  f"Recall={results['recall'][i]:.4f} "
			  f"F1={results['f1'][i]:.4f} "
			  f"AUC={results['auc'][i]:.4f}")
	print("-" * 60)
	print(f"Средний AUC: {results['auc'].mean():.4f}")
	print("=" * 60)



# Пример использования
if __name__ == "__main__":
	import torch
	from torch.utils.data import DataLoader
	from torchvision import datasets, transforms

	# Параметры
	num_classes = 10
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	output_dir = './evaluation_results'


	# Преобразования для данных
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# Загрузка данных (пример: CIFAR-10)
	dataset = datasets.CIFAR10(
		root='./data', train=False, download=True, transform=transform
	)
	data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

	# Загрузка предобученной модели (пример с ResNet18)
	model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
	model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # замена последнего слоя
	model.to(device)

	# Имена классов для CIFAR-10
	class_names = [
		'airplane', 'automobile', 'bird', 'cat', 'deer',
		'dog', 'frog', 'horse', 'ship', 'truck'
	]

	# Полная оценка модели
	results = evaluate_model_full(
		model=model,
		data_loader=data_loader,
		num_classes=num_classes,
		class_names=class_names,
		device=device,
		output_dir=output_dir
	)

	# Вывод сводки
	print_evaluation_summary(results, class_names)


	# Дополнительно: вывод среднего F1
	mean_f1 = results['f1'].mean()
	print(f"\nСредний F1-score: {mean_f1:.4f}")


	# Сохранение результатов в файл
	import json
	with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
		# Преобразуем numpy-массивы в списки для JSON
		results_for_json = {
			k: (v.tolist() if isinstance(v, np.ndarray) else v)
			for k, v in results.items()
		}
		json.dump(results_for_json, f, indent=2)
	print(f"Результаты сохранены в {output_dir}/evaluation_results.json")



def calculate_confusion_matrix(y_true, y_pred, num_classes):
	"""
	Считает матрицу ошибок (confusion matrix).

	Args:
		y_true (np.ndarray): истинные метки (N,)
		y_pred (np.ndarray): предсказанные метки (N,)
		num_classes (int): число классов


	Returns:
		np.ndarray: матрица ошибок размера (num_classes, num_classes)
	"""
	cm = np.zeros((num_classes, num_classes), dtype=int)
	for i in range(len(y_true)):
		cm[y_true[i], y_pred[i]] += 1
	return cm



def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
	"""
	Рисует кривые обучения (потеря и точность).


	Args:
		train_losses (list): тренировочные потери по эпохам
		val_losses (list): валидационные потери по эпохам
		train_accs (list): тренировочная точность по эпохам
		val_accs (list): валидационная точность по эпохам
		output_path (str): путь для сохранения графика
	"""
	epochs = len(train_losses)
	plt.figure(figsize=(14, 5))

	plt.subplot(1, 2, 1)
	plt.plot(range(epochs), train_losses, label='Train Loss')
	plt.plot(range(epochs), val_losses, label='Val Loss')
	plt.xlabel('Эпоха')
	plt.ylabel('Потеря')
	plt.title('Потеря на обучении и валидации')
	plt.legend()
	plt.grid(True)


	plt.subplot(1, 2, 2)
	plt.plot(range(epochs), train_accs, label='Train Acc')
	plt.plot(range(epochs), val_accs, label='Val Acc')
	plt.xlabel('Эпоха')
	plt.ylabel('Точность')
	plt.title('Точность на обучении и валидации')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()
	print(f"Кривые обучения сохранены: {output_path}")


def save_training_log(epoch, train_loss, val_loss, train_acc, val_acc, log_path):
	"""
	Сохраняет лог обучения в CSV-файл.


	Args:
		epoch (int): номер эпохи
		train_loss (float): потеря на обучении
		val_loss (float): потеря на валидации
		train_acc (float): точность на обучении
		val_acc (float): точность на валидации
		log_path (str): путь к файлу лога
	"""
	import csv
	with open(log_path, 'a', newline='') as f:
		writer = csv.writer(f)
		if f.tell() == 0:  # если файл пустой — пишем заголовок
			writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
		writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])


def load_training_log(log_path):
	"""
	Загружает лог обучения из CSV-файла.

	Args:
		log_path (str): путь к файлу лога

	Returns:
		dict: {'epochs': [...], 'train_losses': [...], ...}
	"""
	import pandas as pd
	df = pd.read_csv(log_path)
	return {
		'epochs': df['epoch'].tolist(),
		'train_losses': df['train_loss'].tolist(),
		'val_losses': df['val_loss'].tolist(),
		'train_accs': df['train_acc'].tolist(),
		'val_accs': df['val_acc'].tolist()
	}




