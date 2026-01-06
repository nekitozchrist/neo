print("=" * 80)
print("🧠 ОБУЧЕНИЕ ПСИХОЛОГИЧЕСКОГО БОТА - ПРОДВИНУТАЯ ВЕРСИЯ")
print("=" * 80)

# ДОБАВЬТЕ ВЫБОР РЕЖИМА
print("\n🔧 Выберите режим вывода:")
print("1. 🚀 Быстрый (только LR, Loss, прогресс)")
print("2. 🐛 Отладочный (все детали)")
print("3. 📊 Профессиональный (метрики + графики)")
debug_mode = input("Выберите (1-3, по умолчанию 1): ").strip() or "1"
DEBUG_MODE = int(debug_mode)


import time
import torch
from pathlib import Path
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from datetime import datetime
import os
import sys
import time
import math
import json
import numpy as np

print("=" * 80)
print("🧠 ОБУЧЕНИЕ ПСИХОЛОГИЧЕСКОГО БОТА - ПРОДВИНУТАЯ ВЕРСИЯ")
print("   С АДАПТИВНЫМИ НАСТРОЙКАМИ И МЕТРИКАМИ")
print("=" * 80)

# ================= ПРАВИЛЬНЫЕ ПАРАМЕТРЫ =================
BATCH_SIZE = 3 
MAX_LENGTH = 729
GRADIENT_ACCUMULATION = 9
LEARNING_RATE = 2e-4
print("Введи количество эпох...")
EPOCHS = int(input())
WARMUP_RATIO = 0.9

print(f"\n🎯 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Max length: {MAX_LENGTH}")
print(f"   • Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"   • Learning rate: {LEARNING_RATE:.1e}")
print(f"   • Epochs: {EPOCHS}")
print(f"   • Warmup: {WARMUP_RATIO*100}%")

# ================= КОНТЕКСТНЫЕ МЕНЕДЖЕРЫ ДЛЯ РЕЖИМОВ =================

class TrainingMode:
	"""Контекстный менеджер для режима обучения"""
	def __init__(self, model):
		self.model = model
		self.original_cache = None
		self.original_training = None
	
	def __enter__(self):
		self.original_cache = self.model.config.use_cache
		self.original_training = self.model.training
		self.model.config.use_cache = False  # ⛔ Выключаем кэш
		self.model.gradient_checkpointing_enable()  # ✅ Включаем checkpointing
		self.model.train()  # ✅ Режим обучения
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.model.config.use_cache = self.original_cache
		if not self.original_training:
			self.model.eval()

class ValidationMode:
	"""Контекстный менеджер для режима валидации (только loss)"""
	def __init__(self, model):
		self.model = model
		self.original_cache = None
		self.original_training = None
		self.original_gradient_checkpointing = None
	
	def __enter__(self):
		self.original_cache = self.model.config.use_cache
		self.original_training = self.model.training
		self.original_gradient_checkpointing = self.model.is_gradient_checkpointing
		
		self.model.config.use_cache = False  # ⛔ Кэш не нужен
		if self.model.is_gradient_checkpointing:  # ⛔ Выключаем checkpointing если включен
			try:
				self.model.gradient_checkpointing_disable()
			except:
				self.model.gradient_checkpointing = False
		self.model.eval()  # ✅ Режим оценки
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.model.config.use_cache = self.original_cache
		if self.original_gradient_checkpointing and not self.model.is_gradient_checkpointing:
			self.model.gradient_checkpointing_enable()
		if self.original_training:
			self.model.train()

class GenerationMode:
	"""Контекстный менеджер для режима генерации (тесты/инференс)"""
	def __init__(self, model):
		self.model = model
		self.original_cache = None
		self.original_training = None
		self.original_gradient_checkpointing = None
	
	def __enter__(self):
		self.original_cache = self.model.config.use_cache
		self.original_training = self.model.training
		self.original_gradient_checkpointing = self.model.is_gradient_checkpointing
		
		self.model.config.use_cache = True  # ✅ Включаем кэш для скорости
		if self.model.is_gradient_checkpointing:  # ⛔ Выключаем checkpointing если включен
			try:
				self.model.gradient_checkpointing_disable()
			except:
				self.model.gradient_checkpointing = False
		self.model.eval()  # ✅ Режим оценки
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.model.config.use_cache = self.original_cache
		if self.original_gradient_checkpointing and not self.model.is_gradient_checkpointing:
			self.model.gradient_checkpointing_enable()
		if self.original_training:
			self.model.train()

# ================= ОПТИМАЛЬНЫЙ ШЕДУЛЕР =================

class OptimalScheduler:
	"""
	Оптимальный шедулер для психологической модели
	Warmup → Cosine Decay → Linear Final
	"""
	
	def __init__(self, optimizer, total_steps, initial_lr, warmup_ratio=0.10):
		self.optimizer = optimizer
		self.total_steps = total_steps
		self.initial_lr = initial_lr
		self.warmup_steps = int(total_steps * warmup_ratio)
		self.cosine_steps = int(total_steps * 0.6)
		self.linear_steps = total_steps - self.warmup_steps - self.cosine_steps
		self.current_step = 0
		
		print(f"\n🎯 ОПТИМАЛЬНЫЙ ШЕДУЛЕР (3 фазы):")
		print(f"   • Всего шагов: {total_steps}")
		print(f"   • Warmup: {self.warmup_steps} шагов ({warmup_ratio*100}%)")
		print(f"   • Cosine decay: {self.cosine_steps} шагов (60%)")
		print(f"   • Linear final: {self.linear_steps} шагов (остальное)")
	
	def step(self):
		"""Выполняет один шаг шедулера"""
		self.current_step += 1
		
		if self.current_step <= self.warmup_steps:
			# 1. Warmup: линейный рост
			lr = self.initial_lr * (self.current_step / self.warmup_steps)
			phase = "WARMUP"
			
		elif self.current_step <= self.warmup_steps + self.cosine_steps:
			# 2. Cosine decay
			progress = (self.current_step - self.warmup_steps) / self.cosine_steps
			lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
			phase = "COSINE"
			
		else:
			# 3. Линейное финальное падение
			progress = (self.current_step - self.warmup_steps - self.cosine_steps) / self.linear_steps
			lr = self.initial_lr * 0.1 * (1 - progress * 0.5)
			phase = "FINAL"
		
		# Устанавливаем LR для всех групп параметров
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr
		
		return lr, phase
# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def get_gpu_power():
    """Получение текущей мощности GPU в ваттах"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # мВт → Вт
        pynvml.nvmlShutdown()
        return f"{power:.0f}"
    except:
        return "N/A"


# ================= ПРОДВИНУТЫЙ МОНИТОРИНГ =================
class AdvancedTrainingMonitor:
	def __init__(self, log_dir, debug_mode=1):
		self.debug_mode = debug_mode
		
		if self.debug_mode >= 2:  # Только в отладочном режиме
			print(f"\n🔧 [DEBUG] Создание мониторинга...")
			
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(parents=True, exist_ok=True)
		
		self.metrics = {
			'loss': [], 'lr': [], 'grad_norm': [],
			'step_time': [], 'memory_usage': [],
			'perplexity': [], 'empathy_score': []
		}
		self.quality_scores = []
		self.error_log = self.log_dir / "generation_errors.log"
		
		# Словари для оценки качества
		self.empathy_words = [
			"понимаю", "чувствую", "важно", "ценю", "принимаю", 
			"спасибо", "слышу", "вижу", "замечаю", "уважаю",
			"сопереживаю", "разделяю", "осознаю", "признаю"
		]
		
		self.advice_words = [
			"должен", "надо", "обязан", "рекомендую", 
			"советую", "следует", "нужно", "стоит"
		]

	

	def log_batch(self, step, loss, lr, grad_norm=None, memory_gb=None, 
				  step_time=None, phase="TRAIN", perplexity=None, empathy_score=None):
		"""Логирование метрик батча"""
		self.metrics['loss'].append(loss)
		self.metrics['lr'].append(lr)
		
		if grad_norm is not None:
			self.metrics['grad_norm'].append(grad_norm)
		if memory_gb is not None:
			self.metrics['memory_usage'].append(memory_gb)
		if step_time is not None:
			self.metrics['step_time'].append(step_time)
		if perplexity is not None:
			self.metrics['perplexity'].append(perplexity)
		if empathy_score is not None:
			self.metrics['empathy_score'].append(empathy_score)
		
		# Сохраняем в CSV (всегда, но вывод только в отладочном режиме)
		self.save_to_csv(step, loss, lr, memory_gb, phase, perplexity, empathy_score)
	
	def save_to_csv(self, step, loss, lr, memory_gb, phase, perplexity=None, empathy_score=None):
		"""Сохранение лога в CSV - ТИХАЯ ВЕРСИЯ"""
		csv_file = self.log_dir / "advanced_training_log.csv"
		
		if self.debug_mode >= 2:
			print(f"\n💾 [DEBUG] save_to_csv шаг {step}...")
		
		try:
			write_header = not csv_file.exists()
			
			with open(csv_file, 'a', encoding='utf-8', newline='') as f:
				if write_header:
					f.write("timestamp,step,loss,lr,memory_gb,phase,perplexity,empathy_score\n")
				
				perp_str = f"{perplexity:.2f}" if perplexity is not None else ""
				empathy_str = f"{empathy_score:.3f}" if empathy_score is not None else ""
				line = f"{datetime.now().isoformat()},{step},{loss:.6f},{lr:.6f},{memory_gb:.1f},{phase},{perp_str},{empathy_str}\n"
				f.write(line)
			
			if self.debug_mode >= 2:
				print(f"   ✅ Успешно сохранено")
			
		except Exception as e:
			if self.debug_mode >= 2:
				print(f"   ❌ ОШИБКА save_to_csv: {e}")
			
			# Попробуем записать в другой файл
			backup = Path.cwd() / f"backup_log_{datetime.now().strftime('%H%M%S')}.csv"
			try:
				with open(backup, 'w') as f:
					f.write(f"Ошибка: {e}\n")
				print(f"   💾 Создан backup: {backup}")
			except:
				pass
	
	# Остальные методы остаются без изменений...
	
	def log_problematic_response(self, prompt, response, issue):
		"""Логирование проблемных ответов"""
		with open(self.error_log, 'a', encoding='utf-8') as f:
			f.write(f"\n[{datetime.now()}] {issue}\n")
			f.write(f"Prompt: {prompt}\n")
			f.write(f"Response: {response}\n")
			f.write("-"*80 + "\n")
	
	def calculate_perplexity(self, model, val_data, batch_size=2):
		"""Упрощенный расчет perplexity (без ошибки pad_token_id)"""
		with ValidationMode(model):
			total_loss = 0.0
			num_batches = 0
			
			with torch.no_grad():
				for i in range(0, len(val_data), batch_size):
					if i + batch_size > len(val_data):
						continue
					
					batch = val_data[i:i+batch_size].cuda()
					outputs = model(batch, labels=batch)
					
					if outputs.loss is not None and not torch.isnan(outputs.loss):
						total_loss += outputs.loss.item()
						num_batches += 1
			
			if num_batches == 0:
				return float('inf')
			
			avg_loss = total_loss / num_batches
			# Ограничиваем для численной стабильности
			avg_loss = min(avg_loss, 50)
			return math.exp(avg_loss)
	
	def calculate_empathy_score(self, text):
		"""Расчет оценки эмпатии по словарю"""
		if not text:
			return 0.0
		
		text_lower = text.lower()
		empathy_count = sum(1 for word in self.empathy_words if word in text_lower)
		
		# Нормализуем к 0-1, но не слишком строго
		max_empathy = min(len(self.empathy_words), 5)  # Максимум 5 слов эмпатии в ответе
		return min(empathy_count / max_empathy, 1.0)
	
	def advanced_quality_check(self, model, tokenizer, step, adaptive_temp=True):
		"""Продвинутая проверка качества с адаптивными настройками"""
		test_prompts = [
			"Пациент: Не могу перестать волноваться.",
			"Пациент: Чувствую себя очень одиноко.",
			"Пациент: Как найти смысл в жизни?"
		]
		
		scores = []
		empathy_scores = []
		responses = []
		
		with GenerationMode(model):  # ✅ Используем режим генерации
			# Адаптивная температура на основе предыдущих результатов
			if adaptive_temp and self.quality_scores:
				last_avg_score = self.quality_scores[-1][1] if self.quality_scores else 0.5
				# Динамическая настройка температуры
				temperature = max(0.6, 0.9 - (last_avg_score * 0.3))
			else:
				temperature = 0.729  # Базовое значение
			
			for prompt in test_prompts:
				response = self.generate_adaptive_response(model, tokenizer, prompt, temperature)
				score = self.evaluate_response_comprehensive(prompt, response)
				empathy_score = self.calculate_empathy_score(response)
				
				scores.append(score)
				empathy_scores.append(empathy_score)
				responses.append(response)
		
		avg_score = sum(scores) / len(scores) if scores else 0
		avg_empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0
		
		self.quality_scores.append((step, avg_score, avg_empathy, temperature))
		
		# Сохраняем результаты проверки
		quality_file = self.log_dir / "advanced_quality_checks.json"
		quality_data = {
			'step': step,
			'timestamp': datetime.now().isoformat(),
			'avg_score': avg_score,
			'avg_empathy': avg_empathy,
			'temperature': temperature,
			'tests': []
		}
		
		for prompt, response, score, empathy in zip(test_prompts, responses, scores, empathy_scores):
			quality_data['tests'].append({
				'prompt': prompt,
				'response': response,
				'score': score,
				'empathy_score': empathy
			})
		
		if quality_file.exists():
			with open(quality_file, 'r', encoding='utf-8') as f:
				existing = json.load(f)
		else:
			existing = []
		
		existing.append(quality_data)
		
		with open(quality_file, 'w', encoding='utf-8') as f:
			json.dump(existing, f, ensure_ascii=False, indent=2)
		
		return avg_score, avg_empathy, temperature
	
	def generate_adaptive_response(self, model, tokenizer, prompt, temperature=0.729):
		"""Генерация ответа с адаптивными параметрами"""
		try:
			full_prompt = f"{prompt}\n\nПсихолог:"
			inputs = tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
			
			# Адаптивный top_p на основе температуры
			top_p = 0.95 if temperature > 0.8 else 0.9
			
			with torch.no_grad():
				outputs = model.generate(
					**inputs,
					max_new_tokens=256,
					min_new_tokens=16,
					temperature=temperature,
					do_sample=True,
					top_p=top_p,
					top_k=50,
					pad_token_id=tokenizer.eos_token_id,
					num_return_sequences=1,
					repetition_penalty=1.1,
					length_penalty=0.8
				)
			
			response = tokenizer.decode(outputs[0], skip_special_tokens=True)
			response = response[len(full_prompt):].strip()
			
			# Базовая очистка
			response = self.clean_response(response)
			
			return response
		except Exception as e:
			self.log_problematic_response(prompt, str(e), "Ошибка генерации")
			return ""
	
	def clean_response(self, text):
		"""Очистка ответа"""
		if not text:
			return ""
		
		# Удаляем артефакты
		text = text.replace('�', '').replace('\x00', '')
		
		# Обрезаем по стоп-фразам
		stops = ['\nПациент:', '\nПсихолог:', '\n---', '\n===']
		for stop in stops:
			if stop in text:
				text = text.split(stop)[0].strip()
		
		# Убираем лишние пробелы
		text = ' '.join(text.split())
		
		return text
	
	def evaluate_response_comprehensive(self, prompt, response):
		"""Комплексная оценка качества ответа"""
		if not response:
			return 0.0
		
		score = 0.0
		words = response.split()
		
		# 1. Базовая длина (5-80 слов)
		if 5 <= len(words) <= 80:
			score += 1.0
		
		# 2. Эмпатия (уже считается отдельно, но добавляем бонус)
		empathy_score = self.calculate_empathy_score(response)
		score += empathy_score  # Добавляем прямо как часть оценки
		
		# 3. Вопросы (важно для психолога)
		if '?' in response:
			score += 1.0
		
		# 4. Отсутствие советов
		if not any(word in response.lower() for word in self.advice_words):
			score += 1.0
		
		# 5. Уникальность слов (меньше повторов)
		if len(words) > 5:
			unique_words = len(set(words))
			if unique_words / len(words) > 0.6:
				score += 1.0
		
		# 6. Релевантность запросу
		prompt_words = set(prompt.lower().split()[:10])
		response_words = set(response.lower().split())
		if len(prompt_words.intersection(response_words)) >= 1:
			score += 1.0
		
		# 7. Структура предложений (наличие точек)
		if '.' in response:
			score += 0.5
		
		# Нормализуем к 0-1 (максимум 7.5 баллов)
		return min(score / 7.5, 1.0)

# ================= УЛУЧШЕННОЕ СОХРАНЕНИЕ =================

def save_checkpoint(model, tokenizer, optimizer, step, loss, epoch, checkpoint_dir, 
					is_best=False, scheduler=None, monitor=None):
	"""
	Улучшенное сохранение чекпоинта с метриками
	"""
	try:
		checkpoint_dir = Path(checkpoint_dir)
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		
		print(f"   💾 Сохранение чекпоинта шаг {step}...")
		
		# Сохраняем модель
		model.save_pretrained(str(checkpoint_dir))
		tokenizer.save_pretrained(str(checkpoint_dir))
		
		# Подготовка состояния
		checkpoint_state = {
			'step': step,
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': float(loss),
			'batch_size': BATCH_SIZE,
			'learning_rate': LEARNING_RATE,
			'timestamp': datetime.now().isoformat(),
		}
		
		if scheduler:
			checkpoint_state['scheduler_step'] = scheduler.current_step
		
		if monitor and monitor.quality_scores:
			checkpoint_state['last_quality'] = monitor.quality_scores[-1] if monitor.quality_scores else None
		
		torch.save(checkpoint_state, checkpoint_dir / "checkpoint.pt")
		
		# Сохраняем информацию о чекпоинте
		info_file = checkpoint_dir / "checkpoint_info.txt"
		with open(info_file, 'w', encoding='utf-8') as f:
			f.write(f"ЧЕКПОИНТ {step}\n")
			f.write(f"Эпоха: {epoch}\n")
			f.write(f"Loss: {loss:.6f}\n")
			f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			if is_best:
				f.write(f"\n🏆 СТАТУС: ЛУЧШАЯ МОДЕЛЬ\n")
		
		print(f"   ✅ Чекпоинт сохранён")
		return True
		
	except Exception as e:
		print(f"   ❌ Ошибка при сохранении: {e}")
		return False

def load_last_checkpoint(checkpoint_dir, model, optimizer=None):
	"""Загрузка последнего чекпоинта при ошибках"""
	try:
		checkpoint_dir = Path(checkpoint_dir)
		checkpoints = sorted(checkpoint_dir.glob("step_*"), 
						   key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0,
						   reverse=True)
		
		if checkpoints:
			last_checkpoint = checkpoints[0]
			checkpoint = torch.load(last_checkpoint / "checkpoint.pt", map_location='cpu')
			
			model.load_state_dict(checkpoint['model_state_dict'])
			if optimizer and 'optimizer_state_dict' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			
			print(f"✅ Загружен чекпоинт: {last_checkpoint.name}")
			return checkpoint['step'], checkpoint['loss'], checkpoint['epoch']
	
	except Exception as e:
		print(f"❌ Ошибка загрузки чекпоинта: {e}")
	
	return 0, float('inf'), 0

# ================= ПУТИ =================
BASE_DIR = Path("D:/Ruzanna")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints_advanced"
FINAL_MODEL_DIR = BASE_DIR / "final_model_advanced"
LOGS_DIR = BASE_DIR / "logs_advanced"
DATA_DIR = Path("C:/Files/processed_epitome")

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Инициализируем продвинутый мониторинг с выбранным режимом
monitor = AdvancedTrainingMonitor(LOGS_DIR, debug_mode=DEBUG_MODE)

# Только в отладочном режиме показываем проверку
if DEBUG_MODE >= 2:
	print(f"\n🔍 ПРОВЕРКА МОНИТОРИНГА:")
	print(f"   • log_dir: {monitor.log_dir}")
	
	# Тестовая запись
	monitor.save_to_csv(0, 1.0, 1e-4, 5.0, "TEST", 10.0, 0.5)

# ПРОВЕРКА СРАЗУ ПОСЛЕ СОЗДАНИЯ
print(f"\n🔍 ПРОВЕРКА МОНИТОРИНГА:")
print(f"   • log_dir: {monitor.log_dir}")
print(f"   • Существует: {monitor.log_dir.exists()}")

# Тестовая запись через monitor
monitor.save_to_csv(0, 1.0, 1e-4, 5.0, "TEST", 10.0, 0.5)

# Проверим файл
csv_file = monitor.log_dir / "advanced_training_log.csv"
print(f"   • CSV файл создан: {csv_file.exists()}")
if csv_file.exists():
	print(f"   • Размер файла: {csv_file.stat().st_size} байт")
	with open(csv_file, 'r') as f:
		print(f"   • Содержимое:\n{f.read()}")

# ================= ЗАГРУЗКА ДАННЫХ =================
print(f"\n📂 Загрузка качественных данных...")

data_path = DATA_DIR / "quality_psych_dialogues_enhanced.json"
if not data_path.exists():
	data_path = DATA_DIR / "quality_psych_dialogues.json"

if data_path.exists():
	with open(data_path, 'r', encoding='utf-8') as f:
		dialogues = json.load(f)
	
	print(f"✅ Загружено {len(dialogues)} диалогов")
	
	texts = [dialogue['text'] for dialogue in dialogues]
	
else:
	print(f"❌ Файл не найден: {data_path}")
	sys.exit(1)

# ================= ТОКЕНИЗАЦИЯ =================
print(f"\n🔤 Токенизация данных...")

tokenizer = GPT2Tokenizer.from_pretrained("C:/Files/datasets/neo")
tokenizer.pad_token = tokenizer.eos_token

all_tokens = []
for text in texts:
	tokens = tokenizer.encode(
		text,
		max_length=MAX_LENGTH,
		truncation=True,
		padding='max_length',
		return_tensors='pt'
	)
	all_tokens.append(tokens)

all_tokens = torch.cat(all_tokens, dim=0)

# Разделение
indices = torch.randperm(len(all_tokens))
all_tokens = all_tokens[indices]

split_idx = int(0.85 * len(all_tokens))
train_data = all_tokens[:split_idx]
val_data = all_tokens[split_idx:]

print(f"   Train: {len(train_data)} примеров")
print(f"   Validation: {len(val_data)} примеров")

# ================= ЗАГРУЗКА МОДЕЛИ =================
print(f"\n🧠 Загрузка модели GPT-Neo 2.7B...")

quant_config = BitsAndBytesConfig(
	load_in_8bit=True,
	llm_int8_threshold=6.0,
)

model = GPTNeoForCausalLM.from_pretrained(
	"C:/Files/datasets/neo",
	quantization_config=quant_config,
	device_map="auto",
	torch_dtype=torch.float16,
)

print(f"✅ Модель загружена")

# ================= ОПТИМИЗАТОР =================
print(f"\n⚡ Настройка оптимизатора...")

optimizer = bnb.optim.AdamW8bit(
	model.parameters(),
	lr=LEARNING_RATE,
	betas=(0.9, 0.95),
	weight_decay=0.01,
)

# ================= РАСЧЕТ ШАГОВ И ШЕДУЛЕР =================

# НАЙДИТЕ ЭТУ СТРОКУ (~740) И ИСПРАВЬТЕ:
total_batches = len(train_data) // BATCH_SIZE
# total_steps = (total_batches // GRADIENT_ACCUMULATION) * EPOCHS  # ❌ СТАРОЕ

# ⬇️ НОВОЕ:
if GRADIENT_ACCUMULATION > 0:
	total_steps = max(1, (total_batches + GRADIENT_ACCUMULATION - 1) // GRADIENT_ACCUMULATION * EPOCHS)
else:
	total_steps = max(1, total_batches * EPOCHS)

print(f"\n📈 ПЛАН ОБУЧЕНИЯ:")
print(f"   • Всего шагов: {total_steps}")

scheduler = OptimalScheduler(optimizer, total_steps, LEARNING_RATE, WARMUP_RATIO)

# Настройки для улучшенного раннего стоппинга
checkpoint_steps = [25, 50, 100, 200, 400, 600, 800]
best_loss = float('inf')
best_model_step = 0
patience = 3
patience_counter = 0
previous_val_loss = float('inf')
min_delta = 0.001  # Минимальное значимое улучшение

# Счетчики для обработки ошибок
nan_loss_count = 0
max_nan_losses = 3

# ================= ОБУЧЕНИЕ С АДАПТИВНЫМИ НАСТРОЙКАМИ =================
print(f"\n🎯 НАЧИНАЮ ОБУЧЕНИЕ С АДАПТИВНЫМИ НАСТРОЙКАМИ...")

with TrainingMode(model):  # ✅ Автоматически настраивает use_cache и gradient_checkpointing
	print(f"   • Режим: ОБУЧЕНИЕ")
	print(f"   • use_cache: {model.config.use_cache}")
	print(f"   • gradient_checkpointing: {model.is_gradient_checkpointing}")

global_step = 0
start_time = datetime.now()

# Сохраняем начальный чекпоинт
initial_checkpoint_dir = CHECKPOINTS_DIR / "initial_model"
save_checkpoint(model, tokenizer, optimizer, 0, float('inf'), 0, initial_checkpoint_dir)

for epoch in range(EPOCHS):
	print(f"\n{'='*60}")
	print(f"📚 ЭПОХА {epoch+1}/{EPOCHS}")
	print(f"{'='*60}")
	
	epoch_loss = 0.0
	batch_count = 0
	accumulation_count = 0
	epoch_start_time = time.time()      # ⬅️ ДОБАВЬТЕ
	last_print_time = time.time()       # ⬅️ ДОБАВЬТЕ
	
	# Перемешиваем данные
	train_indices = torch.randperm(len(train_data))
	train_data_shuffled = train_data[train_indices]
	
	with TrainingMode(model):  # ⬅️ УБЕДИТЕСЬ ЧТО ЕСТЬ ОТСТУП (4 пробела)
		for batch_idx in range(0, len(train_data_shuffled), BATCH_SIZE):
			if batch_idx + BATCH_SIZE > len(train_data_shuffled):
				continue
				
			batch_start_time = time.time()
			batch = train_data_shuffled[batch_idx:batch_idx+BATCH_SIZE].cuda()
			
			try:
				# Forward pass
				outputs = model(batch, labels=batch)
				loss = outputs.loss
				
				# Проверка на NaN
				if math.isnan(loss.item()):
					nan_loss_count += 1
					print(f"   ⚠️  NaN loss detected ({nan_loss_count}/{max_nan_losses})")
					
					if nan_loss_count >= max_nan_losses:
						print(f"   🔄 Перезагрузка последнего чекпоинта...")
						global_step, _, _ = load_last_checkpoint(CHECKPOINTS_DIR, model, optimizer)
						nan_loss_count = 0
						continue
					
					# Пропускаем проблемный батч
					optimizer.zero_grad()
					continue
				
				loss_value = loss.item()
				epoch_loss += loss_value
				batch_count += 1
				# ================= ВЫВОД ПРОГРЕССА =================
				current_lr = LEARNING_RATE  # начальное значение

				current_time = time.time()
				if current_time - last_print_time > 10:  # Каждые 10 секунд
					progress = (batch_idx / len(train_data_shuffled)) * 100
					avg_loss_so_far = epoch_loss / (batch_count + 1e-8)
	
					# РАСЧЕТ СКОРОСТИ
					elapsed_since_last_print = current_time - last_print_time
					batches_since_last_print = (batch_idx // BATCH_SIZE) - last_batch_count if 'last_batch_count' in locals() else 1
					last_batch_count = batch_idx // BATCH_SIZE
	
					dialogs_per_second = batches_since_last_print * BATCH_SIZE / elapsed_since_last_print if elapsed_since_last_print > 0 else 0
					tokens_per_second = dialogs_per_second * MAX_LENGTH  # Примерная скорость в токенах
	
					if DEBUG_MODE == 1:
					# Цветной вывод (если терминал поддерживает)
						try:
						# Определяем цвет для скорости
							if dialogs_per_second > 0.5:
								speed_color = "\033[92m"  # зеленый
								speed_icon = "🚀"
							elif dialogs_per_second > 0.2:
								speed_color = "\033[93m"  # желтый
								speed_icon = "⚡"
							else:
								speed_color = "\033[91m"  # красный
								speed_icon = "🐌"
			
							reset_color = "\033[0m"
			
							print(f"\r   🔄 {progress:5.1f}% | 📉 {loss_value:7.4f} | 🎛️ {current_lr:.1e} | 🧺 {batch_idx//BATCH_SIZE:4d} | {speed_icon} {speed_color}{dialogs_per_second:5.2f} диал/с{reset_color}", end='', flush=True)
						except:
							# Без цветов если не поддерживается
							print(f"\r   🔄 {progress:5.1f}% | Loss: {loss_value:7.4f} | LR: {current_lr:.2e} | Батч: {batch_idx//BATCH_SIZE:4d} | 🚀 {dialogs_per_second:5.2f} д/с", end='', flush=True)
	
					elif DEBUG_MODE >= 2:
						# Подробный вывод
						print(f"\n   ⏰ {datetime.now().strftime('%H:%M:%S')}")
						print(f"   📍 Батч {batch_idx//BATCH_SIZE} ({progress:.1f}%)")
						print(f"   📉 Loss: {loss_value:.4f} (средн: {avg_loss_so_far:.4f})")
						print(f"   🚀 Скорость: {dialogs_per_second:.2f} д/с (~{tokens_per_second/1000:.1f}K токенов/сек)")
						print(f"   💾 GPU память: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
						print(f"   ⚡ GPU мощность: {get_gpu_power()}W")  # если есть функция получения мощности
	
					last_print_time = current_time

				elif DEBUG_MODE == 1:
					# Быстрое обновление (без расчета скорости)
					progress = (batch_idx / len(train_data_shuffled)) * 100
					print(f"\r   🔄 {progress:5.1f}% | Loss: {loss_value:7.4f} | LR: {current_lr:.2e} | Батч: {batch_idx//BATCH_SIZE:4d} | ⏳...", end='', flush=True)
				# ================= КОНЕЦ ВЫВОДА ПРОГРЕССА =================
				
				# Gradient accumulation
				loss = loss / GRADIENT_ACCUMULATION
				loss.backward()
				
				accumulation_count += 1
				
				# Step с gradient accumulation
				if accumulation_count % GRADIENT_ACCUMULATION == 0:
					# Gradient clipping
					grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					
					# Оптимизатор
					optimizer.step()
					current_lr, phase = scheduler.step()  # ⬅️ ТЕПЕРЬ current_lr обновлен
					optimizer.zero_grad()
					
					global_step += 1
					step_time = time.time() - batch_start_time
					
					# Мониторинг
					memory_gb = torch.cuda.memory_allocated() / 1024**3
					monitor.log_batch(global_step, loss_value, current_lr, grad_norm, memory_gb, step_time, phase)
					
					# Логирование каждые 10 шагов
					if global_step % 10 == 0:
						avg_loss = epoch_loss / batch_count
						elapsed = (datetime.now() - start_time).seconds / 60
						
						print(f"\n   Шаг {global_step} [{phase}]:")
						print(f"   • Loss: {loss_value:.4f} | Avg: {avg_loss:.4f}")
						print(f"   • LR: {current_lr:.2e}")
						print(f"   • Время: {elapsed:.1f} мин")
					
					# Продвинутая проверка качества
					if global_step % 50 == 0:
						quality_score, empathy_score, current_temp = monitor.advanced_quality_check(
							model, tokenizer, global_step, adaptive_temp=True
						)
						print(f"   • Качество: {quality_score:.2f} | Эмпатия: {empathy_score:.2f} | Temp: {current_temp:.3f}")
					
					# Сохранение чекпоинтов
					if global_step in checkpoint_steps:
						checkpoint_dir = CHECKPOINTS_DIR / f"step_{global_step}_epoch_{epoch+1}"
						save_checkpoint(model, tokenizer, optimizer, global_step, 
									  epoch_loss/batch_count, epoch+1, checkpoint_dir, 
									  scheduler=scheduler, monitor=monitor)
				
			except Exception as e:
				print(f"\n   ❌ Ошибка в батче: {e}")
				optimizer.zero_grad()
				continue
	
	# ================= КОНЕЦ ЭПОХИ =================
	if DEBUG_MODE == 1:
		print()  # Переводим строку после прогресс-бара
	
	# Итоги эпохи
	avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
	print(f"\n✅ ЭПОХА {epoch+1} завершена:")
	print(f"   • Train Loss: {avg_epoch_loss:.4f}")
	print(f"   • Шагов: {global_step}")
	
	# Расчет perplexity на валидации
	perplexity = monitor.calculate_perplexity(model, val_data, BATCH_SIZE)
	print(f"   • Perplexity: {perplexity:.2f}")
	
	# Улучшенный ранний стоппинг
	if previous_val_loss != float('inf'):
		improvement = previous_val_loss - perplexity
		
		if improvement < min_delta:
			patience_counter += 1
			print(f"   ⚠️  Малое улучшение perplexity ({improvement:.4f} < {min_delta}). Patience: {patience_counter}/{patience}")
		else:
			patience_counter = 0
			print(f"   ✅ Значительное улучшение perplexity: {improvement:.4f}")
		
		if patience_counter >= patience:
			print(f"\n🚫 РАННЯЯ ОСТАНОВКА: нет значимых улучшений {patience} эпохи подряд")
			break
	
	# Сохранение лучшей модели
	if perplexity < best_loss:
		best_loss = perplexity
		best_model_step = global_step
		
		best_dir = CHECKPOINTS_DIR / f"BEST_epoch_{epoch+1}_perplexity_{best_loss:.2f}"
		save_checkpoint(model, tokenizer, optimizer, global_step, 
					  best_loss, epoch+1, best_dir, is_best=True, 
					  scheduler=scheduler, monitor=monitor)
		print(f"   🏆 НОВАЯ ЛУЧШАЯ МОДЕЛЬ: perplexity={best_loss:.2f}")
	
	previous_val_loss = perplexity
	
	# Сохраняем чекпоинт эпохи
	epoch_checkpoint_dir = CHECKPOINTS_DIR / f"epoch_{epoch+1}_final"
	save_checkpoint(model, tokenizer, optimizer, global_step, avg_epoch_loss, 
				   epoch+1, epoch_checkpoint_dir, scheduler=scheduler, monitor=monitor)

# ================= СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ =================
print(f"\n💾 Сохранение финальной модели...")

try:
	model.save_pretrained(str(FINAL_MODEL_DIR))
	tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
	
	training_info = {
		'total_steps': global_step,
		'final_train_loss': avg_epoch_loss,
		'best_perplexity': best_loss,
		'best_step': best_model_step,
		'epochs_completed': epoch + 1,
		'early_stopped': patience_counter >= patience,
		'final_perplexity': perplexity,
		'batch_size': BATCH_SIZE,
		'learning_rate': LEARNING_RATE,
		'training_time_minutes': (datetime.now() - start_time).seconds / 60,
		'completion_time': datetime.now().isoformat(),
		'adaptive_training': True,
		'advanced_metrics': True,
		'gradient_checkpointing': True,
		'use_cache_strategy': 'adaptive'
	}
	
	with open(FINAL_MODEL_DIR / "training_info.json", 'w', encoding='utf-8') as f:
		json.dump(training_info, f, ensure_ascii=False, indent=2)
	
	print(f"✅ Финальная модель сохранена")
	
except Exception as e:
	print(f"❌ Ошибка сохранения: {e}")

# ================= ФИНАЛЬНЫЙ ТЕСТ =================
print(f"\n🧪 ФИНАЛЬНЫЙ ТЕСТ С АДАПТИВНЫМИ НАСТРОЙКАМИ...")
with GenerationMode(model):  # ✅ ГЕНЕРАЦИЯ: cache=ON, gc=OFF
	print(f"   • Режим: ГЕНЕРАЦИЯ")
	print(f"   • use_cache: {model.config.use_cache}")
	print(f"   • gradient_checkpointing: {model.is_gradient_checkpointing}")

test_prompts = [
	"Пациент: Не могу перестать волноваться.",
	"Пациент: Чувствую себя очень одиноко.",
	"Пациент: Как найти смысл в жизни?",
	"Пациент: Всё бессмысленно, не вижу причин продолжать.",
	"Пациент: Боюсь, что никогда не изменюсь."
]

for i, prompt in enumerate(test_prompts):
	try:
		# Используем адаптивную температуру на основе качества модели
		last_quality = monitor.quality_scores[-1][1] if monitor.quality_scores else 0.5
		adaptive_temp = max(0.6, 0.9 - (last_quality * 0.3))
		
		with GenerationMode(model):  # Каждый генерационный вызов в правильном режиме
			response = monitor.generate_adaptive_response(model, tokenizer, prompt, adaptive_temp)
			score = monitor.evaluate_response_comprehensive(prompt, response)
			empathy_score = monitor.calculate_empathy_score(response)
		
		print(f"\n{i+1}. 💭 {prompt}")
		print(f"   🌡️  Temp: {adaptive_temp:.3f}")
		print(f"   💬 {response[:120]}{'...' if len(response) > 120 else ''}")
		print(f"   📊 Оценка: {score:.2f} | Эмпатия: {empathy_score:.2f}")
		
	except Exception as e:
		print(f"\n{i+1}. ❌ Ошибка: {e}")

print(f"\n{'='*80}")
print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"{'='*80}")
print(f"📊 ИТОГОВЫЕ МЕТРИКИ:")
print(f"   • Шагов: {global_step}")
print(f"   • Лучший perplexity: {best_loss:.2f}")
print(f"   • Финальный perplexity: {perplexity:.2f}")
print(f"   • Ранняя остановка: {'Да' if patience_counter >= patience else 'Нет'}")
print(f"   • NaN обработок: {nan_loss_count}")
print(f"   • Время: {(datetime.now() - start_time).seconds/60:.1f} мин")
print(f"   • Использованные режимы:")
print(f"      - Обучение: cache=OFF, gradient_checkpointing={model.is_gradient_checkpointing}")
print(f"      - Валидация: cache=OFF, gradient_checkpointing=OFF")
print(f"      - Генерация: cache=ON, gradient_checkpointing=OFF")
print(f"{'='*80}")
