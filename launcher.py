#!/usr/bin/env python3
"""
RUZANNA - Лаунчер для обучения психологического ИИ
Управление конфигурацией и запуск обучения
"""

import os
import sys
import json
import time
from pathlib import Path
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

# Добавляем путь для импорта из core
sys.path.insert(0, str(Path(__file__).parent / "core"))

# Colorama для цветного вывода
try:
    import colorama
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORS_ENABLED = True
except ImportError:
    class DummyColors:
        def __getattr__(self, name):
            return ""
    Fore = Back = Style = DummyColors()
    COLORS_ENABLED = False

# Импорт нашего конфиг-менеджера
try:
    from config_loader import ConfigManager
    CONFIG_MANAGER = ConfigManager("./configs")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что файл core/config_loader.py существует")
    sys.exit(1)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def clear_screen():
    """Очистка экрана"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Печать заголовка"""
    clear_screen()
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*70}")
    print(f"{' '*20}🧠 RUZANNA - ПСИХОЛОГИЧЕСКИЙ ИИ ТРЕНЕР")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'='*70}{Style.RESET_ALL}\n")

def print_success(msg):
    """Успешное сообщение"""
    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")

def print_warning(msg):
    """Предупреждение"""
    print(f"{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}")

def print_error(msg):
    """Ошибка"""
    print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")

def print_info(msg):
    """Информационное сообщение"""
    print(f"{Fore.CYAN}ℹ️  {msg}{Style.RESET_ALL}")

def print_step(msg):
    """Шаг процесса"""
    print(f"{Fore.BLUE}➡️  {msg}{Style.RESET_ALL}")

def progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    """Отображение прогресс-бара"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # Цвет в зависимости от прогресса
    if percent < 33:
        color = Fore.RED
    elif percent < 66:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
    
    print(f'\r{prefix} |{color}{bar}{Style.RESET_ALL}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def get_input(prompt, default=None, input_type=str):
    """Безопасный ввод с дефолтным значением"""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        try:
            value = input(prompt).strip()
            if not value and default is not None:
                return default
            if not value:
                raise ValueError("Значение не может быть пустым")
            return input_type(value)
        except ValueError as e:
            print_error(f"Неверный ввод: {e}")

# ============================================================================
# ФУНКЦИИ УПРАВЛЕНИЯ КОНФИГУРАЦИЕЙ
# ============================================================================

def view_configuration():
    """Просмотр текущей конфигурации"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📋 ТЕКУЩАЯ КОНФИГУРАЦИЯ")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        # Загружаем текущий конфиг
        config = CONFIG_MANAGER.load_full_config()
        params = CONFIG_MANAGER.get_training_params()
        
        # Основные параметры
        print(f"{Fore.YELLOW}🎯 ОСНОВНЫЕ ПАРАМЕТРЫ:{Style.RESET_ALL}")
        print(f"  Модель: {params.get('model_name', 'N/A')}")
        print(f"  Данные: {Path(params.get('data_path', '')).name if params.get('data_path') else 'N/A'}")
        print(f"  Batch size: {params.get('batch_size', 'N/A')}")
        print(f"  Эпохи: {params.get('epochs', 'N/A')}")
        print(f"  Learning rate: {params.get('learning_rate', 'N/A')}")
        print(f"  Max length: {params.get('max_length', 'N/A')}")
        
        # Пути
        print(f"\n{Fore.YELLOW}📁 ПУТИ:{Style.RESET_ALL}")
        paths = config.get('paths', {})
        for key, value in paths.items():
            print(f"  {key}: {value}")
        
        # Модель
        print(f"\n{Fore.YELLOW}🤖 МОДЕЛЬ:{Style.RESET_ALL}")
        model = config.get('model', {})
        for key, value in model.items():
            print(f"  {key}: {value}")
        
        # Дополнительно
        print(f"\n{Fore.YELLOW}⚙️  ДОПОЛНИТЕЛЬНО:{Style.RESET_ALL}")
        print(f"  Device: {config.get('system', {}).get('device', 'N/A')}")
        print(f"  Seed: {config.get('system', {}).get('seed', 'N/A')}")
        print(f"  Precision: {config.get('system', {}).get('precision', 'N/A')}")
        
    except Exception as e:
        print_error(f"Ошибка загрузки конфигурации: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def change_training_params():
    """Изменение параметров обучения"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}✏️  ИЗМЕНЕНИЕ ПАРАМЕТРОВ ОБУЧЕНИЯ")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        config = CONFIG_MANAGER.load_full_config()
        training_config = config.get('training', {})
        
        print_info("Текущие значения:")
        for key, value in training_config.items():
            if key in ['batch_size', 'epochs', 'learning_rate', 'max_length']:
                print(f"  {key}: {value}")
        
        print(f"\n{Fore.YELLOW}Введите новые значения (оставьте пустым для сохранения текущего):{Style.RESET_ALL}")
        
        updates = {'training': {}}
        
        # Batch size
        current_bs = training_config.get('batch_size', 3)
        new_bs = get_input(f"Batch size (1-32)", default=current_bs, input_type=int)
        if 1 <= new_bs <= 32:
            updates['training']['batch_size'] = new_bs
        else:
            print_warning(f"Batch size {new_bs} вне диапазона, оставляем {current_bs}")
            updates['training']['batch_size'] = current_bs
        
        # Learning rate
        current_lr = training_config.get('learning_rate', 0.0002)
        new_lr = get_input(f"Learning rate (1e-5 до 1e-3)", default=current_lr, input_type=float)
        if 1e-5 <= new_lr <= 1e-3:
            updates['training']['learning_rate'] = new_lr
        else:
            print_warning(f"Learning rate {new_lr} вне диапазона, оставляем {current_lr}")
            updates['training']['learning_rate'] = current_lr
        
        # Epochs
        current_epochs = training_config.get('epochs', 3)
        new_epochs = get_input(f"Эпохи (1-10)", default=current_epochs, input_type=int)
        if 1 <= new_epochs <= 10:
            updates['training']['epochs'] = new_epochs
        else:
            print_warning(f"Эпохи {new_epochs} вне диапазона, оставляем {current_epochs}")
            updates['training']['epochs'] = current_epochs
        
        # Max length
        current_ml = config.get('tokenization', {}).get('max_length', 729)
        new_ml = get_input(f"Макс. длина (128-1024)", default=current_ml, input_type=int)
        if 128 <= new_ml <= 1024:
            updates['tokenization'] = {'max_length': new_ml}
        else:
            print_warning(f"Макс. длина {new_ml} вне диапазона, оставляем {current_ml}")
            updates['tokenization'] = {'max_length': current_ml}
        
        # Применяем изменения
        CONFIG_MANAGER.update_custom_config(updates)
        print_success("Параметры успешно обновлены!")
        
        # Показываем новые значения
        print(f"\n{Fore.YELLOW}НОВЫЕ ЗНАЧЕНИЯ:{Style.RESET_ALL}")
        new_config = CONFIG_MANAGER.load_full_config()
        new_training = new_config.get('training', {})
        print(f"  Batch size: {new_training.get('batch_size')}")
        print(f"  Learning rate: {new_training.get('learning_rate')}")
        print(f"  Эпохи: {new_training.get('epochs')}")
        print(f"  Макс. длина: {new_config.get('tokenization', {}).get('max_length')}")
        
    except Exception as e:
        print_error(f"Ошибка изменения параметров: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def select_preset():
    """Выбор пресета"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}🎯 ВЫБОР ПРЕСЕТА ОБУЧЕНИЯ")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        # Получаем список пресетов
        presets_dir = Path("./configs/presets")
        if not presets_dir.exists():
            print_warning("Папка с пресетами не найдена")
            input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
            return
        
        presets = sorted([p.stem for p in presets_dir.glob("*.json")])
        
        if not presets:
            print_warning("Пресеты не найдены")
            input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
            return
        
        print_info("Доступные пресеты:")
        for i, preset in enumerate(presets, 1):
            # Загружаем описание пресета
            preset_path = presets_dir / f"{preset}.json"
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                description = preset_data.get('meta', {}).get('description', 'Без описания')
                print(f"{i}. {preset} - {description}")
            except:
                print(f"{i}. {preset}")
        
        print(f"\n{Fore.YELLOW}0. Вернуться в меню")
        
        while True:
            try:
                choice = int(input(f"\nВыберите пресет (1-{len(presets)}): "))
                if choice == 0:
                    return
                if 1 <= choice <= len(presets):
                    selected = presets[choice-1]
                    
                    # Загружаем пресет для предпросмотра
                    preset_config = CONFIG_MANAGER.load_preset(selected)
                    
                    print(f"\n{Fore.YELLOW}Параметры пресета '{selected}':{Style.RESET_ALL}")
                    if 'training' in preset_config:
                        for key, value in preset_config['training'].items():
                            print(f"  {key}: {value}")
                    
                    confirm = input(f"\n{Fore.YELLOW}Применить пресет '{selected}'? (y/n): {Style.RESET_ALL}").lower()
                    if confirm == 'y':
                        # Обновляем кастомный конфиг
                        CONFIG_MANAGER.update_custom_config(preset_config)
                        print_success(f"Пресет '{selected}' применён!")
                        
                        # Показываем применённые параметры
                        new_config = CONFIG_MANAGER.load_full_config()
                        training = new_config.get('training', {})
                        print(f"\n{Fore.CYAN}Текущие параметры:{Style.RESET_ALL}")
                        print(f"  Batch size: {training.get('batch_size')}")
                        print(f"  Learning rate: {training.get('learning_rate')}")
                        print(f"  Эпохи: {training.get('epochs')}")
                        
                        time.sleep(2)
                    break
                else:
                    print_error(f"Пожалуйста, введите число от 1 до {len(presets)}")
            except ValueError:
                print_error("Неверный ввод. Введите число.")
    
    except Exception as e:
        print_error(f"Ошибка выбора пресета: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def create_preset():
    """Создание нового пресета"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}🆕 СОЗДАНИЕ НОВОГО ПРЕСЕТА")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        preset_name = input(f"{Fore.YELLOW}Название пресета (латинскими буквами, без пробелов): {Style.RESET_ALL}").strip()
        
        if not preset_name:
            print_error("Название не может быть пустым")
            return
        
        # Проверяем, не существует ли уже
        preset_path = Path("./configs/presets") / f"{preset_name}.json"
        if preset_path.exists():
            overwrite = input(f"Пресет '{preset_name}' уже существует. Перезаписать? (y/n): ").lower()
            if overwrite != 'y':
                print_info("Создание отменено")
                return
        
        print_info("\nВведите параметры пресета:")
        print_info("(оставьте пустым для значений по умолчанию из текущей конфигурации)\n")
        
        # Получаем текущие значения
        current_config = CONFIG_MANAGER.load_full_config()
        current_training = current_config.get('training', {})
        
        preset_config = {
            "meta": {
                "description": f"Пресет '{preset_name}'",
                "created_by": "launcher",
                "created_at": datetime.now().isoformat()
            },
            "training": {}
        }
        
        # Собираем параметры
        params = [
            ("batch_size", "Размер батча", int, 1, 32, current_training.get('batch_size', 3)),
            ("learning_rate", "Скорость обучения (например 0.0002)", float, 1e-5, 1e-3, current_training.get('learning_rate', 0.0002)),
            ("epochs", "Количество эпох", int, 1, 10, current_training.get('epochs', 3)),
            ("warmup_ratio", "Доля warmup (0.0-1.0)", float, 0.0, 1.0, current_training.get('warmup_ratio', 0.9)),
        ]
        
        for param_key, param_desc, param_type, min_val, max_val, default in params:
            while True:
                try:
                    prompt = f"{param_desc} [{min_val}-{max_val}] (по умолчанию: {default}): "
                    value_str = input(prompt).strip()
                    
                    if not value_str:
                        value = default
                    else:
                        value = param_type(value_str)
                    
                    if min_val <= value <= max_val:
                        preset_config["training"][param_key] = value
                        break
                    else:
                        print_error(f"Значение должно быть в диапазоне [{min_val}, {max_val}]")
                except ValueError:
                    print_error("Неверный формат")
        
        # Добавляем описание
        description = input(f"\nОписание пресета (опционально): ").strip()
        if description:
            preset_config["meta"]["description"] = description
        
        # Сохраняем пресет
        CONFIG_MANAGER.save_preset(preset_name, preset_config)
        print_success(f"Пресет '{preset_name}' успешно создан!")
        
        # Предлагаем применить
        apply = input(f"\nПрименить созданный пресет? (y/n): ").lower()
        if apply == 'y':
            CONFIG_MANAGER.update_custom_config(preset_config)
            print_success("Пресет применён!")
        
    except Exception as e:
        print_error(f"Ошибка создания пресета: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def manage_configuration():
    """Главное меню управления конфигурацией"""
    while True:
        print_header()
        print(f"{Fore.CYAN}{Style.BRIGHT}⚙️  УПРАВЛЕНИЕ КОНФИГУРАЦИЕЙ")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        print("1. 📋 Просмотреть текущую конфигурацию")
        print("2. ✏️  Изменить параметры обучения")
        print("3. 🎯 Выбрать пресет")
        print("4. 🆕 Создать новый пресет")
        print("5. ↩️  Вернуться в главное меню")
        
        choice = input(f"\n{Fore.YELLOW}Выберите действие (1-5): {Style.RESET_ALL}").strip()
        
        if choice == "1":
            view_configuration()
        elif choice == "2":
            change_training_params()
        elif choice == "3":
            select_preset()
        elif choice == "4":
            create_preset()
        elif choice == "5":
            break
        else:
            print_error("Неверный выбор. Пожалуйста, выберите 1-5.")

# ============================================================================
# ФУНКЦИИ ОБУЧЕНИЯ И МОНИТОРИНГА
# ============================================================================

def start_training():
    """Запуск процесса обучения"""
    print_header()
    print(f"{Fore.CYAN}{Style.Bright}🚀 ЗАПУСК ОБУЧЕНИЯ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # 1. Выбор директории
    output_dir = select_output_directory()
    if output_dir is None:  # Явная проверка на None
        print_info("Запуск отменен")
        time.sleep(1)
        return
    
    # 2. Проверка доступности GPU
    print_step("Проверка оборудования...")
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_success(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print_warning("⚠️  CUDA не доступна, используется CPU")
    except:
        pass

    # 2. Загружаем конфигурацию
    config = CONFIG_MANAGER.load_full_config()
    
    # 3. Обновляем конфиг с выбранным путем
    config['paths']['base'] = output_dir

    try:
        config = CONFIG_MANAGER.load_full_config()
        params = CONFIG_MANAGER.get_training_params()
        
        # Обновляем путь в конфиге
        if 'paths' not in config:
            config['paths'] = {}
        config['paths']['base'] = output_dir
        config['paths']['logs'] = str(Path(output_dir) / 'logs')
        config['paths']['checkpoints'] = str(Path(output_dir) / 'checkpoints')
        
        print_success(f"Конфигурация загружена")
        print_info(f"Директория результатов: {output_dir}")
        
    except Exception as e:
        print_error(f"Ошибка конфигурации: {e}")
        return
    
    # 4. Подтверждение запуска
    print(f"\n{Fore.YELLOW}📋 ПАРАМЕТРЫ ЗАПУСКА:{Style.RESET_ALL}")
    print(f"  Модель: {params.get('model_name', 'N/A')}")
    print(f"  Batch size: {params.get('batch_size', 'N/A')}")
    print(f"  Эпохи: {params.get('epochs', 'N/A')}")
    print(f"  Learning rate: {params.get('learning_rate', 'N/A'):.2e}")
    print(f"  Сохранение в: {output_dir}")
    
    print(f"\n{Fore.RED}{Style.BRIGHT}⚠️  ВНИМАНИЕ: Обучение может занять несколько часов!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    confirm = input(f"\n{Fore.YELLOW}Запустить обучение? (y/n): {Style.RESET_ALL}").lower()
    
    if confirm != 'y':
        print_info("Запуск отменен")
        time.sleep(1)
        return
        
        # Подготовка к запуску
        print_step("Подготовка к запуску...")
        
        # Создаем папки если нужно
        Path("./logs").mkdir(exist_ok=True)
        Path("./checkpoints").mkdir(exist_ok=True)
        
        # Формируем команду для запуска workout.py
        cmd = [
            sys.executable, "workout.py",
            "--config", "./configs/custom.json" if Path("./configs/custom.json").exists() else "./configs/base.json"
        ]
        
        # Добавляем пресет если выбран
        current_preset = None
        if Path("./configs/custom.json").exists():
            with open("./configs/custom.json", 'r') as f:
                custom = json.load(f)
                if custom.get('_preset'):
                    current_preset = custom['_preset']
        
        if current_preset:
            cmd.extend(["--preset", current_preset])
        
        print(f"\n{Fore.CYAN}Команда запуска:{Style.RESET_ALL}")
        print(f"  {' '.join(cmd)}")
        
        # 5. Запуск workout.py
    print(f"\n{Fore.GREEN}{Style.BRIGHT}▶️  ЗАПУСК ОБУЧЕНИЯ...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        # Формируем команду
        cmd = [
            sys.executable, "workout.py",
            "--output_dir", output_dir
        ]
        
        # Добавляем пресет если выбран
        current_config = CONFIG_MANAGER.config
        if current_config.get('_preset'):
            cmd.extend(["--preset", current_config['_preset']])
        
        print_info(f"Команда: {' '.join(cmd)}")
        
        # Запускаем
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # Вывод в реальном времени
        for line in process.stdout:
            print(line, end='')
        
        # Ожидание завершения
        process.wait()
        
        if process.returncode == 0:
            print_success("\n✅ Обучение успешно завершено!")
            print_info(f"Результаты сохранены в: {output_dir}")
        else:
            print_error(f"\n❌ Обучение завершилось с ошибкой (код: {process.returncode})")
            
    except FileNotFoundError:
        print_error("Файл workout.py не найден!")
    except KeyboardInterrupt:
        print_warning("\nОбучение прервано пользователем")
    except Exception as e:
        print_error(f"Ошибка запуска: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def check_logs():
    """Просмотр логов"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📊 ПРОСМОТР ЛОГОВ")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    logs_dir = Path("./logs")
    
    if not logs_dir.exists():
        print_error("Папка логов не найдена")
        input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
        return
    
    # Ищем лог файлы
    log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.csv"))
    
    if not log_files:
        print_info("Лог файлы не найдены")
    else:
        print_info(f"Найдено {len(log_files)} лог файлов:")
        for i, log_file in enumerate(sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10], 1):
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            size_kb = log_file.stat().st_size // 1024
            print(f"{i}. {log_file.name} ({size_kb} KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    print(f"\n{Fore.YELLOW}Опции:{Style.RESET_ALL}")
    print("1. Просмотреть последний лог")
    print("2. Очистить логи")
    print("3. Вернуться в меню")
    
    choice = input(f"\n{Fore.YELLOW}Выберите действие (1-3): {Style.RESET_ALL}").strip()
    
    if choice == "1" and log_files:
        latest_log = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                content = f.readlines()[-100:]  # Последние 100 строк
            print(f"\n{Fore.CYAN}Последние строки из {latest_log.name}:{Style.RESET_ALL}")
            print("-"*60)
            for line in content:
                print(line.rstrip())
        except Exception as e:
            print_error(f"Ошибка чтения лога: {e}")
    elif choice == "2":
        confirm = input(f"\n{Fore.RED}Удалить все логи? (y/n): {Style.RESET_ALL}").lower()
        if confirm == 'y':
            for log_file in log_files:
                try:
                    log_file.unlink()
                except:
                    pass
            print_success("Логи очищены")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def test_model():
    """Тестирование модели"""
    print_header()
    print(f"{Fore.CYAN}{Style.Bright}🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    print_info("Функция тестирования в разработке...")
    print("\nДоступные опции:")
    print("1. Загрузить чекпоинт")
    print("2. Протестировать на валидационных данных")
    print("3. Интерактивный диалог")
    
    choice = input(f"\n{Fore.YELLOW}Выберите действие (1-3): {Style.RESET_ALL}").strip()
    
    if choice == "1":
        checkpoints_dir = Path("./checkpoints")
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*"))
            if checkpoints:
                print_info(f"Найдено {len(checkpoints)} чекпоинтов:")
                for i, cp in enumerate(sorted(checkpoints, reverse=True)[:5], 1):
                    print(f"{i}. {cp.name}")
            else:
                print_warning("Чекпоинты не найдены")
        else:
            print_error("Папка чекпоинтов не найдена")
    elif choice == "2":
        print_info("Тестирование на валидационных данных...")
        # TODO: Реализовать тестирование
    elif choice == "3":
        print_info("Интерактивный диалог...")
        # TODO: Реализовать интерактивный режим
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def get_directory_history() -> List[Tuple[str, str]]:
    """Получает историю директорий"""
    history_file = Path("./configs/directory_history.json")
    
    if not history_file.exists():
        return []
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        history = []
        for item in history_data.get('history', [])[-3:]:  # Последние 3
            path = item.get('path', '')
            if path:
                name = Path(path).name
                history.append((name, path))
        
        return history
    
    except Exception as e:
        print_warning(f"Ошибка чтения истории: {e}")
        return []

def save_to_history(path: str):
    """Сохраняет путь в историю"""
    history_file = Path("./configs/directory_history.json")
    history_file.parent.mkdir(exist_ok=True)
    
    try:
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"history": []}
        
        # Удаляем если уже есть
        path = str(Path(path).resolve())
        data['history'] = [h for h in data['history'] if h.get('path') != path]
        
        # Добавляем в начало (последний выбор будет первым)
        data['history'].insert(0, {
            "path": path,
            "selected": datetime.now().isoformat(),
            "name": Path(path).name
        })
        
        # Ограничиваем 3 элементами
        data['history'] = data['history'][:3]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print_warning(f"Не удалось сохранить историю: {e}")

def create_new_directory_interactive() -> Optional[str]:
    """Интерактивное создание новой директории"""
    print(f"\n{Fore.CYAN}🆕 СОЗДАНИЕ НОВОЙ ДИРЕКТОРИИ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
    
    # 1. Предлагаем умный дефолт
    default_dir = f"./experiments/psych_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"{Fore.YELLOW}Примеры хороших путей:{Style.RESET_ALL}")
    print(f"  • {default_dir}")
    print(f"  • D:/AI_Experiments/psych_training_{datetime.now().strftime('%Y%m%d')}")
    print(f"  • C:/Projects/Ruzanna/runs/session_{datetime.now().strftime('%H%M')}")
    print(f"\n{Fore.YELLOW}Совет:{Style.RESET_ALL} Используйте дату в названии для порядка!")
    
    # 2. Получаем путь
    path_input = input(f"\n{Fore.YELLOW}Введите путь [{default_dir}]: {Style.RESET_ALL}").strip()
    
    if not path_input:
        path_input = default_dir
        print_info(f"Используется путь по умолчанию: {path_input}")
    
    # 3. Обрабатываем путь
    try:
        path = Path(path_input)
        
        # Если путь относительный, делаем абсолютным относительно trener/
        if not path.is_absolute():
            path = (Path(__file__).parent / path).resolve()
        
        # Создаем директорию
        path.mkdir(parents=True, exist_ok=True)
        
        # Создаем структуру внутри
        subdirs = ['logs', 'checkpoints', 'models', 'configs', 'tmp']
        for subdir in subdirs:
            (path / subdir).mkdir(exist_ok=True)
        
        # Создаем info файл
        info = {
            "created": datetime.now().isoformat(),
            "purpose": "Ruzanna психологический AI",
            "training_session": True
        }
        with open(path / 'session_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print_success(f"✅ Директория создана: {path}")
        print_info(f"   Поддиректории: {', '.join(subdirs)}")
        
        # Сохраняем в историю
        save_to_history(str(path))
        
        return str(path)
        
    except Exception as e:
        print_error(f"❌ Ошибка создания директории: {e}")
        return None

def select_output_directory() -> Optional[str]:
    """Интеллектуальный выбор директории для результатов"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📁 КУДА СОХРАНИТЬ РЕЗУЛЬТАТЫ?{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # 1. Проверяем историю
    history = get_directory_history()
    
    if history:
        print(f"{Fore.YELLOW}📚 ИСТОРИЯ ВЫБОРА:{Style.RESET_ALL}")
        for i, (name, path) in enumerate(history, 1):
            # Проверяем существование
            exists = "✅" if Path(path).exists() else "❌"
            print(f"{i}. {exists} {name}")
    
    print(f"\n{Fore.YELLOW}⚡ БЫСТРЫЕ ВАРИАНТЫ:{Style.RESET_ALL}")
    print("n. 📁 Новая директория (ввести путь)")
    
    if history:
        print("0. ↩️  Отмена")
    
    print(f"\n{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    
    # Выбор пользователя
    while True:
        choice = input(f"\n{Fore.YELLOW}Выберите вариант (1-{len(history)}, n, или 0): {Style.RESET_ALL}").strip().lower()
        
        if choice == '0' and history:
            return None  # Отмена
        elif choice == 'n':
            return create_new_directory_interactive()
        elif choice.isdigit() and history:
            idx = int(choice) - 1
            if 0 <= idx < len(history):
                selected_path = history[idx][1]
                if Path(selected_path).exists():
                    print_success(f"Выбрано: {selected_path}")
                    save_to_history(selected_path)
                    return selected_path
                else:
                    print_warning(f"Директория не существует: {selected_path}")
                    recreate = input(f"Создать её заново? (y/n): ").lower()
                    if recreate == 'y':
                        Path(selected_path).mkdir(parents=True, exist_ok=True)
                        print_success(f"Директория создана: {selected_path}")
                        save_to_history(selected_path)
                        return selected_path
        else:
            print_error("Неверный выбор. Попробуйте снова.")

def select_from_history(path_manager):
    """Выбор из истории"""
    menu_items = path_manager.get_history_menu()
    
    if not menu_items:
        print_warning("История пуста")
        return None
    
    print(f"\n{Fore.CYAN}История:{Style.RESET_ALL}")
    for display, path in menu_items:
        if path:  # Это элемент с путем
            print(f"{display}")
        else:     # Это заголовок
            print(f"\n{display}")
    
    print(f"\n{Fore.YELLOW}0. ↩️  Назад{Style.RESET_ALL}")
    
    try:
        choice = int(input(f"\n{Fore.YELLOW}Выберите директорию: {Style.RESET_ALL}"))
        if choice == 0:
            return None
        
        # Считаем реальные элементы с путями
        path_items = [(d, p) for d, p in menu_items if p]
        if 1 <= choice <= len(path_items):
            selected_path = path_items[choice-1][1]
            print_success(f"Выбрано: {selected_path}")
            
            # Показываем содержимое
            if Path(selected_path).exists():
                print_info("Содержимое:")
                for item in Path(selected_path).iterdir():
                    if item.is_dir():
                        print(f"  📁 {item.name}/")
                    else:
                        print(f"  📄 {item.name}")
            
            confirm = input(f"\n{Fore.YELLOW}Использовать эту директорию? (y/n): {Style.RESET_ALL}").lower()
            if confirm == 'y':
                return selected_path
        
    except (ValueError, IndexError):
        print_error("Неверный выбор")
    
    return None

def create_new_directory(path_manager):
    """Создание новой директории"""
    print(f"\n{Fore.CYAN}🆕 СОЗДАНИЕ НОВОЙ ДИРЕКТОРИИ{Style.RESET_ALL}")
    
    # Предлагаем базовый путь
    base_path = input(f"{Fore.YELLOW}Базовый путь [./experiments]: {Style.RESET_ALL}").strip()
    if not base_path:
        base_path = "./experiments"
    
    # Имя эксперимента
    exp_name = input(f"{Fore.YELLOW}Имя эксперимента: {Style.RESET_ALL}").strip()
    if not exp_name:
        exp_name = f"psych_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    try:
        exp_dir = path_manager.create_experiment_dir(base_path, exp_name)
        print_success(f"Создана директория: {exp_dir}")
        return str(exp_dir)
    except Exception as e:
        print_error(f"Ошибка создания: {e}")
        return None

def specify_custom_path(path_manager):
    """Указание своего пути"""
    print(f"\n{Fore.CYAN}📂 УКАЗАНИЕ СВОЕГО ПУТИ{Style.RESET_ALL}")
    
    custom_path = input(f"{Fore.YELLOW}Введите полный путь: {Style.RESET_ALL}").strip()
    
    if not custom_path:
        print_warning("Путь не указан")
        return None
    
    path = Path(custom_path)
    
    # Проверяем существование
    if not path.exists():
        create = input(f"{Fore.YELLOW}Директория не существует. Создать? (y/n): {Style.RESET_ALL}").lower()
        if create == 'y':
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"Директория создана: {path}")
            except Exception as e:
                print_error(f"Ошибка создания: {e}")
                return None
        else:
            return None
    
    # Добавляем в историю
    path_manager._add_to_history('experiments', str(path), 'last_experiment')
    
    return str(path)

# ============================================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================================

def main():
    """Главное меню лаунчера"""
    # Инициализация
    print_header()
    
    # Проверяем наличие workout.py
    if not Path("workout.py").exists():
        print_error("Файл workout.py не найден!")
        print("Убедитесь, что он находится в той же папке")
        input("\nНажмите Enter для выхода...")
        return
    
    # Загружаем начальную конфигурацию
    try:
        CONFIG_MANAGER.load_full_config()
        print_success("Конфигурация загружена")
    except Exception as e:
        print_error(f"Ошибка загрузки конфигурации: {e}")
    
    time.sleep(1)
    
    # Главный цикл меню
    while True:
        print_header()
        print(f"{Fore.CYAN}{Style.BRIGHT}🏠 ГЛАВНОЕ МЕНЮ")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        print("1. 🚀 Запустить обучение")
        print("2. ⚙️  Управление конфигурацией")
        print("3. 📊 Просмотреть логи")
        print("4. 🧪 Тестирование модели")
        print("5. ❌ Выход")
        
        choice = input(f"\n{Fore.YELLOW}Выберите действие (1-5): {Style.RESET_ALL}").strip()
        
        if choice == "1":
            start_training()
        elif choice == "2":
            manage_configuration()
        elif choice == "3":
            check_logs()
        elif choice == "4":
            test_model()
        elif choice == "5":
            print_header()
            print(f"{Fore.GREEN}{Style.BRIGHT}До свидания! Спасибо за использование RUZANNA! 👋{Style.RESET_ALL}\n")
            break
        else:
            print_error("Неверный выбор. Пожалуйста, выберите 1-5.")
            time.sleep(1)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Программа прервана пользователем{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n\n{Fore.RED}Критическая ошибка: {e}{Style.RESET_ALL}")
        input("Нажмите Enter для выхода...")
