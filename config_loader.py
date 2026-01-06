# core/config_loader.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import pprint

class ConfigManager:
    """Менеджер конфигурации с поддержкой пресетов и валидацией"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.base_path = self.config_dir / "base.json"
        self.custom_path = self.config_dir / "custom.json"
        self.presets_dir = self.config_dir / "presets"
        
        self._ensure_dirs()
        self.config = {}

        self.history_file = self.config_dir / "path_history.json"
        self.path_history = self._load_path_history()
        
    def _load_path_history(self) -> Dict:
        """Загружает историю путей"""
        default_history = {
            "output_dirs": [],
            "max_history": 3,
            "last_used": None
        }
        
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # Ограничиваем историю
                    if 'output_dirs' in history:
                        history['output_dirs'] = history['output_dirs'][-3:]
                    return {**default_history, **history}
            except:
                pass
        
        return default_history
    
    def save_path_history(self):
        """Сохраняет историю путей"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.path_history, f, indent=2, ensure_ascii=False)
    
    def add_output_dir(self, path: str):
        """Добавляет путь в историю"""
        path = str(Path(path).resolve())
        
        # Удаляем если уже есть
        if path in self.path_history['output_dirs']:
            self.path_history['output_dirs'].remove(path)
        
        # Добавляем в конец
        self.path_history['output_dirs'].append(path)
        
        # Ограничиваем историю
        max_history = self.path_history.get('max_history', 3)
        self.path_history['output_dirs'] = self.path_history['output_dirs'][-max_history:]
        
        # Обновляем последний использованный
        self.path_history['last_used'] = path
        
        self.save_path_history()
    
    def get_output_dirs_history(self) -> List[str]:
        """Возвращает историю путей"""
        return self.path_history.get('output_dirs', [])
    
    def get_last_output_dir(self) -> Optional[str]:
        """Возвращает последний использованный путь"""
        return self.path_history.get('last_used')
    
    def update_config_paths(self, base_dir: str):
        """Обновляет все пути в конфиге относительно базовой директории"""
        base_path = Path(base_dir).resolve()
        
        # Обновляем секцию paths
        if 'paths' not in self.config:
            self.config['paths'] = {}
        
        paths_config = self.config['paths']
        
        # Основные поддиректории
        subdirs = {
            'logs': 'logs',
            'checkpoints': 'checkpoints', 
            'output': 'output',
            'models': 'models',
            'tmp': 'tmp'
        }
        
        for key, subdir in subdirs.items():
            full_path = base_path / subdir
            paths_config[key] = str(full_path)
            
            # Создаем директорию если её нет
            full_path.mkdir(exist_ok=True, parents=True)
        
        # Сохраняем базовый путь
        paths_config['base'] = str(base_path)
        
        # Добавляем в историю
        self.add_output_dir(base_dir)
        
        print(f"✅ Пути обновлены: {base_path}")
        
        return paths_config

    def _ensure_dirs(self):
        """Создаёт необходимые директории"""
        self.config_dir.mkdir(exist_ok=True)
        self.presets_dir.mkdir(exist_ok=True)
        
        # Создаём пустой base.json если его нет
        if not self.base_path.exists():
            self._create_default_base()
    
    def _create_default_base(self):
        """Создаёт базовый конфиг по умолчанию"""
        default_config = {
            "training": {"batch_size": 3, "learning_rate": 0.0002, "epochs": 3},
            "paths": {"data": "data/dialogues.json"},
            "model": {"name": "gpt-neo-2.7B"}
        }
        self.save_config(default_config, self.base_path)
        print(f"⚠️  Создан базовый конфиг: {self.base_path}")
    
    def load_full_config(self, preset: Optional[str] = None) -> Dict[str, Any]:
        """Загружает полную конфигурацию с учётом пресета"""
        # 1. Загружаем базовый конфиг
        base_config = self._load_json(self.base_path)
        if not base_config:
            raise ValueError(f"Базовый конфиг не найден: {self.base_path}")
        
        # 2. Применяем пресет если указан
        config = base_config.copy()
        if preset:
            preset_config = self.load_preset(preset)
            config = self._deep_merge(config, preset_config)
        
        # 3. Применяем кастомные настройки
        custom_config = self._load_json(self.custom_path)
        if custom_config:
            config = self._deep_merge(config, custom_config)
        
        self.config = config
        return config
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Загружает конфигурацию пресета"""
        preset_path = self.presets_dir / f"{preset_name}.json"
        if not preset_path.exists():
            available = [p.stem for p in self.presets_dir.glob("*.json")]
            raise ValueError(f"Пресет '{preset_name}' не найден. Доступные: {available}")
        
        return self._load_json(preset_path)
    
    def save_preset(self, preset_name: str, config: Dict[str, Any]):
        """Сохраняет пресет"""
        preset_path = self.presets_dir / f"{preset_name}.json"
        self.save_config(config, preset_path)
        print(f"✅ Пресет '{preset_name}' сохранён")
    
    def update_custom_config(self, updates: Dict[str, Any]):
        """Обновляет кастомный конфиг (для лаунчера)"""
        current = self._load_json(self.custom_path) or {}
        merged = self._deep_merge(current, updates)
        self.save_config(merged, self.custom_path)
        print(f"✅ Кастомные настройки обновлены")
    
    def get_training_params(self) -> Dict[str, Any]:
        """Извлекает параметры для тренировки"""
        return {
            "batch_size": self.config.get("training", {}).get("batch_size", 3),
            "epochs": self.config.get("training", {}).get("epochs", 3),
            "learning_rate": self.config.get("training", {}).get("learning_rate", 0.0002),
            "max_length": self.config.get("tokenization", {}).get("max_length", 512),
            "data_path": self.config.get("paths", {}).get("data", ""),
            "model_name": self.config.get("model", {}).get("name", "gpt-neo-2.7B")
        }
    
    def _load_json(self, path: Path) -> Optional[Dict]:
        """Загружает JSON файл"""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Ошибка загрузки {path}: {e}")
        return None
    
    def save_config(self, config: Dict, path: Path):
        """Сохраняет конфиг в JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Рекурсивно объединяет словари"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def print_config(self, section: Optional[str] = None):
        """Печатает конфигурацию"""
        if not self.config:
            print("Конфигурация не загружена")
            return
        
        if section:
            if section in self.config:
                print(f"\n{'='*50}")
                print(f"Секция: {section}")
                print('='*50)
                pprint.pprint(self.config[section], indent=2)
            else:
                print(f"⚠️  Секция '{section}' не найдена")
        else:
            pprint.pprint(self.config, indent=2, width=100)
