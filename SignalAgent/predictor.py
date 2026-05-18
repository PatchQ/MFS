"""
SignalPredictor - 對接 CalAIModel.pkl + AI/RF.py 等推理
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

# 添加專案根目錄到路徑
project_root = os.path.expanduser("~/GitHub/MFS")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 嘗試導入 CommonConfig
try:
    import UTIL.CommonConfig as cc
    MODELLIST = cc.MODELLIST
    OUTPATH = cc.OUTPATH
except ImportError:
    # 備用配置
    MODELLIST = ["SVM", "MLP", "RF"]
    OUTPATH = os.path.expanduser("~/GitHub/SData/P_YFData/")


# 特徵欄位（用於過濾預測時的輸入特徵）
DROP_COLS = [
    "sno", "F10D", "F20D", "F30D", "classification",
    "BOSS_PATTERN", "BOSS_STATUS", "HHHL_PATTERN",
    "LLDate", "HHDate", "WLDate", "WHDate",
    "ICHIMOKU_SIGNAL", "ICHIMOKU_STRENGTH",
    "GBS22C_SIGNAL", "GBS22C_STRENGTH",
    "BREAKOUT200_SIGNAL", "BREAKOUT200_STRENGTH",
    "FISHER_SIGNAL", "FISHER_STRENGTH"
]


class SignalPredictor:
    """載入訓練好的模型，進行推理"""
    
    def __init__(self, model_dir: str = None):
        """
        初始化 SignalPredictor
        
        Args:
            model_dir: 模型目錄，預設 ~/GitHub/MFS/Model/
                       實際路徑為 {model_dir}/{MODEL_NAME}/P_{ticker}.pkl
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.expanduser("~/GitHub/MFS"), 
                "Model"
            )
        
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self._models: Dict[str, object] = {}
    
    def _get_model_path(self, model_name: str, ticker: str) -> str:
        """
        獲取模型檔案路徑
        
        Args:
            model_name: 模型名稱（如 'RF', 'SVM', 'MLP'）
            ticker: 股票代碼
            
        Returns:
            模型檔案完整路徑
        """
        # 對應 CommonConfig 中的命名方式
        model_dir_map = {
            'RF': 'RF',
            'SVM': 'SVM', 
            'MLP': 'MLP',
            'LR': 'LR',
            'DT': 'DT',
            'XGBOOST': 'XGBOOST',
            'LIGHTGBM': 'LIGHTGBM'
        }
        
        dir_name = model_dir_map.get(model_name.upper(), model_name.upper())
        filename = f"P_{ticker}.pkl"
        
        return os.path.join(self.model_dir, dir_name, filename)
    
    def load_model(self, model_name: str, ticker: str = None):
        """
        載入指定模型
        
        Args:
            model_name: 模型名稱（如 'RF', 'SVM', 'MLP'）
            ticker: 股票代碼（可選，若不提供則不載入特定模型檔案）
            
        Returns:
            載入的模型對象，失敗返回 None
        """
        if ticker:
            model_path = self._get_model_path(model_name, ticker)
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    key = f"{model_name.upper()}_{ticker}"
                    self._models[key] = model
                    return model
                except Exception as e:
                    print(f"載入模型失敗 {model_path}: {e}")
                    return None
        
        return None
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        準備預測用的特徵矩陣
        
        Args:
            df: 包含技術指標的 DataFrame
            
        Returns:
            處理後的特徵矩陣
        """
        features = df.copy()
        
        # 移除不需要的欄位
        for col in DROP_COLS:
            if col in features.columns:
                features = features.drop(columns=[col])
        
        # 處理無窮值和缺失值
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 確保所有值為數值型
        features = features.apply(pd.to_numeric, errors='coerce')
        
        # 補充缺失的特徵欄位
        for col in ['Dividends', 'Stock Splits']:
            if col not in features.columns:
                features[col] = 0.0
        
        return features
    
    def _get_model_features(self, model) -> List[str]:
        """獲取模型訓練時的特徵名稱"""
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        return []
    
    def predict(self, ticker: str, df: pd.DataFrame, model: str = None) -> dict:
        """
        對股票進行預測
        
        Args:
            ticker: 股票代碼
            df: 包含技術指標的 DataFrame（需要包含最新一天的數據）
            model: 指定模型（None=使用所有可用模型）
                  可用: 'RF', 'SVM', 'MLP', 'LR', 'DT'
                  
        Returns:
            {
                'ticker': str,
                'date': str,
                'predictions': {
                    'MODEL_NAME': {
                        'signal': bool,      # True=買入信號
                        'confidence': float, # 信心度 0.0-1.0
                        'features': dict      # 使用的特徵值
                    }
                }
            }
        """
        if df.empty:
            return {
                'ticker': ticker,
                'date': None,
                'predictions': {}
            }
        
        # 取得最新一天的數據
        latest_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
        date_str = str(latest_date)[:10] if latest_date else pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # 準備特徵矩陣
        features = self._prepare_features(df.tail(1))
        
        # 確定要使用的模型
        if model:
            models_to_use = [model.upper()]
        else:
            models_to_use = [m.upper() for m in MODELLIST]
        
        predictions = {}
        
        for model_name in models_to_use:
            model_path = self._get_model_path(model_name, ticker)
            
            if not os.path.exists(model_path):
                continue
            
            try:
                # 載入模型
                if f"{model_name}_{ticker}" not in self._models:
                    model_obj = joblib.load(model_path)
                    self._models[f"{model_name}_{ticker}"] = model_obj
                else:
                    model_obj = self._models[f"{model_name}_{ticker}"]
                
                # 獲取模型訓練時的特徵
                model_features = self._get_model_features(model_obj)
                
                # 只保留模型见过的特征
                if model_features:
                    features_filtered = features[[c for c in model_features if c in features.columns]]
                else:
                    features_filtered = features
                
                # 確保沒有額外的特徵
                if model_features:
                    missing_cols = set(model_features) - set(features_filtered.columns)
                    for col in missing_cols:
                        features_filtered[col] = 0.0
                    features_filtered = features_filtered[model_features]
                
                # 預測
                if len(features_filtered) > 0:
                    proba = model_obj.predict_proba(features_filtered)
                    
                    # 提取買入信號的概率
                    if proba.shape[1] > 1:
                        confidence = float(proba[0][1])  # 第二類的概率
                    else:
                        confidence = float(proba[0][0])
                    
                    signal = confidence > 0.5
                    
                    predictions[model_name] = {
                        'signal': bool(signal),
                        'confidence': confidence,
                        'features': features_filtered.iloc[0].to_dict() if len(features_filtered) > 0 else {}
                    }
                    
            except Exception as e:
                print(f"模型 {model_name} 預測失敗: {e}")
                continue
        
        return {
            'ticker': ticker,
            'date': date_str,
            'predictions': predictions
        }
    
    def predict_all(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        使用所有模型預測
        
        Args:
            ticker: 股票代碼
            df: 包含技術指標的 DataFrame
            
        Returns:
            同 predict()，使用所有 MODELLIST 中的模型
        """
        return self.predict(ticker, df, model=None)
    
    def get_feature_importance(self, ticker: str, model_name: str = 'RF') -> Optional[Dict]:
        """
        獲取模型特徵重要性（仅 RandomForest 支持）
        
        Args:
            ticker: 股票代碼
            model_name: 模型名稱
            
        Returns:
            特徵重要性字典，無則返回 None
        """
        key = f"{model_name.upper()}_{ticker}"
        if key not in self._models:
            model_path = self._get_model_path(model_name, ticker)
            if not os.path.exists(model_path):
                return None
            try:
                self._models[key] = joblib.load(model_path)
            except:
                return None
        
        model_obj = self._models[key]
        
        # 僅 RandomForest 有 feature_importances_
        if hasattr(model_obj, 'feature_importances_'):
            feature_names = self._get_model_features(model_obj)
            importances = model_obj.feature_importances_
            
            return dict(zip(feature_names, importances.tolist()))
        
        return None