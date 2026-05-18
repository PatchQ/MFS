"""
SignalMemory - SQLite 持久化歷史信號
"""
import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict


class SignalMemory:
    """SQLite 持久化歷史信號"""
    
    def __init__(self, db_path: str = None):
        """
        初始化 SignalMemory
        
        Args:
            db_path: SQLite 資料庫路徑，預設 ~/GitHub/MFS/SignalAgent/signals.db
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.expanduser("~/GitHub/MFS/SignalAgent"), 
                "signals.db"
            )
        
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self):
        """確保資料庫目錄存在"""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _init_db(self):
        """初始化資料庫表格"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model TEXT NOT NULL,
                    signal INTEGER NOT NULL,
                    confidence REAL,
                    indicators TEXT,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_date 
                ON signals(ticker, date DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model 
                ON signals(model)
            """)
            conn.commit()
    
    def save_signal(self, ticker: str, model: str, signal: bool, 
                    confidence: float, indicators: dict, date: str) -> int:
        """
        保存單個信號到 SQLite
        
        Args:
            ticker: 股票代碼
            model: 模型名稱（如 'RF', 'SVM', 'MLP'）
            signal: 信號（True=買, False=賣）
            confidence: 信心度（0.0-1.0）
            indicators: 技術指標字典
            date: 交易日期（格式：YYYY-MM-DD）
            
        Returns:
            插入記錄的 ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (ticker, model, signal, confidence, indicators, date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker, 
                model, 
                1 if signal else 0, 
                confidence, 
                json.dumps(indicators, ensure_ascii=False),
                date
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_signals(self, ticker: str = None, model: str = None,
                   start_date: str = None, end_date: str = None, 
                   limit: int = 100) -> List[Dict]:
        """
        查詢歷史信號
        
        Args:
            ticker: 股票代碼（None=所有）
            model: 模型名稱（None=所有）
            start_date: 開始日期（格式：YYYY-MM-DD）
            end_date: 結束日期（格式：YYYY-MM-DD）
            limit: 返回記錄數限制
            
        Returns:
            信號記錄列表
        """
        query = "SELECT id, ticker, model, signal, confidence, indicators, date, created_at FROM signals WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if model:
            query += " AND model = ?"
            params.append(model)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC, id DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                {
                    'id': row['id'],
                    'ticker': row['ticker'],
                    'model': row['model'],
                    'signal': bool(row['signal']),
                    'confidence': row['confidence'],
                    'indicators': json.loads(row['indicators']) if row['indicators'] else {},
                    'date': row['date'],
                    'created_at': row['created_at']
                }
                for row in rows
            ]
    
    def get_latest(self, ticker: str, model: str = None) -> Optional[Dict]:
        """
        獲取最新信號
        
        Args:
            ticker: 股票代碼
            model: 模型名稱（None=任何模型）
            
        Returns:
            最新信號記錄，無則返回 None
        """
        if model:
            query = """
                SELECT id, ticker, model, signal, confidence, indicators, date, created_at 
                FROM signals 
                WHERE ticker = ? AND model = ?
                ORDER BY date DESC, id DESC LIMIT 1
            """
            params = (ticker, model)
        else:
            query = """
                SELECT id, ticker, model, signal, confidence, indicators, date, created_at 
                FROM signals 
                WHERE ticker = ?
                ORDER BY date DESC, id DESC LIMIT 1
            """
            params = (ticker,)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'ticker': row['ticker'],
                    'model': row['model'],
                    'signal': bool(row['signal']),
                    'confidence': row['confidence'],
                    'indicators': json.loads(row['indicators']) if row['indicators'] else {},
                    'date': row['date'],
                    'created_at': row['created_at']
                }
            return None
    
    def get_history(self, ticker: str, days: int = 30) -> List[Dict]:
        """
        獲取最近 N 天信號歷史
        
        Args:
            ticker: 股票代碼
            days: 天數
            
        Returns:
            信號歷史列表
        """
        from datetime import timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.get_signals(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            limit=days * 3  # 每天可能有多個模型的信號
        )
    
    def get_all_tickers(self) -> List[str]:
        """獲取所有有記錄的股票代碼"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM signals ORDER BY ticker")
            return [row[0] for row in cursor.fetchall()]
    
    def clear_old_signals(self, days: int = 90) -> int:
        """
        清除舊記錄
        
        Args:
            days: 保留最近 N 天的記錄
            
        Returns:
            刪除的記錄數
        """
        from datetime import timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM signals WHERE date < ?", (cutoff_date,))
            conn.commit()
            return cursor.rowcount