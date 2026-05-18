"""
SignalReasoner - 多指標推理
"""
import os
import sys
from typing import Optional, Dict, List

# 添加專案根目錄到路徑
project_root = os.path.expanduser("~/GitHub/MFS")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .predictor import SignalPredictor
from .memory import SignalMemory


class SignalReasoner:
    """結合多個指標（Ichimoku/RSI/MACD）進行推理"""
    
    def __init__(self, predictor: SignalPredictor, memory: SignalMemory):
        """
        初始化 SignalReasoner
        
        Args:
            predictor: SignalPredictor 實例
            memory: SignalMemory 實例
        """
        self.predictor = predictor
        self.memory = memory
        self._default_rules = self._init_default_rules()
    
    def _init_default_rules(self) -> Dict:
        """初始化預設規則"""
        return {
            'RSI': {
                'overbought': 70,    # RSI > 70 超買
                'oversold': 30       # RSI < 30 超賣
            },
            'MACD': {
                'bullish_cross': True,   # MACD 金叉
                'bearish_cross': True    # MACD 死叉
            },
            'ICHIMOKU': {
                'cloud_bullish': True,   # 價格在雲之上
                'cloud_bearish': True    # 價格在雲之下
            },
            'price_change': {
                'up_threshold': 0.05,    # 漲幅 > 5%
                'down_threshold': -0.05  # 跌幅 > -5%
            }
        }
    
    def _default_rules(self) -> Dict:
        """返回預設規則（供外部調用）"""
        return self._default_rules.copy()
    
    def _extract_indicators(self, df) -> Dict:
        """
        從 DataFrame 提取技術指標值
        
        Args:
            df: 包含技術指標的 DataFrame
            
        Returns:
            指標字典
        """
        indicators = {}
        
        if df is None or df.empty:
            return indicators
        
        # 取得最新一行
        row = df.iloc[-1] if len(df) > 0 else None
        
        # 常見指標
        common_indicators = [
            'RSI', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
            'ICHIMOKU_CONV', 'ICHIMOKU_BASE', 'ICHIMOKU_SPANA', 'ICHIMOKU_SPANB',
            'BOLL_UPPER', 'BOLL_MIDDLE', 'BOLL_LOWER',
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'ATR', 'ADX',
            'Volume', 'F20D', 'F10D', 'F30D'
        ]
        
        for indicator in common_indicators:
            if indicator in df.columns and row is not None:
                try:
                    val = row[indicator]
                    if val is not None and not (isinstance(val, float) and os.name == 'nt' and str(val) == 'nan'):
                        indicators[indicator] = float(val)
                except (ValueError, TypeError):
                    continue
        
        # 計算價格變動
        if len(df) >= 2:
            try:
                close_current = float(df['Close'].iloc[-1]) if 'Close' in df.columns else None
                close_prev = float(df['Close'].iloc[-2]) if 'Close' in df.columns else None
                if close_current and close_prev:
                    indicators['price_change_pct'] = (close_current - close_prev) / close_prev
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        
        return indicators
    
    def _evaluate_rules(self, indicators: Dict, rules: Dict) -> Dict:
        """
        評估規則
        
        Args:
            indicators: 技術指標值
            rules: 規則字典
            
        Returns:
            評估結果
        """
        results = {
            'signals': [],
            'reasons': [],
            'score': 0.0
        }
        
        # RSI 規則
        if 'RSI' in indicators or 'RSI_14' in indicators:
            rsi = indicators.get('RSI', indicators.get('RSI_14', 50))
            if rsi > rules['RSI']['overbought']:
                results['signals'].append('RSI_OVERBOUGHT')
                results['reasons'].append(f'RSI 超買: {rsi:.2f} > {rules["RSI"]["overbought"]}')
                results['score'] -= 0.2
            elif rsi < rules['RSI']['oversold']:
                results['signals'].append('RSI_OVERSOLD')
                results['reasons'].append(f'RSI 超賣: {rsi:.2f} < {rules["RSI"]["oversold"]}')
                results['score'] += 0.3  # 超賣更強的信號
        
        # MACD 規則
        if 'MACD' in indicators and 'MACD_signal' in indicators:
            macd = indicators['MACD']
            macd_signal = indicators['MACD_signal']
            if rules['MACD']['bullish_cross'] and macd > macd_signal:
                results['signals'].append('MACD_BULLISH')
                results['reasons'].append(f'MACD 金叉: MACD={macd:.4f} > Signal={macd_signal:.4f}')
                results['score'] += 0.15
            elif rules['MACD']['bearish_cross'] and macd < macd_signal:
                results['signals'].append('MACD_BEARISH')
                results['reasons'].append(f'MACD 死叉: MACD={macd:.4f} < Signal={macd_signal:.4f}')
                results['score'] -= 0.15
        
        # 價格變動規則
        if 'price_change_pct' in indicators:
            pct = indicators['price_change_pct']
            if pct > rules['price_change']['up_threshold']:
                results['signals'].append('PRICE_UP')
                results['reasons'].append(f'價格上漲: {pct*100:.2f}%')
                results['score'] += 0.1
            elif pct < rules['price_change']['down_threshold']:
                results['signals'].append('PRICE_DOWN')
                results['reasons'].append(f'價格下跌: {pct*100:.2f}%')
                results['score'] -= 0.1
        
        # ICHIMOKU 規則
        ichimoku_fields = ['ICHIMOKU_CONV', 'ICHIMOKU_BASE']
        if all(f in indicators for f in ichimoku_fields):
            conv = indicators['ICHIMOKU_CONV']
            base = indicators['ICHIMOKU_BASE']
            if rules['ICHIMOKU']['cloud_bullish'] and conv > base:
                results['signals'].append('ICHIMOKU_BULLISH')
                results['reasons'].append(f'Ichimoku 看漲: 轉換線={conv:.2f} > 基準線={base:.2f}')
                results['score'] += 0.1
            elif rules['ICHIMOKU']['cloud_bearish'] and conv < base:
                results['signals'].append('ICHIMOKU_BEARISH')
                results['reasons'].append(f'Ichimoku 看跌: 轉換線={conv:.2f} < 基準線={base:.2f}')
                results['score'] -= 0.1
        
        # 限制分數範圍
        results['score'] = max(-1.0, min(1.0, results['score']))
        
        return results
    
    def _combine_signals(self, model_predictions: Dict, rule_evaluation: Dict) -> Dict:
        """
        結合模型預測和規則評估
        
        Args:
            model_predictions: 模型預測結果
            rule_evaluation: 規則評估結果
            
        Returns:
            最終信號
        """
        # 統計模型信號
        buy_count = 0
        sell_count = 0
        total_confidence = 0.0
        model_count = 0
        
        for model_name, pred in model_predictions.items():
            if pred['signal']:
                buy_count += 1
            else:
                sell_count += 1
            total_confidence += pred['confidence']
            model_count += 1
        
        # 計算模型共識
        if model_count > 0:
            avg_confidence = total_confidence / model_count
            consensus_ratio = (buy_count - sell_count) / model_count
        else:
            avg_confidence = 0.5
            consensus_ratio = 0.0
        
        # 結合規則分數
        combined_score = consensus_ratio * 0.7 + rule_evaluation['score'] * 0.3
        
        # 最終判斷
        if combined_score > 0.2:
            final_signal = 'BUY'
        elif combined_score < -0.2:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        return {
            'signal': final_signal,
            'score': combined_score,
            'confidence': avg_confidence,
            'model_consensus': {
                'buy': buy_count,
                'sell': sell_count,
                'total': model_count
            }
        }
    
    def reason(self, ticker: str, df, rules: Dict = None) -> Dict:
        """
        綜合推理
        
        Args:
            ticker: 股票代碼
            df: 包含技術指標的 DataFrame
            rules: 自定義規則（None=使用預設規則）
            
        Returns:
            {
                'ticker': str,
                'signal': 'BUY'|'SELL'|'HOLD',
                'confidence': float,
                'reasons': [str],
                'models': {model_name: {'signal': bool, 'confidence': float}},
                'indicators': {name: value}
            }
        """
        if rules is None:
            rules = self._default_rules()
        
        # 提取技術指標
        indicators = self._extract_indicators(df)
        
        # 獲取模型預測
        model_predictions = self.predictor.predict_all(ticker, df)
        predictions = model_predictions.get('predictions', {})
        
        # 評估規則
        rule_evaluation = self._evaluate_rules(indicators, rules)
        
        # 結合信號
        combined = self._combine_signals(predictions, rule_evaluation)
        
        # 構建原因列表
        reasons = rule_evaluation['reasons'].copy()
        for model_name, pred in predictions.items():
            signal_str = "買入" if pred['signal'] else "賣出"
            reasons.append(f"{model_name}: {signal_str}信號 (信心度: {pred['confidence']:.2%})")
        
        result = {
            'ticker': ticker,
            'signal': combined['signal'],
            'confidence': combined['confidence'],
            'score': combined['score'],
            'reasons': reasons,
            'models': {
                name: {
                    'signal': pred['signal'],
                    'confidence': pred['confidence']
                }
                for name, pred in predictions.items()
            },
            'indicators': indicators,
            'model_consensus': combined['model_consensus']
        }
        
        # 保存到記憶體
        try:
            for model_name, pred in predictions.items():
                self.memory.save_signal(
                    ticker=ticker,
                    model=model_name,
                    signal=pred['signal'],
                    confidence=pred['confidence'],
                    indicators=indicators,
                    date=model_predictions.get('date', '')
                )
        except Exception as e:
            print(f"保存信號失敗: {e}")
        
        return result
    
    def get_recommendation(self, ticker: str, df) -> str:
        """
        獲取簡化的投資建議
        
        Args:
            ticker: 股票代碼
            df: 包含技術指標的 DataFrame
            
        Returns:
            建議字串（'強烈買入', '適量買入', '持有', '適量賣出', '強烈賣出'）
        """
        result = self.reason(ticker, df)
        
        signal = result['signal']
        confidence = result['confidence']
        score = result['score']
        
        if signal == 'BUY':
            if score > 0.5 or confidence > 0.8:
                return "強烈買入"
            else:
                return "適量買入"
        elif signal == 'SELL':
            if score < -0.5 or confidence > 0.8:
                return "強烈賣出"
            else:
                return "適量賣出"
        else:
            return "持有"