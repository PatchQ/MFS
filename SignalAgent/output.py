"""
SignalOutput - 格式化交易信號輸出
"""
import json
from typing import Dict, List, Any
from datetime import datetime


class SignalOutput:
    """格式化交易信號輸出"""
    
    @staticmethod
    def to_dict(signal_reason: Dict) -> Dict:
        """
        轉為字典格式
        
        Args:
            signal_reason: SignalReasoner 返回的信號字典
            
        Returns:
            標準化的字典格式
        """
        return {
            'ticker': signal_reason.get('ticker', ''),
            'signal': signal_reason.get('signal', 'HOLD'),
            'confidence': signal_reason.get('confidence', 0.0),
            'score': signal_reason.get('score', 0.0),
            'reasons': signal_reason.get('reasons', []),
            'models': signal_reason.get('models', {}),
            'indicators': signal_reason.get('indicators', {}),
            'model_consensus': signal_reason.get('model_consensus', {}),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def to_csv_row(signal_reason: Dict) -> Dict:
        """
        轉為 CSV 行格式
        
        Args:
            signal_reason: SignalReasoner 返回的信號字典
            
        Returns:
            CSV 行的字典格式（不含表頭）
        """
        # 提取關鍵欄位
        row = {
            'ticker': signal_reason.get('ticker', ''),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signal': signal_reason.get('signal', 'HOLD'),
            'confidence': signal_reason.get('confidence', 0.0),
            'score': signal_reason.get('score', 0.0),
            'buy_models': 0,
            'sell_models': 0,
        }
        
        # 統計模型
        models = signal_reason.get('models', {})
        if isinstance(models, dict):
            row['buy_models'] = sum(1 for m in models.values() if m.get('signal', False))
            row['sell_models'] = sum(1 for m in models.values() if not m.get('signal', True))
        
        # 主要指標
        indicators = signal_reason.get('indicators', {})
        if isinstance(indicators, dict):
            for key in ['RSI', 'RSI_14', 'MACD', 'F20D', 'price_change_pct']:
                if key in indicators:
                    row[key] = indicators[key]
        
        # 主要原因（取第一個）
        reasons = signal_reason.get('reasons', [])
        if reasons:
            row['primary_reason'] = reasons[0][:100]  # 限制長度
        
        return row
    
    @staticmethod
    def to_text(signal_reason: Dict, verbose: bool = False) -> str:
        """
        轉為可讀文本
        
        Args:
            signal_reason: SignalReasoner 返回的信號字典
            verbose: 是否顯示詳細資訊
            
        Returns:
            格式化的文本字串
        """
        ticker = signal_reason.get('ticker', 'N/A')
        signal = signal_reason.get('signal', 'HOLD')
        confidence = signal_reason.get('confidence', 0.0)
        score = signal_reason.get('score', 0.0)
        
        # 信號表情
        signal_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }
        emoji = signal_emoji.get(signal, '⚪')
        
        # 構建文本
        lines = []
        lines.append(f"{'='*50}")
        lines.append(f"  📊 信號分析報告 - {ticker}")
        lines.append(f"{'='*50}")
        lines.append(f"  信號: {emoji} {signal}")
        lines.append(f"  信心度: {confidence:.2%}")
        lines.append(f"  評分: {score:.3f}")
        
        # 模型共識
        model_consensus = signal_reason.get('model_consensus', {})
        if model_consensus:
            buy = model_consensus.get('buy', 0)
            sell = model_consensus.get('sell', 0)
            total = model_consensus.get('total', 0)
            lines.append(f"  模型共識: 買入={buy} 賣出={sell} 總計={total}")
        
        if verbose:
            # 詳細模式
            models = signal_reason.get('models', {})
            if models:
                lines.append(f"\n  📈 模型詳情:")
                for model_name, pred in models.items():
                    signal_str = "買入" if pred.get('signal', False) else "賣出"
                    conf = pred.get('confidence', 0.0)
                    lines.append(f"    - {model_name}: {signal_str} (信心度: {conf:.2%})")
            
            # 指標
            indicators = signal_reason.get('indicators', {})
            if indicators:
                lines.append(f"\n  📉 關鍵指標:")
                for key in ['RSI', 'RSI_14', 'MACD', 'F20D', 'price_change_pct']:
                    if key in indicators:
                        val = indicators[key]
                        if key == 'price_change_pct':
                            lines.append(f"    - {key}: {val*100:.2f}%")
                        else:
                            lines.append(f"    - {key}: {val:.4f}")
            
            # 原因
            reasons = signal_reason.get('reasons', [])
            if reasons:
                lines.append(f"\n  💡 信號原因:")
                for i, reason in enumerate(reasons[:5], 1):  # 最多5條
                    lines.append(f"    {i}. {reason}")
        
        lines.append(f"{'='*50}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_html(signal_reason: Dict) -> str:
        """
        轉為 HTML 格式
        
        Args:
            signal_reason: SignalReasoner 返回的信號字典
            
        Returns:
            HTML 格式字串
        """
        ticker = signal_reason.get('ticker', 'N/A')
        signal = signal_reason.get('signal', 'HOLD')
        confidence = signal_reason.get('confidence', 0.0)
        score = signal_reason.get('score', 0.0)
        
        # 信號顏色
        signal_colors = {
            'BUY': '#28a745',   # 綠色
            'SELL': '#dc3545',  # 紅色
            'HOLD': '#ffc107'   # 黃色
        }
        color = signal_colors.get(signal, '#6c757d')
        
        # 構建 HTML
        html_parts = []
        html_parts.append(f'''
<div class="signal-report" style="font-family: Arial, sans-serif; max-width: 600px; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px 0;">
    <h2 style="margin: 0 0 16px; color: #333;">📊 {ticker} 信號分析</h2>
    
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div style="text-align: center; padding: 12px; background: {color}; color: white; border-radius: 8px; min-width: 100px;">
            <div style="font-size: 24px; font-weight: bold;">{signal}</div>
        </div>
        <div style="text-align: right;">
            <div>信心度: <strong>{confidence:.2%}</strong></div>
            <div>評分: <strong>{score:.3f}</strong></div>
        </div>
    </div>
''')
        
        # 模型詳情
        models = signal_reason.get('models', {})
        if models:
            html_parts.append('    <h3 style="margin: 16px 0 8px;">📈 模型預測</h3>')
            html_parts.append('    <table style="width: 100%; border-collapse: collapse;">')
            html_parts.append('    <tr style="background: #f8f9fa;"><th style="padding: 8px; text-align: left;">模型</th><th style="padding: 8px;">信號</th><th style="padding: 8px;">信心度</th></tr>')
            
            for model_name, pred in models.items():
                signal_str = "買入 🟢" if pred.get('signal', False) else "賣出 🔴"
                conf = pred.get('confidence', 0.0)
                html_parts.append(f'    <tr><td style="padding: 8px;">{model_name}</td><td style="text-align: center;">{signal_str}</td><td style="text-align: center;">{conf:.2%}</td></tr>')
            
            html_parts.append('    </table>')
        
        # 關鍵指標
        indicators = signal_reason.get('indicators', {})
        if indicators:
            display_indicators = {k: v for k, v in indicators.items() 
                                 if k in ['RSI', 'RSI_14', 'MACD', 'F20D', 'price_change_pct']}
            if display_indicators:
                html_parts.append('    <h3 style="margin: 16px 0 8px;">📉 關鍵指標</h3>')
                html_parts.append('    <ul style="list-style: none; padding: 0; margin: 0;">')
                
                for key, val in display_indicators.items():
                    if key == 'price_change_pct':
                        html_parts.append(f'    <li style="padding: 4px 0;">{key}: <strong>{val*100:.2f}%</strong></li>')
                    else:
                        html_parts.append(f'    <li style="padding: 4px 0;">{key}: <strong>{val:.4f}</strong></li>')
                
                html_parts.append('    </ul>')
        
        # 原因
        reasons = signal_reason.get('reasons', [])
        if reasons:
            html_parts.append('    <h3 style="margin: 16px 0 8px;">💡 信號原因</h3>')
            html_parts.append('    <ol style="margin: 0; padding-left: 20px;">')
            for reason in reasons[:5]:
                html_parts.append(f'    <li style="padding: 4px 0;">{reason}</li>')
            html_parts.append('    </ol>')
        
        html_parts.append(f'''
    <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
        生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
''')
        
        return '\n'.join(html_parts)
    
    @staticmethod
    def to_json(signal_reason: Dict, indent: int = 2) -> str:
        """
        轉為 JSON 格式
        
        Args:
            signal_reason: SignalReasoner 返回的信號字典
            indent: 縮排空格數
            
        Returns:
            JSON 字串
        """
        # 轉換後輸出
        output_dict = SignalOutput.to_dict(signal_reason)
        return json.dumps(output_dict, ensure_ascii=False, indent=indent)
    
    @staticmethod
    def to_markdown(signal_reason: Dict) -> str:
        """
        轉為 Markdown 格式
        
        Args:
            signal_reason: SignalReasoner 返回的信號字典
            
        Returns:
            Markdown 格式字串
        """
        ticker = signal_reason.get('ticker', 'N/A')
        signal = signal_reason.get('signal', 'HOLD')
        confidence = signal_reason.get('confidence', 0.0)
        score = signal_reason.get('score', 0.0)
        
        lines = []
        lines.append(f"## 📊 {ticker} 信號分析")
        lines.append("")
        lines.append(f"| 項目 | 數值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 信號 | **{signal}** |")
        lines.append(f"| 信心度 | {confidence:.2%} |")
        lines.append(f"| 評分 | {score:.3f} |")
        
        # 模型
        models = signal_reason.get('models', {})
        if models:
            lines.append("")
            lines.append("### 📈 模型預測")
            lines.append("")
            lines.append("| 模型 | 信號 | 信心度 |")
            lines.append("|------|------|--------|")
            
            for model_name, pred in models.items():
                signal_str = "買入" if pred.get('signal', False) else "賣出"
                conf = pred.get('confidence', 0.0)
                lines.append(f"| {model_name} | {signal_str} | {conf:.2%} |")
        
        # 指標
        indicators = signal_reason.get('indicators', {})
        display_indicators = {k: v for k, v in indicators.items() 
                             if k in ['RSI', 'RSI_14', 'MACD', 'F20D', 'price_change_pct']}
        if display_indicators:
            lines.append("")
            lines.append("### 📉 關鍵指標")
            lines.append("")
            lines.append("| 指標 | 數值 |")
            lines.append("|------|------|")
            
            for key, val in display_indicators.items():
                if key == 'price_change_pct':
                    lines.append(f"| {key} | {val*100:.2f}% |")
                else:
                    lines.append(f"| {key} | {val:.4f} |")
        
        # 原因
        reasons = signal_reason.get('reasons', [])
        if reasons:
            lines.append("")
            lines.append("### 💡 信號原因")
            lines.append("")
            for i, reason in enumerate(reasons[:5], 1):
                lines.append(f"{i}. {reason}")
        
        lines.append("")
        lines.append(f"*生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return '\n'.join(lines)