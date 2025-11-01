import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config FIRST
st.set_page_config(
    page_title="ULTRA STOCK ANALYZER PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ultra-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
        border-left: 6px solid #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .indicator-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
    }
    .bullish { border-color: #10b981; background: #d1fae5; }
    .bearish { border-color: #ef4444; background: #fee2e2; }
</style>
""", unsafe_allow_html=True)

class UltraStockAnalyzer:
    def __init__(self):
        self.all_symbols = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'LT': 'LT.NS',
            'ITC': 'ITC.NS'
        }
    
    @st.cache_data(ttl=3600)
    def get_stock_data(_self, symbol, period="1y"):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data if not data.empty else None
        except:
            return None

    def calculate_advanced_indicators(self, data):
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # FIBONACCI MOVING AVERAGES
        fib_periods = [5, 8, 13, 21, 34, 55, 89, 144]
        for period in fib_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # ENHANCED RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        df['RSI_21'] = calculate_rsi(df['Close'], 21)
        
        # TRIPLE EMA (TEMA)
        ema1 = df['Close'].ewm(span=5).mean()
        ema2 = ema1.ewm(span=8).mean() 
        ema3 = ema2.ewm(span=13).mean()
        df['TEMA'] = (3 * ema1) - (3 * ema2) + ema3
        
        # MACD WITH HISTOGRAM
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # BOLLINGER BANDS MULTIPLE
        for dev in [1.5, 2, 2.5]:
            df[f'BB_Middle_{dev}'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df[f'BB_Upper_{dev}'] = df[f'BB_Middle_{dev}'] + (bb_std * dev)
            df[f'BB_Lower_{dev}'] = df[f'BB_Middle_{dev}'] - (bb_std * dev)
        
        df['BB_Position'] = (df['Close'] - df['BB_Lower_2']) / (df['BB_Upper_2'] - df['BB_Lower_2'])
        
        # VOLUME INDICATORS
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # ATR FOR VOLATILITY
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df

    def calculate_ai_score(self, df):
        if df is None:
            return 0, []
            
        current_price = df['Close'].iloc[-1]
        score = 0
        reasons = []
        
        # TREND SCORE (Fibonacci MAs)
        bullish_ma_count = 0
        fib_periods = [5, 8, 13, 21, 34, 55, 89, 144]
        for period in fib_periods:
            if f'EMA_{period}' in df and current_price > df[f'EMA_{period}'].iloc[-1]:
                bullish_ma_count += 1
                score += 2
        
        if bullish_ma_count >= 6:
            score += 20
            reasons.append(f"🚀 STRONG FIBONACCI TREND ({bullish_ma_count}/8 MAs bullish)")
        
        # MOMENTUM SCORE
        if 'RSI_14' in df:
            rsi = df['RSI_14'].iloc[-1]
            if 40 <= rsi <= 65:
                score += 15
                reasons.append(f"🎯 Perfect RSI: {rsi:.1f}")
            elif rsi < 35:
                score += 20
                reasons.append(f"📈 Oversold RSI: {rsi:.1f}")
        
        # MACD SCORE
        if all(col in df for col in ['MACD', 'MACD_Signal']):
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                score += 15
                reasons.append("💪 MACD Bullish")
        
        # VOLUME SCORE
        if all(col in df for col in ['Volume', 'Volume_SMA_20']):
            vol_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
            if vol_ratio > 1.8:
                score += 10
                reasons.append(f"💰 High Volume: {vol_ratio:.1f}x")
        
        # BOLLINGER BAND SCORE
        if 'BB_Position' in df:
            bb_pos = df['BB_Position'].iloc[-1]
            if bb_pos < 0.3:
                score += 15
                reasons.append("🎯 Near BB Support - Good Entry")
        
        return min(score, 100), reasons

    def create_advanced_chart(self, df, symbol):
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{symbol} - ADVANCED ANALYSIS', 
                'Volume & OBV',
                'RSI Multi-Timeframe', 
                'MACD & Histogram'
            ),
            row_heights=[0.4, 0.15, 0.2, 0.25]
        )
        
        # Price with Fibonacci MAs
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#00C805', decreasing_line_color='#FF0000'
        ), row=1, col=1)
        
        # Add key Fibonacci MAs
        for period, color in [(8, 'orange'), (21, 'green'), (55, 'blue'), (144, 'purple')]:
            if f'EMA_{period}' in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{period}'], 
                                       name=f'EMA {period}', line=dict(color=color, width=2)), row=1, col=1)
        
        # Bollinger Bands
        if 'BB_Upper_2' in df and 'BB_Lower_2' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper_2'], name='BB Upper', 
                                   line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower_2'], name='BB Lower', 
                                   line=dict(color='gray', dash='dash')), row=1, col=1)
        
        # Volume
        colors = ['#00C805' if row['Close'] >= row['Open'] else '#FF0000' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
        
        # Multi-Timeframe RSI
        for period, color in [(14, 'blue'), (21, 'green')]:
            if f'RSI_{period}' in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[f'RSI_{period}'], name=f'RSI {period}', 
                                       line=dict(color=color, width=2)), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if all(col in df for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                                   line=dict(color='blue', width=2)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                                   line=dict(color='red', width=2)), row=4, col=1)
            
            colors_hist = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', 
                               marker_color=colors_hist), row=4, col=1)
        
        fig.update_layout(height=1000, showlegend=True, template='plotly_dark')
        return fig

def main():
    analyzer = UltraStockAnalyzer()
    
    st.markdown('<h1 class="main-header">🚀 ULTRA STOCK ANALYZER PRO</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #6b7280;">40+ Advanced Indicators • AI-Powered Scoring</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 SELECT STOCK")
        selected_stock = st.selectbox("Choose:", list(analyzer.all_symbols.keys()))
        symbol = analyzer.all_symbols[selected_stock]
        
        st.subheader("⚙️ SETTINGS")
        period = st.selectbox("Period:", ["6mo", "1y", "2y"], index=1)
        
        if st.button("🚀 RUN ULTRA ANALYSIS", type="primary", use_container_width=True):
            with st.spinner("🔄 Running advanced analysis..."):
                data = analyzer.get_stock_data(symbol, period)
                
                if data is not None and not data.empty:
                    df = analyzer.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        score, reasons = analyzer.calculate_ai_score(df)
                        current_price = df['Close'].iloc[-1]
                        
                        # DISPLAY AI SCORE
                        st.markdown('<div class="ultra-card">', unsafe_allow_html=True)
                        st.subheader(f"AI SCORE: {score}/100")
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        
                        if score >= 80:
                            st.success("🎯 STRONG BUY SIGNAL")
                        elif score >= 60:
                            st.info("📈 BUY SIGNAL")
                        elif score >= 40:
                            st.warning("🔄 HOLD")
                        else:
                            st.error("📉 SELL SIGNAL")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # DISPLAY REASONS
                        st.subheader("📊 ANALYSIS BREAKDOWN")
                        for reason in reasons:
                            st.write(f"• {reason}")
                        
                        # INDICATOR GRID
                        st.subheader("🔧 TECHNICAL INDICATORS")
                        cols = st.columns(4)
                        
                        indicators = []
                        if 'RSI_14' in df:
                            rsi_val = df['RSI_14'].iloc[-1]
                            rsi_status = 'bullish' if 40 <= rsi_val <= 65 else 'bearish'
                            indicators.append(('RSI 14', f"{rsi_val:.1f}", rsi_status))
                        
                        if all(col in df for col in ['MACD', 'MACD_Signal']):
                            macd_status = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
                            indicators.append(('MACD', 'BULLISH' if macd_status == 'bullish' else 'BEARISH', macd_status))
                        
                        if 'BB_Position' in df:
                            bb_pos = df['BB_Position'].iloc[-1]
                            bb_status = 'bullish' if bb_pos < 0.3 else 'bearish'
                            indicators.append(('BB Position', f"{bb_pos:.2f}", bb_status))
                        
                        if 'ATR' in df:
                            atr_percent = (df['ATR'].iloc[-1] / current_price) * 100
                            atr_status = 'bullish' if atr_percent < 2 else 'neutral'
                            indicators.append(('ATR %', f"{atr_percent:.1f}%", atr_status))
                        
                        for idx, (name, value, status) in enumerate(indicators):
                            with cols[idx]:
                                st.markdown(f'''
                                <div class="indicator-card {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
            
            with col2:
                if 'df' in locals() and df is not None:
                    st.plotly_chart(analyzer.create_advanced_chart(df, selected_stock), use_container_width=True)

if __name__ == "__main__":
    main()
