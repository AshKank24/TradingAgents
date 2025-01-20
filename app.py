import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from enum import Enum
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Add after loading env vars and before chat state initialization
try:
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    llm = None

# Initialize chat state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load documentation
try:
    loader = PyPDFLoader(r'assets\App Documentation.pdf')
    docs = loader.load()
except Exception as e:
    st.error(f"Error loading documentation: {str(e)}")
    docs = []

# Define prompt template
prompt_template = prompt_template = '''
Human: You are a friendly financial assistant chatbot for a finance application which uses back-testing to test 
users' given trading algorithms. Your job is to answer questions based on the given app documentation. Try to help
and navigate the user to nudge him in the right direction keeping the knowledge of the app documentation in mind.
Do not use any external knowledge. Be friendly and polite at all times. App documentation is enclosed in ##.
 
App Documentation: ##{context}##
 
If the question is not related to the app and is based on financial or trading terms and metrics based on 
stock trading for new users on our application, you can answer using your external knowledge. 
Try to be as friendly as possible while answering the questions and provide details about any financial 
queries they ask. Also, if the question is completely out of context from the world of finance and stock 
trading, politely reply 'I don't know' with explanation that you answer only financial queries. 
The query could be in any language. Try to reply in the same language as the query. If you are not sure 
about the language, answer in English.

Chat History:
{chat_history}

Query: {query}

Answer:
'''

class LogicalOperator(Enum):
    AND = "AND"
    OR = "OR"

def render_chat_interface():
    with st.sidebar:
        st.markdown("---")
        st.subheader("Chat Assistant")
        
        # Initialize chat input state if not exists
        if "chat_input" not in st.session_state:
            st.session_state.chat_input = ""
        
        # Chat history display
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["human"])
            with st.chat_message("assistant"):
                st.write(message["ai"])
        
        # Query input with callback
        def on_input_change():
            user_input = st.session_state.chat_input
            if user_input:
                try:
                    # Format history
                    formatted_history = "\n".join([
                    f"Human: {msg['human']}\nAI: {msg['ai']}" 
                    for msg in st.session_state.chat_history
                    ])
                    
                    # Process query
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    chain = create_stuff_documents_chain(llm, prompt)
                    
                    result = chain.invoke({
                    "context": docs,
                    "query": user_input,
                    "chat_history": formatted_history
                    })
                    
                    # Update history
                    st.session_state.chat_history.append({
                    "human": user_input,
                    "ai": result
                    })
                    
                    # Limit history
                    st.session_state.chat_history = st.session_state.chat_history[-5:]
                    
                    # Clear input
                    st.session_state.chat_input = ""
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Chat input with callback
        st.text_input(
            "Ask me anything!",
            key="chat_input",
            on_change=on_input_change
        )

def get_data(symbol, start_date, end_date):
    try:
        API_KEY = os.getenv("ALPACA_API_KEY")
        API_SECRET = os.getenv("ALPACA_API_SECRET")
        
        if not API_KEY or not API_SECRET:
            st.error("API credentials not found. Please check your .env file.")
            return None
        
        client = StockHistoricalDataClient(API_KEY, API_SECRET)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]
        
        df = add_technical_indicators(df)
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def add_technical_indicators(df):
    """Add all technical indicators"""
    # SMA and EMA
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    
    # VWAP
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df

def implement_strategy(df, params):
    """Implement trading strategy based on parameters"""
    df = df.copy()
    df['position'] = 0
    
    if params['type1'] == 'SMA_Crossover':
        df['position'] = np.where(df['SMA20'] > df['SMA50'], 1, -1)
    
    elif params['type1'] == 'RSI':
        df['position'] = np.where(df['RSI'] < params['oversold'], 1,
                                np.where(df['RSI'] > params['overbought'], -1, 0))
    
    elif params['type1'] == 'MACD':
        df['position'] = np.where(df['MACD_hist'] > 0, 1, -1)
    
    elif params['type1'] == 'Bollinger_Bands':
        df['position'] = np.where(df['close'] < df['BB_lower'], 1,
                                np.where(df['close'] > df['BB_upper'], -1, 0))
    
    elif params['type1'] == 'Stochastic':
        df['position'] = np.where((df['Stoch_k'] < params['oversold']) & 
                                (df['Stoch_d'] < params['oversold']), 1,
                                np.where((df['Stoch_k'] > params['overbought']) & 
                                       (df['Stoch_d'] > params['overbought']), -1, 0))
    
    elif params['type1'] == 'MA_Envelope':
        envelope_upper = df['SMA20'] * (1 + params['envelope_percentage']/100)
        envelope_lower = df['SMA20'] * (1 - params['envelope_percentage']/100)
        df['position'] = np.where(df['close'] < envelope_lower, 1,
                                np.where(df['close'] > envelope_upper, -1, 0))
    
    elif params['type1'] == 'VWAP':
        df['position'] = np.where(df['close'] < df['VWAP'] * 
                                (1 - params['deviation_threshold']/100), 1,
                                np.where(df['close'] > df['VWAP'] * 
                                       (1 + params['deviation_threshold']/100), -1, 0))
    
    # Apply stop loss and take profit
    current_position = 0
    entry_price = 0
    
    for i in range(1, len(df)):
        if df['position'].iloc[i] != 0 and current_position == 0:
            current_position = df['position'].iloc[i]
            entry_price = df['close'].iloc[i]
        elif current_position != 0:
            pnl = (df['close'].iloc[i] - entry_price) / entry_price * 100
            if (pnl <= -params['stop_loss']) or (pnl >= params['take_profit']):
                df.loc[df.index[i], 'position'] = -current_position
                current_position = 0
    
    df['returns'] = df['close'].pct_change() * df['position'].shift(1)
    return df

def calculate_metrics(df):
    returns = df['returns'].dropna()
    metrics = {
        'Total Return': f"{(returns + 1).prod() - 1:.2%}",
        'Annual Return': f"{(returns + 1).prod() ** (252/len(returns)) - 1:.2%}",
        'Sharpe Ratio': f"{np.sqrt(252) * returns.mean() / returns.std():.2f}",
        'Max Drawdown': f"{(df['close'] / df['close'].cummax() - 1).min():.2%}"
    }
    return metrics

def get_trade_feedback(trade_data):
    """Get LLM feedback for a trade"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = f"""
        As an expert in stock trading, provide a brief 2-line analysis of this trade:
        Stock: {trade_data['symbol']}
        Entry: ${trade_data['entry_price']} on {trade_data['entry_date']}
        Exit: ${trade_data['exit_price']} on {trade_data['exit_date']}
        P/L: {trade_data['profit_loss']}
        Strategy: {trade_data['strategy']}
        """
        
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        return f"Error generating feedback: {str(e)}"
    
def update_strategy_params(params, strategy, suffix):
    """Update strategy parameters based on strategy type"""
    if strategy == 'SMA_Crossover':
        params.update({
            f'sma_short_{suffix}': st.slider(f"Short MA Period {suffix}", 5, 50, 20),
            f'sma_long_{suffix}': st.slider(f"Long MA Period {suffix}", 10, 200, 50)
        })
    
    elif strategy == 'RSI':
        params.update({
            f'rsi_period_{suffix}': st.slider(f"RSI Period {suffix}", 5, 30, 14),
            f'oversold_{suffix}': st.slider(f"Oversold Level {suffix}", 20, 40, 30),
            f'overbought_{suffix}': st.slider(f"Overbought Level {suffix}", 60, 80, 70)
        })
    
    elif strategy == 'MACD':
        params.update({
            f'macd_fast_{suffix}': st.slider(f"MACD Fast Period {suffix}", 5, 20, 12),
            f'macd_slow_{suffix}': st.slider(f"MACD Slow Period {suffix}", 15, 50, 26),
            f'macd_signal_{suffix}': st.slider(f"MACD Signal Period {suffix}", 5, 20, 9)
        })
    
    elif strategy == 'Bollinger_Bands':
        params.update({
            f'bb_period_{suffix}': st.slider(f"Bollinger Band Period {suffix}", 5, 50, 20),
            f'bb_std_{suffix}': st.slider(f"Standard Deviations {suffix}", 1.0, 3.0, 2.0, 0.1)
        })
    
    elif strategy == 'Stochastic':
        params.update({
            f'stoch_k_{suffix}': st.slider(f"Stochastic %K Period {suffix}", 5, 30, 14),
            f'stoch_d_{suffix}': st.slider(f"Stochastic %D Period {suffix}", 2, 10, 3),
            f'oversold_{suffix}': st.slider(f"Oversold Level {suffix}", 10, 30, 20),
            f'overbought_{suffix}': st.slider(f"Overbought Level {suffix}", 70, 90, 80)
        })
    
    elif strategy == 'MA_Envelope':
        params.update({
            f'envelope_period_{suffix}': st.slider(f"MA Period {suffix}", 5, 50, 20),
            f'envelope_percentage_{suffix}': st.slider(f"Envelope Percentage {suffix}", 0.5, 5.0, 1.0, 0.1)
        })
    
    elif strategy == 'VWAP':
        params.update({
            f'deviation_threshold_{suffix}': st.slider(f"Deviation Threshold % {suffix}", 0.5, 5.0, 1.0, 0.1)
        })

def main():
    st.set_page_config(layout="wide", page_title="Trade Sarthi")
    st.title("Trade Sarthi")
    
    with st.sidebar:
        st.header("Configuration")
        symbol = st.text_input("Stock Symbol", "AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime('2023-01-01'))
        end_date = st.date_input("End Date", pd.to_datetime('2024-01-01'))
        
        col1, col2, col3 = st.columns([2,1,2])
        with col1:
            strategy1 = st.selectbox("Strategy 1", [
                'SMA_Crossover', 'RSI', 'MACD', 'Bollinger_Bands',
                'Stochastic', 'MA_Envelope', 'VWAP'
            ])
        with col2:
            logical_op = st.selectbox("", [op.value for op in LogicalOperator])
        with col3:
            strategy2 = st.selectbox("Strategy 2", [
                'SMA_Crossover', 'RSI', 'MACD', 'Bollinger_Bands',
                'Stochastic', 'MA_Envelope', 'VWAP'
            ])
        
        strategy_params = {
            'type1': strategy1,
            'type2': strategy2,
            'operator': logical_op
        }
        
        st.subheader("Risk Management")
        strategy_params.update({
            'stop_loss': st.slider("Stop Loss (%)", 1.0, 10.0, 5.0),
            'take_profit': st.slider("Take Profit (%)", 1.0, 20.0, 10.0)
        })
        
        st.subheader("Strategy 1 Parameters")
        update_strategy_params(strategy_params, strategy1, '1')
        
        st.subheader("Strategy 2 Parameters") 
        update_strategy_params(strategy_params, strategy2, '2')

        render_chat_interface()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        df = get_data(symbol, start_date, end_date)
        if df is not None:
            # Plot recent history of the stock
            fig_recent = go.Figure()
            fig_recent.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ))
            fig_recent.update_layout(
                title="Recent Stock Price History",
                xaxis_title="Date",
                yaxis_title="Price(In USD)"
            )
            st.plotly_chart(fig_recent)
        
        if st.button("Run Backtest"):
            if df is not None:
                df = implement_strategy(df, strategy_params)
                
                # Plot price and signals
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                ))
                
                # Add signals to plot
                signals = df[df['position'].diff() != 0].copy()
                fig.add_trace(go.Scatter(
                    x=signals[signals['position'] == 1].index,
                    y=signals[signals['position'] == 1]['close'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name="Buy Signal"
                ))
                fig.add_trace(go.Scatter(
                    x=signals[signals['position'] == -1].index,
                    y=signals[signals['position'] == -1]['close'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name="Sell Signal"
                ))
                
                fig.update_layout(
                    title="Stock Price and Trading Signals",
                    xaxis_title="Date",
                    yaxis_title="Price(In USD)"
                )
                st.plotly_chart(fig)
                
                # Calculate and display metrics
                metrics = calculate_metrics(df)
                st.subheader("Performance Metrics")
                
                if df is not None:
                    for metric, value in metrics.items():
                        st.metric(metric, value)
                    
                    # Add Trade History Table
                    st.subheader("Trade Analysis")
                    signals = df[df['position'].diff() != 0].copy()
                    # Inside main() where trade history is created:
                    trade_history = []
                    entry_price = None
                    entry_date = None

                    for idx, row in signals.iterrows():
                        if row['position'] == 1:  # Buy signal
                            entry_price = row['close']
                            entry_date = idx
                        elif row['position'] == -1 and entry_price is not None:  # Sell signal
                            exit_price = row['close']
                            profit_loss = ((exit_price - entry_price) / entry_price) * 100
                            
                            trade_data = {
                                'symbol': symbol,
                                'strategy': [strategy1,strategy2],
                                'entry_date': entry_date.strftime('%Y-%m-%d'),
                                'entry_price': entry_price,
                                'exit_date': idx.strftime('%Y-%m-%d'),
                                'exit_price': exit_price,
                                'profit_loss': f"{profit_loss:.2f}%"
                            }
                            
                            feedback = get_trade_feedback(trade_data)
                            
                            trade_history.append({
                                'Entry Date': trade_data['entry_date'],
                                'Entry Price': f"${entry_price:.2f}",
                                'Exit Date': trade_data['exit_date'],
                                'Exit Price': f"${exit_price:.2f}",
                                'Profit/Loss (%)': trade_data['profit_loss'],
                                'AI Feedback': feedback
                            })
                            entry_price = None

                    if trade_history:
                        trade_df = pd.DataFrame(trade_history)
                        st.dataframe(
                            trade_df,
                            use_container_width=True,
                            column_config={
                                "AI Feedback": st.column_config.TextColumn(
                                    "AI Analysis",
                                    width="large",
                                    help="AI-generated trade analysis"
                                )
                            }
                        )
                    else:
                        st.write("No trades executed during this period")
                
                # Plot equity curve
                equity_curve = (1 + df['returns']).cumprod()
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve,
                    mode='lines',
                    name='Equity Curve'
                ))
                fig_equity.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Equity"
                )
                st.plotly_chart(fig_equity)

if __name__ == "__main__":
    main()