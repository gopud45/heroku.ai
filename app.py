from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import os
from datetime import datetime, timedelta

app = Flask(__name__)
# Enable CORS for all origins, allowing frontend to access this backend
CORS(app)

# Configure the Gemini API key.
# When deploying, set this as an environment variable (e.g., GEMINI_API_KEY).
# Locally, you can set it via `export GEMINI_API_KEY="YOUR_KEY"` before running.
API_KEY = os.getenv("GEMINI_API_KEY", "") 

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("WARNING: GEMINI_API_KEY environment variable not set. AI assistant may not work.")

# --- Helper function to fetch stock data ---
def get_stock_data(ticker_symbol, period="5y"):
    """Fetches historical stock data using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch data for the specified period (e.g., "5y" for 5 years)
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        # Reset index to make 'Date' a column
        hist.reset_index(inplace=True)
        # Convert Date to string for JSON serialization
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        return hist.to_dict(orient='records')
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None

# --- API Endpoint for Stock Data ---
@app.route('/api/stock_data', methods=['GET'])
def stock_data():
    ticker_symbol = request.args.get('ticker', default='AAPL', type=str).upper()
    data = get_stock_data(ticker_symbol, period="5y") # Always fetch last 5 years

    if data:
        return jsonify({"ticker": ticker_symbol, "data": data})
    else:
        return jsonify({"error": f"Could not retrieve data for {ticker_symbol}. Please check the ticker symbol."}), 404

# --- API Endpoint for AI Assistant Chat ---
@app.route('/api/chat_ai', methods=['POST'])
def chat_ai():
    user_message = request.json.get('message')
    ticker_symbol = request.json.get('ticker')
    stock_data_context = request.json.get('stock_data_context') # Raw data or summary

    if not user_message or not ticker_symbol:
        return jsonify({"error": "Message and ticker symbol are required."}), 400

    # Prepare context for the LLM
    context_string = ""
    if stock_data_context:
        # Convert list of dicts to a more readable string for the LLM
        df_context = pd.DataFrame(stock_data_context)
        # Limit context to recent data or key summaries to avoid token limits
        if not df_context.empty:
            # Get last 30 days of data for context
            recent_data = df_context.tail(30).to_string(index=False)
            context_string = f"Here is some recent historical stock data for {ticker_symbol} (last 30 days):\n{recent_data}\n"
            
            # Add some summary statistics
            start_date = df_context['Date'].min()
            end_date = df_context['Date'].min() # Corrected to min() for start date
            open_price = df_context['Open'].iloc[0]
            close_price = df_context['Close'].iloc[-1]
            high_price = df_context['High'].max()
            low_price = df_context['Low'].min()
            avg_volume = df_context['Volume'].mean()

            context_string += (
                f"\nSummary for {ticker_symbol} from {start_date} to {end_date}:\n"
                f"Start Open: {open_price:.2f}, End Close: {close_price:.2f}\n"
                f"Highest Price: {high_price:.2f}, Lowest Price: {low_price:.2f}\n"
                f"Average Daily Volume: {avg_volume:,.0f}\n"
            )

    try:
        # Check if API key is configured before trying to use the model
        if not API_KEY:
            return jsonify({"error": "AI assistant is not configured. Please set GEMINI_API_KEY environment variable."}), 503

        # Initialize the generative model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Construct the prompt for the LLM
        prompt = (
            f"You are an AI assistant specialized in analyzing stock market data. "
            f"The user is asking a question about {ticker_symbol}. "
            f"Use the provided stock data context to answer the question concisely and accurately. "
            f"If the question cannot be answered with the provided data, state that. "
            f"Do not make up information or give financial advice. "
            f"Focus on factual analysis based on the data.\n\n"
            f"Stock Data Context:\n{context_string}\n"
            f"User's Question: {user_message}"
        )

        # Generate content using the model
        response = model.generate_content(prompt)

        # Extract the text from the response
        ai_response_text = response.candidates[0].content.parts[0].text
        return jsonify({"response": ai_response_text})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": "Failed to get a response from the AI assistant. Please try again."}), 500

if __name__ == '__main__':
    # For local development, run on 0.0.0.0 to be accessible from other devices on the network.
    # For deployment, the PORT environment variable is typically set by the hosting platform.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
