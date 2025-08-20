# Limit Order Book Simulator & Market-Making Strategy

## Overview
An interactive trading simulator with:
- A **Limit Order Book** (price–time priority)
- A **Rule-Based Market Maker** (inventory skew, cash, PnL tracking)
- An **Interactive Dashboard** (PnL, mid-price, inventory plots)
- An **LLM-Powered Chatbot** for Q&A and reasoning

---

## Features
- Limit Order Book simulation  
- Market maker with inventory management  
- Interactive dashboard with plots  
- Performance metrics (volatility, Sharpe ratio, drawdowns)  
- LLM chatbot for performance insights  

---

## Installation
```bash

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

## Running the Simulation
# Run market maker simulation
python -m strategy.rule_based_mm

# Start the dashboard
python dashboard/app.py

## Data Output

Simulation results are stored in:
data/history_rule_based.json


## Dashboard Usage

1. Open http://127.0.0.1:8050

2. Upload JSON data

3. View plots (PnL, inventory, mid-price)

4. Ask the chatbot questions


## Chatbot Setup

Set your OpenAI API key and model:

# PowerShell (Windows)
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"

# Linux / Mac
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4o-mini"


## Example Chatbot Questions

- What was the maximum drawdown?

- Summarize the trading performance.

- What was the average inventory?

- Did the strategy end in profit?

- Explain the risk profile of this market maker.


## Project Structure

lob-mm-project/
│── dashboard/            # Dash app (UI + chatbot + plots)
│   └── app.py
│── lob/                  # Matching engine
│   └── matching_engine.py
│── strategy/             # Trading strategies
│   ├── rule_based_mm.py
│   └── ml_agent.py
│── tests/                # Unit tests
│   └── test_lob.py
│── data/                 # Output JSONs
│── notebooks/            # Jupyter analysis
│   └── strategy_analysis.ipynb
│── requirements.txt
│── README.md


