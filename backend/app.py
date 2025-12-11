"""
Azure Demand Forecasting & Capacity Optimization System
Milestone 4 - Backend API

Flask REST API for CPU demand forecasting with:
- Model deployment
- Recursive forecasting (7 and 30 days)
- Capacity planning
- Model drift monitoring
- Automated reporting
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import traceback

# Load environment variables from .env file
load_dotenv()

# Import custom utility modules
from forecast_utils import recursive_forecast_cpu, recursive_forecast_storage, prepare_single_prediction_features
from capacity_utils import analyze_capacity, detailed_capacity_report, optimization_suggestor
from monitoring_utils import monitoring_stats, comprehensive_model_health, calculate_mape

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ---------------------------------------------------
# 1) LOAD MODEL AND DATASET AT STARTUP
# ---------------------------------------------------
print("=" * 60)
print("= Azure Demand Forecasting API - Starting Up...")
print("=" * 60)

try:
    # Load trained CPU demand model
    cpu_model_path = os.path.join("models", "rf_cpu_model.pkl")
    cpu_model = joblib.load(cpu_model_path)
    print(f"[OK] Loaded CPU model from: {cpu_model_path}")

    # Load trained Storage demand model
    storage_model_path = os.path.join("models", "storage_demand_model.pkl")
    storage_model = joblib.load(storage_model_path)
    print(f"[OK] Loaded Storage model from: {storage_model_path}")
    
    # Load ML-ready dataset
    data_path = os.path.join("data", "feature_engineered", "mlmodeltrainingdataset.csv")
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded dataset from: {data_path}")
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {len(cpu_model.feature_names_in_)}")
    
    print("=" * 60)
    print("[OK] Initialization Complete - API Ready!")
    print("=" * 60)
    
except Exception as e:
    print("=" * 60)
    print(f"[ERROR] during initialization: {str(e)}")
    print("=" * 60)
    raise


# ---------------------------------------------------
# 2) HELPER FUNCTION - Convert numpy types to Python types
# ---------------------------------------------------
def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


# ---------------------------------------------------
# 3) API ENDPOINTS
# ---------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "message": "🚀 Azure Demand Forecasting API is running!",
        "version": "Milestone 4",
        "endpoints": {
            "health": "GET /",
            "metrics": "GET /api/metrics",
            "predict": "POST /api/predict_cpu",
            "forecast_7": "GET /api/forecast_7",
            "forecast_30": "GET /api/forecast_30",
            "capacity": "POST /api/capacity_planning",
            "optimization": "POST /api/optimization",
            "monitoring": "GET /api/monitoring",
            "report": "GET /api/report",
            "multi_region": "GET /api/multi_region"
        }
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Get model status and metadata."""
    try:
        return jsonify({
            "cpu_model": {
                "status": "loaded",
                "type": type(cpu_model).__name__,
                "n_features": len(cpu_model.feature_names_in_),
                "features": cpu_model.feature_names_in_.tolist()
            },
            "storage_model": {
                "status": "loaded",
                "type": type(storage_model).__name__,
                "n_features": len(storage_model.feature_names_in_)
            },
            "dataset": {
                "status": "loaded",
                "shape": df.shape,
                "rows": len(df),
                "columns": len(df.columns)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_cpu", methods=["POST"])
def predict_cpu():
    """
    Single CPU prediction with custom input.
    """
    try:
        # Get input data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Prepare features (calculate lag/rolling if not provided)
        features = prepare_single_prediction_features(input_data, df)
        
        # Reindex to match model's expected feature order
        feature_vector = features.reindex(cpu_model.feature_names_in_).values.reshape(1, -1)
        
        # Make prediction
        prediction = cpu_model.predict(feature_vector)[0]
        
        return jsonify({
            "prediction": float(prediction),
            "input_features": convert_to_python_types(features.to_dict())
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# Region multipliers for simulation
REGION_MULTIPLIERS = {
    "East": 1.0,
    "West": 1.15,
    "North": 0.9,
    "South": 1.05,
    "East US": 1.0, 
    "West Europe": 0.95,
    "Central India": 1.2
}

@app.route("/api/forecast_7", methods=["GET"])
def forecast_7():
    """
    7-day recursive CPU & Storage demand forecast.
    Optional query param: region (default: "East")
    """
    try:
        region = request.args.get("region", "East")
        
        # Generate 7-day forecast
        predictions_cpu = recursive_forecast_cpu(df, cpu_model, n_days=7)
        predictions_storage = recursive_forecast_storage(df, storage_model, n_days=7)
        
        # Apply regional variation
        multiplier = REGION_MULTIPLIERS.get(region, 1.0)
        predictions_cpu = [round(p * multiplier, 2) for p in predictions_cpu]
        predictions_storage = [round(p * multiplier, 2) for p in predictions_storage]
        
        return jsonify({
            "forecast_days": 7,
            "region": region,
            "predictions": predictions_cpu,          # Kept for backward compat
            "predictions_cpu": predictions_cpu,      # Explicit name
            "predictions_storage": predictions_storage # New field
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/forecast_30", methods=["GET"])
def forecast_30():
    """
    30-day recursive CPU & Storage demand forecast.
    Optional query param: region (default: "East")
    """
    try:
        region = request.args.get("region", "East")
        
        # Generate 30-day forecast
        predictions_cpu = recursive_forecast_cpu(df, cpu_model, n_days=30)
        predictions_storage = recursive_forecast_storage(df, storage_model, n_days=30)
        
        # Apply regional variation
        multiplier = REGION_MULTIPLIERS.get(region, 1.0)
        predictions_cpu = [round(p * multiplier, 2) for p in predictions_cpu]
        predictions_storage = [round(p * multiplier, 2) for p in predictions_storage]
        
        return jsonify({
            "forecast_days": 30,
            "region": region,
            "predictions": predictions_cpu,          # Kept for backward compat
            "predictions_cpu": predictions_cpu,      # Explicit name
            "predictions_storage": predictions_storage # New field
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/capacity_planning", methods=["POST"])
def capacity_planning():
    """
    Capacity planning analysis with scaling recommendations.
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data or "capacity" not in data:
            return jsonify({"error": "Missing 'capacity' in request body"}), 400
        
        capacity = data["capacity"]
        forecast_days = data.get("forecast_days", 7)
        
        # Generate forecast
        predictions = recursive_forecast_cpu(df, cpu_model, n_days=forecast_days)
        
        # Analyze capacity
        analysis = analyze_capacity(predictions, capacity)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/optimization", methods=["POST"])
def optimization():
    try:
        data = request.get_json()
        if not data or "capacity" not in data:
            return jsonify({"error": "Missing 'capacity' in request body"}), 400

        capacity = float(data["capacity"])
        forecast_days = int(data.get("forecast_days", 1))
        region = data.get("region", "unknown")

        predictions = recursive_forecast_cpu(df, cpu_model, n_days=forecast_days)
        suggestion = optimization_suggestor(predictions, capacity, region)

        return jsonify(suggestion)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/monitoring", methods=["GET"])
def monitoring():
    """
    Model health monitoring and drift detection.
    """
    try:
        # Get MAPE from query parameters (optional override)
        mape_param = request.args.get("mape")
        
        if mape_param:
            mape = float(mape_param)
        else:
            # CALCULATE REAL MAPE FROM RECENT DATA (Last 30 records)
            recent_data = df.tail(30).copy()
            
            # Prepare features (ensure alignment with model)
            if hasattr(cpu_model, "feature_names_in_"):
                features = recent_data[cpu_model.feature_names_in_]
            else:
                features = recent_data.drop(columns=["usage_cpu", "date", "region"], errors="ignore")
                
            # Make predictions
            predictions = cpu_model.predict(features)
            actuals = recent_data["usage_cpu"].values
            
            # Calculate actual MAPE
            mape = calculate_mape(actuals, predictions)
        
        # Analyze model health
        health = monitoring_stats(mape)
        
        return jsonify(health)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/report", methods=["GET"])
def report():
    """
    Comprehensive automated report combining forecast, capacity, and monitoring.
    """
    try:
        # Get parameters
        capacity = float(request.args.get("capacity", 10000))
        mape = float(request.args.get("mape", 8.5))
        
        # Generate 7-day forecast
        forecast_7 = recursive_forecast_cpu(df, cpu_model, n_days=7)
        
        # Build comprehensive report
        report_data = {
            "report_type": "comprehensive",
            "generated_at": pd.Timestamp.now().isoformat(),
            
            "forecast_summary": {
                "days_forecasted": 7,
                "predictions": forecast_7,
                "avg_forecast": float(np.mean(forecast_7)),
                "min_forecast": float(np.min(forecast_7)),
                "max_forecast": float(np.max(forecast_7)),
                "trend": "increasing" if forecast_7[-1] > forecast_7[0] else "decreasing"
            },
            
            "capacity_analysis": analyze_capacity(forecast_7, capacity),
            
            "model_health": monitoring_stats(mape)
        }
        
        return jsonify(convert_to_python_types(report_data))
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/multi_region", methods=["GET"])
def multi_region():
    """
    Multi-region capacity comparison endpoint.
    """
    try:
        # Get regions from query parameters
        regions_param = request.args.get("regions", "East US,West US,North Europe,Southeast Asia")
        region_names = [r.strip() for r in regions_param.split(",")]
        
        # Generate forecast for base comparison
        forecast_7 = recursive_forecast_cpu(df, cpu_model, n_days=7)
        forecast_4 = forecast_7[:4]  # First 4 days for T+1 to T+4
        
        # Base CPU usage from last known value
        last_cpu = float(df["usage_cpu"].iloc[-1]) if "usage_cpu" in df.columns else float(np.mean(forecast_7))
        last_storage = float(df["usage_storage"].iloc[-1]) if "usage_storage" in df.columns else last_cpu * 0.85
        
        regions_data = []
        
        for idx, region_name in enumerate(region_names):
            # Add regional variation to forecasts (simulate different regions)
            # Each region gets slightly different forecast based on index
            variation_factor = 1.0 + (idx * 0.05)  # 0%, 5%, 10% variation
            region_forecast = [round(v * variation_factor, 1) for v in forecast_4]
            
            # Regional CPU and storage usage with variation
            region_cpu = round(last_cpu * variation_factor, 1)
            region_storage = round(last_storage * variation_factor, 1)
            
            # Generate peak hours (simulate different timezones)
            peak_hour_base = 14 + (idx * 2)  # Different peak hours per region
            peak_hours = [
                f"{(peak_hour_base - 1) % 24:02d}:00",
                f"{peak_hour_base % 24:02d}:00",
                f"{(peak_hour_base + 1) % 24:02d}:00"
            ]
            
            # Get optimization recommendation for this region
            avg_forecast = np.mean(region_forecast)
            capacity = 10000  # Default capacity
            optimization = optimization_suggestor(region_forecast, capacity, region_name)
            recommendation = optimization.get("recommendation", "Monitor usage closely")
            
            regions_data.append({
                "name": region_name,
                "cpuUsage": region_cpu,
                "storageUsage": region_storage,
                "forecast": region_forecast,
                "peakHours": peak_hours,
                "recommendation": recommendation
            })
        
        return jsonify({
            "regions": convert_to_python_types(regions_data)
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ---------------------------------------------------
# 4) ERROR HANDLERS
# ---------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------
# CHAT ASSISTANT ENDPOINT (Gemini Powered)
# ---------------------------------------------------

def generate_offline_response(user_message, context):
    """
    Generate a helpful response without using the Gemini API.
    Uses rule-based logic and context data.
    """
    message_lower = user_message.lower()
    
    # Parse context if available
    forecast_data = context.get("forecastData", []) if context else []
    metrics = context.get("metrics", {}) if context else {}
    
    # Common question patterns
    if any(word in message_lower for word in ["forecast", "prediction", "predict", "future"]):
        if forecast_data and len(forecast_data) > 0:
            avg = sum(forecast_data[:7]) / min(7, len(forecast_data))
            trend = "increasing" if forecast_data[-1] > forecast_data[0] else "decreasing"
            return f"Based on the current forecast data, your CPU usage is {trend} with an average of {avg:.1f}% over the next week. The forecast shows values ranging from {min(forecast_data):.1f}% to {max(forecast_data):.1f}%."
        return "Your forecast shows CPU and storage demand for the next 7-30 days. Use the Forecasts page to see detailed predictions and capacity planning recommendations."
    
    elif any(word in message_lower for word in ["capacity", "scale", "scaling"]):
        return "Capacity planning helps you determine when to scale resources. Check the 'Capacity' section for scaling recommendations based on your forecast. Generally, you should scale up when utilization consistently exceeds 80%, and scale down when it's below 40% for extended periods."
    
    elif any(word in message_lower for word in ["cpu", "processor"]):
        if metrics and "cpu" in str(metrics).lower():
            return "CPU usage metrics show your current processor utilization. Monitor for sustained high usage (>85%) which may indicate need for scaling. Check the Usage Trends page for historical patterns."
        return "CPU (Central Processing Unit) utilization indicates how much processing power your systems are using. High CPU usage may require scaling up resources, while consistently low usage suggests opportunity for cost optimization."
    
    elif any(word in message_lower for word in ["storage", "disk", "memory"]):
        return "Storage metrics track your data storage consumption. Monitor both current usage and growth rate to plan capacity needs. The forecast includes storage predictions to help prevent running out of space."
    
    elif any(word in message_lower for word in ["alert", "threshold", "warning"]):
        return "Alerts are triggered when metrics exceed defined thresholds. You can configure threshold alerts in the Monitoring section. Common thresholds: CPU >85%, Storage >90%, to get early warnings before issues occur."
    
    elif any(word in message_lower for word in ["report", "download", "export"]):
        return "You can download comprehensive PDF reports from the Reports page. These include performance summaries, forecasts, and recommendations. Use the 'Download Report' button to generate a PDF for stakeholders."
    
    elif any(word in message_lower for word in ["model", "accuracy", "drift"]):
        return "The forecasting models use machine learning to predict future demand. Model health is monitored for drift (accuracy degradation). Check the Model Dashboard to see current model performance metrics like MAPE (Mean Absolute Percentage Error)."
    
    elif any(word in message_lower for word in ["region", "multi-region", "geographic"]):
       return "Multi-region monitoring helps you compare resource usage across different Azure regions. This is useful for load balancing and identifying regional capacity bottlenecks. Check the Multi-Region page for cross-region comparisons."
    
    elif any(word in message_lower for word in ["optimize", "optimization", "cost"]):
        return "Optimization recommendations help reduce costs while maintaining performance. Look for: 1) Overprovisioned resources (low utilization), 2) Scaling opportunities (high utilization), 3) Right-sizing suggestions based on usage patterns."
    
    elif any(word in message_lower for word in ["help", "how", "what", "explain"]):
        return """I can help you with:
• **Forecasts**: View CPU & storage predictions for next 7-30 days
• **Capacity Planning**: Get scaling recommendations
• **Reports**: Download performance & forecast reports
• **Monitoring**: Check model health and accuracy
• **Alerts**: Set up threshold-based notifications
• **Multi-Region**: Compare usage across regions

Try asking specific questions like: "What's my CPU forecast?" or "Should I scale up?"
"""
    
    else:
        # Generic helpful response
        return """I'm an Azure capacity planning assistant. I can help with:

📊 **Forecasts** - View demand predictions
🎯 **Capacity** - Get scaling recommendations  
📈 **Reports** - Download summaries
⚠️ **Alerts** - Monitor thresholds
🌍 **Regions** - Compare multi-region usage

What would you like to know about your Azure infrastructure?"""


@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    """
    AI Chat Assistant endpoint using Google Gemini.
    """
    try:
        data = request.json
        user_message = data.get("message", "")
        system_instruction = data.get("system_instruction", "")
        
        print(f"[CHAT] Received message: {user_message[:100]}...")
        
        # Check if API key is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[CHAT] No API key found")
            return jsonify({
                "reply": "API key is not configured. Please contact the administrator."
            }), 500

        print("[CHAT] Configuring Gemini API...")
        genai.configure(api_key=api_key)
        
        # Dynamically discover available model, prefer Gemini 1.5
        print("[CHAT] Discovering available models...")
        available_model = None
        
        # Try Gemini 1.5 models first
        preferred_models = [
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-pro-latest', 
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        try:
            # First check if any preferred model is available
            all_models = list(genai.list_models())
            for preferred in preferred_models:
                for model in all_models:
                    if model.name == preferred and 'generateContent' in model.supported_generation_methods:
                        available_model = model.name
                        print(f"[CHAT] Found preferred model: {available_model}")
                        break
                if available_model:
                    break
            
            # If no preferred model found, use any available model
            if not available_model:
                for model in all_models:
                    if 'generateContent' in model.supported_generation_methods:
                        available_model = model.name
                        print(f"[CHAT] Using available model: {available_model}")
                        break
        except Exception as list_error:
            print(f"[CHAT] Error listing models: {list_error}")
            # Fallback to default
            available_model = 'models/gemini-1.5-flash-latest'
        
        if not available_model:
            return jsonify({
                "reply": "No suitable AI model is available. Please check your API configuration."
            }), 500
        
        print(f"[CHAT] Using model: {available_model}")
        
        # Configure with minimal safety restrictions
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Use discovered model with relaxed safety settings
        model = genai.GenerativeModel(
            available_model,
            generation_config={
                'temperature': 0.9,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 2048,
            },
            safety_settings=safety_settings
        )
        
        # Construct simple prompt (avoid triggering safety filters)
        # Extract and log context for debugging
        context_data = {}
        if system_instruction and "Current page context:" in system_instruction:
            try:
                # Extract the JSON part after "Current page context: "
                context_start = system_instruction.find("Current page context:") + len("Current page context:")
                context_end = system_instruction.find(". Use this context")
                if context_end == -1:
                    context_end = len(system_instruction)
                
                context_str = system_instruction[context_start:context_end].strip()
                print(f"[CHAT] Context string: {context_str[:200]}...")
                
                if context_str and context_str != "{}":
                    context_data = json.loads(context_str)
                    print(f"[CHAT] Parsed context: page={context_data.get('page', 'unknown')}, keys={list(context_data.keys())}")
                else:
                    print("[CHAT] Context is empty {}")
            except Exception as e:
                print(f"[CHAT] Error parsing context: {e}")
        
        # Build prompt with context
        if context_data and context_data.get('page'):
            # Format context nicely for Gemini
            page = context_data.get('page', 'Unknown')
            summary = context_data.get('summary', {})
            
            context_text = f"You are analyzing data from the {page} page of an Azure capacity planning dashboard.\n\n"
            
            if page == "Forecasts":
                context_text += f"Current forecast data:\n"
                context_text += f"- Forecast period: {summary.get('totalDays', 'N/A')} days\n"
                context_text += f"- Average CPU: {summary.get('avgCPU', 'N/A')}%\n"
                context_text += f"- Maximum CPU: {summary.get('maxCPU', 'N/A')}%\n"
                context_text += f"- Minimum CPU: {summary.get('minCPU', 'N/A')}%\n"
                context_text += f"- Average Storage: {summary.get('avgStorage', 'N/A')} TB\n"
                if context_data.get('cpuForecast'):
                    forecast_values = context_data['cpuForecast'][:7]  # First 7 days
                    context_text += f"- Next 7 days CPU forecast: {forecast_values}\n"
            
            elif page == "Reports":
                context_text += f"Performance report data:\n"
                context_text += f"- Average CPU: {summary.get('avgCPU', 'N/A')}%\n"
                context_text +=f"- Average Storage: {summary.get('avgStorage', 'N/A')}\n"
                context_text += f"- Forecast average: {summary.get('avgForecast', 'N/A')}%\n"
                context_text += f"- Total predictions: {summary.get('totalPredictions', 'N/A')}\n"
                
            context_text += f"\nUser question: {user_message}"
            prompt = context_text
        else:
            prompt = user_message
        
        print("[CHAT] Sending request to Gemini...")
        response = model.generate_content(prompt)
        
        # Try to extract text from various response formats
        reply_text = ""
        
        # Method 1: Direct .text attribute
        if hasattr(response, 'text') and response.text:
            reply_text = response.text
        # Method 2: candidates[0].content.parts[0].text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts and hasattr(parts[0], 'text'):
                    reply_text = parts[0].text
        # Method 3: parts attribute directly
        elif hasattr(response, 'parts') and response.parts:
            reply_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
        
        if not reply_text:
            # Check if blocked
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason'):
                    return jsonify({
                        "reply": f"Response was blocked by Google's safety filters: {feedback.block_reason}. Try rephrasing your question or use a different API key with relaxed safety settings."
                    }), 200
            
            return jsonify({
                "reply": "Could not extract response from API. The model may have returned an empty result."
            }), 500
        
        print(f"[CHAT] Got response: {reply_text[:100]}...")
        return jsonify({"reply": reply_text})

    except Exception as e:
        error_msg = str(e)
        print(f"[CHAT ERROR] {type(e).__name__}: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            "reply": f"Error: {error_msg}"
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Flask Development Server...")
    port = int(os.environ.get('PORT', 5000))
    print(f"-> Access the API at: http://localhost:{port}")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
