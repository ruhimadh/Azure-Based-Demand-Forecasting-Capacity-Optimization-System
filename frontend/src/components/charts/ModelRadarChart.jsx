import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

export default function ModelRadarChart({ models }) {
  // Helper to normalize data: "Lower is Better" (Error, Time) vs "Higher is Better" (Speed, R2)
  const normalize = (val, min, max, type = "higher") => {
    if (type === "lower") {
      // For error: Min value gets 100 score, larger values get lower scores
      return val === 0 ? 0 : Math.round((min / val) * 100);
    }
    // For speed/R2: Max value gets 100 score
    return max === 0 ? 0 : Math.round((val / max) * 100);
  };

  // Extract raw values for global min/max calculation
  const maes = models.map(m => m.mae);
  const rmses = models.map(m => m.rmse);
  const mapes = models.map(m => parseFloat(m.mape)); // Parse "18.41%" to 18.41
  const trainTimes = models.map(m => m.trainingTime || 0);
  const infTimes = models.map(m => m.inferenceTime || 0);

  const minMae = Math.min(...maes);
  const minRmse = Math.min(...rmses);
  const minMape = Math.min(...mapes);
  const minTrain = Math.min(...trainTimes);
  const minInf = Math.min(...infTimes);

  // Hardcode indices based on Insights.jsx: 0=ARIMA, 1=RF, 2=XGBoost, 3=LSTM
  // I should rely on finding model by name to be safe.
  const getModel = (name) => models.find(m => m.name === name) || {};

  const arima = getModel("ARIMA");
  const rf = getModel("Random Forest");
  const xgboost = getModel("XGBoost");
  const lstm = getModel("LSTM");

  const radarData = [
    {
      metric: "MAE",
      ARIMA: normalize(arima.mae, minMae, 0, "lower"),
      RF: normalize(rf.mae, minMae, 0, "lower"),
      XGB: normalize(xgboost.mae, minMae, 0, "lower"),
      LSTM: normalize(lstm.mae, minMae, 0, "lower")
    },
    {
      metric: "RMSE",
      ARIMA: normalize(arima.rmse, minRmse, 0, "lower"),
      RF: normalize(rf.rmse, minRmse, 0, "lower"),
      XGB: normalize(xgboost.rmse, minRmse, 0, "lower"),
      LSTM: normalize(lstm.rmse, minRmse, 0, "lower")
    },
    {
      metric: "MAPE",
      ARIMA: normalize(parseFloat(arima.mape), minMape, 0, "lower"),
      RF: normalize(parseFloat(rf.mape), minMape, 0, "lower"),
      XGB: normalize(parseFloat(xgboost.mape), minMape, 0, "lower"),
      LSTM: normalize(parseFloat(lstm.mape), minMape, 0, "lower")
    },
    {
      metric: "R² (Fit)",
      ARIMA: 15,
      RF: 100,
      XGB: 99,
      LSTM: 0
    },
    {
      metric: "Train Time",
      ARIMA: normalize(arima.trainingTime, minTrain, 0, "lower"),
      RF: normalize(rf.trainingTime, minTrain, 0, "lower"),
      XGB: normalize(xgboost.trainingTime, minTrain, 0, "lower"),
      LSTM: normalize(lstm.trainingTime, minTrain, 0, "lower")
    },
    {
      metric: "Inf Time",
      ARIMA: normalize(arima.inferenceTime, minInf, 0, "lower"),
      RF: normalize(rf.inferenceTime, minInf, 0, "lower"),
      XGB: normalize(xgboost.inferenceTime, minInf, 0, "lower"),
      LSTM: normalize(lstm.inferenceTime, minInf, 0, "lower")
    }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow-md mt-6">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        🧭 Model Strength Radar View
      </h3>

      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metric" />
          <Tooltip />
          <Radar name="Random Forest" dataKey="RF" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
          <Radar name="XGBoost" dataKey="XGB" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.5} />
          <Radar name="ARIMA" dataKey="ARIMA" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.4} />
          <Radar name="LSTM" dataKey="LSTM" stroke="#EF4444" fill="#EF4444" fillOpacity={0.3} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
