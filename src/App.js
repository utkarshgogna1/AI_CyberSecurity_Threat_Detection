import React, { useState } from 'react';
import PredictionForm from './PredictionForm';
import PredictionResults from './PredictionResults';
import ChartDisplay from './ChartDisplay';

import './App.css';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [chartData, setChartData] = useState([]);

  const updateResults = (result) => {
    setPredictionResult(result);
    setChartData((prevData) => [...prevData, result.Probability]);
  };

  return (
    <div className="App">
      <h1>Cybersecurity Threat Detector Dashboard</h1>
      <PredictionForm setPredictionResult={updateResults} />
      <PredictionResults result={predictionResult} />
      {chartData.length > 0 && <ChartDisplay chartData={chartData} />}
    </div>
  );
}

export default App;
