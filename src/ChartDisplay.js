import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  LinearScale,
  CategoryScale,
  PointElement,
  Tooltip,
  Legend,
} from 'chart.js';

// Register necessary components
ChartJS.register(LineElement, LinearScale, CategoryScale, PointElement, Tooltip, Legend);

const ChartDisplay = ({ chartData }) => {
  const data = {
    labels: chartData.map((_, index) => `Request ${index + 1}`),
    datasets: [
      {
        label: 'Prediction Probability',
        data: chartData,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  return (
    <div>
      <h3>Prediction Probabilities Over Time</h3>
      <Line data={data} />
    </div>
  );
};

export default ChartDisplay;
