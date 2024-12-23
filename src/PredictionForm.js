import React, { useState } from 'react';
import { predictTraffic } from './services/Api';

const PredictionForm = ({ setPredictionResult }) => {
  const [formData, setFormData] = useState({
    Destination_Port: 8080,
    Flow_Duration: 1000000,
    Total_Fwd_Packets: 1000,
    Total_Length_of_Fwd_Packets: 3000,
    Fwd_Packet_Length_Max: 1500,
    Fwd_Packet_Length_Mean: 1200,
    Fwd_IAT_Total: 50000,
    Flow_Packets_per_s: 500,
    Packet_Length_Mean: 1200
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const result = await predictTraffic(formData);
      setPredictionResult(result);
    } catch (error) {
      alert("Failed to fetch prediction. Check the backend server.");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {Object.keys(formData).map((key) => (
        <div key={key}>
          <label>{key.replace(/_/g, " ")}: </label>
          <input
            type="number"
            name={key}
            value={formData[key]}
            onChange={handleChange}
            required
          />
        </div>
      ))}
      <button type="submit">Predict</button>
    </form>
  );
};

export default PredictionForm;
