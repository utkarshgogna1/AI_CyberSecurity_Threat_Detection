import axios from 'axios';

const API_URL = "http://127.0.0.1:8000/predict/";

export const predictTraffic = async (data) => {
  try {
    const response = await axios.post(API_URL, data);
    return response.data; // Return Prediction and Probability
  } catch (error) {
    console.error("Error fetching prediction:", error);
    throw error;
  }
};
