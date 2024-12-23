import React from 'react';

const PredictionResults = ({ result }) => {
  return (
    <div>
      {result ? (
        <>
          <h3>Prediction Result</h3>
          <p><strong>Prediction:</strong> {result.Prediction}</p>
          <p><strong>Probability:</strong> {result.Probability.toFixed(4)}</p>
        </>
      ) : (
        <p>No results yet. Submit data to get predictions.</p>
      )}
    </div>
  );
};

export default PredictionResults;
