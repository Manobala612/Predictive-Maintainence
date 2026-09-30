import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 5000, // Fail fast if backend is offline
});

export const apiService = {
  /**
   * Diagnoses machine parameters for failure signatures
   */
  async predict(telemetryData) {
    const response = await client.post('/predict', telemetryData);
    return response.data;
  },

  /**
   * Verifies the model registry and API status
   */
  async checkHealth() {
    const response = await client.get('/health');
    return response.data;
  }
};

export default apiService;
