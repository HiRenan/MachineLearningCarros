import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = async () => {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    console.error('Erro no health check:', error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/model-info');
    return response.data;
  } catch (error) {
    console.error('Erro ao obter informações do modelo:', error);
    throw error;
  }
};

export const predictPrice = async (carData) => {
  try {
    const response = await api.post('/api/predict', carData);
    return response.data;
  } catch (error) {
    console.error('Erro na predição:', error);
    throw error;
  }
};

export const getMarcas = async () => {
  try {
    const response = await api.get('/api/options/marcas');
    return response.data.marcas;
  } catch (error) {
    console.error('Erro ao obter marcas:', error);
    throw error;
  }
};

export const getModelos = async () => {
  try {
    const response = await api.get('/api/options/modelos');
    return response.data.modelos;
  } catch (error) {
    console.error('Erro ao obter modelos:', error);
    throw error;
  }
};

export const getCores = async () => {
  try {
    const response = await api.get('/api/options/cores');
    return response.data.cores;
  } catch (error) {
    console.error('Erro ao obter cores:', error);
    throw error;
  }
};

export const getCambios = async () => {
  try {
    const response = await api.get('/api/options/cambios');
    return response.data.cambios;
  } catch (error) {
    console.error('Erro ao obter câmbios:', error);
    throw error;
  }
};

export const getCombustiveis = async () => {
  try {
    const response = await api.get('/api/options/combustiveis');
    return response.data.combustiveis;
  } catch (error) {
    console.error('Erro ao obter combustíveis:', error);
    throw error;
  }
};

export default api;
