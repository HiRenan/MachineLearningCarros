import React, { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const data = await getModelInfo();
        setModelInfo(data);
      } catch (error) {
        console.error('Erro ao carregar informações do modelo:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return <div className="model-info loading">Carregando informações do modelo...</div>;
  }

  if (!modelInfo) {
    return null;
  }

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  return (
    <div className="model-info">
      <h3>Informações do Modelo</h3>
      <div className="model-info-grid">
        <div className="info-item">
          <label>Algoritmo</label>
          <span className="info-value">{modelInfo.nome_modelo}</span>
        </div>

        <div className="info-item highlight">
          <label>R² Score</label>
          <span className="info-value">{formatPercent(modelInfo.r2_score)}</span>
        </div>

        <div className="info-item">
          <label>MAE</label>
          <span className="info-value">{formatCurrency(modelInfo.mae)}</span>
        </div>

        <div className="info-item">
          <label>RMSE</label>
          <span className="info-value">{formatCurrency(modelInfo.rmse)}</span>
        </div>

        <div className="info-item">
          <label>Features</label>
          <span className="info-value">{modelInfo.total_features}</span>
        </div>

        <div className="info-item full-width">
          <label>Descrição</label>
          <span className="info-value">{modelInfo.descricao}</span>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
