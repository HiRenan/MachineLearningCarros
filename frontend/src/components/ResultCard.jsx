import React from 'react';

const ResultCard = ({ prediction }) => {
  if (!prediction) return null;

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  const getConfiancaColor = (confianca) => {
    switch (confianca) {
      case 'alta':
        return 'confianca-alta';
      case 'média':
        return 'confianca-media';
      default:
        return 'confianca-baixa';
    }
  };

  return (
    <div className="result-card">
      <div className="result-header">
        <h2>Resultado da Predição</h2>
        <span className={`confianca-badge ${getConfiancaColor(prediction.confianca)}`}>
          Confiança: {prediction.confianca}
        </span>
      </div>

      <div className="result-body">
        <div className="valor-principal">
          <label>Valor Estimado</label>
          <div className="valor-destaque">
            {formatCurrency(prediction.valor_predito)}
          </div>
        </div>

        <div className="valor-range">
          <div className="range-item">
            <label>Valor Mínimo</label>
            <div className="valor-secundario">
              {formatCurrency(prediction.valor_minimo)}
            </div>
          </div>

          <div className="range-separator">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </div>

          <div className="range-item">
            <label>Valor Máximo</label>
            <div className="valor-secundario">
              {formatCurrency(prediction.valor_maximo)}
            </div>
          </div>
        </div>

        <div className="result-info">
          <svg
            className="info-icon"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p>
            O intervalo de valores considera o erro médio absoluto do modelo (MAE: R$ 2.527,46).
            Este valor é uma estimativa baseada em dados históricos.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
