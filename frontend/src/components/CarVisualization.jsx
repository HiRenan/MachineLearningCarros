import React from 'react';
import { FaCar } from 'react-icons/fa';
import { useCarImage } from '../hooks/useCarImage';
import SkeletonLoader from './SkeletonLoader';

const CarVisualization = ({ marca, modelo, cor }) => {
  const imageState = useCarImage(marca, modelo, cor);

  const colorMap = {
    'Azul': '#3498db',
    'Branco': '#ecf0f1',
    'Cinza': '#95a5a6',
    'Prata': '#bdc3c7',
    'Preto': '#2c3e50',
    'Vermelho': '#e74c3c'
  };

  const getColorHex = (colorName) => {
    return colorMap[colorName] || '#95a5a6';
  };

  if (imageState.type === 'loading') {
    return <SkeletonLoader />;
  }

  if (imageState.type === 'image' && imageState.data) {
    return (
      <div className="car-visualization">
        <div className="car-image-container">
          <img
            src={imageState.data}
            alt={`${marca} ${modelo}`}
            className="car-image"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextElementSibling.style.display = 'flex';
            }}
          />
          <div className="car-placeholder-fallback" style={{ display: 'none' }}>
            <div
              className="car-color-circle"
              style={{ backgroundColor: getColorHex(cor) }}
            >
              <FaCar size={60} color="#fff" />
            </div>
            <div className="car-info-overlay">
              <h3>{marca} {modelo}</h3>
              <span className="color-badge" style={{ backgroundColor: getColorHex(cor) }}>
                {cor || 'Cor não selecionada'}
              </span>
            </div>
          </div>

          <div className="car-info-overlay">
            <h3>{marca} {modelo}</h3>
            {cor && (
              <span className="color-badge" style={{ backgroundColor: getColorHex(cor) }}>
                {cor}
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="car-visualization">
      <div className="car-placeholder" style={{ borderColor: getColorHex(cor) }}>
        <div
          className="car-color-circle"
          style={{ backgroundColor: getColorHex(cor) }}
        >
          <FaCar size={60} color="#fff" />
        </div>
        <div className="car-info">
          <h3>{marca && modelo ? `${marca} ${modelo}` : 'Selecione um veículo'}</h3>
          {cor && (
            <span className="color-badge" style={{ backgroundColor: getColorHex(cor) }}>
              {cor}
            </span>
          )}
          {!marca && !modelo && (
            <p className="hint-text">Escolha a marca e o modelo para visualizar</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default CarVisualization;
