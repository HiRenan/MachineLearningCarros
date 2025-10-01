import React from 'react';

const SkeletonLoader = () => {
  return (
    <div className="skeleton-loader">
      <div className="skeleton-shimmer"></div>
      <div className="skeleton-content">
        <div className="skeleton-icon"></div>
        <div className="skeleton-text">Carregando imagem...</div>
      </div>
    </div>
  );
};

export default SkeletonLoader;
