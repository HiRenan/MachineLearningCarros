import React from 'react';

const Disclaimer = () => {
  return (
    <div className="disclaimer">
      <div className="disclaimer-content">
        <svg
          className="disclaimer-icon"
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
        <p className="disclaimer-text">
          Este conteúdo é destinado apenas para fins educacionais.
          Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
        </p>
      </div>
    </div>
  );
};

export default Disclaimer;
