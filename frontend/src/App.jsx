import { useState } from 'react';
import Header from './components/Header';
import Disclaimer from './components/Disclaimer';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import ModelInfo from './components/ModelInfo';
import { useTheme } from './hooks/useTheme';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const handlePrediction = (data) => {
    setPrediction(data);
  };

  const handleLoading = (loading) => {
    setIsLoading(loading);
  };

  return (
    <div className="app">
      <Header theme={theme} toggleTheme={toggleTheme} />

      <Disclaimer />

      <main className="app-main">
        <div className="container">
          <div className="content-grid">
            <div className="form-section">
              <PredictionForm
                onPrediction={handlePrediction}
                onLoading={handleLoading}
              />
            </div>

            <div className="result-section">
              {isLoading && (
                <div className="loading-card">
                  <div className="loading-spinner"></div>
                  <p>Calculando valor estimado...</p>
                </div>
              )}

              {!isLoading && prediction && (
                <ResultCard prediction={prediction} />
              )}

              {!isLoading && !prediction && (
                <div className="empty-state">
                  <svg
                    className="empty-icon"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  <h3>Nenhuma predição ainda</h3>
                  <p>Preencha o formulário ao lado para calcular o valor estimado do veículo</p>
                </div>
              )}
            </div>
          </div>

          <div className="model-section">
            <ModelInfo />
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Desenvolvido por Renan Mocelin, Nathan Arrais, Lucas Porfilio, João Casemiro | Projeto Educacional de Machine Learning | 2025</p>
      </footer>
    </div>
  );
}

export default App;
