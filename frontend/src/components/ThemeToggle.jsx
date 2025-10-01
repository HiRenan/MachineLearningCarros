import React from 'react';
import { FiSun, FiMoon } from 'react-icons/fi';

const ThemeToggle = ({ theme, toggleTheme }) => {
  return (
    <button
      onClick={toggleTheme}
      className="theme-toggle-button"
      aria-label={`Mudar para tema ${theme === 'light' ? 'escuro' : 'claro'}`}
      title={`Mudar para tema ${theme === 'light' ? 'escuro' : 'claro'}`}
    >
      <div className="theme-toggle-track">
        <div className={`theme-toggle-thumb ${theme === 'dark' ? 'theme-toggle-thumb-dark' : ''}`}>
          {theme === 'light' ? (
            <FiSun size={14} className="theme-icon" />
          ) : (
            <FiMoon size={14} className="theme-icon" />
          )}
        </div>
      </div>
    </button>
  );
};

export default ThemeToggle;
