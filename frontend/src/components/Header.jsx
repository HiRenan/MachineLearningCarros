import React, { useState, useEffect } from 'react';
import { FiActivity } from 'react-icons/fi';
import ThemeToggle from './ThemeToggle';

const Header = ({ theme, toggleTheme }) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const offset = window.scrollY;
      setScrolled(offset > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`header-sticky ${scrolled ? 'header-scrolled' : ''}`}>
      <nav className="nav-container">
        <div className="nav-logo">
          <FiActivity size={28} className="logo-icon" />
          <span className="logo-text">AutoPredict</span>
        </div>

        <div className="nav-actions">
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
        </div>
      </nav>
    </header>
  );
};

export default Header;
