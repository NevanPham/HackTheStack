import { useState, useEffect } from 'react';
import './SplashScreen.css';

function SplashScreen({ onUnlock }) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [lockHovered, setLockHovered] = useState(false);

  const handleUnlock = () => {
    setIsAnimating(true);
    // Add a small delay for animation before unlocking
    setTimeout(() => {
      onUnlock();
    }, 300);
  };

  return (
    <div className="splash-screen">
      <div className="splash-content">
        <h1 className="splash-title">HackTheStack</h1>
        <div 
          className={`splash-lock ${isAnimating ? 'unlocking' : ''} ${lockHovered ? 'hovered' : ''}`}
          onClick={handleUnlock}
          onMouseEnter={() => setLockHovered(true)}
          onMouseLeave={() => setLockHovered(false)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleUnlock();
            }
          }}
          aria-label="Click to unlock and enter HackTheStack"
        >
          <svg 
            width="80" 
            height="80" 
            viewBox="0 0 24 24" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg"
            className="lock-icon"
          >
            <path 
              d="M6 10V8C6 5.79086 7.79086 4 10 4H14C16.2091 4 18 5.79086 18 8V10M6 10H4C2.89543 10 2 10.8954 2 12V19C2 20.1046 2.89543 21 4 21H20C21.1046 21 22 20.1046 22 19V12C22 10.8954 21.1046 10 20 10H18M6 10H18" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            />
          </svg>
          <p className="splash-hint">Click to unlock</p>
        </div>
      </div>
    </div>
  );
}

export default SplashScreen;

