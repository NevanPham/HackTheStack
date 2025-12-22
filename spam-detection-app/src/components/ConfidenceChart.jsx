import { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale
} from 'chart.js';
import { Doughnut } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, Title, CategoryScale, LinearScale);

function ConfidenceChart({ modelId = 'xgboost', predictionData = null, modelName = 'Model', modelDescription = '' }) {
  const [prediction, setPrediction] = useState({ spamProbability: 0.5 });

  // Use prediction data from parent component, or fallback to default
  useEffect(() => {
    if (predictionData) {
      setPrediction({
        spamProbability: predictionData.spamProbability
      });
    } else {
      // Fallback to default if no data provided
      setPrediction({
        spamProbability: 0.5
      });
    }
  }, [predictionData]);

  // Determine confidence level with clear color gradient
  const getClassification = (spamProb) => {
    const spamPercent = spamProb * 100;
    
    if (spamProb > 0.8) return { 
      type: 'spam', 
      confidence: 'Very High Risk', 
      label: 'Total Spam Bomb',
      color: '#b91c1c' // Very red (red-700)
    };
    if (spamProb > 0.65) return { 
      type: 'spam', 
      confidence: 'High Risk', 
      label: 'Looks Spammy',
      color: '#dc2626' // Red (red-600)
    };
    if (spamProb > 0.5) return { 
      type: 'spam', 
      confidence: 'Medium Risk', 
      label: 'Hmm... Suspicious',
      color: '#ea580c' // Orange (orange-600)
    };
    if (spamProb > 0.35) return { 
      type: 'mixed', 
      confidence: 'Low Risk', 
      label: 'Kinda Legit',
      color: '#ca8a04' // Yellow (yellow-600)
    };
    return { 
      type: 'ham', 
      confidence: 'Safe', 
      label: 'Pure Ham Energy',
      color: '#16a34a' // Green (green-600)
    };
  };

  const classification = getClassification(prediction.spamProbability);
  const spamPercent = Math.round(prediction.spamProbability * 100);
  const hamPercent = 100 - spamPercent;

  // Chart.js configuration with confidence-based color gradient
  const chartData = {
    labels: ['Confidence Level', 'Remaining'],
    datasets: [
      {
        data: [spamPercent, hamPercent],
        backgroundColor: [
          classification.color, // Dynamic color based on confidence level
          '#e2e8f0'  // Light grey for unused portion
        ],
        borderColor: [
          classification.color,
          '#cbd5e1' // Slightly darker grey border
        ],
        borderWidth: 4,
        borderRadius: 12,
        hoverOffset: 12,
        hoverBackgroundColor: [
          classification.color,
          '#e2e8f0'
        ],
        hoverBorderWidth: 6,
        // Keep inner hole size fixed while reducing outer radius for a thinner ring
        radius: '90%',
        cutout: 65,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 2000,
      easing: 'easeOutBounce', // More playful animation
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
        external: (context) => {
          const { chart, tooltip } = context;
          let tooltipEl = chart.canvas.parentNode.querySelector('.confidence-tooltip');
          if (!tooltipEl) {
            tooltipEl = document.createElement('div');
            tooltipEl.className = 'confidence-tooltip';
            tooltipEl.style.position = 'fixed';
            tooltipEl.style.pointerEvents = 'none';
            tooltipEl.style.background = 'rgba(0, 0, 0, 0.9)';
            tooltipEl.style.color = 'white';
            tooltipEl.style.border = `3px solid ${classification.color}`;
            tooltipEl.style.borderRadius = '12px';
            tooltipEl.style.padding = '8px 10px';
            tooltipEl.style.font = '500 13px Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial';
            tooltipEl.style.zIndex = '9999';
            tooltipEl.style.whiteSpace = 'nowrap';
            tooltipEl.style.transition = 'opacity 0.08s ease';
            chart.canvas.parentNode.appendChild(tooltipEl);
          }

          if (tooltip.opacity === 0) {
            tooltipEl.style.opacity = '0';
            return;
          }

          const dataIndex = tooltip.dataPoints?.[0]?.dataIndex;
          const lines = [];
          lines.push('Spam Detection Confidence');
          if (dataIndex === 0) {
            lines.push(`Confidence Level: ${spamPercent}%`);
          } else {
            lines.push(`Remaining: ${hamPercent}%`);
          }
          lines.push(`Risk Level: ${classification.confidence}`);

          tooltipEl.innerHTML = `<div style="display:flex;flex-direction:column;gap:4px">${lines
            .map((l, i) => `<div style=\"${i===0?'font-weight:700;font-size:14px;':''}\">${l}</div>`) 
            .join('')}</div>`;

          const rect = chart.canvas.getBoundingClientRect();
          const centerY = rect.top + rect.height / 2;
          tooltipEl.style.opacity = '1';
          const tooltipWidth = tooltipEl.offsetWidth;
          const offset = 14;
          let left;
          if (dataIndex === 0) {
            left = rect.right + offset;
          } else {
            left = rect.left - tooltipWidth - offset;
          }
          tooltipEl.style.left = `${Math.round(left)}px`;
          tooltipEl.style.top = `${Math.round(centerY - tooltipEl.offsetHeight / 2)}px`;
        }
      },
    },
    onHover: (event, elements) => {
      event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
    },
  };

  return (
    <div className="confidence-chart-container">
      <div className="chart-wrapper">
        {/* Moved header into ConfidenceChart */}
        {(modelName || modelDescription) && (
          <div className="model-chart-header">
            {modelName && <h4 className="model-chart-title">{modelName}</h4>}
            {modelDescription && <p className="model-chart-description">{modelDescription}</p>}
          </div>
        )}
        {/* Interactive Chart.js Doughnut Chart */}
        <div className="chart-container">
          <div className="chart-canvas">
            <Doughnut 
              data={chartData} 
              options={chartOptions}
            />
          </div>
          {/* Enhanced animated pulse ring with dynamic glow */}
          <div 
            className="chart-pulse-ring"
            style={{ 
              borderColor: classification.color,
              boxShadow: `
                0 0 0 6px ${classification.color}15,
                0 0 20px ${classification.color}40,
                0 0 40px ${classification.color}20
              `
            }}
          />
          
          {/* Additional outer glow ring for high-risk spam */}
          {spamPercent > 70 && (
            <div 
              className="chart-danger-glow"
              style={{ 
                borderColor: classification.color,
                boxShadow: `
                  0 0 0 3px ${classification.color}30,
                  0 0 30px ${classification.color}60,
                  0 0 60px ${classification.color}40
                `
              }}
            />
          )}
          
          {/* Center content overlay */}
          <div className="chart-center-overlay">
            <div className="confidence-display">
              <div 
                className="confidence-number"
                style={{ color: classification.color }}
              >
                {spamPercent}%
              </div>
              <div className="confidence-level">Risk Level</div>
            </div>
          </div>
        </div>
        
        {/* Interactive Prediction Result */}
        <div className="prediction-result-interactive">
          <div 
            className="result-text-interactive"
            style={{ 
              color: classification.color,
              backgroundColor: `${classification.color}10`,
              borderColor: classification.color
            }}
          >
            {classification.label}
          </div>
          
          {/* Additional Info Panel */}
          <div className="confidence-details">
            <div className="detail-item">
              <span className="detail-label">Confidence Level:</span>
              <span className="detail-value" style={{ color: classification.color }}>
                {spamPercent}%
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Risk Assessment:</span>
              <span className="detail-value" style={{ color: classification.color }}>
                {classification.confidence}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Classification:</span>
              <span className="detail-value" style={{ color: classification.color }}>
                {classification.label}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfidenceChart;