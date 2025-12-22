import { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend);

function KMeansClusterChart({ modelId = 'kmeans', predictionData = null, modelName = 'K-Means', modelDescription = '' }) {
  const chartRef = useRef(null);
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1); // 1 = full view, >1 zoomed in
  const [panX, setPanX] = useState(50); // center percentage 0-100
  const [panY, setPanY] = useState(50); // center percentage 0-100

  // Deterministic PRNG (Mulberry32) for fixed cluster points
  const mulberry32 = (seed) => {
    return function() {
      let t = (seed += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  // Generate sample cluster data for visualization
  const generateClusterData = () => {
    // Use real K-Means k=3 cluster summary (from models/kmeans/k3/tfidf_1000)
    // cluster_id: 0 (mixed), 1 (high-risk spam), 2 (safe)
    const clusterSummaries = [
      { id: 0, size: 2161, spamRate: 21.9, dominantSource: 'sms' },
      { id: 1, size: 570, spamRate: 99.8, dominantSource: 'mix_email_sms' },
      { id: 2, size: 481, spamRate: 9.4, dominantSource: 'email' }
    ];

    const idToStyle = {
      1: { label: 'High-Risk Spam', color: '#dc2626', center: { x: 92, y: 88 } },
      0: { label: 'Mixed Messages', color: '#f59e0b', center: { x: 35, y: 58 } },
      2: { label: 'Safe Content', color: '#16a34a', center: { x: 12, y: 86 } }
    };

    const clusters = clusterSummaries.map(cs => ({
      id: cs.id,
      label: idToStyle[cs.id].label,
      spamRate: cs.spamRate,
      size: cs.size,
      dominantSource: cs.dominantSource,
      color: idToStyle[cs.id].color,
      center: idToStyle[cs.id].center
    }));

    // Generate scatter points for each cluster using seeded RNG so they are fixed
    const scatterData = clusters.map(cluster => {
      // Use a stable seed per cluster (change the base number to re-layout deterministically)
      const rng = mulberry32(12345 + cluster.id * 1000);
      const points = [];
      // Increased number of points to make clusters look larger and more realistic
      const numPoints = Math.min(cluster.size / 15, 150); // More points per cluster
      
      for (let i = 0; i < numPoints; i++) {
        // Generate points around cluster center with some spread (in percentage space)
        const spreadX = cluster.id === 0 ? 22 : 12; // mixed cluster is wider, increased spread
        const spreadY = 10; // increased vertical spread
        const x = Math.max(0, Math.min(100, cluster.center.x + (rng() - 0.5) * spreadX));
        const y = Math.max(0, Math.min(100, cluster.center.y + (rng() - 0.5) * spreadY));
        
        points.push({
          x,
          y,
          clusterId: cluster.id,
          isSpam: rng() < (cluster.spamRate / 100)
        });
      }
      
      return {
        label: cluster.label,
        data: points,
        backgroundColor: cluster.color + '80', // Add transparency
        borderColor: cluster.color,
        borderWidth: 3, // Thicker borders
        pointRadius: 6, // Larger points
        pointHoverRadius: 6, // Same size on hover
        clusterId: cluster.id,
        clusterInfo: cluster
      };
    });

    // Add user message point from parent-provided data (no internal randomness)
    const userX = typeof predictionData?.kmeans?.userPointX === 'number' ? predictionData.kmeans.userPointX : 50;
    const userY = typeof predictionData?.kmeans?.userPointY === 'number' ? predictionData.kmeans.userPointY : 50;
    
    // Debug logging
    if (predictionData?.kmeans) {
      console.log('KMeansClusterChart - predictionData.kmeans:', {
        userPointX: predictionData.kmeans.userPointX,
        userPointY: predictionData.kmeans.userPointY,
        hasUserPointX: typeof predictionData.kmeans.userPointX === 'number',
        hasUserPointY: typeof predictionData.kmeans.userPointY === 'number'
      });
    } else {
      console.log('KMeansClusterChart - No kmeans data in predictionData:', predictionData);
    }

    const userPoint = {
      label: 'Your Message',
      data: [{
        x: userX,
        y: userY,
        isUserMessage: true
      }],
      backgroundColor: '#000000',
      borderColor: '#000000',
      borderWidth: 4,
      pointRadius: 9,
      pointHoverRadius: 12,
      showLine: false,
      pointStyle: 'star'
    };
    scatterData.push(userPoint);

    return { clusters, scatterData };
  };

  useEffect(() => {
    const data = generateClusterData();
    setClusterData(data);
  }, [predictionData]);

  if (!clusterData) return null;

  const chartData = {
    datasets: clusterData.scatterData
  };

  // Compute axis ranges based on zoom and pan
  const computeAxisRange = (centerPercent, zoom) => {
    const span = 100 / Math.max(1, zoom);
    const half = span / 2;
    let min = centerPercent - half;
    let max = centerPercent + half;
    if (min < 0) {
      max -= min; // shift right
      min = 0;
    }
    if (max > 100) {
      const overflow = max - 100;
      min = Math.max(0, min - overflow);
      max = 100;
    }
    return { min: Math.max(0, Math.min(100, min)), max: Math.max(0, Math.min(100, max)) };
  };

  const xRange = computeAxisRange(panX, zoomLevel);
  // Invert Y so dragging slider down moves view down
  const yRange = computeAxisRange(100 - panY, zoomLevel);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1200,
      easing: 'easeOutQuart'
    },
    interaction: {
      intersect: true,
      mode: 'point' // Only hover when cursor is directly over a point
    },
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          usePointStyle: true,
          font: {
            size: 11,
            weight: '600'
          },
          color: '#374151',
          padding: 15,
          generateLabels: (chart) => {
            const datasets = chart.data.datasets;
            // Sorted order for clusters: ham (2), mixed (0), spam (1); user point last
            const clusterOrder = [2, 0, 1];
            const sortedDatasets = [...datasets].sort((a, b) => {
              const aCluster = a.clusterId;
              const bCluster = b.clusterId;
              // If user message (no clusterId), always last
              if (aCluster === undefined) return 1;
              if (bCluster === undefined) return -1;
              return (
                clusterOrder.indexOf(aCluster) - clusterOrder.indexOf(bCluster)
              );
            });
            return sortedDatasets.map((dataset, i) => ({
              text: dataset.label,
              fillStyle: dataset.backgroundColor,
              strokeStyle: dataset.borderColor,
              lineWidth: dataset.borderWidth,
              pointStyle: dataset.pointStyle || 'circle',
              hidden: false,
              datasetIndex: datasets.indexOf(dataset) // use original index for correct toggling
            }));
          }
        }
      },
      tooltip: {
        enabled: false,
        external: (context) => {
          const { chart, tooltip } = context;
          let tooltipEl = chart.canvas.parentNode.querySelector('.kmeans-tooltip');
          
          if (!tooltipEl) {
            tooltipEl = document.createElement('div');
            tooltipEl.className = 'kmeans-tooltip';
            tooltipEl.style.position = 'fixed';
            tooltipEl.style.pointerEvents = 'none';
            tooltipEl.style.background = 'rgba(0, 0, 0, 0.9)';
            tooltipEl.style.color = 'white';
            tooltipEl.style.border = '3px solid #7c3aed';
            tooltipEl.style.borderRadius = '12px';
            tooltipEl.style.padding = '8px 12px';
            tooltipEl.style.font = '500 13px Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial';
            tooltipEl.style.zIndex = '9999';
            tooltipEl.style.whiteSpace = 'nowrap';
            tooltipEl.style.transition = 'opacity 0.2s ease';
            chart.canvas.parentNode.appendChild(tooltipEl);
          }

          if (tooltip.opacity === 0) {
            tooltipEl.style.opacity = '0';
            return;
          }

          const dataPoint = tooltip.dataPoints?.[0];
          if (dataPoint) {
            const dataset = chart.data.datasets[dataPoint.datasetIndex];
            const point = dataset.data[dataPoint.dataIndex];
            
            let content = '';
            if (point.isUserMessage) {
              content = `
                <div style="display:flex;flex-direction:column;gap:4px">
                  <div style="font-weight:700;font-size:14px;">Your Message</div>
                  <div>Spam Likelihood: ${point.x.toFixed(1)}%</div>
                  <div>Confidence: ${point.y.toFixed(1)}%</div>
                </div>
              `;
            } else {
              const cluster = dataset.clusterInfo;
              content = `
                <div style="display:flex;flex-direction:column;gap:4px">
                  <div style="font-weight:700;font-size:14px;">${cluster?.label || 'Cluster'}</div>
                  <div>Cluster Spam Rate: ${cluster?.spamRate?.toFixed(1)}%</div>
                  <div>Cluster Size: ${cluster?.size} messages</div>
                  <div>Center Position: (${cluster?.center?.x?.toFixed(1)}, ${cluster?.center?.y?.toFixed(1)})</div>
                </div>
              `;
            }
            
            tooltipEl.innerHTML = content;
            tooltipEl.style.borderColor = dataset.borderColor;
          }

          const rect = chart.canvas.getBoundingClientRect();
          tooltipEl.style.left = `${rect.left + tooltip.caretX + 12}px`;
          tooltipEl.style.top = `${rect.top + tooltip.caretY - tooltipEl.offsetHeight / 2}px`;
          tooltipEl.style.opacity = '1';
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        min: xRange.min,
        max: xRange.max,
        title: {
          display: true,
          text: 'Spam Likelihood (%)',
          font: {
            size: 13,
            weight: '600'
          },
          color: '#374151'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
          lineWidth: 1
        },
        ticks: {
          font: {
            size: 11
          },
          color: '#6b7280',
          callback: function(value) {
            return parseFloat(value).toFixed(1) + '%';
          }
        }
      },
      y: {
        type: 'linear',
        min: yRange.min,
        max: yRange.max,
        title: {
          display: true,
          text: 'Confidence Level (%)',
          font: {
            size: 13,
            weight: '600'
          },
          color: '#374151'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
          lineWidth: 1
        },
        ticks: {
          font: {
            size: 11
          },
          color: '#6b7280',
          callback: function(value) {
            return parseFloat(value).toFixed(1) + '%';
          }
        }
      }
    },
    onHover: (event, elements) => {
      event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
      if (elements && elements.length > 0) {
        const el = elements[0];
        const dataset = chartData.datasets[el.datasetIndex];
        if (dataset.clusterInfo) {
          setHoveredCluster(dataset.clusterInfo);
        }
      } else {
        setHoveredCluster(null);
      }
    }
  };

  return (
    <div className="kmeans-cluster-chart-container">
      <div className="chart-wrapper">
        {/* Chart Header */}
        {(modelName || modelDescription) && (
          <div className="model-chart-header">
            {modelName && <h4 className="model-chart-title">{modelName} Cluster Analysis (Not Recommended)</h4>}
            {modelDescription && <p className="model-chart-description">{modelDescription}</p>}
          </div>
        )}

        {/* Controls header: Zoom buttons above Pan X slider */}
        <div className="chart-controls-header">
          <div className="controls-mid">
            <div className="zoom-controls inline">
              <div className="zoom-buttons">
                <div className="zoom-pair">
                  <button
                    type="button"
                    className="zoom-btn"
                    onClick={() => {
                      setZoomLevel((z) => Math.min(4, parseFloat((z + 0.25).toFixed(1))));
                    }}
                  >
                    +
                  </button>
                  <button
                    type="button"
                    className="zoom-btn"
                    onClick={() => {
                      setZoomLevel((z) => Math.max(1, parseFloat((z - 0.25).toFixed(1))));
                    }}
                  >
                    -
                  </button>
                </div>
                <button
                  type="button"
                  className="reset-btn"
                  onClick={() => {
                    setZoomLevel(1);
                    setPanX(50);
                    setPanY(50);
                  }}
                  disabled={zoomLevel === 1 && panX === 50 && panY === 50}
                >
                  Reset
                </button>
              </div>
            </div>
            <div className="panx-group">
              <label className="slider-group">
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={panX}
                  onChange={(e) => setPanX(parseFloat(e.target.value))}
                  disabled={zoomLevel <= 1}
                />
              </label>
            </div>
          </div>
        </div>

        {/* Main Scatter Plot */}
        <div className="kmeans-chart-container">
          <div
            className="chart-canvas"
            onMouseLeave={() => {
              const chart = chartRef.current;
              if (chart) {
                chart.setActiveElements([]);
                chart.update();
              }
              setHoveredCluster(null);
              const tooltipEl = chart?.canvas?.parentNode?.querySelector?.('.kmeans-tooltip');
              if (tooltipEl) tooltipEl.style.opacity = '0';
            }}
          >
            <Scatter 
              ref={chartRef}
              data={chartData} 
              options={chartOptions}
            />
            {/* Vertical Pan Y slider on right side */}
            <div className="pany-controls">
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={panY}
                onChange={(e) => setPanY(parseFloat(e.target.value))}
                onTouchStart={(e) => {
                  e.stopPropagation();
                }}
                onTouchMove={(e) => {
                  e.stopPropagation();
                  // Only prevent default if we're actually dragging the slider
                  if (e.touches.length === 1) {
                    e.preventDefault();
                  }
                }}
                onTouchEnd={(e) => {
                  e.stopPropagation();
                }}
                disabled={zoomLevel <= 1}
                className="pany-slider"
              />
            </div>
            {/* Pulse ring overlay for hovered cluster (compute exact px from chart scales) */}
            {hoveredCluster && (() => {
              const chart = chartRef.current;
              const xScale = chart?.scales?.x;
              const yScale = chart?.scales?.y;
              if (!xScale || !yScale) return null;

              // Approximate cluster envelope based on generation spreads
              const spreadX = hoveredCluster.id === 0 ? 22 : 12; // wider for mixed cluster
              const spreadY = 10;

              // Compute center in pixels
              const cx = xScale.getPixelForValue(hoveredCluster.center.x);
              const cy = yScale.getPixelForValue(hoveredCluster.center.y);

              // Compute width/height in pixels using spreads (slightly larger multiplier to nicely wrap)
              const leftPx = xScale.getPixelForValue(hoveredCluster.center.x - spreadX);
              const rightPx = xScale.getPixelForValue(hoveredCluster.center.x + spreadX);
              const topVal = hoveredCluster.center.y + spreadY;
              const bottomVal = hoveredCluster.center.y - spreadY;
              const topPx = yScale.getPixelForValue(topVal);
              const bottomPx = yScale.getPixelForValue(bottomVal);

              const width = Math.abs(rightPx - leftPx) * 0.9; // trim a bit for aesthetics
              const height = Math.abs(bottomPx - topPx) * 0.9;

              const style = {
                position: 'absolute',
                left: `${cx - width / 2}px`,
                top: `${cy - height / 2}px`,
                width: `${width}px`,
                height: `${height}px`,
                borderColor: hoveredCluster.color,
                borderRadius: '50%',
                pointerEvents: 'none',
                zIndex: 1,
              };

              return <div className="cluster-pulse-ring" style={style} />;
            })()}
          </div>
        </div>

        {/* Footer controls area (bottom-right, aligned with legend line) */}
        {/* Footer removed; pan sliders moved to header/canvas */}

        {/* Cluster Summary Cards */}
        <div className="cluster-summary-section">
          <h5 className="cluster-summary-title">Cluster Characteristics</h5>
          <div className="cluster-cards-grid">
            {(() => {
              // Sort by desired order: ham (2), mixed (0), spam (1)
              const clusterOrder = [2, 0, 1];
              const sortedClusters = [...clusterData.clusters].sort(
                (a, b) => clusterOrder.indexOf(a.id) - clusterOrder.indexOf(b.id)
              );
              return sortedClusters.map((cluster) => (
                <div 
                  key={cluster.id}
                  className="cluster-summary-card"
                  style={{ borderColor: cluster.color }}
                >
                  <div className="cluster-card-header">
                    <div 
                      className="cluster-color-indicator"
                      style={{ backgroundColor: cluster.color }}
                    ></div>
                    <h6 className="cluster-name">{cluster.label}</h6>
                  </div>
                  <div className="cluster-stats">
                    <div className="cluster-stat">
                      <span className="stat-label">Spam Rate:</span>
                      <span 
                        className="stat-value"
                        style={{ color: cluster.color }}
                      >
                        {cluster.spamRate.toFixed(1)}%
                      </span>
                    </div>
                    <div className="cluster-stat">
                      <span className="stat-label">Size:</span>
                      <span 
                        className="stat-value"
                        style={{ color: cluster.color }}
                      >
                        {cluster.size} messages
                      </span>
                    </div>
                    
                  </div>
                </div>
              ));
            })()}
          </div>
        </div>

        {/* Chart Description */}
        <div className="chart-description">
          <p>
            Each point represents messages grouped by similarity. Your message appears as a "*" and shows which cluster it's closest to.
            Distance from cluster centers indicates confidence in the classification.
          </p>
        </div>
      </div>
    </div>
  );
}

export default KMeansClusterChart;