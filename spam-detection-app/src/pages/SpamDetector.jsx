import { useState, useRef, useEffect } from 'react';
import ConfidenceChart from '../components/ConfidenceChart';
import ModelComparisonChart from '../components/ModelComparisonChart';
import KMeansClusterChart from '../components/KMeansClusterChart';
import '../styles/confidence-chart.css';
import '../styles/model-comparison-chart.css';
import '../styles/kmeans-cluster-chart.css';

function SpamDetector({ labMode = false }) {
  const [text, setText] = useState('');
  const [selectedModels, setSelectedModels] = useState(['xgboost']); // Changed to array
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [predictionData, setPredictionData] = useState(null); // Shared prediction data
  const [savedAnalyses, setSavedAnalyses] = useState([]);
  const [savedAnalysesLoading, setSavedAnalysesLoading] = useState(false);
  const [savedAnalysesError, setSavedAnalysesError] = useState(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const analysisSectionRef = useRef(null);
  const [modelTooltip, setModelTooltip] = useState({ visible: false, text: '', x: 0, y: 0 });
  const [wordLimitError, setWordLimitError] = useState(false);

  const models = [
    { 
      id: 'lstm', 
      name: 'LSTM (Recommended)', 
      description: 'Deep learning neural network',
      tooltip: 'LSTM (sequence model): slower than XGBoost but captures word order/context; useful for nuanced language. (Recommned model)'
    },
    { 
      id: 'xgboost', 
      name: 'XGBoost', 
      description: 'Fast gradient boosting',
      tooltip: 'XGBoost (gradient-boosted trees): very fast inference and strong accuracy; great baseline for spam detection.'
    },
    { 
      id: 'kmeans', 
      name: 'K-Means (Not Recommended)', 
      description: 'Unsupervised clustering approach',
      tooltip: 'K-Means (clustering): not a classifier; visualizes how close your message is to spam/ham clusters.'
    }
  ];

  // Generate shared prediction data for all charts
  const generateModelPredictions = () => {
    const predictions = {};
    
    selectedModels.forEach(modelId => {
      let spamProb;
      
      // Generate model-specific random values with different characteristics
      switch(modelId) {
        case 'xgboost':
          // XGBoost tends to be more confident in predictions
          spamProb = Math.random() > 0.5 ? Math.random() * 0.3 + 0.65 : Math.random() * 0.3 + 0.15;
          break;
        case 'lstm':
          // LSTM might have more moderate predictions
          spamProb = Math.random() * 0.6 + 0.2;
          break;
        case 'kmeans':
          // K-Means clustering might be less certain
          spamProb = Math.random() * 0.5 + 0.25;
          break;
        default:
          spamProb = Math.random() * 0.7 + 0.15;
      }
      
      predictions[modelId] = {
        spamProbability: spamProb,
        // K-Means specific: user message point (delegated random here)
        ...(modelId === 'kmeans' ? {
          userPointX: Math.random() * 100,
          userPointY: Math.random() * 100,
        } : {}),
        // Model performance metrics for Chart 2
        accuracy: modelId === 'xgboost' ? 94 + Math.random() * 4 - 2 : // 92-96%
                  modelId === 'lstm' ? 89 + Math.random() * 4 - 2 :     // 87-91%
                  78 + Math.random() * 4 - 2,                          // 76-80%
        spamDetectionRate: modelId === 'xgboost' ? 92 + Math.random() * 4 - 2 :
                          modelId === 'lstm' ? 87 + Math.random() * 4 - 2 :
                          75 + Math.random() * 4 - 2,
        falseAlarmRate: modelId === 'xgboost' ? 3 + Math.random() * 2 - 1 :    // 2-4%
                       modelId === 'lstm' ? 5 + Math.random() * 2 - 1 :        // 4-6%
                       12 + Math.random() * 4 - 2,                             // 10-14%
        confidenceLevel: modelId === 'xgboost' ? 89 + Math.random() * 6 - 3 :
                        modelId === 'lstm' ? 85 + Math.random() * 6 - 3 :
                        70 + Math.random() * 6 - 3
      };
    });
    
    return predictions;
  };

  const countWords = (text) => {
    if (!text.trim()) return 0;
    return text.trim().split(/\s+/).length;
  };

  // Auto-scroll to analysis section when result is available
  useEffect(() => {
    if (result && analysisSectionRef.current) {
      // Small delay to ensure the DOM has updated
      setTimeout(() => {
        analysisSectionRef.current.scrollIntoView({ 
          behavior: 'smooth',
          block: 'start'
        });
      }, 100);
    }
  }, [result]);

  // Handle scroll to show/hide scroll to top button
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      setShowScrollTop(scrollTop > 300); // Show button after scrolling 300px
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  const fetchSavedAnalyses = async () => {
    setSavedAnalysesLoading(true);
    setSavedAnalysesError(null);
    try {
      const resp = await fetch('http://localhost:8000/analysis/list');
      if (!resp.ok) {
        throw new Error(`Failed to load saved analyses: ${resp.status}`);
      }
      const data = await resp.json();
      setSavedAnalyses(data || []);
    } catch (err) {
      console.error('Error loading saved analyses:', err);
      setSavedAnalysesError(err.message || 'Failed to load saved analyses.');
    } finally {
      setSavedAnalysesLoading(false);
    }
  };

  useEffect(() => {
    // Load history on first mount
    fetchSavedAnalyses();
  }, []);

  // Note: Removed useEffect that was regenerating random predictions
  // The backend provides real, consistent predictions now

  const showModelTooltip = (e, text) => {
    setModelTooltip({ visible: true, text, x: e.clientX - 12, y: e.clientY });
  };

  const moveModelTooltip = (e) => {
    setModelTooltip(prev => ({ ...prev, x: e.clientX - 12, y: e.clientY }));
  };

  const hideModelTooltip = () => {
    setModelTooltip(prev => ({ ...prev, visible: false }));
  };

  const handleModelToggle = (modelId) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        // Remove if already selected (but keep at least one)
        return prev.length > 1 ? prev.filter(id => id !== modelId) : prev;
      } else {
        // Add if not selected
        return [...prev, modelId];
      }
    });
  };

  const handleTextChange = (e) => {
    const value = e.target.value;
    const wordCount = countWords(value);
    
    if (wordCount > 1000) {
      setWordLimitError(true);
    } else {
      setWordLimitError(false);
    }
    
    setText(value);
  };

  const handlePasteText = async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();
      setText(clipboardText);
    } catch (err) {
      console.error('Failed to read clipboard:', err);
      alert('Unable to read clipboard. Please paste manually or grant clipboard permissions.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Check word limit before proceeding
    const wordCount = countWords(text);
    if (wordCount > 1000) {
      setError('Error: Message exceeds 1000 word limit. Please shorten your message.');
      setWordLimitError(true);
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    setWordLimitError(false);

    try {
      // Call backend API
      console.log('Calling backend API with:', { text: text.substring(0, 50), models: selectedModels });

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          models: selectedModels
        })
      });

      console.log('Backend response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Backend error:', errorText);
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Backend returned data:', data);

      // Transform backend response to match frontend expectations
      const predictions = {};
      data.predictions.forEach(pred => {
        // Debug logging for K-Means
        if (pred.model_id === 'kmeans') {
          console.log('K-Means prediction data:', {
            user_point_2d: pred.user_point_2d,
            cluster_id: pred.cluster_id,
            has_point_2d: !!pred.user_point_2d
          });
        }
        
        predictions[pred.model_id] = {
          spamProbability: pred.spam_probability,
          prediction: pred.prediction,
          confidence: pred.confidence,
          processingTime: pred.processing_time_ms,
          // K-Means specific data
          ...(pred.user_point_2d ? {
            userPointX: pred.user_point_2d[0],
            userPointY: pred.user_point_2d[1],
            clusterId: pred.cluster_id,
            clusterDistances: pred.cluster_distances
          } : {}),
          // Model performance metrics (from metadata)
          accuracy: pred.model_id === 'xgboost' ? 94 : pred.model_id === 'lstm' ? 90 : 78,
          spamDetectionRate: pred.model_id === 'xgboost' ? 92 : pred.model_id === 'lstm' ? 87 : 75,
          falseAlarmRate: pred.model_id === 'xgboost' ? 3 : pred.model_id === 'lstm' ? 5 : 12,
          confidenceLevel: pred.confidence * 100
        };
      });

      setPredictionData(predictions);

      // Set result to show the analysis panel
      setResult({
        isEmpty: true, // Keep using visualization mode
        models: selectedModels,
        text: text,
        textStats: data.text_stats,
        totalProcessingTime: data.total_processing_time_ms
      });

    } catch (err) {
      console.error('Prediction error:', err);
      console.error('Error details:', {
        message: err.message,
        stack: err.stack,
        type: err.constructor.name
      });

      setError(err.message || 'Failed to connect to backend. Make sure the backend server is running on port 8000.');

      // Fall back to mock data on error
      console.warn('Falling back to mock data due to error');
      const predictions = generateModelPredictions();
      setPredictionData(predictions);
      setResult({
        isEmpty: true,
        models: selectedModels,
        text: text
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAnalysis = async () => {
    if (!result || !predictionData) return;

    try {
      const response = await fetch('http://localhost:8000/analysis/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_text: text,
          selected_models: selectedModels,
          prediction_summary: {
            predictions: predictionData,
            text_stats: result.textStats || null,
            total_processing_time_ms: result.totalProcessingTime || null
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Save analysis error:', errorText);
        throw new Error(`Failed to save analysis: ${response.status}`);
      }

      const saved = await response.json();
      // Optimistically refresh list and select the newly saved analysis
      await fetchSavedAnalyses();
      setSelectedAnalysis(saved);
    } catch (err) {
      console.error('Error saving analysis:', err);
      alert(err.message || 'Failed to save analysis.');
    }
  };

  // Enforce consistent model order for all charts
  const modelOrder = { lstm: 0, xgboost: 1, kmeans: 2 };
  const orderedModels = [...selectedModels].sort((a, b) => (modelOrder[a] ?? 99) - (modelOrder[b] ?? 99));

  // Only allow comparison chart for xgboost and lstm (preserve order)
  const selectedModelsForChart = orderedModels.filter((id) => id === 'xgboost' || id === 'lstm');
  // Only allow confidence chart for xgboost and lstm
  const selectedModelsForConfidence = orderedModels.filter((id) => id === 'xgboost' || id === 'lstm');
  // Only show K-Means chart when K-Means is selected
  const showKMeansChart = selectedModels.includes('kmeans');

  return (
    <div className="spam-detector-container">
      <h1 className="detector-title">Spam Detector</h1>
      {(
        <div
          className="model-tooltip"
          style={{
            left: `${Math.round(modelTooltip.x)}px`,
            top: `${Math.round(modelTooltip.y)}px`,
            opacity: modelTooltip.visible ? 1 : 0
          }}
        >
          <div style={{ fontWeight: 700, fontSize: '14px', marginBottom: '2px' }}>Model Info</div>
          <div>{modelTooltip.text}</div>
        </div>
      )}
      
      {/* Top Section: Input and Model Selection */}
      <div className="detector-top-layout">
        {/* Left Panel: Text Input */}
        <div className="input-section">
          <h3 className="section-title">Message Input</h3>
          <div className="email-preview">
            <textarea
              className="email-body"
              placeholder="Paste your message here to check if it's spam..."
              value={text}
              onChange={handleTextChange}
              style={wordLimitError ? { borderColor: '#e74c3c' } : {}}
            />
            <div 
              className="word-count" 
              style={wordLimitError ? { color: '#e74c3c', fontWeight: 'bold' } : {}}
            >
              {countWords(text)}/1000 words {wordLimitError && '(Limit exceeded!)'}
            </div>
          </div>
          
          <div className="action-buttons">
            <button 
              className="btn btn-secondary"
              onClick={handlePasteText}
            >
              PASTE TEXT
            </button>
          </div>
        </div>

        {/* Middle Panel: Model Selection */}
        <div className="model-selection-section">
          <h3 className="section-title">Choose AI Model(s) To Detect Spam</h3>
          <div className="model-options">
            {models.map((model) => (
              <div 
                key={model.id}
                className={`model-card ${selectedModels.includes(model.id) ? 'selected' : ''}`}
                onMouseEnter={(e) => showModelTooltip(e, model.tooltip)}
                onMouseMove={moveModelTooltip}
                onMouseLeave={hideModelTooltip}
                onClick={() => handleModelToggle(model.id)}
              >
                <div className="model-name">{model.name}</div>
                <div className="model-description">{model.description}</div>
                <div className="model-checkbox">
                  <input
                    type="checkbox"
                    name="models"
                    value={model.id}
                    checked={selectedModels.includes(model.id)}
                    onChange={() => handleModelToggle(model.id)}
                  />
                </div>
              </div>
            ))}
          </div>
          
          <div className="selected-models-info">
            <p>{selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''} selected</p>
          </div>
          
          <button 
            className="btn btn-primary predict-btn" 
            onClick={handleSubmit}
            disabled={!text.trim() || loading || selectedModels.length === 0 || wordLimitError}
          >
            {loading ? 'ANALYZING...' : 'PREDICT'}
          </button>
          
          {wordLimitError && (
            <div style={{ 
              color: '#e74c3c', 
              marginTop: '10px', 
              fontSize: '14px',
              textAlign: 'center',
              fontWeight: 'bold'
            }}>
              Please reduce text to 1000 words or less
            </div>
          )}
        </div>
      </div>

      {/* Bottom Section: Results (only shown after prediction) */}
      {(result || error || loading) && (
        <div className="analysis-section" ref={analysisSectionRef}>
          <h3 className="section-title">Analysis Results</h3>
          <div className="analysis-panel">
            {error && (
              <div className="error" style={{ color: '#e74c3c', padding: '20px', textAlign: 'center' }}>
                <p>{error}</p>
              </div>
            )}

            {loading && (
              <div className="loading">
                <p>Analyzing...</p>
              </div>
            )}

            {!loading && result && (
              <div className="results-grid">
                {result.isEmpty ? (
                  // Show confidence charts for selected models
                  <div className="multi-chart-analysis-panel">
                    <div className="message-preview-card">
                      <div className="message-preview-header">
                        <h4>Message Preview</h4>
                        <span className={`mode-pill ${labMode ? 'mode-pill--lab' : 'mode-pill--secure'}`}>
                          {labMode ? 'Lab Mode: HTML renders (unsafe)' : 'Secure Mode: HTML escaped'}
                        </span>
                      </div>
                      <div className="message-preview-body">
                        {labMode ? (
                          <div
                            className="message-preview-content"
                            dangerouslySetInnerHTML={{ __html: result.text || '' }}
                          />
                        ) : (
                          <div className="message-preview-content message-preview-content--safe">
                            {result.text || ''}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Charts section heading for Classification Models */}
                    {selectedModelsForConfidence.length > 0 && (
                      <>
                        <div className="charts-section-header charts-section-header--tight">
                          <h3 className="charts-section-title">Spam Confidence Meter</h3>
                        </div>
                        
                        {/* Charts container */}
                        <div className={`charts-container charts-count-${selectedModelsForConfidence.length}`}>
                          {selectedModelsForConfidence.map((modelId) => {
                            const model = models.find(m => m.id === modelId);
                            return (
                              <div key={modelId} className="model-chart-wrapper">
                                <ConfidenceChart 
                                  modelId={modelId}
                                  modelName={model.name}
                                  modelDescription={model.description}
                                  predictionData={predictionData ? predictionData[modelId] : null}
                                />
                              </div>
                            );
                          })}
                        </div>
                      </>
                    )}
                    
                    {/* Second Chart: Model Performance Comparison (only for xgboost/lstm) */}
                    {selectedModelsForChart.length > 0 && (
                      <>
                        <div className="charts-section-header charts-section-header--tight">
                          <h3 className="charts-section-title">Model Accuracy Comparison</h3>
                        </div>
                        <ModelComparisonChart 
                          selectedModels={selectedModelsForChart}
                          predictionData={predictionData}
                        />
                      </>
                    )}

                    {/* Third Chart: K-Means Cluster Analysis (only when K-Means is selected) */}
                    {showKMeansChart && (
                      <>
                        <div className="charts-section-header charts-section-header--tight">
                          <h3 className="charts-section-title">Cluster Analysis</h3>
                        </div>
                        <KMeansClusterChart 
                          modelId="kmeans"
                          modelName="K-Means"
                          modelDescription="Unsupervised clustering approach"
                          predictionData={predictionData}
                        />
                      </>
                    )}

                    {/* Saved Analyses (History) - secure-only, plain text rendering */}
                    <div className={`saved-analyses-section ${labMode ? 'saved-analyses-section--lab' : 'saved-analyses-section--secure'}`}>
                      <div className="saved-analyses-header">
                        <h3 className="charts-section-title">Saved Analyses</h3>
                        <button
                          className={`btn ${labMode ? 'btn-lab' : 'btn-secure'}`}
                          onClick={handleSaveAnalysis}
                          disabled={!text.trim() || !predictionData}
                        >
                          Save Analysis
                        </button>
                      </div>
                      <div className="saved-analyses-layout">
                        <div className="saved-analyses-list">
                          {savedAnalysesLoading && <p>Loading history...</p>}
                          {savedAnalysesError && (
                            <p style={{ color: '#e74c3c' }}>{savedAnalysesError}</p>
                          )}
                          {!savedAnalysesLoading && !savedAnalysesError && savedAnalyses.length === 0 && (
                            <p style={{ fontSize: '14px', color: '#666' }}>No saved analyses yet.</p>
                          )}
                          <div className="saved-analyses-items">
                            {savedAnalyses.map((item) => (
                              <button
                                key={item.id}
                                type="button"
                                className={`saved-analysis-item${
                                  selectedAnalysis && selectedAnalysis.id === item.id ? ' saved-analysis-item--active' : ''
                                }`}
                                onClick={async () => {
                                  try {
                                    const resp = await fetch(`http://localhost:8000/analysis/${item.id}`);
                                    if (!resp.ok) {
                                      throw new Error(`Failed to load analysis ${item.id}`);
                                    }
                                    const detail = await resp.json();
                                    setSelectedAnalysis(detail);
                                  } catch (err) {
                                    console.error('Error loading analysis detail:', err);
                                    alert(err.message || 'Failed to load analysis details.');
                                  }
                                }}
                              >
                                <div className="saved-analysis-meta">
                                  <span className="saved-analysis-timestamp">
                                    {item.created_at}
                                  </span>
                                </div>
                                <div className="saved-analysis-snippet">
                                  {item.snippet}
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="saved-analyses-detail">
                          {selectedAnalysis ? (
                            <div className="saved-analysis-detail-card">
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                                <h4>Saved Analysis Detail</h4>
                                <span className={`mode-pill ${labMode ? 'mode-pill--lab' : 'mode-pill--secure'}`}>
                                  {labMode ? 'Lab Mode: HTML renders (unsafe)' : 'Secure Mode: HTML escaped'}
                                </span>
                              </div>
                              <div className="saved-analysis-detail-row">
                                <span className="saved-analysis-detail-label">Saved At:</span>
                                <span className="saved-analysis-detail-value">
                                  {selectedAnalysis.created_at}
                                </span>
                              </div>
                              <div className="saved-analysis-detail-row">
                                <span className="saved-analysis-detail-label">Models:</span>
                                <span className="saved-analysis-detail-value">
                                  {Array.isArray(selectedAnalysis.selected_models)
                                    ? selectedAnalysis.selected_models.join(', ')
                                    : ''}
                                </span>
                              </div>
                              <div className="saved-analysis-detail-row">
                                <span className="saved-analysis-detail-label">Message Text:</span>
                                {labMode ? (
                                  <div
                                    className="saved-analysis-message-text"
                                    dangerouslySetInnerHTML={{ __html: selectedAnalysis.message_text || '' }}
                                  />
                                ) : (
                                  <div className="saved-analysis-message-text">
                                    {selectedAnalysis.message_text}
                                  </div>
                                )}
                              </div>
                            </div>
                          ) : (
                            <div className="saved-analysis-detail-placeholder">
                              <p style={{ fontSize: '14px', color: '#666' }}>
                                Select a saved analysis to view details.
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                  </div>
                ) : (
                  // Real results (when backend is connected)
                  <>
                    {/* Main Result */}
                    <div className={`main-result ${result.prediction === 1 ? 'spam' : 'ham'}`}>
                      <p className="result-label">Prediction Result</p>
                      <h2 className="probability">
                        {(result.probability * 100).toFixed(2)}%
                      </h2>
                      <p className="result-text">
                        {result.prediction === 1 ? 'SPAM' : 'HAM'}
                      </p>
                      <p className="confidence-text">
                        Confidence: {result.prediction === 1 ? 'High' : 'High'}
                      </p>
                    </div>

                    {/* Model Information */}
                    <div className="model-info">
                      <h4>Models Used</h4>
                      {selectedModels.map(modelId => {
                        const model = models.find(m => m.id === modelId);
                        return (
                          <div key={modelId} className="model-used-item">
                            <p className="model-used">{model.name}</p>
                            <p className="model-desc">{model.description}</p>
                          </div>
                        );
                      })}
                    </div>

                    {/* Message Analysis */}
                    <div className="message-analysis">
                      <h4>Message Analysis</h4>
                      <div className="analysis-stats">
                        <div className="stat">
                          <span className="stat-label">Word Count:</span>
                          <span className="stat-value">{countWords(text)}</span>
                        </div>
                        <div className="stat">
                          <span className="stat-label">Character Count:</span>
                          <span className="stat-value">{text.length}</span>
                        </div>
                        <div className="stat">
                          <span className="stat-label">Processing Time:</span>
                          <span className="stat-value">{result.processing_time || 'N/A'}ms</span>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Scroll to Top Button */}
      {showScrollTop && (
        <button 
          className="scroll-to-top-btn"
          onClick={scrollToTop}
          title="Scroll to top"
        >
          ↑
        </button>
      )}
    </div>
  );
}

export default SpamDetector;