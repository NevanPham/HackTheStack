import { useNavigate } from 'react-router-dom';
import spamEmailImg from '../assets/images/spam-email_16843165.png';
import DatasetOverview from '../components/DatasetOverview';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import FeatureDistributionChart from '../components/FeatureDistributionChart';
import CorrelationHeatmap from '../components/CorrelationHeatmap';

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">Hack The Stack</h1>
          <div className="cta-buttons">
            <button
              className="btn btn-primary"
              onClick={() => navigate('/spam-detector')}
            >
              START USING SPAM DETECTOR
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => navigate('/about')}
            >
              WHO WE ARE
            </button>
          </div>
        </div>
        <div className="hero-image">
          <img src={spamEmailImg} alt="Spam Detection Warning" />
        </div>
      </div>

      <div className="visualizations-section">
        <div className="section-intro">
          <h2>Understanding Spam Detection Features</h2>
          <p>Explore the data and features that power our spam detection models</p>
        </div>

        <DatasetOverview />

        <div className="features-section">
          <h2 className="section-title">What Makes a Message Spam?</h2>
          <p className="section-description">
            Our machine learning models analyze dozens of features to identify spam.
            These interactive visualizations show which features are most important for detection.
          </p>

          <div className="chart-grid-vertical">
            <FeatureImportanceChart />
            <FeatureDistributionChart />
            <CorrelationHeatmap />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
