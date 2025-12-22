import trangAnhImg from '../assets/images/trang-anh.jpg';
import khoiNguyenImg from '../assets/images/khoi-nguyen.jpg';
import minhAnImg from '../assets/images/minh-an.jpg';

function About() {
  return (
    <div className="about-content">
      <section className="about-section">
        <h2>How It Works</h2>
        <p>
          Our advanced spam detection system delivers a comprehensive, multi-layered approach to identifying malicious messages
          across various platforms including SMS, email, YouTube comments, and product reviews. The system leverages three
          complementary machine learning algorithms, each bringing unique strengths to create a robust detection framework:
        </p>
        <div className="features-grid">
          <div className="feature-card">
            <h3>K-Means Clustering</h3>
            <p>Unsupervised learning for pattern detection with 83.5% accuracy</p>
          </div>
          <div className="feature-card">
            <h3>XGBoost</h3>
            <p>Gradient boosting with optimized thresholds achieving 91.8% accuracy</p>
          </div>
          <div className="feature-card">
            <h3>BiLSTM</h3>
            <p>Deep learning LSTM model with bidirectional architecture reaching 97.1% accuracy</p>
          </div>
        </div>
      </section>

      <section className="team-section">
        <h1 className="page-title">Acknowledgement</h1>
        <div className="about-description">
          <br />
          <p>
            I would like to acknowledge Andy Nguyen and Anne Pham for working with me to develop this website and supporting every step of
            the journey, from shaping the initial idea to polishing the final experience.
          </p>
          <p>
            Their hard work, creativity, and constant encouragement have been invaluable, and I&apos;m deeply grateful for their
            collaboration and belief in this project.
          </p>
          <br />
        </div>
      </section>
    </div>
  );
}

export default About;