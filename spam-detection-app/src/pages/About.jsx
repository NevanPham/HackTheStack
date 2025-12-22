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
            I would like to thank{' '}
            <a
              href="https://www.linkedin.com/in/minhan6559/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Andy Nguyen
            </a>{' '}
            and{' '}
            <a
              href="https://www.linkedin.com/in/annetranhpham/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Anne Pham
            </a>{' '}
            for their contributions to the development of the core functionality of this website application. Their support throughout the project, from shaping the
            initial concept to refining the final experience, laid a strong foundation upon which this work was built.
          </p>
          <p>
            The solid baseline of the website they created made it possible for me to further develop the website by exploring and integrating intentional security
            vulnerabilities as part of a hands-on learning experience.
          </p>
          <br />
        </div>
      </section>
    </div>
  );
}

export default About;