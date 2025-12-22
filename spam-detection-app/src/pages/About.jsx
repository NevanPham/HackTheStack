import trangAnhImg from '../assets/images/trang-anh.jpg';
import khoiNguyenImg from '../assets/images/khoi-nguyen.jpg';
import minhAnImg from '../assets/images/minh-an.jpg';

function About() {
  return (
    <div className="about-content">
      <h1 className="page-title">Meet Our Team</h1>

      <section className="team-section">
        <div className="team-grid">
          <div className="team-member">
            <div className="team-avatar">
              <img src={trangAnhImg} alt="Hoang Trang Anh Pham" />
            </div>
            <h3 className="team-name">Hoang Trang Anh Pham</h3>
            <p className="team-role">Data Analyst Lead</p>
          </div>

          <div className="team-member">
            <div className="team-avatar">
              <img src={khoiNguyenImg} alt="Khoi Nguyen Pham" />
            </div>
            <h3 className="team-name">Khoi Nguyen Pham</h3>
            <p className="team-role">Technical Lead</p>
          </div>

          <div className="team-member">
            <div className="team-avatar">
              <img src={minhAnImg} alt="Minh An Nguyen" />
            </div>
            <h3 className="team-name">Minh An Nguyen</h3>
            <p className="team-role">Manager Lead</p>
          </div>
        </div>
      </section>

      <section className="about-description">
        <br />
        <p>
          We're a team of cybersecurity and machine learning specialists dedicated to making email security simple and effective. Our goal
          is to protect users from spam and malicious emails while helping them understand the threats they encounter.
        </p>
        <p>
          We believe security tools should be both powerful and easy to use. That's why we've built a spam detector that not only identifies
          threats but also explains why something is dangerous, empowering users with knowledge to stay safe online.
        </p>
        <br />
        <p className="cta-text">
          <strong>Ready to test our technology? Upload an email or paste text to get started.</strong>
        </p>
        <br />
      </section>

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
    </div>
  );
}

export default About;