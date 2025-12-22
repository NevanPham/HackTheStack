import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function FeatureDistributionChart() {
  const svgRef = useRef();
  const [data, setData] = useState(null);
  const [selectedFeature, setSelectedFeature] = useState('money_count');

  useEffect(() => {
    fetch('/data/feature_analysis.json')
      .then(res => res.json())
      .then(data => setData(data))
      .catch(err => console.error('Error loading feature analysis:', err));
  }, []);

  useEffect(() => {
    if (!data) return;

    const container = svgRef.current?.parentElement || svgRef.current;
    let resizeObserver;
    let tooltip = null;

    // Hide tooltip on scroll - defined outside draw so it's accessible in cleanup
    const hideTooltipOnScroll = () => {
      if (tooltip) {
        tooltip.style('visibility', 'hidden');
      }
    };

    const draw = () => {
      // clear previous drawing + tooltips
      d3.select(svgRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-dist').remove();

      const distributions = data.distributions;
      const featureData = distributions[selectedFeature];

      const containerWidth = container?.clientWidth || 1000;
      const margin = { top: 40, right: 50, bottom: 70, left: 80 };
      const width = Math.max(300, Math.min(containerWidth - 40, 900) - margin.left - margin.right);
      const height = 400 - margin.top - margin.bottom;

      const svg = d3.select(svgRef.current)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .style('width', '100%')
        .style('height', 'auto')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const categories = ['Mean', 'Median', 'Max'];
      const chartData = categories.map(cat => ({
        category: cat,
        spam: featureData.spam[cat.toLowerCase()],
        ham: featureData.ham[cat.toLowerCase()]
      }));

      const x0 = d3.scaleBand().domain(categories).range([0, width]).padding(0.3);
      const x1 = d3.scaleBand().domain(['spam', 'ham']).range([0, x0.bandwidth()]).padding(0.1);
      const y = d3.scaleLinear().domain([0, d3.max(chartData, d => Math.max(d.spam, d.ham)) * 1.1]).range([height, 0]);
      const colorScale = d3.scaleOrdinal().domain(['spam', 'ham']).range(['#f44336', '#4CAF50']);

      // tooltip
      tooltip = d3.select('body')
        .append('div')
        .attr('class', 'd3-tooltip d3-tooltip-dist')
        .style('position', 'fixed')
        .style('visibility', 'hidden')
        .style('background-color', 'rgba(0, 0, 0, 0.9)')
        .style('color', 'white')
        .style('padding', '10px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('pointer-events', 'none')
        .style('z-index', '1000')
        .style('max-width', '200px');

      // Helper function to constrain tooltip within viewport
      const positionTooltip = (event) => {
        const tooltipNode = tooltip.node();
        if (!tooltipNode) return;
        
        const tooltipRect = tooltipNode.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const footerHeight = 80; // Approximate footer height
        const padding = 10;
        
        let left = event.clientX + 10;
        let top = event.clientY - 10;
        
        // Constrain horizontally
        if (left + tooltipRect.width > viewportWidth - padding) {
          left = event.clientX - tooltipRect.width - 10;
        }
        if (left < padding) {
          left = padding;
        }
        
        // Constrain vertically - ensure it doesn't go into footer area
        if (top + tooltipRect.height > viewportHeight - footerHeight - padding) {
          top = event.clientY - tooltipRect.height - 10;
        }
        if (top < padding) {
          top = padding;
        }
        
        tooltip.style('left', left + 'px').style('top', top + 'px');
      };

      // bars
      const groups = svg.selectAll('.group').data(chartData).enter().append('g').attr('class', 'group').attr('transform', d => `translate(${x0(d.category)},0)`);

      ['spam', 'ham'].forEach(type => {
        groups.append('rect')
          .attr('x', d => x1(type))
          .attr('width', x1.bandwidth())
          .attr('y', height)
          .attr('height', 0)
          .attr('fill', colorScale(type))
          .attr('opacity', 0.8)
          .style('cursor', 'pointer')
          .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 1);
            tooltip.style('visibility', 'visible').html(`
              <strong>${d.category} - ${type.toUpperCase()}</strong><br/>
              Value: <strong>${d[type].toFixed(3)}</strong>
            `);
            positionTooltip(event);
          })
          .on('mousemove', function(event) {
            positionTooltip(event);
          })
          .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.8);
            tooltip.style('visibility', 'hidden');
          })
          .transition()
          .duration(800)
          .delay(300)
          .attr('y', d => y(d[type]))
          .attr('height', d => height - y(d[type]));

        // value labels
        groups.append('text')
          .attr('x', d => x1(type) + x1.bandwidth() / 2)
          .attr('y', d => y(d[type]) - 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('font-weight', 'bold')
          .attr('fill', colorScale(type))
          .attr('opacity', 0)
          .text(d => d[type].toFixed(2))
          .transition()
          .duration(800)
          .delay(800)
          .attr('opacity', 1);
      });

      // axes
      svg.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x0)).selectAll('text').attr('font-size', '13px').attr('font-weight', '500');
      svg.append('g').attr('class', 'y-axis').call(d3.axisLeft(y).ticks(8)).selectAll('text').attr('font-size', '12px');

      svg.append('text').attr('x', width / 2).attr('y', height + 50).attr('text-anchor', 'middle').attr('font-size', '13px').attr('font-weight', 'bold').text('Statistics');
      svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -50).attr('text-anchor', 'middle').attr('font-size', '13px').attr('font-weight', 'bold').text('Value');

      // legend
      const legend = svg.append('g').attr('class', 'legend').attr('transform', `translate(${Math.max(0, width - 180)}, -30)`);
      ['spam', 'ham'].forEach((type, i) => {
        const g = legend.append('g').attr('transform', `translate(${i * 70}, 0)`);
        g.append('rect').attr('width', 16).attr('height', 16).attr('fill', colorScale(type)).attr('opacity', 0.8);
        g.append('text').attr('x', 20).attr('y', 8).attr('dy', '0.35em').attr('font-size', '12px').attr('font-weight', '500').text(type.toUpperCase());
      });
    };

    draw();

    // Add scroll listener to hide tooltips
    window.addEventListener('scroll', hideTooltipOnScroll, true);

    const onResize = () => draw();
    window.addEventListener('resize', onResize);

    if ('ResizeObserver' in window && container) {
      resizeObserver = new ResizeObserver(() => draw());
      resizeObserver.observe(container);
    }

    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', hideTooltipOnScroll, true);
      if (resizeObserver && container) resizeObserver.unobserve(container);
      d3.select(svgRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-dist').remove();
    };
  }, [data, selectedFeature]);

  if (!data) {
    return <div className="loading">Loading distributions...</div>;
  }

  const features = Object.keys(data.distributions);

  return (
    <div className="feature-distribution-chart">
      <div className="chart-header">
        <h3>Feature Distribution: Spam vs Ham</h3>
        <p className="chart-description">
          Compare statistical distributions of features between spam and ham messages
        </p>
        <div className="feature-selector">
          <label htmlFor="feature-select">Select Feature: </label>
          <select
            id="feature-select"
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
          >
            {features.map(feat => (
              <option key={feat} value={feat}>
                {feat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>
      <svg ref={svgRef}></svg>
    </div>
  );
}

export default FeatureDistributionChart;
