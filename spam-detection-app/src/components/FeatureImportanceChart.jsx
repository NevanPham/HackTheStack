import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function FeatureImportanceChart() {
  const svgRef = useRef();
  const [data, setData] = useState(null);
  const [selectedGroup, setSelectedGroup] = useState('all');

  useEffect(() => {
    fetch('/data/feature_analysis.json')
      .then(res => res.json())
      .then(data => setData(data))
      .catch(err => console.error('Error loading feature analysis:', err));
  }, []);

  useEffect(() => {
    if (!data) return;

    // Filter data by group
    let filteredData = data.top_discriminative.slice(0, 15);
    if (selectedGroup !== 'all') {
      filteredData = data.feature_comparison
        .filter(d => d.group === selectedGroup)
        .slice(0, 15);
    }

    // Setup dimensions - responsive & smaller
    const containerWidth = svgRef.current?.parentElement?.offsetWidth || 720;
    const margin = { top: 16, right: 16, bottom: 56, left: 140 }; // reduced left margin
    const maxChartWidth = Math.min(containerWidth - 32, 760);    // smaller max width
    const width = Math.max(320, maxChartWidth) - margin.left - margin.right;
    const height = 360 - margin.top - margin.bottom;             // reduced height

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Create SVG with viewBox for responsiveness
    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('height', 'auto')
      .append('g')
      .attr('transform', `translate(${margin.left - 35},${margin.top - 25})`);

    // Scales (unchanged logic but ensure safe domain)
    const xMax = d3.max(filteredData, d => d.ratio) || 1;
    const x = d3.scaleLinear()
      .domain([0, xMax])
      .range([0, width]);

    const y = d3.scaleBand()
      .domain(filteredData.map(d => d.feature))
      .range([0, height])
      .padding(0.18);

    const colorScale = d3.scaleOrdinal()
      .domain(['Basic', 'Spam Indicators', 'Character', 'Lexical'])
      .range(['#4CAF50', '#f44336', '#FF9800', '#2196F3']);

    // Add bars (use smaller font/spacing)
    const bars = svg.selectAll('.bar')
      .data(filteredData)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('y', d => y(d.feature))
      .attr('height', y.bandwidth())
      .attr('x', 0)
      .attr('width', 0)
      .attr('fill', d => colorScale(d.group))
      .attr('opacity', 0.9)
      .style('cursor', 'pointer');

    bars.transition()
      .duration(700)
      .delay((d, i) => i * 40)
      .attr('width', d => x(d.ratio));

    // Tooltip (no change)
    const tooltip = d3.select('body')
      .append('div')
      .attr('class', 'd3-tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background-color', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '6px')
      .style('font-size', '13px')
      .style('pointer-events', 'none')
      .style('z-index', '1000');

    bars.on('mouseover', function(event, d) {
      d3.select(this).attr('opacity', 1).attr('stroke', '#333').attr('stroke-width', 1.5);
      tooltip.style('visibility', 'visible').html(`
        <strong>${d.feature.replace(/_/g, ' ')}</strong><br/>
        <span style="color: ${colorScale(d.group)}">${d.group}</span><br/>
        Spam/Ham Ratio: <strong>${d.ratio.toFixed(2)}x</strong>
      `);
    }).on('mousemove', function(event) {
      tooltip.style('top', (event.pageY - 10) + 'px').style('left', (event.pageX + 10) + 'px');
    }).on('mouseout', function() {
      d3.select(this).attr('opacity', 0.9).attr('stroke', 'none');
      tooltip.style('visibility', 'hidden');
    });

    // Value labels (smaller)
    svg.selectAll('.value-label')
      .data(filteredData)
      .enter()
      .append('text')
      .attr('class', 'value-label')
      .attr('x', d => x(d.ratio) + 6)
      .attr('y', d => y(d.feature) + y.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('font-size', '11px')
      .attr('fill', '#222')
      .attr('opacity', 0)
      .text(d => `${d.ratio.toFixed(2)}x`)
      .transition()
      .duration(600)
      .delay((d, i) => i * 40 + 300)
      .attr('opacity', 1);

    // Axes with smaller fonts
    const xAxis = d3.axisBottom(x).ticks(6);
    const yAxis = d3.axisLeft(y).tickFormat(d => d.replace(/_/g, ' '));

    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .selectAll('text')
      .attr('font-size', '11px');

    svg.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .selectAll('text')
      .attr('font-size', '11px')
      .attr('font-weight', '500');

    // Axis label adjusted
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 36)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .text('Spam/Ham Ratio (Higher = More Spam-Indicative)');

    // Legend below chart (compact)
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(0, ${height + 50})`);

    const groups = ['Basic', 'Spam Indicators', 'Character', 'Lexical'];
    const legendItemWidth = Math.max(90, width / groups.length);

    groups.forEach((group, i) => {
      const g = legend.append('g')
        .attr('transform', `translate(${i * legendItemWidth}, 0)`);

      g.append('rect')
        .attr('width', 14)
        .attr('height', 14)
        .attr('fill', colorScale(group))
        .attr('opacity', 0.9);

      g.append('text')
        .attr('x', 20)
        .attr('y', 9)
        .attr('dy', '0.35em')
        .attr('font-size', '11px')
        .text(group);
    });

    // Cleanup
    return () => {
      tooltip.remove();
    };

  }, [data, selectedGroup]);

  if (!data) {
    return <div className="loading">Loading feature analysis...</div>;
  }

  return (
    <div className="feature-importance-chart">
      <div className="chart-header">
        <h3>Feature Importance: Spam vs Ham</h3>
        <p className="chart-description">
          Shows which features are most different between spam and ham messages
        </p>
        <div className="filter-buttons">
          <button
            className={selectedGroup === 'all' ? 'active' : ''}
            onClick={() => setSelectedGroup('all')}
          >
            All Features
          </button>
          <button
            className={selectedGroup === 'Spam Indicators' ? 'active' : ''}
            onClick={() => setSelectedGroup('Spam Indicators')}
          >
            Spam Indicators
          </button>
          <button
            className={selectedGroup === 'Character' ? 'active' : ''}
            onClick={() => setSelectedGroup('Character')}
          >
            Character
          </button>
          <button
            className={selectedGroup === 'Basic' ? 'active' : ''}
            onClick={() => setSelectedGroup('Basic')}
          >
            Basic
          </button>
          <button
            className={selectedGroup === 'Lexical' ? 'active' : ''}
            onClick={() => setSelectedGroup('Lexical')}
          >
            Lexical
          </button>
        </div>
      </div>
      <svg ref={svgRef}></svg>
    </div>
  );
}

export default FeatureImportanceChart;
