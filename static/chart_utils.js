// static/chart_utils.js

/**
 * Initialize a Chart.js chart
 * @param {string} canvasId - HTML canvas element id
 * @param {string} label - Dataset label
 * @param {Array} labels - X-axis labels
 * @param {Array} data - Y-axis values
 * @param {string} color - Line color
 * @returns {Chart} - Chart.js instance
 */
function initLineChart(canvasId, label, labels = [], data = [], color = '#3b82f6') {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: 'rgba(0,0,0,0)',
                fill: false,
                tension: 0.1,
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: { display: true },
                tooltip: { enabled: true }
            },
            scales: {
                x: { display: true, title: { display: true, text: 'Date' } },
                y: { display: true, title: { display: true, text: 'Price ($)' } }
            }
        }
    });
}

/**
 * Update an existing chart with new data
 * @param {Chart} chartInstance - Chart.js instance
 * @param {Array} labels - X-axis labels
 * @param {Array} data - Y-axis values
 */
function updateChart(chartInstance, labels, data) {
    chartInstance.data.labels = labels;
    chartInstance.data.datasets[0].data = data;
    chartInstance.update();
}

/**
 * Example usage:
 * 
 * // Initialize chart
 * let myChart = initLineChart('priceChart', 'AAPL Close Price');
 * 
 * // Update chart dynamically
 * updateChart(myChart, ['2025-10-01', '2025-10-02'], [150, 152]);
 */
