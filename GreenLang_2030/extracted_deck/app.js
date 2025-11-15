// Presentation State
let currentSlide = 1;
const totalSlides = 15;
let charts = {};

// Initialize presentation
function init() {
    createParticles();
    createSlideDots();
    setupEventListeners();
    setupKeyboardNavigation();
    
    // Delay chart initialization to ensure proper rendering
    setTimeout(() => {
        initializeCharts();
    }, 500);
}

// Create particle effect for title slide
function createParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 3}s`;
        particle.style.animationDuration = `${3 + Math.random() * 2}s`;
        container.appendChild(particle);
    }
}

// Create navigation dots
function createSlideDots() {
    const container = document.getElementById('slideDots');
    for (let i = 1; i <= totalSlides; i++) {
        const dot = document.createElement('div');
        dot.className = `dot ${i === 1 ? 'active' : ''}`;
        dot.setAttribute('data-slide', i);
        dot.addEventListener('click', () => goToSlide(i));
        container.appendChild(dot);
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('prevBtn').addEventListener('click', prevSlide);
    document.getElementById('nextBtn').addEventListener('click', nextSlide);
    document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);
}

// Keyboard navigation
function setupKeyboardNavigation() {
    document.addEventListener('keydown', (e) => {
        switch(e.key) {
            case 'ArrowLeft':
                prevSlide();
                break;
            case 'ArrowRight':
                nextSlide();
                break;
            case 'h':
            case 'H':
                goToSlide(1);
                break;
            case 'f':
            case 'F':
                toggleFullscreen();
                break;
        }
    });
}

// Navigation functions
function nextSlide() {
    if (currentSlide < totalSlides) {
        goToSlide(currentSlide + 1);
    }
}

function prevSlide() {
    if (currentSlide > 1) {
        goToSlide(currentSlide - 1);
    }
}

function goToSlide(slideNumber) {
    // Remove active class from current slide
    document.getElementById(`slide-${currentSlide}`).classList.remove('active');
    
    // Update current slide
    currentSlide = slideNumber;
    
    // Add active class to new slide
    document.getElementById(`slide-${currentSlide}`).classList.add('active');
    
    // Update navigation dots
    document.querySelectorAll('.dot').forEach((dot, index) => {
        dot.classList.toggle('active', index + 1 === currentSlide);
    });
    
    // Animate counters when entering specific slides
    if ([6, 7, 11].includes(currentSlide)) {
        setTimeout(() => animateCounters(), 300);
    }
}

// Counter animation
function animateCounters() {
    const counters = document.querySelectorAll('.metric-number[data-target]');
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 1500;
        const step = target / (duration / 16);
        let current = 0;
        
        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                counter.textContent = target.toLocaleString();
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current).toLocaleString();
            }
        }, 16);
    });
}

// Fullscreen toggle
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

// Initialize all charts
function initializeCharts() {
    createCompetitorChart();
    createMarketChart();
    createRadarChart();
    createRevenueChart();
    createFundsChart();
}

// Competitor positioning chart
function createCompetitorChart() {
    const ctx = document.getElementById('competitorChart');
    if (!ctx) return;
    
    charts.competitor = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'GreenLang',
                    data: [{x: 10, y: 10}],
                    backgroundColor: 'rgba(198, 255, 0, 0.8)',
                    borderColor: '#C6FF00',
                    borderWidth: 3,
                    pointRadius: 20,
                    pointHoverRadius: 25
                },
                {
                    label: 'Persefoni',
                    data: [{x: 9, y: 6}],
                    backgroundColor: 'rgba(100, 100, 100, 0.5)',
                    borderColor: '#666',
                    borderWidth: 2,
                    pointRadius: 12
                },
                {
                    label: 'Watershed',
                    data: [{x: 8, y: 5}],
                    backgroundColor: 'rgba(100, 100, 100, 0.5)',
                    borderColor: '#666',
                    borderWidth: 2,
                    pointRadius: 12
                },
                {
                    label: 'Workiva',
                    data: [{x: 6, y: 4}],
                    backgroundColor: 'rgba(100, 100, 100, 0.5)',
                    borderColor: '#666',
                    borderWidth: 2,
                    pointRadius: 12
                },
                {
                    label: 'SAP',
                    data: [{x: 5, y: 3}],
                    backgroundColor: 'rgba(100, 100, 100, 0.5)',
                    borderColor: '#666',
                    borderWidth: 2,
                    pointRadius: 12
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#FFFFFF',
                        font: { size: 14, weight: '600' },
                        padding: 20
                    }
                },
                title: {
                    display: true,
                    text: 'Competitive Positioning',
                    color: '#C6FF00',
                    font: { size: 24, weight: '900' },
                    padding: 20
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Regulatory Accuracy',
                        color: '#FFFFFF',
                        font: { size: 14, weight: '600' }
                    },
                    min: 0,
                    max: 10,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF', font: { size: 12 } }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Agent Ecosystem',
                        color: '#FFFFFF',
                        font: { size: 14, weight: '600' }
                    },
                    min: 0,
                    max: 10,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF', font: { size: 12 } }
                }
            }
        }
    });
}

// Market size chart
function createMarketChart() {
    const ctx = document.getElementById('marketChart');
    if (!ctx) return;
    
    charts.market = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2025', '2026', '2027', '2028', '2029', '2030'],
            datasets: [
                {
                    label: 'Corporate Carbon Management',
                    data: [50, 55, 62, 70, 80, 90],
                    backgroundColor: 'rgba(198, 255, 0, 0.8)',
                    borderColor: '#C6FF00',
                    borderWidth: 2
                },
                {
                    label: 'Sustainability Reporting',
                    data: [35, 40, 45, 52, 60, 70],
                    backgroundColor: 'rgba(10, 58, 42, 0.8)',
                    borderColor: '#0A3A2A',
                    borderWidth: 2
                },
                {
                    label: 'Supply Chain Intelligence',
                    data: [20, 24, 28, 34, 40, 48],
                    backgroundColor: 'rgba(100, 180, 100, 0.6)',
                    borderColor: '#64B464',
                    borderWidth: 2
                },
                {
                    label: 'ESG Data Infrastructure',
                    data: [15, 18, 22, 27, 33, 40],
                    backgroundColor: 'rgba(150, 150, 150, 0.6)',
                    borderColor: '#969696',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#FFFFFF',
                        font: { size: 13 },
                        padding: 15
                    }
                },
                title: {
                    display: true,
                    text: 'Total Addressable Market Growth ($ Billions)',
                    color: '#C6FF00',
                    font: { size: 20, weight: '900' },
                    padding: 20
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF', font: { size: 12 } }
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Market Size ($B)',
                        color: '#FFFFFF',
                        font: { size: 12, weight: '600' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF', font: { size: 12 } }
                }
            }
        }
    });
}

// Radar chart for competitive moats
function createRadarChart() {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;
    
    charts.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                'Regulatory Compliance',
                'Supply Chain Transparency',
                'Developer Ecosystem',
                'AI-Powered Insights',
                'Enterprise Performance'
            ],
            datasets: [{
                label: 'GreenLang',
                data: [9, 9, 9, 8, 8],
                backgroundColor: 'rgba(198, 255, 0, 0.3)',
                borderColor: '#C6FF00',
                borderWidth: 3,
                pointBackgroundColor: '#C6FF00',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#C6FF00',
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Competitive Advantages (Scale: 0-10)',
                    color: '#C6FF00',
                    font: { size: 24, weight: '900' },
                    padding: 20
                }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.2)' },
                    grid: { color: 'rgba(255, 255, 255, 0.2)' },
                    pointLabels: {
                        color: '#FFFFFF',
                        font: { size: 12, weight: '600' }
                    },
                    ticks: {
                        color: '#FFFFFF',
                        backdropColor: 'transparent',
                        font: { size: 12 }
                    },
                    min: 0,
                    max: 10
                }
            }
        }
    });
}

// Revenue projections chart
function createRevenueChart() {
    const ctx = document.getElementById('revenueChart');
    if (!ctx) return;
    
    charts.revenue = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['2026', '2027', '2028', '2029', '2030'],
            datasets: [
                {
                    label: 'ARR ($M)',
                    data: [18, 50, 150, 300, 500],
                    backgroundColor: 'rgba(198, 255, 0, 0.2)',
                    borderColor: '#C6FF00',
                    borderWidth: 4,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 8,
                    pointHoverRadius: 12,
                    pointBackgroundColor: '#C6FF00',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'Customers',
                    data: [750, 5000, 10000, 25000, 50000],
                    backgroundColor: 'rgba(10, 58, 42, 0.2)',
                    borderColor: '#0A3A2A',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 10,
                    pointBackgroundColor: '#0A3A2A',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#FFFFFF',
                        font: { size: 16, weight: '600' },
                        padding: 20
                    }
                },
                title: {
                    display: true,
                    text: '5-Year Revenue Projection',
                    color: '#C6FF00',
                    font: { size: 24, weight: '900' },
                    padding: 20
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF', font: { size: 14, weight: '600' } }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'ARR ($M)',
                        color: '#C6FF00',
                        font: { size: 14, weight: '600' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF', font: { size: 12 } }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Customers',
                        color: '#0A3A2A',
                        font: { size: 14, weight: '600' }
                    },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#FFFFFF', font: { size: 12 } }
                }
            }
        }
    });
}

// Use of funds chart
function createFundsChart() {
    const ctx = document.getElementById('fundsChart');
    if (!ctx) return;
    
    charts.funds = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Engineering (60%)', 'Sales & Marketing (20%)', 'Infrastructure (10%)', 'G&A (10%)'],
            datasets: [{
                data: [60, 20, 10, 10],
                backgroundColor: [
                    'rgba(198, 255, 0, 0.8)',
                    'rgba(10, 58, 42, 0.8)',
                    'rgba(100, 180, 100, 0.7)',
                    'rgba(150, 150, 150, 0.7)'
                ],
                borderColor: [
                    '#C6FF00',
                    '#0A3A2A',
                    '#64B464',
                    '#969696'
                ],
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#FFFFFF',
                        font: { size: 16, weight: '600' },
                        padding: 20
                    }
                },
                title: {
                    display: true,
                    text: 'Use of Funds',
                    color: '#C6FF00',
                    font: { size: 24, weight: '900' },
                    padding: 20
                }
            }
        }
    });
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);