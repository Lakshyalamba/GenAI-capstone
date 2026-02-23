// ─── Sidebar Toggle (mobile) ─────────────────────────────────────────
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('overlay');
    const hamburger = document.getElementById('hamburger');
    const isOpen = sidebar.classList.toggle('open');
    overlay.classList.toggle('active', isOpen);
    hamburger.classList.toggle('open', isOpen);
}

function closeSidebar() {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('overlay').classList.remove('active');
    document.getElementById('hamburger').classList.remove('open');
}

// ─── Page Navigation ────────────────────────────────────────────────
function showPage(id, el) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + id).classList.add('active');
    el.classList.add('active');
    // Auto-close sidebar on mobile after navigation
    if (window.innerWidth <= 768) closeSidebar();
}

// ─── Animated KPI Counter ────────────────────────────────────────────
function animateCount(id, target, suffix = '%', duration = 1200) {
    const el = document.getElementById(id);
    const start = performance.now();
    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(eased * target) + suffix;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// Model metrics from the trained logistic regression (actual values from Capstone.ipynb)
const METRICS = {
    accuracy: 92.50,
    precision: 92.00,
    recall: 93.00,
    f1: 92.00,
    auc: 89.00
};

window.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        animateCount('acc-val', METRICS.accuracy);
        animateCount('prec-val', METRICS.precision);
        animateCount('rec-val', METRICS.recall);
        animateCount('f1-val', METRICS.f1);
        animateCount('auc-val', METRICS.auc);
    }, 300);

    buildDistChart();
    buildFeatureTypeChart();
    buildROCChart();
    buildCoefChart();
});

// ─── Chart Defaults ─────────────────────────────────────────────────
Chart.defaults.color = '#6b7a99';
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.font.size = 12;

// ─── KPI Page Charts ─────────────────────────────────────────────────

function buildDistChart() {
    new Chart(document.getElementById('distChart'), {
        type: 'doughnut',
        data: {
            labels: ['High Risk (1)', 'Low Risk (0)'],
            datasets: [{
                data: [206, 194],
                backgroundColor: ['rgba(239,68,68,0.8)', 'rgba(34,197,94,0.8)'],
                borderColor: ['rgba(239,68,68,1)', 'rgba(34,197,94,1)'],
                borderWidth: 2,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, pointStyleWidth: 10 } },
                tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} patients (${((ctx.raw / 400) * 100).toFixed(1)}%)` } }
            },
            cutout: '65%'
        }
    });
}

function buildFeatureTypeChart() {
    new Chart(document.getElementById('featureTypeChart'), {
        type: 'doughnut',
        data: {
            labels: ['Numerical (5)', 'Categorical (5)'],
            datasets: [{
                data: [5, 5],
                backgroundColor: ['rgba(79,142,247,0.8)', 'rgba(124,95,247,0.8)'],
                borderColor: ['rgba(79,142,247,1)', 'rgba(124,95,247,1)'],
                borderWidth: 2,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, pointStyleWidth: 10 } }
            },
            cutout: '65%'
        }
    });
}

// ─── Visual Analysis Charts ──────────────────────────────────────────

function buildROCChart() {
    // Representative ROC curve points for AUC ≈ 0.89
    const fpr = [0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.38, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0];
    const tpr = [0, 0.28, 0.48, 0.60, 0.70, 0.76, 0.81, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0];

    new Chart(document.getElementById('rocChart'), {
        type: 'line',
        data: {
            labels: fpr,
            datasets: [
                {
                    label: `ROC Curve (AUC = 0.89)`,
                    data: tpr,
                    borderColor: '#4f8ef7',
                    backgroundColor: 'rgba(79,142,247,0.08)',
                    borderWidth: 3,
                    pointRadius: 3,
                    pointBackgroundColor: '#4f8ef7',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Random Classifier',
                    data: fpr,
                    borderColor: 'rgba(107,122,153,0.5)',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'False Positive Rate', color: '#6b7a99', font: { size: 12 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 6, callback: v => (+v).toFixed(1) }
                },
                y: {
                    title: { display: true, text: 'True Positive Rate', color: '#6b7a99', font: { size: 12 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    min: 0, max: 1
                }
            },
            plugins: {
                legend: { labels: { usePointStyle: true, padding: 20 } },
                tooltip: { callbacks: { title: ctx => `FPR: ${(+ctx[0].label).toFixed(2)}`, label: ctx => ` TPR: ${ctx.raw.toFixed(2)}` } }
            }
        }
    });
}

function buildCoefChart() {
    const features = [
        'Chest Pain\n(Typical)',
        'Exercise\nAngina',
        'Age',
        'Systolic BP',
        'Smoker',
        'Cholesterol',
        'Diabetes',
        'BMI',
        'Max HR',
        'Sex (Male)'
    ];
    const coefficients = [1.42, 1.18, 0.85, 0.73, 0.62, 0.54, 0.48, 0.31, -0.56, -0.28];
    const colors = coefficients.map(v => v > 0 ? 'rgba(239,68,68,0.8)' : 'rgba(34,197,94,0.8)');
    const borders = coefficients.map(v => v > 0 ? 'rgba(239,68,68,1)' : 'rgba(34,197,94,1)');

    new Chart(document.getElementById('coefChart'), {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Coefficient Weight',
                data: coefficients,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    title: { display: true, text: 'Coefficient Value', color: '#6b7a99' }
                },
                y: { grid: { display: false }, ticks: { font: { size: 11 } } }
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => ` Weight: ${ctx.raw.toFixed(2)}` } }
            }
        }
    });
}

// ─── Prediction System ───────────────────────────────────────────────

let gaugeChart = null;

function runPrediction() {
    const age = parseFloat(document.getElementById('f-age').value);
    const sbp = parseFloat(document.getElementById('f-sbp').value);
    const chol = parseFloat(document.getElementById('f-chol').value);
    const hr = parseFloat(document.getElementById('f-hr').value);
    const bmi = parseFloat(document.getElementById('f-bmi').value);
    const sex = document.getElementById('f-sex').value;
    const cp = document.getElementById('f-cp').value;
    const smoker = document.getElementById('f-smoker').value;
    const diab = document.getElementById('f-diabetes').value;
    const angina = document.getElementById('f-angina').value;

    if ([age, sbp, chol, hr, bmi].some(isNaN) || !sex || !cp || !smoker || !diab || !angina) {
        alert('Please fill in all fields before predicting.');
        return;
    }

    // Logistic regression scoring with approximate feature weights
    // Intercept ≈ -2.8 (baseline log-odds)
    let logit = -2.8;

    // Numerical features (standardised roughly)
    logit += 0.85 * ((age - 55) / 14);
    logit += 0.73 * ((sbp - 135) / 25);
    logit += 0.54 * ((chol - 225) / 43);
    logit += 0.31 * ((bmi - 29) / 6.4);
    logit -= 0.56 * ((hr - 145) / 32);  // higher max HR → lower risk

    // Categorical features
    if (cp === 'typical') logit += 1.42;
    if (cp === 'atypical') logit += 0.60;
    if (cp === 'asymptomatic') logit += 0.30;
    if (angina === 'yes') logit += 1.18;
    if (smoker === 'yes') logit += 0.62;
    if (diab === 'yes') logit += 0.48;
    if (sex === 'male') logit -= 0.28;

    // Sigmoid
    const prob = Math.round((1 / (1 + Math.exp(-logit))) * 100);
    const clamped = Math.min(Math.max(prob, 2), 98);

    showResult(clamped, { age, sbp, chol, hr, bmi, sex, cp, smoker, diab, angina });
}

function showResult(score, data) {
    document.getElementById('result-placeholder').style.display = 'none';
    document.getElementById('result-output').style.display = 'block';

    document.getElementById('gauge-pct').textContent = score + '%';

    let color, level, desc;
    if (score < 30) {
        color = '#22c55e'; level = 'Low Risk'; desc = 'The patient shows low cardiovascular risk. Maintaining a healthy lifestyle is recommended.';
    } else if (score < 60) {
        color = '#f59e0b'; level = 'Moderate Risk'; desc = 'Moderate cardiovascular risk detected. Regular check-ups and lifestyle adjustments are advised.';
    } else {
        color = '#ef4444'; level = 'High Risk'; desc = 'High cardiovascular risk detected. Immediate medical consultation and intervention are strongly recommended.';
    }

    document.getElementById('risk-level').textContent = level;
    document.getElementById('risk-level').style.color = color;
    document.getElementById('risk-desc').textContent = desc;

    // Build gauge (doughnut)
    if (gaugeChart) gaugeChart.destroy();
    gaugeChart = new Chart(document.getElementById('gaugeChart'), {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                borderWidth: 0,
                hoverOffset: 0
            }]
        },
        options: {
            responsive: false,
            cutout: '78%',
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            animation: { duration: 900, easing: 'easeOutQuart' }
        }
    });

    // Key risk indicators
    const factors = [
        { label: 'Age', val: `${data.age} yrs`, risk: data.age > 65 },
        { label: 'Systolic BP', val: `${data.sbp} mmHg`, risk: data.sbp > 140 },
        { label: 'Cholesterol', val: `${data.chol} mg/dL`, risk: data.chol > 240 },
        { label: 'BMI', val: data.bmi.toFixed(1), risk: data.bmi > 30 },
        { label: 'Chest Pain', val: data.cp, risk: data.cp === 'typical' || data.cp === 'asymptomatic' },
        { label: 'Smoker', val: data.smoker, risk: data.smoker === 'yes' },
        { label: 'Diabetes', val: data.diab, risk: data.diab === 'yes' },
        { label: 'Exer. Angina', val: data.angina, risk: data.angina === 'yes' }
    ];

    const list = document.getElementById('risk-factors-list');
    list.innerHTML = factors.map(f => `
    <div class="factor-row">
      <div class="factor-dot" style="background:${f.risk ? '#ef4444' : '#22c55e'}"></div>
      <span>${f.label}</span>
      <span class="factor-val" style="color:${f.risk ? '#ef4444' : '#22c55e'}">${f.val}</span>
    </div>
  `).join('');
}
