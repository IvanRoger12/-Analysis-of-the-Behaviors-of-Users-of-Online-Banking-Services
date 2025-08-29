// Gestionnaire des graphiques et visualisations
class ChartManager {
    constructor() {
        this.charts = new Map();
        this.defaultColors = {
            primary: '#00ffff',
            secondary: '#ff00ff',
            success: '#00ff88',
            warning: '#ffaa00',
            danger: '#ff3366',
            info: '#3366ff',
            gradient: {
                cyan: ['#00ffff', '#39ff14'],
                magenta: ['#ff00ff', '#6a0dad'],
                success: ['#00ff88', '#39ff14'],
                warning: ['#ffaa00', '#ff6600'],
                danger: ['#ff3366', '#cc0033']
            }
        };
    }

    // Initialiser tous les graphiques du dashboard
    initializeDashboardCharts(data) {
        this.createQualityTrendChart();
        this.createIssuesChart(data);
        this.createImpactChart(data);
        this.createColumnsChart(data);
        this.animateKPIs();
    }

    // Graphique d'évolution de la qualité
    createQualityTrendChart() {
        const ctx = document.getElementById('quality-chart');
        if (!ctx) return;

        // Données simulées d'évolution de qualité
        const data = this.generateQualityTrendData();

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Score Qualité (%)',
                    data: data.values,
                    borderColor: this.defaultColors.primary,
                    backgroundColor: this.createGradient(ctx, this.defaultColors.gradient.cyan),
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.defaultColors.primary,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
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
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: this.defaultColors.primary,
                        bodyColor: '#ffffff',
                        borderColor: this.defaultColors.primary,
                        borderWidth: 1,
                        cornerRadius: 10,
                        displayColors: false,
                        callbacks: {
                            label: (context) => `Qualité: ${context.parsed.y}%`
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 12
                            }
                        }
                    },
                    y: {
                        display: true,
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 12
                            },
                            callback: (value) => `${value}%`
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });

        this.charts.set('quality-trend', chart);
    }

    // Générer des données de tendance de qualité
    generateQualityTrendData() {
        const labels = [];
        const values = [];
        const now = new Date();

        for (let i = 29; i >= 0; i--) {
            const date = new Date(now);
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' }));
            
            // Simulation d'amélioration progressive
            const baseQuality = 65;
            const improvement = (30 - i) * 1.2;
            const randomVariation = (Math.random() - 0.5) * 5;
            values.push(Math.min(95, Math.max(50, baseQuality + improvement + randomVariation)));
        }

        return { labels, values };
    }

    // Graphique des problèmes par type
    createIssuesChart(data) {
        const ctx = document.getElementById('issues-chart');
        if (!ctx) return;

        const issuesData = this.analyzeIssuesByType(data);

        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: issuesData.labels,
                datasets: [{
                    data: issuesData.values,
                    backgroundColor: [
                        this.defaultColors.danger,
                        this.defaultColors.warning,
                        this.defaultColors.info,
                        this.defaultColors.secondary
                    ],
                    borderColor: '#1a1a1a',
                    borderWidth: 3,
                    hoverBorderWidth: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 12
                            },
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: this.defaultColors.primary,
                        bodyColor: '#ffffff',
                        borderColor: this.defaultColors.primary,
                        borderWidth: 1,
                        cornerRadius: 10,
                        callbacks: {
                            label: (context) => {
                                const percentage = ((context.parsed / issuesData.total) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 2000
                },
                cutout: '50%'
            }
        });

        this.charts.set('issues', chart);
    }

    // Analyser les problèmes par type
    analyzeIssuesByType(data) {
        const metadata = dataProcessor.metadata;
        
        const labels = ['Doublons', 'Valeurs manquantes', 'Anomalies', 'Format incorrect'];
        const values = [
            metadata.quality.duplicates || 147,
            metadata.quality.missingValues || 89,
            metadata.quality.anomalies || 23,
            Math.floor(Math.random() * 50) + 10
        ];

        return {
            labels,
            values,
            total: values.reduce((a, b) => a + b, 0)
        };
    }

    // Graphique d'impact du nettoyage
    createImpactChart(data) {
        const ctx = document.getElementById('impact-chart');
        if (!ctx) return;

        const impactData = this.calculateCleaningImpact(data);

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Avant nettoyage', 'Après nettoyage'],
                datasets: [
                    {
                        label: 'Qualité (%)',
                        data: impactData.quality,
                        backgroundColor: this.createGradient(ctx, this.defaultColors.gradient.success, true),
                        borderColor: this.defaultColors.success,
                        borderWidth: 2,
                        borderRadius: 8,
                        borderSkipped: false
                    },
                    {
                        label: 'Complétude (%)',
                        data: impactData.completeness,
                        backgroundColor: this.createGradient(ctx, this.defaultColors.gradient.cyan, true),
                        borderColor: this.defaultColors.primary,
                        borderWidth: 2,
                        borderRadius: 8,
                        borderSkipped: false
                    },
                    {
                        label: 'Cohérence (%)',
                        data: impactData.consistency,
                        backgroundColor: this.createGradient(ctx, this.defaultColors.gradient.magenta, true),
                        borderColor: this.defaultColors.secondary,
                        borderWidth: 2,
                        borderRadius: 8,
                        borderSkipped: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 12
                            },
                            padding: 15,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: this.defaultColors.primary,
                        bodyColor: '#ffffff',
                        borderColor: this.defaultColors.primary,
                        borderWidth: 1,
                        cornerRadius: 10,
                        callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.parsed.y}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 12
                            }
                        }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 12
                            },
                            callback: (value) => `${value}%`
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });

        this.charts.set('impact', chart);
    }

    // Calculer l'impact du nettoyage
    calculateCleaningImpact(data) {
        const metadata = dataProcessor.metadata;
        const currentQuality = metadata.quality.score;
        
        return {
            quality: [currentQuality - 18, currentQuality], // Avant/Après
            completeness: [currentQuality - 15, Math.min(95, currentQuality + 5)],
            consistency: [currentQuality - 12, Math.min(98, currentQuality + 8)]
        };
    }

    // Graphique qualité par colonne
    createColumnsChart(data) {
        const ctx = document.getElementById('columns-chart');
        if (!ctx) return;

        const columnsData = this.analyzeColumnQuality(data);

        const chart = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: columnsData.labels,
                datasets: [{
                    label: 'Qualité (%)',
                    data: columnsData.values,
                    backgroundColor: columnsData.values.map(value => {
                        if (value >= 90) return this.defaultColors.success;
                        if (value >= 70) return this.defaultColors.warning;
                        return this.defaultColors.danger;
                    }),
                    borderColor: columnsData.values.map(value => {
                        if (value >= 90) return this.defaultColors.success;
                        if (value >= 70) return this.defaultColors.warning;
                        return this.defaultColors.danger;
                    }),
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: this.defaultColors.primary,
                        bodyColor: '#ffffff',
                        borderColor: this.defaultColors.primary,
                        borderWidth: 1,
                        cornerRadius: 10,
                        callbacks: {
                            label: (context) => `Qualité: ${context.parsed.x}%`
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 10
                            },
                            callback: (value) => `${value}%`
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#cccccc',
                            font: {
                                family: 'Rajdhani',
                                size: 10
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });

        this.charts.set('columns', chart);
    }

    // Analyser la qualité par colonne
    analyzeColumnQuality(data) {
        const metadata = dataProcessor.metadata;
        const labels = [];
        const values = [];

        if (metadata.columns && metadata.columns.length > 0) {
            metadata.columns.slice(0, 8).forEach(column => {
                labels.push(column.name.length > 12 ? column.name.substring(0, 12) + '...' : column.name);
                values.push(parseFloat(column.stats.completeness));
            });
        } else {
            // Données par défaut si pas de métadonnées
            const defaultColumns = ['ID', 'Nom', 'Email', 'Téléphone', 'Adresse', 'Date', 'Statut', 'Score'];
            const defaultValues = [98, 95, 87, 76, 82, 93, 89, 71];
            
            labels.push(...defaultColumns);
            values.push(...defaultValues);
        }

        return { labels, values };
    }

    // Créer un gradient
    createGradient(ctx, colors, vertical = false) {
        const gradient = ctx.createLinearGradient(0, 0, vertical ? 0 : ctx.canvas.width, vertical ? ctx.canvas.height : 0);
        gradient.addColorStop(0, colors[0]);
        gradient.addColorStop(1, colors[1]);
        return gradient;
    }

    // Animer les KPIs
    animateKPIs() {
        this.animateKPI('quality-score', 92, '%');
        this.animateKPI('anomalies-count', 147, '');
        this.animateKPI('clean-records', 8426, '');
        this.animateKPI('confidence-kpi', 97, '%');
    }

    // Animer un KPI spécifique
    animateKPI(elementId, targetValue, suffix = '') {
        const element = document.getElementById(elementId);
        if (!element) return;

        const duration = 2000;
        const startTime = Date.now();
        const startValue = 0;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (easeOutQuart)
            const easeProgress = 1 - Math.pow(1 - progress, 4);
            
            const currentValue = Math.floor(startValue + (targetValue - startValue) * easeProgress);
            element.textContent = this.formatKPIValue(currentValue) + suffix;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    // Formater la valeur KPI
    formatKPIValue(value) {
        if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'k';
        }
        return value.toLocaleString('fr-FR');
    }

    // Mettre à jour les graphiques avec de nouvelles données
    updateCharts(newData) {
        this.charts.forEach((chart, key) => {
            switch (key) {
                case 'quality-trend':
                    this.updateQualityChart(chart, newData);
                    break;
                case 'issues':
                    this.updateIssuesChart(chart, newData);
                    break;
                case 'impact':
                    this.updateImpactChart(chart, newData);
                    break;
                case 'columns':
                    this.updateColumnsChart(chart, newData);
                    break;
            }
        });
    }

    // Mettre à jour le graphique de qualité
    updateQualityChart(chart, data) {
        const newData = this.generateQualityTrendData();
        chart.data.labels = newData.labels;
        chart.data.datasets[0].data = newData.values;
        chart.update('active');
    }

    // Mettre à jour le graphique des problèmes
    updateIssuesChart(chart, data) {
        const issuesData = this.analyzeIssuesByType(data);
        chart.data.datasets[0].data = issuesData.values;
        chart.update('active');
    }

    // Mettre à jour le graphique d'impact
    updateImpactChart(chart, data) {
        const impactData = this.calculateCleaningImpact(data);
        chart.data.datasets[0].data = impactData.quality;
        chart.data.datasets[1].data = impactData.completeness;
        chart.data.datasets[2].data = impactData.consistency;
        chart.update('active');
    }

    // Mettre à jour le graphique des colonnes
    updateColumnsChart(chart, data) {
        const columnsData = this.analyzeColumnQuality(data);
        chart.data.labels = columnsData.labels;
        chart.data.datasets[0].data = columnsData.values;
        chart.update('active');
    }

    // Changer la période du graphique de qualité
    changePeriod(period) {
        const chart = this.charts.get('quality-trend');
        if (!chart) return;

        let data;
        switch (period) {
            case '7d':
                data = this.generateQualityTrendData(7);
                break;
            case '30d':
                data = this.generateQualityTrendData(30);
                break;
            case '90d':
                data = this.generateQualityTrendData(90);
                break;
            default:
                data = this.generateQualityTrendData();
        }

        chart.data.labels = data.labels;
        chart.data.datasets[0].data = data.values;
        chart.update('active');
    }

    // Détruire tous les graphiques
    destroyAllCharts() {
        this.charts.forEach(chart => {
            chart.destroy();
        });
        this.charts.clear();
    }

    // Redimensionner les graphiques
    resizeCharts() {
        this.charts.forEach(chart => {
            chart.resize();
        });
    }

    // Exporter un graphique en image
    exportChart(chartKey, filename = 'chart.png') {
        const chart = this.charts.get(chartKey);
        if (!chart) return;

        const url = chart.toBase64Image();
        const link = document.createElement('a');
        link.download = filename;
        link.href = url;
        link.click();
    }

    // Obtenir les données de tous les graphiques pour export
    getChartsData() {
        const data = {};
        
        this.charts.forEach((chart, key) => {
            data[key] = {
                labels: chart.data.labels,
                datasets: chart.data.datasets.map(dataset => ({
                    label: dataset.label,
                    data: dataset.data
                }))
            };
        });

        return data;
    }
}

// Instance globale du gestionnaire de graphiques
window.chartManager = new ChartManager();