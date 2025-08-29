// Application principale - AI DataClean Pro
class App {
    constructor() {
        this.currentSection = 'upload';
        this.isProcessing = false;
        this.sidebarOpen = false;
        this.currentData = null;
        this.currentPage = 1;
        this.pageSize = 100;
        this.processingMetrics = {
            processedRows: 0,
            duplicatesFound: 0,
            missingFilled: 0,
            anomaliesDetected: 0
        };
        
        this.init();
    }

    // Initialiser l'application
    async init() {
        try {
            // Afficher le loading
            this.showLoadingOverlay();
            
            // Initialiser les composants
            await this.initializeComponents();
            
            // Configurer les événements
            this.setupEventListeners();
            
            // Initialiser les traductions
            i18n.init();
            
            // Créer les particules d'arrière-plan
            this.createParticles();
            
            // Masquer le loading après animation
            setTimeout(() => {
                this.hideLoadingOverlay();
            }, 2000);
            
        } catch (error) {
            console.error('Erreur lors de l\'initialisation:', error);
            this.hideLoadingOverlay();
        }
    }

    // Initialiser les composants
    async initializeComponents() {
        // Les instances sont déjà créées dans les fichiers individuels
        console.log('Composants initialisés:', {
            dataProcessor: !!window.dataProcessor,
            aiEngine: !!window.aiEngine,
            chartManager: !!window.chartManager,
            chatAI: !!window.chatAI,
            i18n: !!window.i18n
        });
    }

    // Configurer les écouteurs d'événements
    setupEventListeners() {
        // Navigation
        this.setupNavigation();
        
        // Upload
        this.setupFileUpload();
        
        // Sample data
        this.setupSampleData();
        
        // Processing
        this.setupProcessing();
        
        // Dashboard
        this.setupDashboard();
        
        // Export
        this.setupExport();
        
        // Demo
        this.setupDemo();
        
        // Responsive
        this.setupResponsive();
    }

    // Configuration de la navigation
    setupNavigation() {
        // Sidebar toggle
        const sidebarToggle = document.getElementById('sidebar-toggle');
        const sidebar = document.getElementById('sidebar');
        
        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                this.toggleSidebar();
            });
        }

        // Menu items
        const menuItems = document.querySelectorAll('.menu-item');
        menuItems.forEach(item => {
            item.addEventListener('click', () => {
                const section = item.getAttribute('data-section');
                if (section) {
                    this.navigateToSection(section);
                }
            });
        });

        // Section buttons
        const backToUploadBtn = document.getElementById('back-to-upload');
        const startCleaningBtn = document.getElementById('start-cleaning');

        if (backToUploadBtn) {
            backToUploadBtn.addEventListener('click', () => {
                this.navigateToSection('upload');
            });
        }

        if (startCleaningBtn) {
            startCleaningBtn.addEventListener('click', () => {
                this.startProcessing();
            });
        }
    }

    // Configuration de l'upload
    setupFileUpload() {
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');

        if (uploadZone && fileInput) {
            // Drag & Drop
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });

            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });

            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            });

            // Click to upload
            uploadZone.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }
    }

    // Configuration des données d'exemple
    setupSampleData() {
        const sampleBtns = document.querySelectorAll('.sample-btn');
        sampleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const sampleType = btn.getAttribute('data-sample');
                this.loadSampleData(sampleType);
            });
        });
    }

    // Configuration du traitement
    setupProcessing() {
        // Écouter les événements du moteur IA
        if (window.aiEngine) {
            aiEngine.on('processingStart', (data) => {
                this.onProcessingStart(data);
            });

            aiEngine.on('stepStart', (data) => {
                this.onStepStart(data);
            });

            aiEngine.on('stepProgress', (data) => {
                this.onStepProgress(data);
            });

            aiEngine.on('stepComplete', (data) => {
                this.onStepComplete(data);
            });

            aiEngine.on('processingComplete', (data) => {
                this.onProcessingComplete(data);
            });

            aiEngine.on('status', (message) => {
                this.updateProcessingStatus(message);
            });
        }
    }

    // Configuration du dashboard
    setupDashboard() {
        // Contrôles des graphiques
        const chartBtns = document.querySelectorAll('.chart-btn');
        chartBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Retirer la classe active de tous les boutons
                chartBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const period = btn.getAttribute('data-period');
                if (period && window.chartManager) {
                    chartManager.changePeriod(period);
                }
            });
        });
    }

    // Configuration de l'export
    setupExport() {
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.showExportModal();
            });
        }
    }

    // Configuration du mode démo
    setupDemo() {
        const demoBtn = document.getElementById('demo-btn');
        if (demoBtn) {
            demoBtn.addEventListener('click', () => {
                this.startDemo();
            });
        }
    }

    // Configuration responsive
    setupResponsive() {
        window.addEventListener('resize', () => {
            if (window.chartManager) {
                chartManager.resizeCharts();
            }
        });
    }

    // Gestion de l'upload de fichier
    async handleFileUpload(file) {
        try {
            // Vérifications
            if (file.size > 50 * 1024 * 1024) { // 50MB
                this.showNotification(i18n.t('file_too_large'), 'error');
                return;
            }

            const allowedTypes = ['.csv', '.xlsx', '.xls', '.json', '.xml', '.txt'];
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            
            if (!allowedTypes.includes(fileExtension)) {
                this.showNotification(i18n.t('invalid_format'), 'error');
                return;
            }

            // Afficher un indicateur de chargement
            this.showNotification('Traitement du fichier...', 'info');

            // Traiter le fichier
            await dataProcessor.processFile(file);
            
            // Mettre à jour l'interface
            this.updateFileInfo();
            this.updateDataPreview();
            
            // Naviguer vers la section preview
            this.navigateToSection('preview');
            
            this.showNotification('Fichier traité avec succès !', 'success');
            
        } catch (error) {
            console.error('Erreur lors du traitement du fichier:', error);
            this.showNotification(i18n.t('upload_error'), 'error');
        }
    }

    // Charger des données d'exemple
    async loadSampleData(type) {
        try {
            this.showNotification('Chargement des données d\'exemple...', 'info');
            
            const success = dataProcessor.loadSampleDataset(type);
            
            if (success) {
                this.updateFileInfo();
                this.updateDataPreview();
                this.navigateToSection('preview');
                this.showNotification('Données d\'exemple chargées !', 'success');
            } else {
                this.showNotification('Erreur lors du chargement', 'error');
            }
            
        } catch (error) {
            console.error('Erreur lors du chargement des données d\'exemple:', error);
            this.showNotification('Erreur lors du chargement', 'error');
        }
    }

    // Mettre à jour les informations du fichier
    updateFileInfo() {
        const metadata = dataProcessor.metadata;
        
        this.updateElement('file-name', metadata.fileName);
        this.updateElement('total-rows', this.formatNumber(metadata.totalRows));
        this.updateElement('total-columns', metadata.totalColumns);
        this.updateElement('file-size', this.formatFileSize(metadata.fileSize));
        
        // Mettre à jour les colonnes
        this.updateColumnsAnalysis();
        
        // Mettre à jour les status de validation
        const formatStatus = document.getElementById('format-status');
        const structureStatus = document.getElementById('structure-status');
        
        if (formatStatus) formatStatus.className = 'fas fa-check-circle';
        if (structureStatus) structureStatus.className = 'fas fa-check-circle';
    }

    // Mettre à jour l'analyse des colonnes
    updateColumnsAnalysis() {
        const columnsGrid = document.getElementById('columns-grid');
        if (!columnsGrid) return;

        columnsGrid.innerHTML = '';
        
        const metadata = dataProcessor.metadata;
        if (!metadata.columns) return;

        metadata.columns.forEach(column => {
            const card = document.createElement('div');
            card.className = 'column-card';
            
            card.innerHTML = `
                <div class="column-name">${column.name}</div>
                <div class="column-type">${this.getTypeLabel(column.type)}</div>
                <div class="column-stats">
                    <div>Complétude: ${column.stats.completeness}%</div>
                    <div>Valeurs uniques: ${column.stats.uniqueValues}</div>
                    ${column.stats.duplicates > 0 ? `<div>Doublons: ${column.stats.duplicates}</div>` : ''}
                </div>
            `;
            
            columnsGrid.appendChild(card);
        });
    }

    // Mettre à jour l'aperçu des données
    updateDataPreview() {
        const preview = dataProcessor.getDataPreview(this.currentPage, this.pageSize);
        
        // Mettre à jour les informations de pagination
        this.updateElement('rows-from', (this.currentPage - 1) * this.pageSize + 1);
        this.updateElement('rows-to', Math.min(this.currentPage * this.pageSize, preview.totalRows));
        this.updateElement('total-rows-info', this.formatNumber(preview.totalRows));
        
        // Mettre à jour le tableau
        this.updatePreviewTable(preview);
        
        // Mettre à jour la pagination
        this.updatePagination(preview);
    }

    // Mettre à jour le tableau d'aperçu
    updatePreviewTable(preview) {
        const tableHeader = document.getElementById('table-header');
        const tableBody = document.getElementById('table-body');
        
        if (!tableHeader || !tableBody) return;

        // Header
        tableHeader.innerHTML = '';
        const headerRow = document.createElement('tr');
        preview.headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        tableHeader.appendChild(headerRow);

        // Body
        tableBody.innerHTML = '';
        preview.rows.forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell || '';
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });
    }

    // Mettre à jour la pagination
    updatePagination(preview) {
        const pagination = document.getElementById('pagination');
        if (!pagination) return;

        pagination.innerHTML = '';

        // Bouton précédent
        const prevBtn = document.createElement('button');
        prevBtn.textContent = '‹';
        prevBtn.disabled = this.currentPage === 1;
        prevBtn.addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.updateDataPreview();
            }
        });
        pagination.appendChild(prevBtn);

        // Numéros de page
        const totalPages = preview.totalPages;
        const startPage = Math.max(1, this.currentPage - 2);
        const endPage = Math.min(totalPages, this.currentPage + 2);

        for (let i = startPage; i <= endPage; i++) {
            const pageBtn = document.createElement('button');
            pageBtn.textContent = i;
            pageBtn.className = i === this.currentPage ? 'active' : '';
            pageBtn.addEventListener('click', () => {
                this.currentPage = i;
                this.updateDataPreview();
            });
            pagination.appendChild(pageBtn);
        }

        // Bouton suivant
        const nextBtn = document.createElement('button');
        nextBtn.textContent = '›';
        nextBtn.disabled = this.currentPage === totalPages;
        nextBtn.addEventListener('click', () => {
            if (this.currentPage < totalPages) {
                this.currentPage++;
                this.updateDataPreview();
            }
        });
        pagination.appendChild(nextBtn);
    }

    // Démarrer le traitement
    async startProcessing() {
        if (!dataProcessor.rawData) {
            this.showNotification('Aucune donnée à traiter', 'error');
            return;
        }

        try {
            this.isProcessing = true;
            this.navigateToSection('processing');
            
            // Lancer l'analyse IA
            await aiEngine.analyzeAndClean(dataProcessor.rawData);
            
        } catch (error) {
            console.error('Erreur lors du traitement:', error);
            this.showNotification(i18n.t('processing_error'), 'error');
            this.isProcessing = false;
        }
    }

    // Événements du traitement
    onProcessingStart(data) {
        console.log('Traitement démarré:', data);
        this.resetProcessingMetrics();
    }

    onStepStart(data) {
        const step = document.getElementById(`step-${data.step + 1}`);
        if (step) {
            // Retirer les classes des autres étapes
            document.querySelectorAll('.step').forEach(s => {
                s.classList.remove('active');
            });
            
            step.classList.add('active');
            const status = step.querySelector('.step-status');
            if (status) {
                status.textContent = i18n.t('in_progress');
            }
        }
        
        // Simuler la mise à jour des métriques en temps réel
        this.startMetricsAnimation();
    }

    onStepProgress(data) {
        const progressBar = document.querySelector(`#step-${data.step + 1} .progress-fill`);
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
        }
        
        // Mettre à jour la progression globale
        const overallProgress = (data.step * 25) + (data.progress * 0.25);
        this.updateOverallProgress(overallProgress);
    }

    onStepComplete(data) {
        const step = document.getElementById(`step-${data.step + 1}`);
        if (step) {
            step.classList.remove('active');
            step.classList.add('completed');
            
            const status = step.querySelector('.step-status');
            if (status) {
                status.textContent = i18n.t('completed');
            }
        }
    }

    onProcessingComplete(data) {
        this.isProcessing = false;
        this.stopMetricsAnimation();
        
        // Finaliser les métriques
        this.finalizeMetrics();
        
        // Générer les insights
        this.generateInsights(data.insights);
        
        // Naviguer vers le dashboard
        setTimeout(() => {
            this.navigateToSection('dashboard');
            this.initializeDashboard();
        }, 2000);
    }

    // Initialiser le dashboard
    initializeDashboard() {
        if (window.chartManager) {
            chartManager.initializeDashboardCharts(dataProcessor.rawData);
        }
        
        this.renderInsights();
    }

    // Générer les insights
    generateInsights(insights) {
        const insightsGrid = document.getElementById('insights-grid');
        if (!insightsGrid || !insights) return;

        insightsGrid.innerHTML = '';

        insights.forEach(insight => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            
            card.innerHTML = `
                <div class="insight-header">
                    <div class="insight-icon">
                        <i class="${insight.icon}"></i>
                    </div>
                    <div class="insight-type">${insight.type}</div>
                </div>
                <div class="insight-title">${insight.title}</div>
                <div class="insight-description">${insight.description}</div>
                ${insight.metrics ? this.renderInsightMetrics(insight.metrics) : ''}
            `;
            
            insightsGrid.appendChild(card);
        });
    }

    // Rendre les métriques d'insight
    renderInsightMetrics(metrics) {
        const metricsHtml = Object.entries(metrics).map(([key, value]) => `
            <div class="insight-metric">
                <div class="insight-metric-value">${value}</div>
                <div class="insight-metric-label">${key}</div>
            </div>
        `).join('');
        
        return `<div class="insight-metrics">${metricsHtml}</div>`;
    }

    // Navigation entre sections
    navigateToSection(sectionName) {
        // Masquer toutes les sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // Afficher la section demandée
        const targetSection = document.getElementById(`${sectionName}-section`);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionName;
        }

        // Mettre à jour le menu
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.remove('active');
        });

        const menuItem = document.querySelector(`[data-section="${sectionName}"]`);
        if (menuItem) {
            menuItem.classList.add('active');
        }
    }

    // Toggle sidebar
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const main = document.querySelector('.main');
        
        if (sidebar && main) {
            this.sidebarOpen = !this.sidebarOpen;
            
            if (this.sidebarOpen) {
                sidebar.classList.add('open');
                main.classList.add('sidebar-open');
            } else {
                sidebar.classList.remove('open');
                main.classList.remove('sidebar-open');
            }
        }
    }

    // Afficher/masquer l'overlay de chargement
    showLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('hidden');
        }
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }

    // Créer les particules d'arrière-plan
    createParticles() {
        const particlesBg = document.getElementById('particles-bg');
        if (!particlesBg) return;

        // Créer des particules animées
        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.style.position = 'absolute';
            particle.style.width = Math.random() * 3 + 1 + 'px';
            particle.style.height = particle.style.width;
            particle.style.backgroundColor = Math.random() > 0.5 ? '#00ffff' : '#ff00ff';
            particle.style.borderRadius = '50%';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.opacity = Math.random() * 0.5 + 0.2;
            particle.style.animation = `particleFloat ${Math.random() * 10 + 10}s ease-in-out infinite`;
            particle.style.animationDelay = Math.random() * 5 + 's';
            
            particlesBg.appendChild(particle);
        }
    }

    // Gestion des métriques de traitement
    resetProcessingMetrics() {
        this.processingMetrics = {
            processedRows: 0,
            duplicatesFound: 0,
            missingFilled: 0,
            anomaliesDetected: 0
        };
        this.updateProcessingMetrics();
    }

    startMetricsAnimation() {
        const metadata = dataProcessor.metadata;
        const totalRows = metadata.totalRows || 8426;
        
        this.metricsInterval = setInterval(() => {
            if (this.processingMetrics.processedRows < totalRows) {
                this.processingMetrics.processedRows += Math.floor(Math.random() * 100) + 50;
                this.processingMetrics.duplicatesFound += Math.floor(Math.random() * 5);
                this.processingMetrics.missingFilled += Math.floor(Math.random() * 8);
                this.processingMetrics.anomaliesDetected += Math.floor(Math.random() * 2);
                
                this.updateProcessingMetrics();
            }
        }, 300);
    }

    stopMetricsAnimation() {
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
        }
    }

    finalizeMetrics() {
        const metadata = dataProcessor.metadata;
        this.processingMetrics = {
            processedRows: metadata.totalRows || 8426,
            duplicatesFound: metadata.quality?.duplicates || 147,
            missingFilled: metadata.quality?.missingValues || 89,
            anomaliesDetected: metadata.quality?.anomalies || 23
        };
        this.updateProcessingMetrics();
    }

    updateProcessingMetrics() {
        this.updateElement('processed-rows', this.formatNumber(this.processingMetrics.processedRows));
        this.updateElement('duplicates-found', this.formatNumber(this.processingMetrics.duplicatesFound));
        this.updateElement('missing-filled', this.formatNumber(this.processingMetrics.missingFilled));
        this.updateElement('anomalies-detected', this.formatNumber(this.processingMetrics.anomaliesDetected));
    }

    updateOverallProgress(progress) {
        const progressBar = document.getElementById('overall-progress');
        const progressPercent = document.getElementById('overall-percent');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }
    }

    updateProcessingStatus(message) {
        // Mettre à jour le status si nécessaire
        console.log('Status:', message);
    }

    // Modal d'export
    showExportModal() {
        // Créer un modal simple pour l'export
        const modal = document.createElement('div');
        modal.className = 'export-modal';
        modal.innerHTML = `
            <div class="modal-overlay">
                <div class="modal-content">
                    <h3>Exporter les données</h3>
                    <div class="export-options">
                        <button class="btn primary" data-format="csv">
                            <i class="fas fa-file-csv"></i>
                            CSV
                        </button>
                        <button class="btn primary" data-format="excel">
                            <i class="fas fa-file-excel"></i>
                            Excel
                        </button>
                        <button class="btn primary" data-format="json">
                            <i class="fas fa-file-code"></i>
                            JSON
                        </button>
                        <button class="btn primary" data-format="pdf">
                            <i class="fas fa-file-pdf"></i>
                            Rapport PDF
                        </button>
                    </div>
                    <button class="btn secondary close-modal">Fermer</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Événements
        modal.querySelectorAll('[data-format]').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.getAttribute('data-format');
                this.exportData(format);
                document.body.removeChild(modal);
            });
        });
        
        modal.querySelector('.close-modal').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal.querySelector('.modal-overlay')) {
                document.body.removeChild(modal);
            }
        });
    }

    // Exporter les données
    async exportData(format) {
        try {
            this.showNotification('Export en cours...', 'info');
            
            let blob;
            let filename;
            
            switch (format) {
                case 'csv':
                    blob = dataProcessor.exportCleanedData('csv');
                    filename = 'cleaned_data.csv';
                    break;
                case 'excel':
                    blob = dataProcessor.exportCleanedData('excel');
                    filename = 'cleaned_data.xlsx';
                    break;
                case 'json':
                    blob = dataProcessor.exportCleanedData('json');
                    filename = 'cleaned_data.json';
                    break;
                case 'pdf':
                    blob = await this.generatePDFReport();
                    filename = 'data_report.pdf';
                    break;
                default:
                    throw new Error('Format non supporté');
            }
            
            // Télécharger le fichier
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
            
            this.showNotification('Export réussi !', 'success');
            
        } catch (error) {
            console.error('Erreur lors de l\'export:', error);
            this.showNotification('Erreur lors de l\'export', 'error');
        }
    }

    // Générer un rapport PDF
    async generatePDFReport() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Titre
        doc.setFontSize(20);
        doc.text('Rapport de Nettoyage IA', 20, 30);
        
        // Métadonnées
        const metadata = dataProcessor.metadata;
        doc.setFontSize(12);
        doc.text(`Fichier: ${metadata.fileName}`, 20, 50);
        doc.text(`Lignes: ${metadata.totalRows}`, 20, 60);
        doc.text(`Colonnes: ${metadata.totalColumns}`, 20, 70);
        doc.text(`Score qualité: ${metadata.quality?.score || 'N/A'}%`, 20, 80);
        
        // Plus de contenu...
        doc.text('Résumé du traitement:', 20, 100);
        doc.text(`• Doublons supprimés: ${metadata.quality?.duplicates || 0}`, 30, 110);
        doc.text(`• Valeurs manquantes traitées: ${metadata.quality?.missingValues || 0}`, 30, 120);
        doc.text(`• Anomalies corrigées: ${metadata.quality?.anomalies || 0}`, 30, 130);
        
        return doc.output('blob');
    }

    // Mode démo
    async startDemo() {
        this.showNotification('Démarrage de la démo...', 'info');
        
        // Charger des données d'exemple
        await this.loadSampleData('customers');
        
        // Auto-play du processus
        setTimeout(() => {
            this.startProcessing();
        }, 1000);
    }

    // Utilitaires
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    formatNumber(number) {
        return new Intl.NumberFormat('fr-FR').format(number);
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    getTypeLabel(type) {
        const labels = {
            'text': 'Texte',
            'number': 'Nombre',
            'date': 'Date',
            'email': 'Email',
            'phone': 'Téléphone',
            'url': 'URL',
            'boolean': 'Booléen'
        };
        return labels[type] || type;
    }

    showNotification(message, type = 'info') {
        // Créer une notification simple
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: ${type === 'success' ? '#00ff88' : type === 'error' ? '#ff3366' : '#3366ff'};
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    renderInsights() {
        // Cette méthode sera appelée pour rendre les insights dans le dashboard
        if (aiEngine.insights && aiEngine.insights.length > 0) {
            this.generateInsights(aiEngine.insights);
        }
    }
}

// Initialiser l'application quand le DOM est chargé
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});

// CSS pour les notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .export-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 10000;
    }
    
    .modal-overlay {
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal-content {
        background: var(--card-bg);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid var(--border-glow);
        max-width: 500px;
        width: 90%;
        text-align: center;
    }
    
    .modal-content h3 {
        margin-bottom: 30px;
        color: var(--primary-cyan);
        font-family: var(--font-primary);
    }
    
    .export-options {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-bottom: 30px;
    }
    
    .export-options .btn {
        padding: 15px;
        flex-direction: column;
        gap: 8px;
    }
    
    .export-options .btn i {
        font-size: 24px;
    }
`;

document.head.appendChild(notificationStyles);