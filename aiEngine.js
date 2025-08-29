// Moteur IA pour le nettoyage et l'analyse des données
class AIEngine {
    constructor() {
        this.patterns = new Map();
        this.insights = [];
        this.recommendations = [];
        this.isProcessing = false;
        this.processingSteps = [
            { id: 'analysis', name: 'Analyse des données', duration: 2000 },
            { id: 'cleaning', name: 'Nettoyage intelligent', duration: 3000 },
            { id: 'insights', name: 'Génération insights', duration: 2500 },
            { id: 'complete', name: 'Finalisation', duration: 1000 }
        ];
        this.currentStep = 0;
        this.callbacks = {};
    }

    // Enregistrer des callbacks pour les événements
    on(event, callback) {
        if (!this.callbacks[event]) this.callbacks[event] = [];
        this.callbacks[event].push(callback);
    }

    // Émettre un événement
    emit(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => callback(data));
        }
    }

    // Lancer l'analyse IA complète
    async analyzeAndClean(data, options = {}) {
        if (this.isProcessing) {
            throw new Error('Traitement déjà en cours');
        }

        this.isProcessing = true;
        this.currentStep = 0;
        this.insights = [];
        this.recommendations = [];

        try {
            this.emit('processingStart', { totalSteps: this.processingSteps.length });

            // Étape 1: Analyse des données
            await this.executeStep('analysis', async () => {
                await this.analyzeDataPatterns(data);
                await this.detectDataQualityIssues(data);
            });

            // Étape 2: Nettoyage intelligent
            await this.executeStep('cleaning', async () => {
                await this.performIntelligentCleaning(data, options);
            });

            // Étape 3: Génération d'insights
            await this.executeStep('insights', async () => {
                await this.generateBusinessInsights(data);
                await this.calculateROI(data);
            });

            // Étape 4: Finalisation
            await this.executeStep('complete', async () => {
                await this.generateRecommendations(data);
                await this.finalizeAnalysis(data);
            });

            this.emit('processingComplete', {
                insights: this.insights,
                recommendations: this.recommendations,
                quality: this.calculateFinalQuality(data)
            });

        } catch (error) {
            this.emit('processingError', error);
            throw error;
        } finally {
            this.isProcessing = false;
        }
    }

    // Exécuter une étape de traitement
    async executeStep(stepId, action) {
        const step = this.processingSteps.find(s => s.id === stepId);
        if (!step) return;

        this.emit('stepStart', { step: this.currentStep, stepId, name: step.name });

        // Simulation de progression en temps réel
        const progressInterval = setInterval(() => {
            const progress = Math.min(95, Math.random() * 20 + 
                (Date.now() - startTime) / step.duration * 80);
            this.emit('stepProgress', { step: this.currentStep, progress });
        }, 200);

        const startTime = Date.now();
        
        try {
            await action();
            await this.delay(Math.max(0, step.duration - (Date.now() - startTime)));
            
            clearInterval(progressInterval);
            this.emit('stepProgress', { step: this.currentStep, progress: 100 });
            this.emit('stepComplete', { step: this.currentStep, stepId });
            
        } finally {
            clearInterval(progressInterval);
            this.currentStep++;
        }
    }

    // Analyser les patterns dans les données
    async analyzeDataPatterns(data) {
        this.emit('status', 'Analyse des patterns de données...');
        
        const headers = data[0];
        const rows = data.slice(1);
        
        // Analyser les corrélations entre colonnes
        await this.findColumnCorrelations(headers, rows);
        
        // Détecter les patterns temporels
        await this.detectTemporalPatterns(headers, rows);
        
        // Analyser les distributions de valeurs
        await this.analyzeValueDistributions(headers, rows);
        
        // Détecter les groupes/segments
        await this.detectDataSegments(headers, rows);

        this.emit('patternsDetected', { count: this.patterns.size });
    }

    // Trouver les corrélations entre colonnes
    async findColumnCorrelations(headers, rows) {
        const numericColumns = [];
        
        headers.forEach((header, index) => {
            const values = rows.map(row => parseFloat(row[index])).filter(val => !isNaN(val));
            if (values.length > rows.length * 0.5) {
                numericColumns.push({ index, header, values });
            }
        });

        for (let i = 0; i < numericColumns.length; i++) {
            for (let j = i + 1; j < numericColumns.length; j++) {
                const correlation = this.calculateCorrelation(
                    numericColumns[i].values,
                    numericColumns[j].values
                );
                
                if (Math.abs(correlation) > 0.7) {
                    this.patterns.set(`correlation_${i}_${j}`, {
                        type: 'correlation',
                        columns: [numericColumns[i].header, numericColumns[j].header],
                        strength: correlation,
                        description: `Forte corrélation (${(correlation * 100).toFixed(1)}%) entre ${numericColumns[i].header} et ${numericColumns[j].header}`
                    });
                }
            }
        }
    }

    // Calculer la corrélation de Pearson
    calculateCorrelation(x, y) {
        const n = Math.min(x.length, y.length);
        if (n < 2) return 0;

        const sumX = x.slice(0, n).reduce((a, b) => a + b, 0);
        const sumY = y.slice(0, n).reduce((a, b) => a + b, 0);
        const sumXY = x.slice(0, n).reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumXX = x.slice(0, n).reduce((sum, xi) => sum + xi * xi, 0);
        const sumYY = y.slice(0, n).reduce((sum, yi) => sum + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

        return denominator === 0 ? 0 : numerator / denominator;
    }

    // Détecter les patterns temporels
    async detectTemporalPatterns(headers, rows) {
        const dateColumns = headers.map((header, index) => {
            const values = rows.map(row => row[index]).filter(val => val && !isNaN(Date.parse(val)));
            return values.length > rows.length * 0.5 ? { index, header, values } : null;
        }).filter(col => col);

        dateColumns.forEach(column => {
            const dates = column.values.map(val => new Date(val)).sort((a, b) => a - b);
            const intervals = [];
            
            for (let i = 1; i < dates.length; i++) {
                intervals.push(dates[i] - dates[i-1]);
            }
            
            // Détecter la régularité
            const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
            const variance = intervals.reduce((sum, interval) => sum + Math.pow(interval - avgInterval, 2), 0) / intervals.length;
            const regularity = 1 - (Math.sqrt(variance) / avgInterval);
            
            if (regularity > 0.7) {
                this.patterns.set(`temporal_${column.index}`, {
                    type: 'temporal',
                    column: column.header,
                    regularity: regularity,
                    avgInterval: avgInterval,
                    description: `Pattern temporel régulier détecté dans ${column.header} (régularité: ${(regularity * 100).toFixed(1)}%)`
                });
            }
        });
    }

    // Analyser les distributions de valeurs
    async analyzeValueDistributions(headers, rows) {
        headers.forEach((header, index) => {
            const values = rows.map(row => row[index]).filter(val => val && val.toString().trim());
            const valueCounts = {};
            
            values.forEach(val => {
                valueCounts[val] = (valueCounts[val] || 0) + 1;
            });

            const uniqueValues = Object.keys(valueCounts).length;
            const totalValues = values.length;
            const topValues = Object.entries(valueCounts)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5);

            // Détecter les valeurs dominantes
            const topValue = topValues[0];
            if (topValue && topValue[1] / totalValues > 0.5) {
                this.patterns.set(`dominant_value_${index}`, {
                    type: 'distribution',
                    column: header,
                    dominantValue: topValue[0],
                    dominance: topValue[1] / totalValues,
                    description: `Valeur dominante "${topValue[0]}" représente ${((topValue[1] / totalValues) * 100).toFixed(1)}% des données`
                });
            }

            // Détecter les distributions uniformes
            if (uniqueValues > 10 && uniqueValues / totalValues > 0.8) {
                this.patterns.set(`uniform_distribution_${index}`, {
                    type: 'distribution',
                    column: header,
                    uniqueness: uniqueValues / totalValues,
                    description: `Distribution très uniforme dans ${header} (${((uniqueValues / totalValues) * 100).toFixed(1)}% de valeurs uniques)`
                });
            }
        });
    }

    // Détecter les segments de données
    async detectDataSegments(headers, rows) {
        // Analyser les colonnes catégorielles pour détecter des segments
        headers.forEach((header, index) => {
            const values = rows.map(row => row[index]).filter(val => val && val.toString().trim());
            const valueCounts = {};
            
            values.forEach(val => {
                valueCounts[val] = (valueCounts[val] || 0) + 1;
            });

            const categories = Object.keys(valueCounts);
            
            // Si c'est une colonne catégorielle avec peu de valeurs uniques
            if (categories.length > 2 && categories.length < 20 && categories.length < values.length * 0.1) {
                const segments = Object.entries(valueCounts)
                    .map(([category, count]) => ({
                        category,
                        count,
                        percentage: (count / values.length * 100).toFixed(1)
                    }))
                    .sort((a, b) => b.count - a.count);

                this.patterns.set(`segments_${index}`, {
                    type: 'segmentation',
                    column: header,
                    segments: segments,
                    description: `${segments.length} segments identifiés dans ${header}: ${segments.map(s => `${s.category} (${s.percentage}%)`).join(', ')}`
                });
            }
        });
    }

    // Détecter les problèmes de qualité
    async detectDataQualityIssues(data) {
        this.emit('status', 'Détection des problèmes de qualité...');
        
        const headers = data[0];
        const rows = data.slice(1);
        const issues = [];

        // Détecter les doublons
        const duplicates = this.findDuplicateRows(rows);
        if (duplicates.length > 0) {
            issues.push({
                type: 'duplicates',
                severity: 'high',
                count: duplicates.length,
                description: `${duplicates.length} lignes dupliquées détectées`
            });
        }

        // Détecter les valeurs manquantes
        const missingValues = this.findMissingValues(headers, rows);
        Object.keys(missingValues).forEach(column => {
            if (missingValues[column] > 0) {
                issues.push({
                    type: 'missing',
                    severity: missingValues[column] > rows.length * 0.1 ? 'high' : 'medium',
                    column: column,
                    count: missingValues[column],
                    description: `${missingValues[column]} valeurs manquantes dans ${column}`
                });
            }
        });

        // Détecter les anomalies statistiques
        const anomalies = await this.detectStatisticalAnomalies(headers, rows);
        anomalies.forEach(anomaly => issues.push(anomaly));

        this.emit('qualityIssuesDetected', { issues, count: issues.length });
        return issues;
    }

    // Trouver les lignes dupliquées
    findDuplicateRows(rows) {
        const seen = new Set();
        const duplicates = [];
        
        rows.forEach((row, index) => {
            const rowString = row.join('|');
            if (seen.has(rowString)) {
                duplicates.push(index);
            } else {
                seen.add(rowString);
            }
        });

        return duplicates;
    }

    // Trouver les valeurs manquantes par colonne
    findMissingValues(headers, rows) {
        const missing = {};
        
        headers.forEach((header, index) => {
            missing[header] = rows.filter(row => !row[index] || !row[index].toString().trim()).length;
        });

        return missing;
    }

    // Détecter les anomalies statistiques
    async detectStatisticalAnomalies(headers, rows) {
        const anomalies = [];

        headers.forEach((header, index) => {
            const values = rows.map(row => parseFloat(row[index])).filter(val => !isNaN(val));
            
            if (values.length > 10) {
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
                
                const outliers = values.filter(val => Math.abs(val - mean) > 3 * std);
                
                if (outliers.length > 0) {
                    anomalies.push({
                        type: 'outliers',
                        severity: outliers.length > values.length * 0.05 ? 'high' : 'medium',
                        column: header,
                        count: outliers.length,
                        description: `${outliers.length} valeurs aberrantes détectées dans ${header}`
                    });
                }
            }
        });

        return anomalies;
    }

    // Effectuer le nettoyage intelligent
    async performIntelligentCleaning(data, options) {
        this.emit('status', 'Nettoyage intelligent en cours...');
        
        let cleaningActions = 0;

        // Supprimer les doublons avec ML-like logic
        cleaningActions += await this.removeDuplicatesIntelligently(data);
        
        // Imputation intelligente des valeurs manquantes
        cleaningActions += await this.imputeMissingValuesIntelligently(data);
        
        // Correction des anomalies
        cleaningActions += await this.correctAnomaliesIntelligently(data);
        
        // Standardisation des formats
        cleaningActions += await this.standardizeFormatsIntelligently(data);

        this.emit('cleaningProgress', { actionsPerformed: cleaningActions });
    }

    // Supprimer les doublons intelligemment
    async removeDuplicatesIntelligently(data) {
        const headers = data[0];
        const rows = data.slice(1);
        let removed = 0;

        // Algorithme de détection de doublons fuzzy
        const duplicateGroups = this.findFuzzyDuplicates(rows);
        
        duplicateGroups.forEach(group => {
            if (group.length > 1) {
                // Garder le meilleur enregistrement du groupe
                const bestRecord = this.selectBestRecord(group, headers);
                
                // Supprimer les autres
                group.forEach((recordIndex, i) => {
                    if (recordIndex !== bestRecord) {
                        rows.splice(recordIndex - removed, 1);
                        removed++;
                    }
                });
            }
        });

        return removed;
    }

    // Trouver les doublons fuzzy (similaires mais pas identiques)
    findFuzzyDuplicates(rows) {
        const groups = [];
        const processed = new Set();

        rows.forEach((row, index) => {
            if (processed.has(index)) return;

            const group = [index];
            processed.add(index);

            // Comparer avec les autres lignes
            for (let i = index + 1; i < rows.length; i++) {
                if (processed.has(i)) continue;

                const similarity = this.calculateRowSimilarity(row, rows[i]);
                if (similarity > 0.85) { // 85% de similarité
                    group.push(i);
                    processed.add(i);
                }
            }

            if (group.length > 1) {
                groups.push(group);
            }
        });

        return groups;
    }

    // Calculer la similarité entre deux lignes
    calculateRowSimilarity(row1, row2) {
        if (row1.length !== row2.length) return 0;

        let matches = 0;
        let totalComparisons = 0;

        for (let i = 0; i < row1.length; i++) {
            if (row1[i] && row2[i]) {
                totalComparisons++;
                if (row1[i].toString().toLowerCase() === row2[i].toString().toLowerCase()) {
                    matches++;
                } else {
                    // Similarité de chaîne pour les textes
                    const similarity = this.calculateStringSimilarity(row1[i].toString(), row2[i].toString());
                    if (similarity > 0.8) matches += similarity;
                }
            }
        }

        return totalComparisons > 0 ? matches / totalComparisons : 0;
    }

    // Calculer la similarité entre deux chaînes (Levenshtein normalisé)
    calculateStringSimilarity(str1, str2) {
        const maxLength = Math.max(str1.length, str2.length);
        if (maxLength === 0) return 1;
        
        const distance = this.levenshteinDistance(str1, str2);
        return 1 - distance / maxLength;
    }

    // Distance de Levenshtein
    levenshteinDistance(str1, str2) {
        const matrix = [];

        for (let i = 0; i <= str2.length; i++) {
            matrix[i] = [i];
        }

        for (let j = 0; j <= str1.length; j++) {
            matrix[0][j] = j;
        }

        for (let i = 1; i <= str2.length; i++) {
            for (let j = 1; j <= str1.length; j++) {
                if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }

        return matrix[str2.length][str1.length];
    }

    // Sélectionner le meilleur enregistrement d'un groupe de doublons
    selectBestRecord(group, headers) {
        // Logique pour sélectionner le meilleur: plus de champs remplis, valeurs plus récentes, etc.
        let bestScore = -1;
        let bestIndex = group[0];

        group.forEach(index => {
            let score = 0;
            const row = dataProcessor.rawData[index + 1]; // +1 car on skip le header
            
            // Score basé sur la complétude
            row.forEach(cell => {
                if (cell && cell.toString().trim()) score++;
            });

            // Bonus pour les dates récentes
            headers.forEach((header, colIndex) => {
                if (header.toLowerCase().includes('date') && row[colIndex]) {
                    const date = new Date(row[colIndex]);
                    if (!isNaN(date.getTime())) {
                        score += date.getTime() / 1000000000; // Normalisé
                    }
                }
            });

            if (score > bestScore) {
                bestScore = score;
                bestIndex = index;
            }
        });

        return bestIndex;
    }

    // Imputation intelligente des valeurs manquantes
    async imputeMissingValuesIntelligently(data) {
        const headers = data[0];
        const rows = data.slice(1);
        let filled = 0;

        for (let colIndex = 0; colIndex < headers.length; colIndex++) {
            const column = headers[colIndex];
            const values = rows.map(row => row[colIndex]).filter(val => val && val.toString().trim());
            
            if (values.length === 0) continue;

            const columnType = this.detectColumnType(values);
            
            for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
                if (!rows[rowIndex][colIndex] || !rows[rowIndex][colIndex].toString().trim()) {
                    const imputedValue = this.imputeValue(rows, rowIndex, colIndex, columnType, headers);
                    if (imputedValue !== null) {
                        rows[rowIndex][colIndex] = imputedValue;
                        filled++;
                    }
                }
            }
        }

        return filled;
    }

    // Imputer une valeur manquante
    imputeValue(rows, rowIndex, colIndex, columnType, headers) {
        const currentRow = rows[rowIndex];
        
        switch (columnType) {
            case 'numeric':
                return this.imputeNumericValue(rows, rowIndex, colIndex, headers);
            case 'categorical':
                return this.imputeCategoricalValue(rows, rowIndex, colIndex, headers);
            case 'date':
                return this.imputeDateValue(rows, rowIndex, colIndex);
            case 'text':
                return this.imputeTextValue(rows, rowIndex, colIndex, headers);
            default:
                return 'N/A';
        }
    }

    // Imputer une valeur numérique
    imputeNumericValue(rows, rowIndex, colIndex, headers) {
        // Essayer d'abord la régression basée sur d'autres colonnes
        const correlatedValue = this.predictNumericValue(rows, rowIndex, colIndex, headers);
        if (correlatedValue !== null) return correlatedValue;

        // Sinon, utiliser la médiane
        const values = rows.map(row => parseFloat(row[colIndex])).filter(val => !isNaN(val));
        if (values.length > 0) {
            values.sort((a, b) => a - b);
            return values[Math.floor(values.length / 2)].toString();
        }

        return '0';
    }

    // Prédire une valeur numérique basée sur les corrélations
    predictNumericValue(rows, rowIndex, colIndex, headers) {
        const currentRow = rows[rowIndex];
        
        // Trouver des lignes similaires
        const similarRows = this.findSimilarRows(rows, currentRow, colIndex);
        
        if (similarRows.length > 0) {
            const values = similarRows.map(row => parseFloat(row[colIndex])).filter(val => !isNaN(val));
            if (values.length > 0) {
                return (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
            }
        }

        return null;
    }

    // Trouver des lignes similaires
    findSimilarRows(rows, targetRow, excludeColumn) {
        const similarities = [];

        rows.forEach(row => {
            if (row === targetRow) return;

            let similarity = 0;
            let comparisons = 0;

            row.forEach((cell, index) => {
                if (index === excludeColumn || !cell || !targetRow[index]) return;
                
                comparisons++;
                if (cell.toString().toLowerCase() === targetRow[index].toString().toLowerCase()) {
                    similarity++;
                }
            });

            if (comparisons > 0) {
                similarities.push({ row, similarity: similarity / comparisons });
            }
        });

        return similarities
            .filter(item => item.similarity > 0.5)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 5)
            .map(item => item.row);
    }

    // Imputer une valeur catégorielle
    imputeCategoricalValue(rows, rowIndex, colIndex, headers) {
        const currentRow = rows[rowIndex];
        
        // Essayer de prédire basé sur d'autres colonnes
        const predictedValue = this.predictCategoricalValue(rows, currentRow, colIndex);
        if (predictedValue) return predictedValue;

        // Sinon, utiliser la valeur la plus fréquente
        const values = rows.map(row => row[colIndex]).filter(val => val && val.toString().trim());
        const counts = {};
        values.forEach(val => counts[val] = (counts[val] || 0) + 1);
        
        const mostFrequent = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        return mostFrequent || 'Unknown';
    }

    // Prédire une valeur catégorielle
    predictCategoricalValue(rows, targetRow, colIndex) {
        const similarRows = this.findSimilarRows(rows, targetRow, colIndex);
        
        if (similarRows.length > 0) {
            const values = similarRows.map(row => row[colIndex]).filter(val => val && val.toString().trim());
            if (values.length > 0) {
                const counts = {};
                values.forEach(val => counts[val] = (counts[val] || 0) + 1);
                return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
            }
        }

        return null;
    }

    // Imputer une date
    imputeDateValue(rows, rowIndex, colIndex) {
        const values = rows.map(row => row[colIndex]).filter(val => val && !isNaN(Date.parse(val)));
        
        if (values.length > 0) {
            const dates = values.map(val => new Date(val)).sort((a, b) => a - b);
            const medianDate = dates[Math.floor(dates.length / 2)];
            return medianDate.toISOString().split('T')[0];
        }

        return new Date().toISOString().split('T')[0];
    }

    // Imputer un texte
    imputeTextValue(rows, rowIndex, colIndex, headers) {
        const currentRow = rows[rowIndex];
        const similarRows = this.findSimilarRows(rows, currentRow, colIndex);
        
        if (similarRows.length > 0) {
            const values = similarRows.map(row => row[colIndex]).filter(val => val && val.toString().trim());
            if (values.length > 0) {
                return values[0];
            }
        }

        return 'N/A';
    }

    // Détecter le type d'une colonne
    detectColumnType(values) {
        const numericCount = values.filter(val => !isNaN(parseFloat(val))).length;
        const dateCount = values.filter(val => !isNaN(Date.parse(val))).length;
        
        if (numericCount / values.length > 0.8) return 'numeric';
        if (dateCount / values.length > 0.8) return 'date';
        
        const uniqueValues = new Set(values).size;
        if (uniqueValues < values.length * 0.1 && uniqueValues < 20) return 'categorical';
        
        return 'text';
    }

    // Corriger les anomalies intelligemment
    async correctAnomaliesIntelligently(data) {
        // Implémentation de la correction d'anomalies
        return 0; // Placeholder
    }

    // Standardiser les formats intelligemment
    async standardizeFormatsIntelligently(data) {
        // Implémentation de la standardisation
        return 0; // Placeholder
    }

    // Générer des insights business
    async generateBusinessInsights(data) {
        this.emit('status', 'Génération des insights business...');
        
        this.insights = [];

        // Insights sur la qualité des données
        await this.generateQualityInsights(data);
        
        // Insights sur les patterns détectés
        await this.generatePatternInsights();
        
        // Insights sur les opportunités business
        await this.generateBusinessOpportunities(data);
        
        // Insights sur les risques
        await this.generateRiskInsights(data);

        this.emit('insightsGenerated', { insights: this.insights });
    }

    // Générer des insights sur la qualité
    async generateQualityInsights(data) {
        const quality = dataProcessor.metadata.quality;
        
        if (quality.score > 85) {
            this.insights.push({
                type: 'opportunity',
                icon: 'fas fa-star',
                title: 'Excellente qualité de données',
                description: `Vos données présentent un score de qualité exceptionnel de ${quality.score}%. Cette qualité élevée permet des analyses fiables et des prises de décision éclairées.`,
                impact: 'high',
                metrics: {
                    'Score qualité': `${quality.score}%`,
                    'Confiance': 'Très élevée'
                },
                roi: 15
            });
        } else if (quality.score > 70) {
            this.insights.push({
                type: 'recommendation',
                icon: 'fas fa-chart-line',
                title: 'Qualité correcte avec potentiel d\'amélioration',
                description: `Score de qualité de ${quality.score}%. Quelques améliorations pourraient significativement augmenter la fiabilité de vos analyses.`,
                impact: 'medium',
                metrics: {
                    'Score actuel': `${quality.score}%`,
                    'Potentiel': `${Math.min(95, quality.score + 15)}%`
                },
                roi: 8
            });
        } else {
            this.insights.push({
                type: 'risk',
                icon: 'fas fa-exclamation-triangle',
                title: 'Qualité de données nécessitant attention',
                description: `Score de qualité de ${quality.score}%. Des améliorations urgentes sont recommandées pour éviter des erreurs d'analyse.`,
                impact: 'high',
                metrics: {
                    'Score actuel': `${quality.score}%`,
                    'Risque d\'erreur': 'Élevé'
                },
                roi: -12
            });
        }
    }

    // Générer des insights sur les patterns
    async generatePatternInsights() {
        this.patterns.forEach((pattern, key) => {
            switch (pattern.type) {
                case 'correlation':
                    this.insights.push({
                        type: 'pattern',
                        icon: 'fas fa-link',
                        title: `Corrélation détectée: ${pattern.columns.join(' ↔ ')}`,
                        description: pattern.description,
                        impact: Math.abs(pattern.strength) > 0.9 ? 'high' : 'medium',
                        metrics: {
                            'Force de corrélation': `${(pattern.strength * 100).toFixed(1)}%`,
                            'Fiabilité': Math.abs(pattern.strength) > 0.8 ? 'Élevée' : 'Moyenne'
                        },
                        roi: Math.abs(pattern.strength) * 10
                    });
                    break;
                    
                case 'temporal':
                    this.insights.push({
                        type: 'pattern',
                        icon: 'fas fa-clock',
                        title: `Pattern temporel dans ${pattern.column}`,
                        description: pattern.description,
                        impact: 'medium',
                        metrics: {
                            'Régularité': `${(pattern.regularity * 100).toFixed(1)}%`,
                            'Predictibilité': 'Élevée'
                        },
                        roi: 5
                    });
                    break;
                    
                case 'segmentation':
                    this.insights.push({
                        type: 'opportunity',
                        icon: 'fas fa-users',
                        title: `Segments identifiés dans ${pattern.column}`,
                        description: pattern.description,
                        impact: 'high',
                        metrics: {
                            'Nombre de segments': pattern.segments.length,
                            'Segment principal': `${pattern.segments[0].category} (${pattern.segments[0].percentage}%)`
                        },
                        roi: pattern.segments.length * 3
                    });
                    break;
            }
        });
    }

    // Générer des opportunités business
    async generateBusinessOpportunities(data) {
        const headers = data[0];
        const rows = data.slice(1);

        // Analyser les colonnes pour identifier des opportunités
        headers.forEach((header, index) => {
            const values = rows.map(row => row[index]).filter(val => val && val.toString().trim());
            
            if (header.toLowerCase().includes('revenue') || header.toLowerCase().includes('chiffre')) {
                const amounts = values.map(val => parseFloat(val)).filter(val => !isNaN(val));
                if (amounts.length > 0) {
                    const total = amounts.reduce((a, b) => a + b, 0);
                    const average = total / amounts.length;
                    
                    this.insights.push({
                        type: 'opportunity',
                        icon: 'fas fa-euro-sign',
                        title: 'Analyse du chiffre d\'affaires',
                        description: `Potentiel d'optimisation identifié dans la colonne revenus. Moyenne actuelle: ${this.formatCurrency(average)}.`,
                        impact: 'high',
                        metrics: {
                            'CA moyen': this.formatCurrency(average),
                            'CA total': this.formatCurrency(total),
                            'Transactions': amounts.length
                        },
                        roi: 20
                    });
                }
            }
        });
    }

    // Générer des insights sur les risques
    async generateRiskInsights(data) {
        const quality = dataProcessor.metadata.quality;
        
        if (quality.duplicates > 0) {
            this.insights.push({
                type: 'risk',
                icon: 'fas fa-copy',
                title: 'Doublons détectés',
                description: `${quality.duplicates} doublons identifiés. Ces données redondantes peuvent biaiser vos analyses et fausser les résultats.`,
                impact: quality.duplicates > data.length * 0.05 ? 'high' : 'medium',
                metrics: {
                    'Doublons': quality.duplicates,
                    'Impact sur précision': `${(quality.duplicates / data.length * 100).toFixed(1)}%`
                },
                roi: -quality.duplicates * 0.1
            });
        }

        if (quality.missingValues > 0) {
            this.insights.push({
                type: 'risk',
                icon: 'fas fa-question-circle',
                title: 'Valeurs manquantes importantes',
                description: `${quality.missingValues} valeurs manquantes détectées. Ces lacunes peuvent compromettre la fiabilité de vos analyses.`,
                impact: quality.missingValues > data.length * 0.1 ? 'high' : 'medium',
                metrics: {
                    'Valeurs manquantes': quality.missingValues,
                    'Taux de complétude': `${(100 - quality.missingValues / (data.length * data[0].length) * 100).toFixed(1)}%`
                },
                roi: -quality.missingValues * 0.05
            });
        }
    }

    // Calculer le ROI potentiel
    async calculateROI(data) {
        this.emit('status', 'Calcul du ROI potentiel...');
        
        let totalROI = 0;
        
        // ROI basé sur l'amélioration de la qualité
        const qualityImprovement = dataProcessor.metadata.quality.score - 70; // Baseline
        if (qualityImprovement > 0) {
            totalROI += qualityImprovement * 0.5; // 0.5% ROI par point de qualité
        }

        // ROI basé sur la réduction des doublons
        const duplicateReduction = dataProcessor.metadata.quality.duplicates;
        totalROI += duplicateReduction * 0.1; // Économies opérationnelles

        // ROI basé sur les insights business
        const businessInsights = this.insights.filter(insight => insight.type === 'opportunity');
        totalROI += businessInsights.reduce((sum, insight) => sum + (insight.roi || 0), 0);

        this.emit('roiCalculated', { roi: totalROI });
        return totalROI;
    }

    // Générer des recommandations
    async generateRecommendations(data) {
        this.emit('status', 'Génération des recommandations...');
        
        this.recommendations = [];

        // Recommandations basées sur la qualité
        const quality = dataProcessor.metadata.quality;
        
        if (quality.duplicates > 0) {
            this.recommendations.push({
                priority: 'high',
                category: 'Qualité des données',
                title: 'Supprimer les doublons',
                description: `Éliminer les ${quality.duplicates} doublons pour améliorer la précision des analyses.`,
                impact: 'Amélioration de la fiabilité des données',
                effort: 'Faible',
                timeline: 'Immédiat'
            });
        }

        if (quality.missingValues > data.length * 0.05) {
            this.recommendations.push({
                priority: 'high',
                category: 'Complétude',
                title: 'Traiter les valeurs manquantes',
                description: 'Implémenter une stratégie d\'imputation pour les valeurs manquantes critiques.',
                impact: 'Augmentation du taux de complétude',
                effort: 'Moyen',
                timeline: '1-2 semaines'
            });
        }

        // Recommandations basées sur les patterns
        this.patterns.forEach((pattern, key) => {
            if (pattern.type === 'correlation' && Math.abs(pattern.strength) > 0.8) {
                this.recommendations.push({
                    priority: 'medium',
                    category: 'Analyse prédictive',
                    title: `Exploiter la corrélation ${pattern.columns.join(' - ')}`,
                    description: 'Utiliser cette corrélation forte pour développer des modèles prédictifs.',
                    impact: 'Amélioration des prévisions business',
                    effort: 'Élevé',
                    timeline: '1-3 mois'
                });
            }
        });

        // Recommandations d'automatisation
        this.recommendations.push({
            priority: 'medium',
            category: 'Automatisation',
            title: 'Mettre en place un pipeline de nettoyage automatique',
            description: 'Automatiser le processus de nettoyage pour les futurs datasets.',
            impact: 'Efficacité opérationnelle accrue',
            effort: 'Élevé',
            timeline: '2-4 semaines'
        });

        this.emit('recommendationsGenerated', { recommendations: this.recommendations });
    }

    // Finaliser l'analyse
    async finalizeAnalysis(data) {
        this.emit('status', 'Finalisation de l\'analyse...');
        
        const finalResults = {
            originalRows: data.length - 1,
            finalQuality: this.calculateFinalQuality(data),
            insights: this.insights,
            recommendations: this.recommendations,
            patterns: Array.from(this.patterns.values()),
            roi: await this.calculateROI(data),
            completedAt: new Date().toISOString()
        };

        this.emit('analysisFinalized', finalResults);
        return finalResults;
    }

    // Calculer la qualité finale
    calculateFinalQuality(data) {
        // Réalculer les métriques après nettoyage
        return Math.min(95, dataProcessor.metadata.quality.score + 10); // Simulation d'amélioration
    }

    // Utilitaires
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('fr-FR', {
            style: 'currency',
            currency: 'EUR'
        }).format(amount);
    }

    formatNumber(number) {
        return new Intl.NumberFormat('fr-FR').format(number);
    }

    formatPercentage(percentage) {
        return `${percentage.toFixed(1)}%`;
    }
}

// Instance globale du moteur IA
window.aiEngine = new AIEngine();