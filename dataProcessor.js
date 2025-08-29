// Gestionnaire de traitement des données
class DataProcessor {
    constructor() {
        this.rawData = null;
        this.cleanedData = null;
        this.metadata = {
            fileName: '',
            fileSize: 0,
            totalRows: 0,
            totalColumns: 0,
            columns: [],
            dataTypes: {},
            quality: {
                score: 0,
                duplicates: 0,
                missingValues: 0,
                anomalies: 0,
                consistency: 0
            }
        };
        this.sampleDatasets = this.initSampleDatasets();
    }

    // Initialiser les datasets d'exemple
    initSampleDatasets() {
        return {
            customers: {
                name: 'customers_sample.csv',
                data: this.generateCustomersData(5000),
                description: 'Base de données clients avec informations personnelles'
            },
            sales: {
                name: 'sales_sample.csv', 
                data: this.generateSalesData(10000),
                description: 'Données de ventes avec produits et montants'
            },
            products: {
                name: 'products_sample.csv',
                data: this.generateProductsData(2000),
                description: 'Catalogue produits avec prix et catégories'
            }
        };
    }

    // Générer des données clients d'exemple
    generateCustomersData(count) {
        const data = [];
        const firstNames = ['Jean', 'Marie', 'Pierre', 'Sophie', 'Luc', 'Claire', 'Michel', 'Julie', 'David', 'Sarah'];
        const lastNames = ['Martin', 'Bernard', 'Dubois', 'Thomas', 'Robert', 'Petit', 'Durand', 'Leroy', 'Moreau', 'Simon'];
        const cities = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg', 'Montpellier', 'Bordeaux', 'Lille'];
        const domains = ['gmail.com', 'yahoo.fr', 'hotmail.com', 'outlook.fr', 'free.fr'];

        // Headers
        data.push(['ID', 'Prénom', 'Nom', 'Email', 'Téléphone', 'Ville', 'Age', 'Statut', 'Date_Inscription', 'Score_Credit']);

        for (let i = 1; i <= count; i++) {
            const firstName = firstNames[Math.floor(Math.random() * firstNames.length)];
            const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
            const email = Math.random() > 0.05 ? `${firstName.toLowerCase()}.${lastName.toLowerCase()}@${domains[Math.floor(Math.random() * domains.length)]}` : ''; // 5% d'emails manquants
            const phone = Math.random() > 0.03 ? `0${Math.floor(Math.random() * 6) + 1}${Math.floor(Math.random() * 100000000).toString().padStart(8, '0')}` : ''; // 3% de téléphones manquants
            const city = cities[Math.floor(Math.random() * cities.length)];
            const age = Math.floor(Math.random() * 60) + 18;
            const status = Math.random() > 0.7 ? 'VIP' : Math.random() > 0.3 ? 'Standard' : 'Bronze';
            const registrationDate = new Date(2020 + Math.floor(Math.random() * 4), Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0];
            const creditScore = Math.floor(Math.random() * 900) + 100;

            // Introduire quelques doublons (2%)
            if (Math.random() < 0.02 && i > 100) {
                const duplicateIndex = Math.floor(Math.random() * (i - 100)) + 1;
                data.push([...data[duplicateIndex]]);
            } else {
                data.push([i, firstName, lastName, email, phone, city, age, status, registrationDate, creditScore]);
            }
        }

        return data;
    }

    // Générer des données de ventes d'exemple
    generateSalesData(count) {
        const data = [];
        const products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Mouse', 'Keyboard', 'Monitor', 'Printer', 'Camera', 'Speaker'];
        const categories = ['Electronics', 'Accessories', 'Computing', 'Audio', 'Photo'];
        const salespeople = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'];

        // Headers
        data.push(['Order_ID', 'Product', 'Category', 'Quantity', 'Unit_Price', 'Total', 'Salesperson', 'Customer_ID', 'Order_Date', 'Region']);

        for (let i = 1; i <= count; i++) {
            const product = products[Math.floor(Math.random() * products.length)];
            const category = categories[Math.floor(Math.random() * categories.length)];
            const quantity = Math.floor(Math.random() * 10) + 1;
            const unitPrice = Math.random() > 0.02 ? (Math.random() * 2000 + 50).toFixed(2) : ''; // 2% de prix manquants
            const total = unitPrice ? (quantity * parseFloat(unitPrice)).toFixed(2) : '';
            const salesperson = salespeople[Math.floor(Math.random() * salespeople.length)];
            const customerId = Math.floor(Math.random() * 5000) + 1;
            const orderDate = new Date(2023, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0];
            const region = ['North', 'South', 'East', 'West'][Math.floor(Math.random() * 4)];

            data.push([`ORD-${i.toString().padStart(6, '0')}`, product, category, quantity, unitPrice, total, salesperson, customerId, orderDate, region]);
        }

        return data;
    }

    // Générer des données produits d'exemple
    generateProductsData(count) {
        const data = [];
        const categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Automotive', 'Garden'];
        const brands = ['TechPro', 'StyleMax', 'HomeComfort', 'SportElite', 'BeautyPlus', 'AutoParts', 'GreenThumb'];
        const statuses = ['Active', 'Discontinued', 'Out of Stock', 'Coming Soon'];

        // Headers
        data.push(['SKU', 'Product_Name', 'Category', 'Brand', 'Price', 'Cost', 'Stock', 'Status', 'Launch_Date', 'Rating']);

        for (let i = 1; i <= count; i++) {
            const sku = `SKU-${i.toString().padStart(6, '0')}`;
            const productName = `Product ${i}`;
            const category = categories[Math.floor(Math.random() * categories.length)];
            const brand = brands[Math.floor(Math.random() * brands.length)];
            const price = Math.random() > 0.01 ? (Math.random() * 1000 + 10).toFixed(2) : ''; // 1% de prix manquants
            const cost = price ? (parseFloat(price) * (0.4 + Math.random() * 0.3)).toFixed(2) : '';
            const stock = Math.floor(Math.random() * 1000);
            const status = statuses[Math.floor(Math.random() * statuses.length)];
            const launchDate = new Date(2020 + Math.floor(Math.random() * 4), Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0];
            const rating = Math.random() > 0.05 ? (Math.random() * 5).toFixed(1) : ''; // 5% de ratings manquants

            data.push([sku, productName, category, brand, price, cost, stock, status, launchDate, rating]);
        }

        return data;
    }

    // Charger un dataset d'exemple
    loadSampleDataset(type) {
        if (this.sampleDatasets[type]) {
            const dataset = this.sampleDatasets[type];
            this.rawData = dataset.data;
            this.metadata.fileName = dataset.name;
            this.metadata.fileSize = JSON.stringify(dataset.data).length;
            this.analyzeData();
            return true;
        }
        return false;
    }

    // Traiter un fichier uploadé
    async processFile(file) {
        try {
            this.metadata.fileName = file.name;
            this.metadata.fileSize = file.size;

            const fileExtension = file.name.split('.').pop().toLowerCase();
            let data;

            switch (fileExtension) {
                case 'csv':
                    data = await this.parseCSV(file);
                    break;
                case 'xlsx':
                case 'xls':
                    data = await this.parseExcel(file);
                    break;
                case 'json':
                    data = await this.parseJSON(file);
                    break;
                case 'xml':
                    data = await this.parseXML(file);
                    break;
                case 'txt':
                    data = await this.parseTXT(file);
                    break;
                default:
                    throw new Error(`Format ${fileExtension} non supporté`);
            }

            this.rawData = data;
            this.analyzeData();
            return true;
        } catch (error) {
            console.error('Erreur lors du traitement du fichier:', error);
            throw error;
        }
    }

    // Parser CSV
    async parseCSV(file) {
        const text = await file.text();
        const lines = text.split('\n').filter(line => line.trim());
        return lines.map(line => {
            // Parser CSV simple (ne gère pas les guillemets complexes)
            return line.split(',').map(cell => cell.trim().replace(/^"|"$/g, ''));
        });
    }

    // Parser Excel
    async parseExcel(file) {
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: 'array' });
        const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
        const jsonData = XLSX.utils.sheet_to_json(firstSheet, { header: 1 });
        return jsonData.filter(row => row.some(cell => cell !== undefined && cell !== ''));
    }

    // Parser JSON
    async parseJSON(file) {
        const text = await file.text();
        const jsonData = JSON.parse(text);
        
        if (Array.isArray(jsonData)) {
            if (jsonData.length > 0 && typeof jsonData[0] === 'object') {
                // Convertir array d'objets en array 2D
                const headers = Object.keys(jsonData[0]);
                const data = [headers];
                jsonData.forEach(obj => {
                    data.push(headers.map(header => obj[header] || ''));
                });
                return data;
            }
            return jsonData;
        } else {
            throw new Error('Format JSON non supporté');
        }
    }

    // Parser XML
    async parseXML(file) {
        const text = await file.text();
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(text, 'text/xml');
        
        // Conversion XML basique vers array 2D
        const rows = xmlDoc.querySelectorAll('row, record, item');
        if (rows.length === 0) return [['Données XML']];
        
        const data = [];
        const headers = Array.from(rows[0].children).map(child => child.tagName);
        data.push(headers);
        
        rows.forEach(row => {
            const rowData = headers.map(header => {
                const element = row.querySelector(header);
                return element ? element.textContent : '';
            });
            data.push(rowData);
        });
        
        return data;
    }

    // Parser TXT
    async parseTXT(file) {
        const text = await file.text();
        const lines = text.split('\n').filter(line => line.trim());
        
        // Essayer de détecter le séparateur
        const firstLine = lines[0];
        let separator = ',';
        if (firstLine.includes('\t')) separator = '\t';
        else if (firstLine.includes(';')) separator = ';';
        else if (firstLine.includes('|')) separator = '|';
        
        return lines.map(line => line.split(separator).map(cell => cell.trim()));
    }

    // Analyser les données
    analyzeData() {
        if (!this.rawData || this.rawData.length === 0) return;

        const headers = this.rawData[0];
        const dataRows = this.rawData.slice(1);

        this.metadata.totalRows = dataRows.length;
        this.metadata.totalColumns = headers.length;
        this.metadata.columns = headers.map((header, index) => ({
            name: header,
            index: index,
            type: this.detectColumnType(dataRows, index),
            stats: this.calculateColumnStats(dataRows, index)
        }));

        this.calculateQualityMetrics(dataRows);
    }

    // Détecter le type d'une colonne
    detectColumnType(dataRows, columnIndex) {
        const sample = dataRows.slice(0, 100).map(row => row[columnIndex]).filter(val => val && val.toString().trim());
        
        if (sample.length === 0) return 'unknown';

        // Test pour nombre
        const numberCount = sample.filter(val => !isNaN(val) && !isNaN(parseFloat(val))).length;
        if (numberCount / sample.length > 0.8) return 'number';

        // Test pour date
        const dateCount = sample.filter(val => !isNaN(Date.parse(val))).length;
        if (dateCount / sample.length > 0.8) return 'date';

        // Test pour email
        const emailCount = sample.filter(val => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val)).length;
        if (emailCount / sample.length > 0.8) return 'email';

        // Test pour téléphone
        const phoneCount = sample.filter(val => /^[\+]?[\d\s\-\(\)]{8,}$/.test(val)).length;
        if (phoneCount / sample.length > 0.8) return 'phone';

        // Test pour URL
        const urlCount = sample.filter(val => /^https?:\/\//.test(val)).length;
        if (urlCount / sample.length > 0.8) return 'url';

        return 'text';
    }

    // Calculer les statistiques d'une colonne
    calculateColumnStats(dataRows, columnIndex) {
        const values = dataRows.map(row => row[columnIndex]);
        const nonEmptyValues = values.filter(val => val && val.toString().trim());
        
        return {
            total: values.length,
            filled: nonEmptyValues.length,
            empty: values.length - nonEmptyValues.length,
            completeness: (nonEmptyValues.length / values.length * 100).toFixed(1),
            uniqueValues: new Set(nonEmptyValues).size,
            duplicates: nonEmptyValues.length - new Set(nonEmptyValues).size
        };
    }

    // Calculer les métriques de qualité
    calculateQualityMetrics(dataRows) {
        const totalCells = dataRows.length * this.metadata.totalColumns;
        let filledCells = 0;
        let duplicateRows = 0;
        let anomalies = 0;

        // Compter les cellules remplies
        dataRows.forEach(row => {
            row.forEach(cell => {
                if (cell && cell.toString().trim()) filledCells++;
            });
        });

        // Détecter les doublons de lignes
        const rowStrings = dataRows.map(row => row.join('|'));
        const uniqueRows = new Set(rowStrings);
        duplicateRows = rowStrings.length - uniqueRows.size;

        // Détecter les anomalies (valeurs aberrantes)
        this.metadata.columns.forEach(column => {
            if (column.type === 'number') {
                const values = dataRows.map(row => parseFloat(row[column.index])).filter(val => !isNaN(val));
                if (values.length > 0) {
                    const mean = values.reduce((a, b) => a + b, 0) / values.length;
                    const std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
                    anomalies += values.filter(val => Math.abs(val - mean) > 3 * std).length;
                }
            }
        });

        const completeness = (filledCells / totalCells * 100);
        const uniqueness = ((dataRows.length - duplicateRows) / dataRows.length * 100);
        const accuracy = Math.max(0, 100 - (anomalies / dataRows.length * 100));
        
        this.metadata.quality = {
            score: Math.round((completeness + uniqueness + accuracy) / 3),
            completeness: Math.round(completeness),
            uniqueness: Math.round(uniqueness),
            accuracy: Math.round(accuracy),
            duplicates: duplicateRows,
            missingValues: totalCells - filledCells,
            anomalies: anomalies
        };
    }

    // Obtenir un aperçu des données avec pagination
    getDataPreview(page = 1, pageSize = 100) {
        if (!this.rawData) return { headers: [], rows: [], totalRows: 0 };

        const headers = this.rawData[0];
        const dataRows = this.rawData.slice(1);
        const startIndex = (page - 1) * pageSize;
        const endIndex = Math.min(startIndex + pageSize, dataRows.length);
        const rows = dataRows.slice(startIndex, endIndex);

        return {
            headers,
            rows,
            totalRows: dataRows.length,
            currentPage: page,
            totalPages: Math.ceil(dataRows.length / pageSize),
            pageSize
        };
    }

    // Nettoyer les données
    async cleanData(options = {}) {
        if (!this.rawData) throw new Error('Aucune donnée à nettoyer');

        const cleanedData = JSON.parse(JSON.stringify(this.rawData));
        const headers = cleanedData[0];
        const dataRows = cleanedData.slice(1);

        // Supprimer les doublons
        if (options.removeDuplicates !== false) {
            const seen = new Set();
            const uniqueRows = [];
            dataRows.forEach(row => {
                const rowString = row.join('|');
                if (!seen.has(rowString)) {
                    seen.add(rowString);
                    uniqueRows.push(row);
                }
            });
            cleanedData.splice(1, dataRows.length, ...uniqueRows);
        }

        // Traiter les valeurs manquantes
        if (options.fillMissingValues !== false) {
            this.metadata.columns.forEach(column => {
                this.fillMissingValues(cleanedData, column);
            });
        }

        // Standardiser les formats
        if (options.standardizeFormats !== false) {
            this.standardizeFormats(cleanedData);
        }

        // Nettoyer les anomalies
        if (options.cleanAnomalies !== false) {
            this.cleanAnomalies(cleanedData);
        }

        this.cleanedData = cleanedData;
        return cleanedData;
    }

    // Remplir les valeurs manquantes
    fillMissingValues(data, column) {
        const dataRows = data.slice(1);
        const columnValues = dataRows.map(row => row[column.index]).filter(val => val && val.toString().trim());

        if (columnValues.length === 0) return;

        let fillValue;
        switch (column.type) {
            case 'number':
                const numbers = columnValues.map(val => parseFloat(val)).filter(val => !isNaN(val));
                fillValue = numbers.length > 0 ? (numbers.reduce((a, b) => a + b, 0) / numbers.length).toFixed(2) : '0';
                break;
            case 'text':
                const counts = {};
                columnValues.forEach(val => counts[val] = (counts[val] || 0) + 1);
                fillValue = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
                break;
            case 'date':
                fillValue = new Date().toISOString().split('T')[0];
                break;
            default:
                fillValue = 'N/A';
        }

        dataRows.forEach(row => {
            if (!row[column.index] || !row[column.index].toString().trim()) {
                row[column.index] = fillValue;
            }
        });
    }

    // Standardiser les formats
    standardizeFormats(data) {
        const dataRows = data.slice(1);

        this.metadata.columns.forEach(column => {
            if (column.type === 'email') {
                dataRows.forEach(row => {
                    if (row[column.index]) {
                        row[column.index] = row[column.index].toLowerCase().trim();
                    }
                });
            } else if (column.type === 'phone') {
                dataRows.forEach(row => {
                    if (row[column.index]) {
                        // Standardiser le format téléphone français
                        let phone = row[column.index].replace(/[\s\-\(\)]/g, '');
                        if (phone.startsWith('33')) phone = '0' + phone.substring(2);
                        if (phone.length === 10 && phone.startsWith('0')) {
                            row[column.index] = phone.replace(/(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/, '$1 $2 $3 $4 $5');
                        }
                    }
                });
            } else if (column.type === 'text') {
                dataRows.forEach(row => {
                    if (row[column.index]) {
                        // Normaliser les noms (première lettre majuscule)
                        if (column.name.toLowerCase().includes('nom') || column.name.toLowerCase().includes('name')) {
                            row[column.index] = row[column.index].toLowerCase().replace(/\b\w/g, l => l.toUpperCase());
                        }
                    }
                });
            }
        });
    }

    // Nettoyer les anomalies
    cleanAnomalies(data) {
        const dataRows = data.slice(1);

        this.metadata.columns.forEach(column => {
            if (column.type === 'number') {
                const values = dataRows.map(row => parseFloat(row[column.index])).filter(val => !isNaN(val));
                if (values.length > 0) {
                    const mean = values.reduce((a, b) => a + b, 0) / values.length;
                    const std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
                    
                    dataRows.forEach(row => {
                        const value = parseFloat(row[column.index]);
                        if (!isNaN(value) && Math.abs(value - mean) > 3 * std) {
                            // Remplacer par la médiane
                            const sortedValues = [...values].sort((a, b) => a - b);
                            const median = sortedValues[Math.floor(sortedValues.length / 2)];
                            row[column.index] = median.toString();
                        }
                    });
                }
            }
        });
    }

    // Exporter les données nettoyées
    exportCleanedData(format = 'csv') {
        if (!this.cleanedData) throw new Error('Aucune donnée nettoyée disponible');

        switch (format.toLowerCase()) {
            case 'csv':
                return this.exportToCSV(this.cleanedData);
            case 'excel':
                return this.exportToExcel(this.cleanedData);
            case 'json':
                return this.exportToJSON(this.cleanedData);
            default:
                throw new Error(`Format d'export ${format} non supporté`);
        }
    }

    // Exporter vers CSV
    exportToCSV(data) {
        const csvContent = data.map(row => 
            row.map(cell => `"${cell.toString().replace(/"/g, '""')}"`).join(',')
        ).join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        return blob;
    }

    // Exporter vers Excel
    exportToExcel(data) {
        const ws = XLSX.utils.aoa_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'Cleaned Data');
        
        const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
        const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
        return blob;
    }

    // Exporter vers JSON
    exportToJSON(data) {
        const headers = data[0];
        const rows = data.slice(1);
        const jsonData = rows.map(row => {
            const obj = {};
            headers.forEach((header, index) => {
                obj[header] = row[index];
            });
            return obj;
        });
        
        const jsonString = JSON.stringify(jsonData, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        return blob;
    }

    // Obtenir les statistiques de nettoyage
    getCleaningStats() {
        if (!this.rawData || !this.cleanedData) return null;

        const originalRows = this.rawData.length - 1;
        const cleanedRows = this.cleanedData.length - 1;
        const duplicatesRemoved = originalRows - cleanedRows;

        return {
            originalRows,
            cleanedRows,
            duplicatesRemoved,
            qualityImprovement: this.metadata.quality.score,
            processing: {
                missingValuesFilled: this.metadata.quality.missingValues,
                anomaliesCleaned: this.metadata.quality.anomalies,
                formatsStandardized: this.metadata.totalColumns
            }
        };
    }
}

// Instance globale du processeur de données
window.dataProcessor = new DataProcessor();