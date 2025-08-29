// Système d'internationalisation FR/EN
class TranslationManager {
    constructor() {
        this.currentLang = 'fr';
        this.translations = {
            fr: {
                // Navigation
                demo: "DÉMO",
                upgrade: "UPGRADE VIP",
                nav_upload: "Upload",
                nav_preview: "Aperçu",
                nav_processing: "Traitement",
                nav_dashboard: "Dashboard",
                nav_chat: "Chat IA",
                
                // Titres principaux
                main_title: "Agent IA de Nettoyage de Données",
                main_subtitle: "Nettoyage intelligent et automatisé de vos données avec IA avancée",
                
                // Section Upload
                upload_title: "Glissez vos fichiers ici",
                upload_subtitle: "ou cliquez pour sélectionner",
                file_limit: "Fichiers jusqu'à 50MB",
                sample_data_title: "Ou essayez avec nos données d'exemple :",
                sample_customers: "Clients (5k lignes)",
                sample_sales: "Ventes (10k lignes)",
                sample_products: "Produits (2k lignes)",
                
                // Section Preview
                preview_title: "Aperçu des Données",
                back: "Retour",
                start_cleaning: "Lancer le Nettoyage IA",
                filename: "Fichier :",
                rows: "Lignes :",
                columns: "Colonnes :",
                size: "Taille :",
                format_valid: "Format valide",
                structure_valid: "Structure valide",
                columns_analysis: "Analyse des Colonnes",
                data_preview: "Aperçu des Données",
                showing: "Affichage de",
                of: "sur",
                rows_label: "lignes",
                
                // Section Processing
                processing_title: "Nettoyage IA en Cours",
                live: "LIVE",
                overall_progress: "Progression Globale",
                step_analysis: "Analyse des données",
                step_cleaning: "Nettoyage intelligent",
                step_insights: "Génération insights",
                step_complete: "Finalisation",
                pending: "En attente...",
                in_progress: "En cours...",
                completed: "Terminé",
                processed_rows: "Lignes traitées",
                duplicates_found: "Doublons détectés",
                missing_filled: "Valeurs corrigées",
                anomalies_detected: "Anomalies détectées",
                
                // Section Dashboard
                dashboard_title: "Dashboard & Insights IA",
                complete: "COMPLET",
                export: "Exporter",
                quality_score: "Score Qualité",
                anomalies: "Anomalies",
                detected: "détectées",
                clean_records: "Enregistrements",
                clean: "propres",
                confidence: "Confiance",
                data_quality_trend: "Évolution Qualité des Données",
                issues_by_type: "Problèmes par Type",
                cleaning_impact: "Impact du Nettoyage",
                column_quality: "Qualité par Colonne",
                ai_insights_title: "🧠 Insights IA Automatiques",
                
                // Section Chat
                chat_title: "💬 Chat IA Conversationnel",
                ai_online: "IA en ligne",
                welcome_message: "Bonjour ! Je suis votre assistant IA spécialisé en nettoyage de données. Posez-moi des questions sur vos données ou demandez-moi des analyses spécifiques !",
                suggested_questions: "Questions suggérées :",
                q_quality: "Résumé de la qualité des données",
                q_issues: "Quels sont les principaux problèmes ?",
                q_roi: "Calcul du ROI potentiel",
                q_recommendations: "Recommandations d'amélioration",
                chat_placeholder: "Posez votre question...",
                ai_typing: "L'IA est en train d'écrire",
                
                // Messages IA
                analyzing_data: "Analyse des données en cours...",
                detecting_patterns: "Détection des patterns...",
                cleaning_records: "Nettoyage des enregistrements...",
                generating_insights: "Génération des insights...",
                
                // Insights types
                insight_opportunity: "Opportunité",
                insight_risk: "Risque",
                insight_recommendation: "Recommandation",
                insight_pattern: "Pattern",
                
                // Métriques
                data_quality: "Qualité des données",
                completion_rate: "Taux de complétude",
                accuracy_score: "Score de précision",
                consistency_rate: "Taux de cohérence",
                
                // Boutons et actions
                upload_file: "Télécharger un fichier",
                process_data: "Traiter les données",
                download_results: "Télécharger les résultats",
                view_report: "Voir le rapport",
                
                // Messages d'erreur
                file_too_large: "Le fichier est trop volumineux (max 50MB)",
                invalid_format: "Format de fichier non supporté",
                upload_error: "Erreur lors du téléchargement",
                processing_error: "Erreur lors du traitement"
            },
            en: {
                // Navigation
                demo: "DEMO",
                upgrade: "UPGRADE VIP",
                nav_upload: "Upload",
                nav_preview: "Preview",
                nav_processing: "Processing",
                nav_dashboard: "Dashboard",
                nav_chat: "AI Chat",
                
                // Main titles
                main_title: "AI Data Cleaning Agent",
                main_subtitle: "Intelligent and automated data cleaning with advanced AI",
                
                // Upload Section
                upload_title: "Drop your files here",
                upload_subtitle: "or click to select",
                file_limit: "Files up to 50MB",
                sample_data_title: "Or try with our sample data:",
                sample_customers: "Customers (5k rows)",
                sample_sales: "Sales (10k rows)",
                sample_products: "Products (2k rows)",
                
                // Preview Section
                preview_title: "Data Preview",
                back: "Back",
                start_cleaning: "Start AI Cleaning",
                filename: "File:",
                rows: "Rows:",
                columns: "Columns:",
                size: "Size:",
                format_valid: "Valid format",
                structure_valid: "Valid structure",
                columns_analysis: "Columns Analysis",
                data_preview: "Data Preview",
                showing: "Showing",
                of: "of",
                rows_label: "rows",
                
                // Processing Section
                processing_title: "AI Cleaning in Progress",
                live: "LIVE",
                overall_progress: "Overall Progress",
                step_analysis: "Data analysis",
                step_cleaning: "Intelligent cleaning",
                step_insights: "Insights generation",
                step_complete: "Finalization",
                pending: "Pending...",
                in_progress: "In progress...",
                completed: "Completed",
                processed_rows: "Processed rows",
                duplicates_found: "Duplicates found",
                missing_filled: "Values corrected",
                anomalies_detected: "Anomalies detected",
                
                // Dashboard Section
                dashboard_title: "Dashboard & AI Insights",
                complete: "COMPLETE",
                export: "Export",
                quality_score: "Quality Score",
                anomalies: "Anomalies",
                detected: "detected",
                clean_records: "Records",
                clean: "clean",
                confidence: "Confidence",
                data_quality_trend: "Data Quality Trend",
                issues_by_type: "Issues by Type",
                cleaning_impact: "Cleaning Impact",
                column_quality: "Quality by Column",
                ai_insights_title: "🧠 Automatic AI Insights",
                
                // Chat Section
                chat_title: "💬 Conversational AI Chat",
                ai_online: "AI online",
                welcome_message: "Hello! I'm your AI assistant specialized in data cleaning. Ask me questions about your data or request specific analyses!",
                suggested_questions: "Suggested questions:",
                q_quality: "Data quality summary",
                q_issues: "What are the main issues?",
                q_roi: "Potential ROI calculation",
                q_recommendations: "Improvement recommendations",
                chat_placeholder: "Ask your question...",
                ai_typing: "AI is typing",
                
                // AI Messages
                analyzing_data: "Analyzing data...",
                detecting_patterns: "Detecting patterns...",
                cleaning_records: "Cleaning records...",
                generating_insights: "Generating insights...",
                
                // Insights types
                insight_opportunity: "Opportunity",
                insight_risk: "Risk",
                insight_recommendation: "Recommendation",
                insight_pattern: "Pattern",
                
                // Metrics
                data_quality: "Data quality",
                completion_rate: "Completion rate",
                accuracy_score: "Accuracy score",
                consistency_rate: "Consistency rate",
                
                // Buttons and actions
                upload_file: "Upload file",
                process_data: "Process data",
                download_results: "Download results",
                view_report: "View report",
                
                // Error messages
                file_too_large: "File too large (max 50MB)",
                invalid_format: "Unsupported file format",
                upload_error: "Upload error",
                processing_error: "Processing error"
            }
        };
    }

    // Obtenir la traduction d'une clé
    t(key, lang = null) {
        const language = lang || this.currentLang;
        return this.translations[language]?.[key] || key;
    }

    // Changer la langue
    setLanguage(lang) {
        if (this.translations[lang]) {
            this.currentLang = lang;
            this.updateDOM();
            this.updatePlaceholders();
            localStorage.setItem('ai-dataclean-lang', lang);
        }
    }

    // Mettre à jour tous les éléments avec data-i18n
    updateDOM() {
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.setAttribute('placeholder', translation);
            } else {
                element.textContent = translation;
            }
        });
    }

    // Mettre à jour les placeholders
    updatePlaceholders() {
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            const translation = this.t(key);
            element.setAttribute('placeholder', translation);
        });
    }

    // Initialiser les traductions
    init() {
        // Récupérer la langue sauvegardée ou utiliser le navigateur
        const savedLang = localStorage.getItem('ai-dataclean-lang');
        const browserLang = navigator.language.startsWith('fr') ? 'fr' : 'en';
        const defaultLang = savedLang || browserLang;
        
        this.setLanguage(defaultLang);
        this.setupLanguageToggle();
    }

    // Configurer le bouton de changement de langue
    setupLanguageToggle() {
        const langToggle = document.getElementById('lang-toggle');
        if (langToggle) {
            this.updateLanguageToggle(langToggle);
            
            langToggle.addEventListener('click', () => {
                const newLang = this.currentLang === 'fr' ? 'en' : 'fr';
                this.setLanguage(newLang);
                this.updateLanguageToggle(langToggle);
            });
        }
    }

    // Mettre à jour l'affichage du bouton de langue
    updateLanguageToggle(button) {
        const flag = button.querySelector('.flag');
        const text = button.querySelector('.lang-text');
        
        if (this.currentLang === 'fr') {
            flag.textContent = '🇫🇷';
            text.textContent = 'FR';
        } else {
            flag.textContent = '🇺🇸';
            text.textContent = 'EN';
        }
    }

    // Obtenir les formats de date localisés
    getDateFormat() {
        return this.currentLang === 'fr' ? 'dd/MM/yyyy' : 'MM/dd/yyyy';
    }

    // Obtenir les formats de nombre localisés
    getNumberFormat(number) {
        return new Intl.NumberFormat(this.currentLang === 'fr' ? 'fr-FR' : 'en-US').format(number);
    }

    // Obtenir les formats de pourcentage localisés
    getPercentFormat(number) {
        return new Intl.NumberFormat(this.currentLang === 'fr' ? 'fr-FR' : 'en-US', {
            style: 'percent',
            minimumFractionDigits: 1,
            maximumFractionDigits: 1
        }).format(number / 100);
    }

    // Obtenir les formats de devise localisés
    getCurrencyFormat(number, currency = 'EUR') {
        const locale = this.currentLang === 'fr' ? 'fr-FR' : 'en-US';
        const curr = this.currentLang === 'fr' ? 'EUR' : 'USD';
        
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currency || curr
        }).format(number);
    }

    // Formater les dates selon la locale
    formatDate(date, options = {}) {
        const locale = this.currentLang === 'fr' ? 'fr-FR' : 'en-US';
        const defaultOptions = {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        };
        
        return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(new Date(date));
    }

    // Formater les durées
    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (this.currentLang === 'fr') {
            if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
            if (minutes > 0) return `${minutes}m ${secs}s`;
            return `${secs}s`;
        } else {
            if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
            if (minutes > 0) return `${minutes}m ${secs}s`;
            return `${secs}s`;
        }
    }

    // Messages contextuels pour l'IA
    getAIMessage(type, context = {}) {
        const messages = {
            fr: {
                welcome: "Bonjour ! Je suis prêt à analyser vos données.",
                processing: "Traitement en cours, veuillez patienter...",
                complete: "Analyse terminée ! Voici vos résultats.",
                error: "Une erreur s'est produite lors du traitement.",
                quality_good: `Excellente qualité ! Score: ${context.score}%`,
                quality_average: `Qualité correcte. Score: ${context.score}%`,
                quality_poor: `Qualité à améliorer. Score: ${context.score}%`,
                duplicates: `${context.count} doublons détectés et traités.`,
                missing_values: `${context.count} valeurs manquantes corrigées.`,
                anomalies: `${context.count} anomalies identifiées.`,
                roi_positive: `ROI potentiel estimé: +${context.roi}%`,
                recommendation: "Voici mes recommandations d'amélioration:",
                pattern_found: "Pattern intéressant détecté dans vos données."
            },
            en: {
                welcome: "Hello! I'm ready to analyze your data.",
                processing: "Processing in progress, please wait...",
                complete: "Analysis complete! Here are your results.",
                error: "An error occurred during processing.",
                quality_good: `Excellent quality! Score: ${context.score}%`,
                quality_average: `Good quality. Score: ${context.score}%`,
                quality_poor: `Quality needs improvement. Score: ${context.score}%`,
                duplicates: `${context.count} duplicates detected and processed.`,
                missing_values: `${context.count} missing values corrected.`,
                anomalies: `${context.count} anomalies identified.`,
                roi_positive: `Estimated potential ROI: +${context.roi}%`,
                recommendation: "Here are my improvement recommendations:",
                pattern_found: "Interesting pattern detected in your data."
            }
        };
        
        return messages[this.currentLang]?.[type] || type;
    }
}

// Instance globale du gestionnaire de traductions
window.i18n = new TranslationManager();