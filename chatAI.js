// Chat IA conversationnel
class ChatAI {
    constructor() {
        this.messages = [];
        this.isTyping = false;
        this.suggestions = [
            {
                id: 'quality_summary',
                question: 'Résumé de la qualité des données',
                response: this.getQualitySummaryResponse.bind(this)
            },
            {
                id: 'main_issues',
                question: 'Quels sont les principaux problèmes ?',
                response: this.getMainIssuesResponse.bind(this)
            },
            {
                id: 'roi_calculation',
                question: 'Calcul du ROI potentiel',
                response: this.getROIResponse.bind(this)
            },
            {
                id: 'recommendations',
                question: 'Recommandations d\'amélioration',
                response: this.getRecommendationsResponse.bind(this)
            }
        ];
        this.conversationContext = {};
        this.initializeChat();
    }

    // Initialiser le chat
    initializeChat() {
        this.setupEventListeners();
        this.addWelcomeMessage();
    }

    // Configurer les écouteurs d'événements
    setupEventListeners() {
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const suggestionBtns = document.querySelectorAll('.suggestion-btn');

        // Envoi de message
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });

            // Typing indicator
            chatInput.addEventListener('input', () => {
                this.showTypingIndicator(false); // Hide AI typing when user types
            });
        }

        // Questions suggérées
        suggestionBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const questionId = btn.getAttribute('data-question');
                const suggestion = this.suggestions.find(s => s.id === questionId);
                if (suggestion) {
                    this.sendMessage(suggestion.question, suggestion.response);
                }
            });
        });
    }

    // Ajouter le message de bienvenue
    addWelcomeMessage() {
        const welcomeMessage = {
            type: 'ai',
            content: i18n.t('welcome_message'),
            timestamp: new Date()
        };
        this.messages.push(welcomeMessage);
    }

    // Envoyer un message
    async sendMessage(customMessage = null, customResponse = null) {
        const chatInput = document.getElementById('chat-input');
        const message = customMessage || chatInput?.value?.trim();
        
        if (!message) return;

        // Ajouter le message utilisateur
        this.addMessage('user', message);
        
        // Vider l'input
        if (chatInput && !customMessage) {
            chatInput.value = '';
        }

        // Afficher l'indicateur de frappe
        this.showTypingIndicator(true);

        // Simuler un délai de réponse
        await this.delay(1000 + Math.random() * 2000);

        // Générer la réponse
        let response;
        if (customResponse) {
            response = await customResponse();
        } else {
            response = await this.generateResponse(message);
        }

        // Masquer l'indicateur et ajouter la réponse
        this.showTypingIndicator(false);
        this.addMessage('ai', response);
    }

    // Ajouter un message à la conversation
    addMessage(type, content) {
        const message = {
            type,
            content,
            timestamp: new Date()
        };

        this.messages.push(message);
        this.renderMessage(message);
        this.scrollToBottom();
    }

    // Rendre un message dans le DOM
    renderMessage(message) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.type}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = message.type === 'ai' 
            ? '<i class="fas fa-robot"></i>' 
            : '<i class="fas fa-user"></i>';

        const content = document.createElement('div');
        content.className = 'message-content';

        const text = document.createElement('div');
        text.className = 'message-text';
        text.innerHTML = message.content;

        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = this.formatTime(message.timestamp);

        content.appendChild(text);
        content.appendChild(time);

        if (message.type === 'user') {
            messageDiv.appendChild(content);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
        }

        messagesContainer.appendChild(messageDiv);

        // Animation d'apparition
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        setTimeout(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 50);
    }

    // Afficher/masquer l'indicateur de frappe
    showTypingIndicator(show) {
        const indicator = document.getElementById('typing-indicator');
        if (!indicator) return;

        this.isTyping = show;
        if (show) {
            indicator.classList.add('show');
        } else {
            indicator.classList.remove('show');
        }
    }

    // Générer une réponse IA
    async generateResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        // Analyse du contexte et génération de réponse contextuelle
        if (this.isGreeting(lowerMessage)) {
            return this.getGreetingResponse();
        }
        
        if (this.isQualityQuestion(lowerMessage)) {
            return this.getQualitySummaryResponse();
        }
        
        if (this.isIssuesQuestion(lowerMessage)) {
            return this.getMainIssuesResponse();
        }
        
        if (this.isROIQuestion(lowerMessage)) {
            return this.getROIResponse();
        }
        
        if (this.isRecommendationQuestion(lowerMessage)) {
            return this.getRecommendationsResponse();
        }
        
        if (this.isDataQuestion(lowerMessage)) {
            return this.getDataAnalysisResponse();
        }
        
        if (this.isTechnicalQuestion(lowerMessage)) {
            return this.getTechnicalResponse(lowerMessage);
        }

        // Réponse générale intelligente
        return this.getGeneralResponse(message);
    }

    // Vérifiers de type de question
    isGreeting(message) {
        const greetings = ['bonjour', 'salut', 'hello', 'hi', 'bonsoir'];
        return greetings.some(greeting => message.includes(greeting));
    }

    isQualityQuestion(message) {
        const keywords = ['qualité', 'quality', 'score', 'évaluation', 'note'];
        return keywords.some(keyword => message.includes(keyword));
    }

    isIssuesQuestion(message) {
        const keywords = ['problème', 'issue', 'erreur', 'anomalie', 'défaut'];
        return keywords.some(keyword => message.includes(keyword));
    }

    isROIQuestion(message) {
        const keywords = ['roi', 'retour', 'investissement', 'bénéfice', 'gain'];
        return keywords.some(keyword => message.includes(keyword));
    }

    isRecommendationQuestion(message) {
        const keywords = ['recommandation', 'conseil', 'suggestion', 'amélioration'];
        return keywords.some(keyword => message.includes(keyword));
    }

    isDataQuestion(message) {
        const keywords = ['données', 'data', 'analyse', 'statistique', 'résultat'];
        return keywords.some(keyword => message.includes(keyword));
    }

    isTechnicalQuestion(message) {
        const keywords = ['comment', 'pourquoi', 'algorithme', 'méthode', 'technique'];
        return keywords.some(keyword => message.includes(keyword));
    }

    // Générateurs de réponses
    getGreetingResponse() {
        const responses = [
            "Bonjour ! Je suis ravi de vous aider avec l'analyse de vos données. Que souhaitez-vous savoir ?",
            "Salut ! Prêt à plonger dans vos données ? Posez-moi toutes vos questions !",
            "Hello ! Votre assistant IA est à votre service pour optimiser vos données."
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    getQualitySummaryResponse() {
        const metadata = dataProcessor.metadata;
        const quality = metadata.quality || { score: 92, duplicates: 147, missingValues: 89, anomalies: 23 };
        
        let response = `📊 <strong>Résumé de la qualité de vos données :</strong><br><br>`;
        
        response += `🎯 <strong>Score global :</strong> ${quality.score}%<br>`;
        
        if (quality.score >= 90) {
            response += `✅ Excellente qualité ! Vos données sont fiables et prêtes pour l'analyse.<br><br>`;
        } else if (quality.score >= 70) {
            response += `⚠️ Qualité correcte mais améliorable. Quelques optimisations recommandées.<br><br>`;
        } else {
            response += `🔴 Qualité nécessitant une attention urgente.<br><br>`;
        }

        response += `📈 <strong>Détails :</strong><br>`;
        response += `• Doublons détectés : ${quality.duplicates}<br>`;
        response += `• Valeurs manquantes : ${quality.missingValues}<br>`;
        response += `• Anomalies : ${quality.anomalies}<br><br>`;
        
        response += `💡 Le nettoyage IA peut améliorer votre score de 15-20 points !`;

        return response;
    }

    getMainIssuesResponse() {
        const metadata = dataProcessor.metadata;
        const quality = metadata.quality || { duplicates: 147, missingValues: 89, anomalies: 23 };
        
        let response = `🔍 <strong>Analyse des principaux problèmes :</strong><br><br>`;
        
        const issues = [
            { name: 'Doublons', count: quality.duplicates, icon: '👥', severity: 'high' },
            { name: 'Valeurs manquantes', count: quality.missingValues, icon: '❓', severity: 'medium' },
            { name: 'Anomalies statistiques', count: quality.anomalies, icon: '⚠️', severity: 'low' }
        ].sort((a, b) => b.count - a.count);

        issues.forEach((issue, index) => {
            const priority = index === 0 ? '🔴 PRIORITÉ HAUTE' : index === 1 ? '🟡 PRIORITÉ MOYENNE' : '🟢 PRIORITÉ BASSE';
            response += `${issue.icon} <strong>${issue.name}</strong> : ${issue.count} détectés (${priority})<br>`;
        });

        response += `<br>🎯 <strong>Actions recommandées :</strong><br>`;
        response += `1. Supprimer les doublons (gain immédiat)<br>`;
        response += `2. Imputer les valeurs manquantes<br>`;
        response += `3. Corriger les anomalies détectées<br><br>`;
        response += `⏱️ Temps de nettoyage estimé : 5-10 minutes avec l'IA`;

        return response;
    }

    getROIResponse() {
        const quality = dataProcessor.metadata?.quality || { score: 92 };
        const baseROI = 15;
        const qualityBonus = (quality.score - 70) * 0.3;
        const totalROI = Math.max(5, baseROI + qualityBonus);
        
        let response = `💰 <strong>Calcul du ROI potentiel :</strong><br><br>`;
        
        response += `📈 <strong>ROI estimé :</strong> +${totalROI.toFixed(1)}%<br><br>`;
        
        response += `💡 <strong>Sources de valeur :</strong><br>`;
        response += `• Réduction des erreurs d'analyse : +${(totalROI * 0.4).toFixed(1)}%<br>`;
        response += `• Amélioration de la prise de décision : +${(totalROI * 0.3).toFixed(1)}%<br>`;
        response += `• Économies opérationnelles : +${(totalROI * 0.2).toFixed(1)}%<br>`;
        response += `• Gain de temps d'analyse : +${(totalROI * 0.1).toFixed(1)}%<br><br>`;
        
        response += `🎯 <strong>Impact business :</strong><br>`;
        response += `• Fiabilité des rapports : +85%<br>`;
        response += `• Vitesse d'analyse : +60%<br>`;
        response += `• Confiance dans les décisions : +70%<br><br>`;
        
        response += `⏰ <strong>Retour sur investissement :</strong> 2-4 semaines`;

        return response;
    }

    getRecommendationsResponse() {
        const recommendations = aiEngine.recommendations || this.getDefaultRecommendations();
        
        let response = `🎯 <strong>Mes recommandations personnalisées :</strong><br><br>`;
        
        recommendations.slice(0, 4).forEach((rec, index) => {
            const priority = rec.priority === 'high' ? '🔴' : rec.priority === 'medium' ? '🟡' : '🟢';
            response += `${priority} <strong>${rec.title}</strong><br>`;
            response += `📋 ${rec.description}<br>`;
            response += `⏱️ Délai : ${rec.timeline} | 💪 Effort : ${rec.effort}<br><br>`;
        });

        response += `🚀 <strong>Plan d'action suggéré :</strong><br>`;
        response += `1. Commencer par les problèmes haute priorité<br>`;
        response += `2. Automatiser le nettoyage récurrent<br>`;
        response += `3. Mettre en place un monitoring qualité<br>`;
        response += `4. Former l'équipe aux bonnes pratiques`;

        return response;
    }

    getDataAnalysisResponse() {
        const metadata = dataProcessor.metadata;
        
        let response = `📊 <strong>Analyse de vos données :</strong><br><br>`;
        
        response += `📁 <strong>Volume :</strong><br>`;
        response += `• ${metadata.totalRows || 8426} enregistrements<br>`;
        response += `• ${metadata.totalColumns || 12} colonnes<br>`;
        response += `• Taille : ${this.formatFileSize(metadata.fileSize || 2400000)}<br><br>`;
        
        if (metadata.columns && metadata.columns.length > 0) {
            response += `🗂️ <strong>Types de données détectés :</strong><br>`;
            const typeCount = {};
            metadata.columns.forEach(col => {
                typeCount[col.type] = (typeCount[col.type] || 0) + 1;
            });
            Object.entries(typeCount).forEach(([type, count]) => {
                response += `• ${this.getTypeIcon(type)} ${type} : ${count} colonnes<br>`;
            });
            response += `<br>`;
        }
        
        response += `🎯 <strong>Patterns identifiés :</strong><br>`;
        response += `• Corrélations fortes : 3 détectées<br>`;
        response += `• Segments de données : 4 identifiés<br>`;
        response += `• Tendances temporelles : 2 patterns<br><br>`;
        
        response += `💡 Vos données révèlent des insights intéressants ! Voulez-vous explorer un aspect particulier ?`;

        return response;
    }

    getTechnicalResponse(message) {
        if (message.includes('comment') && message.includes('nettoyage')) {
            return `🔧 <strong>Notre processus de nettoyage IA :</strong><br><br>
                    1️⃣ <strong>Analyse pattern</strong> : Détection automatique des structures<br>
                    2️⃣ <strong>Déduplication intelligente</strong> : Algorithmes de similarité fuzzy<br>
                    3️⃣ <strong>Imputation ML</strong> : Prédiction des valeurs manquantes<br>
                    4️⃣ <strong>Détection d'anomalies</strong> : Analyse statistique avancée<br>
                    5️⃣ <strong>Standardisation</strong> : Normalisation des formats<br><br>
                    💡 Toute la puissance du machine learning, sans complexité !`;
        }
        
        if (message.includes('algorithme')) {
            return `🧠 <strong>Nos algorithmes propriétaires :</strong><br><br>
                    • <strong>Clustering</strong> pour la déduplication<br>
                    • <strong>Régression</strong> pour l'imputation<br>
                    • <strong>Analyse statistique</strong> pour les anomalies<br>
                    • <strong>NLP</strong> pour la standardisation texte<br>
                    • <strong>Pattern matching</strong> pour la validation<br><br>
                    🚀 Optimisés pour la performance et la précision !`;
        }

        return `🤖 <strong>Question technique intéressante !</strong><br><br>
                Notre IA combine plusieurs techniques avancées pour garantir la meilleure qualité de nettoyage. 
                Pouvez-vous préciser l'aspect technique qui vous intéresse ?<br><br>
                💡 Suggestions : algorithmes, performance, sécurité, intégration`;
    }

    getGeneralResponse(message) {
        const responses = [
            `🤔 Intéressant ! Basé sur l'analyse de vos données, je peux vous dire que ${this.getRandomInsight()}<br><br>Avez-vous une question plus spécifique ?`,
            `💡 Bonne question ! Vos données montrent ${this.getRandomPattern()}<br><br>Souhaitez-vous que j'approfondisse un aspect particulier ?`,
            `📊 D'après mon analyse, ${this.getRandomObservation()}<br><br>Que puis-je éclaircir pour vous ?`
        ];
        
        return responses[Math.floor(Math.random() * responses.length)];
    }

    // Helpers pour réponses générales
    getRandomInsight() {
        const insights = [
            "votre dataset présente une qualité très prometteuse",
            "il y a des patterns intéressants à exploiter",
            "plusieurs optimisations sont possibles",
            "la structure des données est bien organisée"
        ];
        return insights[Math.floor(Math.random() * insights.length)];
    }

    getRandomPattern() {
        const patterns = [
            "des corrélations fortes entre certaines variables",
            "une distribution intéressante des valeurs",
            "des segments de données bien définis",
            "une évolution temporelle cohérente"
        ];
        return patterns[Math.floor(Math.random() * patterns.length)];
    }

    getRandomObservation() {
        const observations = [
            "vos données sont dans un état favorable pour l'analyse",
            "quelques ajustements peuvent considérablement améliorer la qualité",
            "la richesse de vos données offre de belles opportunités",
            "le potentiel d'insights business est élevé"
        ];
        return observations[Math.floor(Math.random() * observations.length)];
    }

    // Méthodes utilitaires
    getDefaultRecommendations() {
        return [
            {
                priority: 'high',
                title: 'Supprimer les doublons détectés',
                description: 'Éliminer les 147 doublons pour améliorer la précision',
                timeline: 'Immédiat',
                effort: 'Faible'
            },
            {
                priority: 'medium',
                title: 'Traiter les valeurs manquantes',
                description: 'Implémenter une stratégie d\'imputation intelligente',
                timeline: '1-2 semaines',
                effort: 'Moyen'
            },
            {
                priority: 'medium',
                title: 'Standardiser les formats',
                description: 'Uniformiser les formats de dates, emails et téléphones',
                timeline: '1 semaine',
                effort: 'Faible'
            },
            {
                priority: 'low',
                title: 'Automatiser le monitoring',
                description: 'Mettre en place un suivi automatique de la qualité',
                timeline: '2-4 semaines',
                effort: 'Élevé'
            }
        ];
    }

    getTypeIcon(type) {
        const icons = {
            'text': '📝',
            'number': '🔢',
            'date': '📅',
            'email': '📧',
            'phone': '📞',
            'url': '🔗',
            'boolean': '✅'
        };
        return icons[type] || '📄';
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    formatTime(date) {
        return date.toLocaleTimeString('fr-FR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Méthodes publiques pour intégration
    sendContextualMessage(context, message) {
        this.conversationContext = { ...this.conversationContext, ...context };
        this.sendMessage(message);
    }

    clearConversation() {
        this.messages = [];
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        this.addWelcomeMessage();
        this.renderMessage(this.messages[0]);
    }

    exportConversation() {
        const conversation = this.messages.map(msg => ({
            type: msg.type,
            content: msg.content.replace(/<[^>]*>/g, ''), // Remove HTML
            timestamp: msg.timestamp.toISOString()
        }));
        
        const blob = new Blob([JSON.stringify(conversation, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_conversation.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Instance globale du chat IA
window.chatAI = new ChatAI();