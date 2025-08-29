import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  fr: {
    translation: {
      upload: 'Téléversement',
      preview: 'Aperçu',
      dashboard: 'Tableau de bord',
      insights: 'Insights',
      chat: 'Chat',
      export: 'Export',
    }
  },
  en: {
    translation: {
      upload: 'Upload',
      preview: 'Preview',
      dashboard: 'Dashboard',
      insights: 'Insights',
      chat: 'Chat',
      export: 'Export',
    }
  }
}

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: 'fr',
    fallbackLng: 'en',
    interpolation: { escapeValue: false }
  })

export default i18n

