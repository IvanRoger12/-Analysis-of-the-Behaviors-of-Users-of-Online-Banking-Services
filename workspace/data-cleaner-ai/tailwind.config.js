/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Orbitron', 'ui-sans-serif', 'system-ui'],
        mono: ['Share Tech Mono', 'ui-monospace', 'SFMono-Regular'],
      },
    },
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: ['cyberpunk', 'dark', 'business'],
    darkTheme: 'cyberpunk',
  },
};
